"""Session helps participating devices stream to the same Rerun recording."""

import logging
from collections.abc import Sequence
from typing import Optional

import DracoPy
import numpy as np
import numpy.typing as npt
import rerun as rr

from arflow._types import (
    ARFrameType,
    Timeline,
)
from arflow._utils import (
    group_color_frames_by_format_and_dims,
    group_depth_frames_by_format_dims_and_smoothness,
)
from cakelab.arflow_grpc.v1.ar_trackable_pb2 import ARTrackable
from cakelab.arflow_grpc.v1.audio_frame_pb2 import AudioFrame
from cakelab.arflow_grpc.v1.color_frame_pb2 import ColorFrame
from cakelab.arflow_grpc.v1.depth_frame_pb2 import DepthFrame
from cakelab.arflow_grpc.v1.device_pb2 import Device
from cakelab.arflow_grpc.v1.gyroscope_frame_pb2 import GyroscopeFrame
from cakelab.arflow_grpc.v1.mesh_detection_frame_pb2 import MeshDetectionFrame
from cakelab.arflow_grpc.v1.plane_detection_frame_pb2 import PlaneDetectionFrame
from cakelab.arflow_grpc.v1.point_cloud_detection_frame_pb2 import (
    PointCloudDetectionFrame,
)
from cakelab.arflow_grpc.v1.session_pb2 import Session
from cakelab.arflow_grpc.v1.transform_frame_pb2 import TransformFrame
from cakelab.arflow_grpc.v1.vector2_pb2 import Vector2
from cakelab.arflow_grpc.v1.vector3_pb2 import Vector3
from cakelab.arflow_grpc.v1.xr_cpu_image_pb2 import XRCpuImage

# Import compression module with fallback
try:
    from arflow._compression import FFmpegVideoCompressor, create_compressor_for_format

    COMPRESSION_AVAILABLE = True
except ImportError:
    FFmpegVideoCompressor = None
    create_compressor_for_format = None
    COMPRESSION_AVAILABLE = False

logger = logging.getLogger(__name__)
y_down_to_y_up = np.array(
    [
        [1.0, -0.0, 0.0, 0],
        [0.0, -1.0, 0.0, 0],
        [0.0, 0.0, 1.0, 0],
        [0.0, 0.0, 0, 1.0],
    ],
    dtype=np.float32,
)


class SessionStream:
    """All devices in a session share a stream."""

    def __init__(
        self,
        info: Session,
        stream: rr.RecordingStream,
        enable_compression: bool = False,
        compression_quality: str = "medium",
    ):
        self.info = info
        """Session information."""
        self.stream = stream
        """Stream handle to the Rerun recording associated with this session."""

        # Video compression settings
        self.enable_compression = enable_compression and COMPRESSION_AVAILABLE
        self.compression_quality = compression_quality
        self._compressors = {}  # Cache compressors by resolution
        self._frame_buffers = {}  # Buffer frames for batch compression
        self._compression_batch_size = (
            30  # Compress every 30 frames (1 second at 30fps)
        )

        if self.enable_compression:
            if not COMPRESSION_AVAILABLE:
                logger.warning(
                    "Compression requested but not available - FFmpeg or OpenCV missing"
                )
            else:
                logger.info(
                    "RGB compression enabled with quality: %s", compression_quality
                )
        else:
            logger.info("RGB compression disabled")

    def save_transform_frames(
        self,
        frames: Sequence[TransformFrame],
        device: Device,
    ):
        if len(frames) == 0:
            logger.warning("No transform frames to save.")
            return

        entity_path = rr.new_entity_path(
            [
                f"{self.info.metadata.name}_{self.info.id.value}",
                f"{device.model}_{device.name}_{device.uid}",
                ARFrameType.TRANSFORM_FRAME,
            ]
        )
        rr.log(
            entity_path,
            [rr.Transform3D.indicator()],
            static=True,
            recording=self.stream,
        )
        t = np.array([np.frombuffer(frame.data, dtype=np.float32) for frame in frames])
        transforms = np.array([np.eye(4, dtype=np.float32) for _ in range(len(frames))])
        transforms[:, :3, :] = t.reshape((len(frames), 3, 4))

        # TODO: Do we need to flip Y?
        transforms = y_down_to_y_up @ transforms
        rr.send_columns(
            entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=[
                        f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9
                        for f in frames
                    ],
                ),
            ],
            components=[
                rr.components.TransformMat3x3Batch(
                    data=[transform[:3, :3] for transform in transforms]
                ),
                rr.components.Translation3DBatch(
                    data=[transform[:3, 3] for transform in transforms]
                ),
            ],
            # TODO: Remove when this stabilizes. See https://github.com/rerun-io/rerun/issues/8167
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )

    def save_color_frames(
        self,
        frames: Sequence[ColorFrame],
        device: Device,
    ):
        """Assumes that the device is in the session and all frames have the same format, width, height, and originating device.

        @private
        """
        if len(frames) == 0:
            logger.warning("No color frames to save.")
            return
        grouped_frames = group_color_frames_by_format_and_dims(frames)
        for (format, width, height), homogenous_frames in grouped_frames.items():
            if len(homogenous_frames) == 0:
                continue

            entity_path = rr.new_entity_path(
                [
                    f"{self.info.metadata.name}_{self.info.id.value}",
                    f"{device.model}_{device.name}_{device.uid}",
                    ARFrameType.COLOR_FRAME,
                    f"{width}x{height}",
                ]
            )
            intrinsics_entity_path = rr.new_entity_path(
                [
                    f"{self.info.metadata.name}_{self.info.id.value}",
                    f"{device.model}_{device.name}_{device.uid}",
                    ARFrameType.COLOR_FRAME,
                    f"{homogenous_frames[0].intrinsics.resolution.x}x{homogenous_frames[0].intrinsics.resolution.y}",
                ]
            )

            # Check if compression is enabled for RGB formats
            should_compress = (
                self.enable_compression
                and format
                in [
                    XRCpuImage.FORMAT_RGB24,
                    XRCpuImage.FORMAT_RGBA32,
                    XRCpuImage.FORMAT_BGRA32,
                    XRCpuImage.FORMAT_ARGB32,
                ]
                and len(homogenous_frames)
                > 1  # Only compress if we have multiple frames
            )

            if format == XRCpuImage.FORMAT_ANDROID_YUV_420_888:
                format_static = rr.components.ImageFormat(
                    width=width,
                    height=height,
                    pixel_format=rr.PixelFormat.Y_U_V12_LimitedRange,
                )
                data = np.array([_to_i420_format(f.image) for f in homogenous_frames])
            elif format == XRCpuImage.FORMAT_RGB24:
                format_static = rr.components.ImageFormat(
                    width=width,
                    height=height,
                    pixel_format=rr.PixelFormat.RGB,
                )
                # Assume each frame's image.planes[0].data is RGB packed
                raw_frames = [
                    np.frombuffer(f.image.planes[0].data, dtype=np.uint8).reshape(
                        (height, width, 3)
                    )
                    for f in homogenous_frames
                ]

                # Apply compression if enabled
                if should_compress:
                    compressed_data = self._compress_rgb_frames(
                        raw_frames,
                        width,
                        height,
                        f"{self.info.id.value}",
                        f"{device.uid}",
                    )
                    if compressed_data:
                        # Store compressed data for potential transmission
                        logger.info(
                            "RGB24 frames compressed: %d frames -> %d bytes",
                            len(raw_frames),
                            len(compressed_data),
                        )

                data = np.array(raw_frames)
            elif format == XRCpuImage.FORMAT_RGBA32:
                format_static = rr.components.ImageFormat(
                    width=width,
                    height=height,
                    pixel_format=rr.PixelFormat.RGBA,
                )
                data = np.array(
                    [
                        np.frombuffer(f.image.planes[0].data, dtype=np.uint8).reshape(
                            (height, width, 4)
                        )
                        for f in homogenous_frames
                    ]
                )
            elif format == XRCpuImage.FORMAT_BGRA32:
                format_static = rr.components.ImageFormat(
                    width=width,
                    height=height,
                    pixel_format=rr.PixelFormat.RGBA,
                )
                # Convert BGRA to RGBA for rerun
                data = np.array(
                    [
                        _bgra_to_rgba(
                            np.frombuffer(
                                f.image.planes[0].data, dtype=np.uint8
                            ).reshape((height, width, 4))
                        )
                        for f in homogenous_frames
                    ]
                )
            elif format == XRCpuImage.FORMAT_ARGB32:
                format_static = rr.components.ImageFormat(
                    width=width,
                    height=height,
                    pixel_format=rr.PixelFormat.RGBA,
                )
                # Convert ARGB to RGBA for rerun
                data = np.array(
                    [
                        _argb_to_rgba(
                            np.frombuffer(
                                f.image.planes[0].data, dtype=np.uint8
                            ).reshape((height, width, 4))
                        )
                        for f in homogenous_frames
                    ]
                )
            elif format == XRCpuImage.FORMAT_IOS_YP_CBCR_420_8BI_PLANAR_FULL_RANGE:
                format_static = rr.components.ImageFormat(
                    width=width,
                    height=height,
                    pixel_format=rr.PixelFormat.NV12,
                )
                data = np.array([f.image.planes[0].data for f in homogenous_frames])
            else:
                logger.warning(f"Unsupported color frame format: {format}")
                continue

            rr.log(
                intrinsics_entity_path,
                [rr.Pinhole.indicator()],
                static=True,
                recording=self.stream,
            )
            rr.send_columns(
                intrinsics_entity_path,
                times=[
                    rr.TimeSecondsColumn(
                        timeline=Timeline.DEVICE,
                        times=[
                            f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9
                            for f in homogenous_frames
                        ],
                    ),
                ],
                components=[
                    rr.components.PinholeProjectionBatch(
                        data=[
                            np.array(
                                [
                                    [
                                        f.intrinsics.focal_length.x,
                                        0,
                                        f.intrinsics.principal_point.x,
                                    ],
                                    [
                                        0,
                                        f.intrinsics.focal_length.y,
                                        f.intrinsics.principal_point.y,
                                    ],
                                    [0, 0, 1],
                                ],
                                dtype=np.float32,
                            )
                            for f in homogenous_frames
                        ]
                    )
                ],
                recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
            )
            rr.log(
                entity_path,
                [format_static, rr.Image.indicator()],
                static=True,
                recording=self.stream,
            )
            rr.send_columns(
                entity_path,
                times=[
                    rr.TimeSecondsColumn(
                        timeline=Timeline.DEVICE,
                        times=[
                            f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9
                            for f in homogenous_frames
                        ],
                    ),
                    rr.TimeSecondsColumn(
                        timeline=Timeline.IMAGE,
                        times=[f.image.timestamp for f in homogenous_frames],
                    ),
                ],
                components=[
                    rr.components.ImageBufferBatch(
                        data=data,
                        strict=True,
                    ),
                ],
                recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
            )

    def _compress_rgb_frames(
        self,
        frames: list,
        width: int,
        height: int,
        session_id: str,
        device_id: str,
    ) -> Optional[bytes]:
        """Compress RGB frames using FFmpeg if available.

        Args:
            frames: List of RGB frame arrays
            width: Frame width
            height: Frame height
            session_id: Session identifier
            device_id: Device identifier

        Returns:
            Compressed video data as bytes or None if compression failed
        """
        if not self.enable_compression or not COMPRESSION_AVAILABLE:
            return None

        # Get or create compressor for this resolution
        compressor_key = f"{width}x{height}"
        if compressor_key not in self._compressors:
            self._compressors[compressor_key] = create_compressor_for_format(
                width, height, self.compression_quality
            )

        compressor = self._compressors[compressor_key]

        try:
            return compressor.compress_frame_sequence(frames, session_id, device_id)
        except Exception as e:
            logger.error("RGB compression failed: %s", e)
            return None

    def save_depth_frames(
        self,
        frames: Sequence[DepthFrame],
        device: Device,
    ):
        """Assumes that the device is in the session and all frames have the same format, width, height, smoothness, and and originating device."""
        if len(frames) == 0:
            logger.warning("No depth frames to save.")
            return

        grouped_frames = group_depth_frames_by_format_dims_and_smoothness(frames)
        for (
            format,
            width,
            height,
            environment_depth_temporal_smoothing_enabled,
        ), homogenous_frames in grouped_frames.items():
            entity_path = rr.new_entity_path(
                [
                    f"{self.info.metadata.name}_{self.info.id.value}",
                    f"{device.model}_{device.name}_{device.uid}",
                    ARFrameType.DEPTH_FRAME,
                    f"{width}x{height}",
                    "smoothed"
                    if environment_depth_temporal_smoothing_enabled
                    else "raw",
                ]
            )

            if format == XRCpuImage.FORMAT_DEPTHFLOAT32:
                format_static = rr.components.ImageFormat(
                    width=width,
                    height=height,
                    color_model=rr.ColorModel.L,
                    channel_datatype=rr.ChannelDatatype.F32,
                )
            elif format == XRCpuImage.FORMAT_DEPTHUINT16:
                format_static = rr.components.ImageFormat(
                    width=width,
                    height=height,
                    color_model=rr.ColorModel.L,
                    channel_datatype=rr.ChannelDatatype.U16,
                )
            else:
                logger.warning(f"Unsupported depth frame format: {format}")
                continue

            rr.log(
                entity_path,
                [format_static, rr.DepthImage.indicator()],
                [rr.components.DepthMeter(1.0)],
                static=True,
                recording=self.stream,
            )
            rr.send_columns(
                entity_path,
                times=[
                    rr.TimeSecondsColumn(
                        timeline=Timeline.DEVICE,
                        times=[
                            f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9
                            for f in homogenous_frames
                        ],
                    ),
                    rr.TimeSecondsColumn(
                        timeline=Timeline.IMAGE,
                        times=[f.image.timestamp for f in homogenous_frames],
                    ),
                ],
                components=[
                    rr.components.ImageBufferBatch(
                        data=[f.image.planes[0].data for f in homogenous_frames],
                    ),
                ],
                recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
            )

    def save_gyroscope_frames(
        self,
        frames: Sequence[GyroscopeFrame],
        device: Device,
    ):
        if len(frames) == 0:
            return

        entity_path = rr.new_entity_path(
            [
                f"{self.info.metadata.name}_{self.info.id.value}",
                f"{device.model}_{device.name}_{device.uid}",
                ARFrameType.GYROSCOPE_FRAME,
            ]
        )
        device_timestamps = [
            f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9 for f in frames
        ]
        attitude_entity_path = f"{entity_path}/attitude"
        rr.log(
            attitude_entity_path,
            [rr.Boxes3D.indicator()],
            [rr.components.HalfSize3D([0.5, 0.5, 0.5])],
            static=True,
            recording=self.stream,
        )
        rr.send_columns(
            attitude_entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=device_timestamps,
                ),
            ],
            components=[
                rr.components.RotationQuatBatch(
                    data=[
                        [
                            frame.attitude.x,
                            frame.attitude.y,
                            frame.attitude.z,
                            frame.attitude.w,
                        ]
                        for frame in frames
                    ]
                ),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )
        rotation_rate_entity_path = f"{entity_path}/rotation_rate"
        rr.log(
            rotation_rate_entity_path,
            [rr.Arrows3D.indicator()],
            [rr.components.Color([0, 255, 0])],
            static=True,
            recording=self.stream,
        )
        rr.send_columns(
            rotation_rate_entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=device_timestamps,
                ),
            ],
            components=[
                rr.components.Vector3DBatch(
                    data=[
                        [
                            frame.rotation_rate.x,
                            frame.rotation_rate.y,
                            frame.rotation_rate.z,
                        ]
                        for frame in frames
                    ]
                ),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )
        gravity_entity_path = f"{entity_path}/gravity"
        rr.log(
            gravity_entity_path,
            [rr.Arrows3D.indicator()],
            [rr.components.Color([0, 0, 255])],
            static=True,
            recording=self.stream,
        )
        rr.send_columns(
            gravity_entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=device_timestamps,
                ),
            ],
            components=[
                rr.components.Vector3DBatch(
                    data=[
                        [
                            frame.gravity.x,
                            frame.gravity.y,
                            frame.gravity.z,
                        ]
                        for frame in frames
                    ]
                ),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )
        acceleration_entity_path = f"{entity_path}/acceleration"
        rr.log(
            acceleration_entity_path,
            [rr.Arrows3D.indicator()],
            [rr.components.Color([255, 255, 0])],
            static=True,
            recording=self.stream,
        )
        rr.send_columns(
            acceleration_entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=device_timestamps,
                ),
            ],
            components=[
                rr.components.Vector3DBatch(
                    data=[
                        [
                            frame.acceleration.x,
                            frame.acceleration.y,
                            frame.acceleration.z,
                        ]
                        for frame in frames
                    ]
                ),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )

    def save_audio_frames(
        self,
        frames: Sequence[AudioFrame],
        device: Device,
    ):
        if len(frames) == 0:
            logger.warning("No audio frames to save.")
            return

        entity_path = rr.new_entity_path(
            [
                f"{self.info.metadata.name}_{self.info.id.value}",
                f"{device.model}_{device.name}_{device.uid}",
                ARFrameType.AUDIO_FRAME,
            ]
        )
        rr.log(
            entity_path,
            [rr.Scalar.indicator()],
            static=True,
            recording=self.stream,
        )
        rr.send_columns(
            entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=[
                        f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9
                        for f in frames
                    ],
                ),
            ],
            components=[
                rr.components.ScalarBatch(
                    data=[frame.data for frame in frames]
                ).partition([len(frame.data) for frame in frames]),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )

    def save_plane_detection_frames(
        self,
        frames: Sequence[PlaneDetectionFrame],
        device: Device,
    ):
        if len(frames) == 0:
            logger.warning("No plane detection frames to save.")
            return

        entity_path = rr.new_entity_path(
            [
                f"{self.info.metadata.name}_{self.info.id.value}",
                f"{device.model}_{device.name}_{device.uid}",
                ARFrameType.PLANE_DETECTION_FRAME,
            ]
        )
        rr.log(
            entity_path,
            [rr.LineStrips3D.indicator()],
            static=True,
            recording=self.stream,
        )
        positively_changed_frames = list(
            filter(
                lambda f: f.state == PlaneDetectionFrame.STATE_ADDED
                or f.state == PlaneDetectionFrame.STATE_UPDATED
                and len(f.plane.boundary)
                > 0,  # boundary can sometimes 0 points for some reason
                frames,
            )
        )
        rr.send_columns(
            entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=[
                        f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9
                        for f in positively_changed_frames
                    ],
                ),
            ],
            components=[
                # TODO: notice ARTrackable.Pose
                rr.components.LineStrip3DBatch(
                    data=[
                        _convert_2d_to_3d_boundary_points(
                            boundary=f.plane.boundary,
                            normal=f.plane.normal,
                            center=f.plane.center,
                        )
                        for f in positively_changed_frames
                    ]
                ),
                rr.components.EntityPathBatch(
                    data=[
                        f"{f.plane.trackable.trackable_id.sub_id_1}_{f.plane.trackable.trackable_id.sub_id_2}"
                        for f in positively_changed_frames
                    ],
                ),
                rr.components.ColorBatch(
                    data=[
                        [0, 255, 0]  # green
                        if f.plane.trackable.tracking_state
                        == ARTrackable.TRACKING_STATE_TRACKING
                        else [255, 0, 0]  # red
                        for f in positively_changed_frames
                    ],
                ),
                rr.components.TextBatch(
                    data=[
                        ARTrackable.TrackingState.Name(f.plane.trackable.tracking_state)
                        for f in positively_changed_frames
                    ]
                ),
                # rr.components.TextBatch(
                #     data=[
                #         PlaneDetectionFrame.State.Name(f.state)
                #         for f in positively_changed_frames
                #     ]
                # ),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )
        negatively_changed_frames = list(
            filter(lambda f: f.state == PlaneDetectionFrame.STATE_REMOVED, frames)
        )
        rr.send_columns(
            entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=[
                        f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9
                        for f in negatively_changed_frames
                    ],
                ),
            ],
            components=[
                rr.components.EntityPathBatch(
                    data=[
                        f"{f.plane.trackable.trackable_id.sub_id_1}_{f.plane.trackable.trackable_id.sub_id_2}"
                        for f in negatively_changed_frames
                    ]
                ),
                rr.components.ClearIsRecursiveBatch(
                    data=[True for _ in negatively_changed_frames]
                ),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )

    def save_point_cloud_detection_frames(
        self,
        frames: Sequence[PointCloudDetectionFrame],
        device: Device,
    ):
        if len(frames) == 0:
            logger.warning("No point cloud detection frames to save.")
            return

        entity_path = rr.new_entity_path(
            [
                f"{self.info.metadata.name}_{self.info.id.value}",
                f"{device.model}_{device.name}_{device.uid}",
                ARFrameType.POINT_CLOUD_DETECTION_FRAME,
            ]
        )
        rr.log(
            entity_path,
            [rr.Points3D.indicator()],
            static=True,
            recording=self.stream,
        )
        positively_changed_frames = list(
            filter(
                lambda f: f.state == PointCloudDetectionFrame.STATE_ADDED
                or f.state == PointCloudDetectionFrame.STATE_UPDATED,
                frames,
            )
        )
        # for each point cloud
        rr.send_columns(
            entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=[
                        f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9
                        for f in positively_changed_frames
                    ],
                ),
            ],
            components=[
                rr.components.EntityPathBatch(
                    data=[
                        f"{f.point_cloud.trackable.trackable_id.sub_id_1}_{f.point_cloud.trackable.trackable_id.sub_id_2}"
                        for f in positively_changed_frames
                    ]
                ),
                # TODO: notice ARTrackable.Pose
                rr.components.ColorBatch(
                    data=[
                        [0, 255, 0]  # green
                        if f.point_cloud.trackable.tracking_state
                        == ARTrackable.TRACKING_STATE_TRACKING
                        else [255, 0, 0]  # red
                        for f in positively_changed_frames
                    ],
                ),
                rr.components.TextBatch(
                    data=[
                        ARTrackable.TrackingState.Name(
                            f.point_cloud.trackable.tracking_state
                        )
                        for f in positively_changed_frames
                    ]
                ),
                # rr.components.TextBatch(
                #     data=[
                #         PointCloudDetectionFrame.State.Name(f.state)
                #         for f in positively_changed_frames
                #     ]
                # ),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )
        # for each point in the cloud
        rr.send_columns(
            entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=[
                        f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9
                        for f in positively_changed_frames
                        for _ in f.point_cloud.identifiers
                    ],
                ),
            ],
            components=[
                rr.components.EntityPathBatch(
                    data=[
                        rr.new_entity_path(
                            [
                                f"{f.point_cloud.trackable.trackable_id.sub_id_1}_{f.point_cloud.trackable.trackable_id.sub_id_2}",
                                i,
                            ]
                        )
                        for f in positively_changed_frames
                        for i in f.point_cloud.identifiers
                    ]
                ),
                rr.components.Position3DBatch(
                    data=[
                        [
                            p.x,
                            p.y,
                            p.z,
                        ]
                        for f in positively_changed_frames
                        for p in f.point_cloud.positions
                    ]
                ),
                # TODO: Can use AnyBatchValue once this is released https://github.com/rerun-io/rerun/pull/8163.
                # rr.components.TextBatch(
                #     data=[
                #         f.point_cloud.confidence_values
                #         for f in positively_changed_frames
                #     ],
                # ),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )
        negatively_changed_frames = list(
            filter(lambda f: f.state == PointCloudDetectionFrame.STATE_REMOVED, frames)
        )
        rr.send_columns(
            entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=[
                        f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9
                        for f in negatively_changed_frames
                    ],
                ),
            ],
            components=[
                rr.components.EntityPathBatch(
                    data=[
                        f"{f.point_cloud.trackable.trackable_id.sub_id_1}_{f.point_cloud.trackable.trackable_id.sub_id_2}"
                        for f in negatively_changed_frames
                    ]
                ),
                rr.components.ClearIsRecursiveBatch(
                    data=[True for _ in negatively_changed_frames]
                ),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )

    def save_mesh_detection_frames(
        self,
        frames: Sequence[MeshDetectionFrame],
        device: Device,
    ):
        if len(frames) == 0:
            logger.warning("No mesh detection frames to save.")
            return

        entity_path = rr.new_entity_path(
            [
                f"{self.info.metadata.name}_{self.info.id.value}",
                f"{device.model}_{device.name}_{device.uid}",
                ARFrameType.MESH_DETECTION_FRAME,
            ]
        )
        rr.log(
            entity_path,
            [rr.Mesh3D.indicator()],
            static=True,
            recording=self.stream,
        )
        positively_changed_frames = list(
            filter(
                lambda f: f.state == MeshDetectionFrame.STATE_ADDED
                or f.state == MeshDetectionFrame.STATE_UPDATED,
                frames,
            )
        )
        for f in positively_changed_frames:
            rr.set_time_seconds(
                Timeline.DEVICE,
                seconds=f.device_timestamp.seconds + f.device_timestamp.nanos / 1e9,
                recording=self.stream,
            )
            for sub_mesh in f.mesh_filter.mesh.sub_meshes:
                # We are ignoring type because DracoPy is written with Cython, and Pyright cannot infer types from a native module.
                decoded_mesh = DracoPy.decode(sub_mesh.data)  # pyright: ignore [reportUnknownMemberType, reportUnknownVariableType]
                rr.log(
                    f"{entity_path}/{rr.escape_entity_path_part(str(f.mesh_filter.instance_id))}",
                    rr.Mesh3D(
                        vertex_positions=decoded_mesh.points,  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
                        triangle_indices=decoded_mesh.faces,  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
                        vertex_normals=decoded_mesh.normals,  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
                        vertex_colors=decoded_mesh.colors,  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
                        vertex_texcoords=decoded_mesh.tex_coord,  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
                    ),
                    recording=self.stream,
                )
        negatively_changed_frames = filter(
            lambda f: f.state == PointCloudDetectionFrame.STATE_REMOVED, frames
        )
        rr.send_columns(
            entity_path,
            times=[
                rr.TimeSecondsColumn(
                    timeline=Timeline.DEVICE,
                    times=[
                        f.device_timestamp.seconds for f in negatively_changed_frames
                    ],
                ),
            ],
            components=[
                rr.components.EntityPathBatch(
                    data=[
                        rr.new_entity_path([f.mesh_filter.instance_id])
                        for f in negatively_changed_frames
                    ]
                ),
                rr.components.ClearIsRecursiveBatch(
                    data=[True for _ in negatively_changed_frames]
                ),
            ],
            recording=self.stream.to_native(),  # pyright: ignore [reportUnknownMemberType, reportUnknownArgumentType]
        )


# TODO: Performance opportunity for hot path. Can operate on a batch of images at once instead of one at a time.
def _to_i420_format(image: XRCpuImage) -> npt.NDArray[np.uint8]:
    if len(image.planes) != 3:
        logger.warning(
            f"Skipping bad image. Expected 3 planes, got {len(image.planes)}."
        )
        return np.array([], dtype=np.uint8)

    height = image.dimensions.y
    width = image.dimensions.x
    uv_height = height // 2
    uv_width = width // 2
    y_plane, u_plane, v_plane = (
        image.planes[0],
        image.planes[1],
        image.planes[2],
    )
    y_data = (
        np.frombuffer(y_plane.data, dtype=np.uint8)
        .reshape((height, y_plane.row_stride))[:, :width]
        .flatten()
    )
    # Downsample and pack U and V planes
    u_data = (
        # Have to pad an extra byte due to how the Android image format was captured. Check:
        # https://stackoverflow.com/questions/51399908/yuv-420-888-byte-format/62090742
        np.frombuffer(u_plane.data + b"\x00", dtype=np.uint8)
        # pad an extra byte
        .reshape((uv_height, u_plane.row_stride))[
            :,
            : uv_width * u_plane.pixel_stride : u_plane.pixel_stride,
        ]
        .flatten()
    )
    v_data = (
        np.frombuffer(v_plane.data + b"\x00", dtype=np.uint8)
        .reshape((uv_height, v_plane.row_stride))[
            :, : uv_width * v_plane.pixel_stride : v_plane.pixel_stride
        ]
        .flatten()
    )
    return np.concatenate([y_data, u_data, v_data])


def _convert_2d_to_3d_boundary_points(
    boundary: Sequence[Vector2],
    normal: Vector3,
    center: Vector3,
) -> npt.NDArray[np.float32]:
    if len(boundary) == 0:
        logger.warning("Skipping plane with no boundary points.")
        return np.array([], dtype=np.float32)

    normal_as_np = np.array([normal.x, normal.y, normal.z])
    normalized_normal_as_np = normal_as_np / np.linalg.norm(normal_as_np)
    arbitary_vector = (
        np.array([1, 0, 0])
        if not np.allclose(normalized_normal_as_np, [1, 0, 0])
        else np.array([0, 1, 0])
    )
    u = np.cross(normalized_normal_as_np, arbitary_vector)
    u = u / np.linalg.norm(u)
    v = np.cross(normalized_normal_as_np, u)
    center_as_np = np.array([center.x, center.y, center.z])
    boundary_points_3d = np.array(
        [center_as_np + point_2d.x * u + point_2d.y * v for point_2d in boundary]
        # close off boundary
        + [center_as_np + boundary[0].x * u + boundary[0].y * v],
        dtype=np.float32,
    )
    return boundary_points_3d


def _yuv420_to_rgb_cv2(image: XRCpuImage) -> npt.NDArray[np.uint8]:
    try:
        import cv2
    except ImportError:
        logger.warning("OpenCV (cv2) is not installed. Cannot convert YUV to RGB.")
        return np.zeros((image.dimensions.y, image.dimensions.x, 3), dtype=np.uint8)
    height = image.dimensions.y
    width = image.dimensions.x
    yuv = _to_i420_format(image)
    yuv = yuv.reshape((height * 3) // 2, width)
    rgb = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_I420)
    return rgb


def _bgra_to_rgba(bgra_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Convert BGRA image to RGBA format."""
    # BGRA -> RGBA: swap B and R channels
    rgba_image = bgra_image.copy()
    rgba_image[:, :, [0, 2]] = rgba_image[:, :, [2, 0]]  # Swap R and B
    return rgba_image


def _argb_to_rgba(argb_image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Convert ARGB image to RGBA format."""
    # ARGB -> RGBA: move alpha channel from first to last position
    rgba_image = np.zeros_like(argb_image)
    rgba_image[:, :, 0] = argb_image[:, :, 1]  # R
    rgba_image[:, :, 1] = argb_image[:, :, 2]  # G
    rgba_image[:, :, 2] = argb_image[:, :, 3]  # B
    rgba_image[:, :, 3] = argb_image[:, :, 0]  # A
    return rgba_image
