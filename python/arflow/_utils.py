from collections import defaultdict
from collections.abc import Sequence
from typing import DefaultDict, Tuple

import numpy as np
import numpy.typing as npt

from cakelab.arflow_grpc.v1.color_frame_pb2 import ColorFrame
from cakelab.arflow_grpc.v1.depth_frame_pb2 import DepthFrame
from cakelab.arflow_grpc.v1.xr_cpu_image_pb2 import XRCpuImage


def group_color_frames_by_format_and_dims(
    frames: Sequence[ColorFrame],
) -> DefaultDict[Tuple[XRCpuImage.Format, int, int], list[ColorFrame]]:
    """Group color frames by format and dimensions (width x height)."""
    color_frames_grouped_by_format_and_dims: DefaultDict[
        Tuple[XRCpuImage.Format, int, int], list[ColorFrame]
    ] = defaultdict(list)
    for frame in frames:
        color_frames_grouped_by_format_and_dims[
            (
                frame.image.format,
                frame.image.dimensions.x,
                frame.image.dimensions.y,
            )
        ].append(frame)
    return color_frames_grouped_by_format_and_dims


def group_depth_frames_by_format_dims_and_smoothness(
    frames: Sequence[DepthFrame],
) -> DefaultDict[Tuple[XRCpuImage.Format, int, int, bool], list[DepthFrame]]:
    """Group depth frames by format, dimensions (width x height), and smoothness."""
    depth_frames_grouped_by_format_dims_and_smoothness: DefaultDict[
        Tuple[XRCpuImage.Format, int, int, bool], list[DepthFrame]
    ] = defaultdict(list)
    for frame in frames:
        depth_frames_grouped_by_format_dims_and_smoothness[
            (
                frame.image.format,
                frame.image.dimensions.x,
                frame.image.dimensions.y,
                frame.environment_depth_temporal_smoothing_enabled,
            )
        ].append(frame)
    return depth_frames_grouped_by_format_dims_and_smoothness


def create_rgb_frame_data(
    rgb_array: npt.NDArray[np.uint8], timestamp: float = 0.0
) -> XRCpuImage:
    """Create an XRCpuImage from an RGB numpy array.

    Args:
        rgb_array: RGB image data as numpy array with shape (height, width, 3)
        timestamp: Timestamp for the frame

    Returns:
        XRCpuImage with RGB24 format
    """
    height, width, channels = rgb_array.shape
    if channels != 3:
        raise ValueError("RGB array must have 3 channels")

    from cakelab.arflow_grpc.v1.vector2_int_pb2 import Vector2Int

    return XRCpuImage(
        dimensions=Vector2Int(x=width, y=height),
        format=XRCpuImage.FORMAT_RGB24,
        timestamp=timestamp,
        planes=[
            XRCpuImage.Plane(
                row_stride=width * 3,
                pixel_stride=3,
                data=rgb_array.tobytes(),
            )
        ],
    )


def create_rgba_frame_data(
    rgba_array: npt.NDArray[np.uint8], timestamp: float = 0.0
) -> XRCpuImage:
    """Create an XRCpuImage from an RGBA numpy array.

    Args:
        rgba_array: RGBA image data as numpy array with shape (height, width, 4)
        timestamp: Timestamp for the frame

    Returns:
        XRCpuImage with RGBA32 format
    """
    height, width, channels = rgba_array.shape
    if channels != 4:
        raise ValueError("RGBA array must have 4 channels")

    from cakelab.arflow_grpc.v1.vector2_int_pb2 import Vector2Int

    return XRCpuImage(
        dimensions=Vector2Int(x=width, y=height),
        format=XRCpuImage.FORMAT_RGBA32,
        timestamp=timestamp,
        planes=[
            XRCpuImage.Plane(
                row_stride=width * 4,
                pixel_stride=4,
                data=rgba_array.tobytes(),
            )
        ],
    )


def is_rgb_format(format: XRCpuImage.Format) -> bool:
    """Check if the given format is an RGB-based format."""
    rgb_formats = {
        XRCpuImage.FORMAT_RGB24,
        XRCpuImage.FORMAT_RGBA32,
        XRCpuImage.FORMAT_BGRA32,
        XRCpuImage.FORMAT_ARGB32,
    }
    return format in rgb_formats


def is_yuv_format(format: XRCpuImage.Format) -> bool:
    """Check if the given format is a YUV-based format."""
    yuv_formats = {
        XRCpuImage.FORMAT_ANDROID_YUV_420_888,
        XRCpuImage.FORMAT_IOS_YP_CBCR_420_8BI_PLANAR_FULL_RANGE,
    }
    return format in yuv_formats
