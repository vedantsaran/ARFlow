"""Video compression utilities for ARFlow RGB streaming.

This module provides FFmpeg-based compression for RGB data streams to reduce
bandwidth usage while maintaining visual quality for AR applications.
"""

import logging
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

# Optional imports with proper fallbacks
try:
    import numpy as np  # type: ignore
    import numpy.typing as npt  # type: ignore

    NUMPY_AVAILABLE = True  # type: ignore
except ImportError:
    NUMPY_AVAILABLE = False  # type: ignore
    np = None  # type: ignore
    npt = None  # type: ignore

try:
    import cv2  # type: ignore

    OPENCV_AVAILABLE = True  # type: ignore
except ImportError:
    OPENCV_AVAILABLE = False  # type: ignore
    cv2 = None  # type: ignore

logger = logging.getLogger(__name__)


class FFmpegVideoCompressor:
    """FFmpeg-based video compressor for RGB frame sequences.

    This compressor creates temporal video streams from RGB frames using H.264
    encoding to significantly reduce bandwidth while maintaining visual quality.
    """

    def __init__(
        self,
        width: int,
        height: int,
        fps: int = 30,
        bitrate: str = "2M",
        preset: str = "ultrafast",
        output_dir: Optional[str] = None,
    ):
        """Initialize the video compressor.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
            bitrate: Target bitrate (e.g., "2M" for 2 Mbps)
            preset: FFmpeg encoding preset (ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow)
            output_dir: Directory for temporary files (uses system temp if None)
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.bitrate = bitrate
        self.preset = preset
        self.output_dir = (
            Path(output_dir) if output_dir else Path(tempfile.gettempdir())
        )

        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Track active compression sessions
        self._active_sessions: Dict[str, "VideoSession"] = {}
        self._session_lock = threading.Lock()

        # Verify FFmpeg availability
        self._verify_ffmpeg()

        logger.info(
            "Initialized FFmpeg compressor: %dx%d @ %dfps, bitrate=%s, preset=%s",
            width,
            height,
            fps,
            bitrate,
            preset,
        )

    def _verify_ffmpeg(self) -> None:
        """Verify that FFmpeg is available on the system."""
        try:
            result = subprocess.run(
                ["ffmpeg", "-version"], capture_output=True, text=True, check=True
            )
            logger.info("FFmpeg available: %s", result.stdout.split("\n")[0])
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            raise RuntimeError(
                "FFmpeg not found. Please install FFmpeg to use video compression. "
                "Visit https://ffmpeg.org/download.html for installation instructions."
            ) from e

    def compress_frame_sequence(
        self,
        frames: List[Any],  # Use Any to support both numpy arrays and other formats
        session_id: str,
        device_id: str,
    ) -> Optional[bytes]:
        """Compress a sequence of RGB frames into a video segment.

        Args:
            frames: List of RGB frames as numpy arrays (H, W, 3) or (H, W, 4)
            session_id: Unique session identifier
            device_id: Device identifier

        Returns:
            Compressed video data as bytes, or None if compression failed
        """
        if not frames:
            logger.warning("No frames provided for compression")
            return None

        session_key = f"{session_id}_{device_id}"

        with self._session_lock:
            if session_key not in self._active_sessions:
                self._active_sessions[session_key] = VideoSession(
                    session_id, device_id, self.output_dir
                )
            session = self._active_sessions[session_key]

        return session.compress_frames(frames, self.fps, self.bitrate, self.preset)

    def cleanup_session(self, session_id: str, device_id: str) -> None:
        """Clean up resources for a specific session.

        Args:
            session_id: Session identifier
            device_id: Device identifier
        """
        session_key = f"{session_id}_{device_id}"

        with self._session_lock:
            if session_key in self._active_sessions:
                self._active_sessions[session_key].cleanup()
                del self._active_sessions[session_key]
                logger.info("Cleaned up compression session: %s", session_key)

    def cleanup_all_sessions(self) -> None:
        """Clean up all active compression sessions."""
        with self._session_lock:
            for session in self._active_sessions.values():
                session.cleanup()
            self._active_sessions.clear()
            logger.info("Cleaned up all compression sessions")


class VideoSession:
    """Manages compression for a single video session."""

    def __init__(self, session_id: str, device_id: str, output_dir: Path):
        self.session_id = session_id
        self.device_id = device_id
        self.output_dir = output_dir
        self.frame_count = 0
        self.temp_files: List[Path] = []

        # Create session-specific directory
        self.session_dir = output_dir / f"arflow_compression_{session_id}_{device_id}"
        self.session_dir.mkdir(parents=True, exist_ok=True)

        logger.debug("Created video session: %s", self.session_dir)

    def compress_frames(
        self,
        frames: List[Any],  # Use Any to support both numpy arrays and other formats
        fps: int,
        bitrate: str,
        preset: str,
    ) -> Optional[bytes]:
        """Compress frames using FFmpeg.

        Args:
            frames: RGB frames to compress
            fps: Target framerate
            bitrate: Target bitrate
            preset: Encoding preset

        Returns:
            Compressed video bytes or None if failed
        """
        if not frames:
            return None

        # Generate unique filenames
        timestamp = int(time.time() * 1000)
        input_pattern = self.session_dir / f"frame_%06d_{timestamp}.png"
        output_file = self.session_dir / f"compressed_{timestamp}.mp4"

        saved_frames: List[Path] = []

        try:
            # Save frames as individual images
            saved_frames = self._save_frames_as_images(frames, input_pattern)
            if not saved_frames:
                return None

            # Run FFmpeg compression
            compressed_data = self._run_ffmpeg_compression(
                saved_frames[0].parent / f"frame_%06d_{timestamp}.png",
                output_file,
                fps,
                bitrate,
                preset,
                len(frames),
            )

            return compressed_data

        except Exception as e:
            logger.error("Frame compression failed: %s", e)
            return None
        finally:
            # Clean up temporary files
            self._cleanup_temp_files([*saved_frames, output_file])

    def _save_frames_as_images(
        self,
        frames: List[Any],
        pattern: Path,
    ) -> List[Path]:
        """Save numpy arrays as image files for FFmpeg input."""
        if not OPENCV_AVAILABLE:
            logger.error("OpenCV required for image saving in compression")
            return []

        if not NUMPY_AVAILABLE:
            logger.error("NumPy required for frame processing")
            return []

        saved_files: List[Path] = []
        pattern_str = str(pattern)

        for i, frame in enumerate(frames):
            try:
                # Ensure frame is a numpy array
                if not hasattr(frame, "shape"):
                    logger.error("Frame %d is not a valid numpy array", i)
                    self._cleanup_temp_files(saved_files)
                    return []

                # Convert RGBA to RGB if necessary
                if len(frame.shape) == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]

                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Generate filename
                filename = pattern_str % (i + 1)
                filepath = Path(filename)

                # Save image
                if cv2.imwrite(str(filepath), frame_bgr):
                    saved_files.append(filepath)
                    self.temp_files.append(filepath)
                else:
                    logger.error("Failed to save frame: %s", filepath)
                    # Clean up partially saved files
                    self._cleanup_temp_files(saved_files)
                    return []

            except Exception as e:
                logger.error("Error processing frame %d: %s", i, e)
                # Clean up partially saved files
                self._cleanup_temp_files(saved_files)
                return []

        return saved_files

    def _run_ffmpeg_compression(
        self,
        input_pattern: Path,
        output_file: Path,
        fps: int,
        bitrate: str,
        preset: str,
        frame_count: int,
    ) -> Optional[bytes]:
        """Run FFmpeg to compress the frame sequence."""
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file
            "-r",
            str(fps),  # Input framerate
            "-i",
            str(input_pattern),  # Input pattern
            "-c:v",
            "libx264",  # Video codec
            "-preset",
            preset,  # Encoding preset
            "-b:v",
            bitrate,  # Target bitrate
            "-pix_fmt",
            "yuv420p",  # Pixel format for compatibility
            "-vframes",
            str(frame_count),  # Number of frames
            str(output_file),
        ]

        try:
            # Run FFmpeg
            subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=30,  # 30 second timeout
            )

            # Read compressed data
            if output_file.exists():
                with open(output_file, "rb") as f:
                    compressed_data = f.read()

                # Log compression info
                if compressed_data:
                    compressed_size = len(compressed_data)
                    # Estimate original size (assuming RGB24 format for calculation)
                    estimated_original_size = frame_count * 640 * 480 * 3
                    compression_ratio = estimated_original_size / max(
                        compressed_size, 1
                    )

                    logger.info(
                        "Compression successful: %d frames, %d bytes (est. ratio: %.2f:1)",
                        frame_count,
                        compressed_size,
                        compression_ratio,
                    )

                return compressed_data
            else:
                logger.error("FFmpeg output file not created")
                return None

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg compression timed out")
            return None
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else "Unknown error"
            logger.error("FFmpeg compression failed: %s", error_msg)
            return None
        except Exception as e:
            logger.error("Unexpected error during compression: %s", e)
            return None

    def _cleanup_temp_files(self, files: List[Path]) -> None:
        """Remove temporary files."""
        for file in files:
            try:
                if file.exists():
                    file.unlink()
            except Exception as e:
                logger.warning("Failed to remove temp file %s: %s", file, e)

    def cleanup(self) -> None:
        """Clean up all session resources."""
        # Remove all temporary files
        self._cleanup_temp_files(self.temp_files)

        # Remove session directory if empty
        try:
            if self.session_dir.exists():
                # Remove any remaining files
                for file in self.session_dir.iterdir():
                    if file.is_file():
                        file.unlink()
                # Only remove directory if it's empty
                try:
                    self.session_dir.rmdir()
                except OSError:
                    # Directory not empty, that's ok
                    pass
        except Exception as e:
            logger.warning("Failed to cleanup session directory: %s", e)


def create_compressor_for_format(
    width: int,
    height: int,
    quality: str = "medium",
) -> FFmpegVideoCompressor:
    """Create a video compressor optimized for the given format and quality.

    Args:
        width: Frame width
        height: Frame height
        quality: Quality preset ("low", "medium", "high")

    Returns:
        Configured FFmpegVideoCompressor instance
    """
    quality_settings = {
        "low": {"bitrate": "1M", "preset": "ultrafast"},
        "medium": {"bitrate": "2M", "preset": "fast"},
        "high": {"bitrate": "4M", "preset": "medium"},
    }

    settings = quality_settings.get(quality, quality_settings["medium"])

    return FFmpegVideoCompressor(
        width=width,
        height=height,
        bitrate=settings["bitrate"],
        preset=settings["preset"],
    )
