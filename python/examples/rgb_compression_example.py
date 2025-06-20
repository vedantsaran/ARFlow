#!/usr/bin/env python3
"""Example demonstrating RGB compression in ARFlow.

This example shows how to:
1. Enable video compression for RGB streams
2. Configure compression quality settings
3. Stream compressed RGB data to ARFlow server
4. Monitor compression performance and bandwidth savings
"""

import logging
import time
from typing import List

import numpy as np
import numpy.typing as npt

from arflow import ARFlowServicer, ColorFrame, run_server
from arflow._utils import create_rgb_frame_data, create_rgba_frame_data, is_rgb_format
from cakelab.arflow_grpc.v1.device_pb2 import Device
from cakelab.arflow_grpc.v1.intrinsics_pb2 import Intrinsics
from cakelab.arflow_grpc.v1.vector2_pb2 import Vector2
from cakelab.arflow_grpc.v1.vector2_int_pb2 import Vector2Int
from cakelab.arflow_grpc.v1.xr_cpu_image_pb2 import XRCpuImage

logger = logging.getLogger(__name__)


class CompressionDemoServicer(ARFlowServicer):
    """Demo servicer that demonstrates RGB compression capabilities."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.frame_count = 0
        self.total_raw_bytes = 0
        self.total_compressed_bytes = 0
        self.compression_start_time = time.time()
    
    def on_save_color_frames(self, frames, session_stream, device):
        """Process incoming color frames and track compression metrics."""
        super().on_save_color_frames(frames, session_stream, device)
        
        for frame in frames:
            self.frame_count += 1
            format_name = XRCpuImage.Format.Name(frame.image.format)
            is_rgb = is_rgb_format(frame.image.format)
            
            # Calculate raw frame size
            width = frame.image.dimensions.x
            height = frame.image.dimensions.y
            if is_rgb:
                channels = 4 if frame.image.format == XRCpuImage.FORMAT_RGBA32 else 3
                raw_size = width * height * channels
                self.total_raw_bytes += raw_size
                
                logger.info(
                    "Frame %d: %s (%dx%d) - Raw size: %d bytes - Compression: %s",
                    self.frame_count,
                    format_name,
                    width,
                    height,
                    raw_size,
                    "ENABLED" if session_stream.enable_compression else "DISABLED"
                )
                
                # Log compression statistics every 30 frames
                if self.frame_count % 30 == 0:
                    self._log_compression_stats()
    
    def _log_compression_stats(self):
        """Log compression performance statistics."""
        elapsed_time = time.time() - self.compression_start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        logger.info("=== COMPRESSION STATISTICS ===")
        logger.info("Frames processed: %d", self.frame_count)
        logger.info("Processing rate: %.2f FPS", fps)
        logger.info("Total raw data: %.2f MB", self.total_raw_bytes / (1024 * 1024))
        
        if self.total_compressed_bytes > 0:
            compression_ratio = self.total_raw_bytes / self.total_compressed_bytes
            bandwidth_savings = (1 - self.total_compressed_bytes / self.total_raw_bytes) * 100
            
            logger.info("Total compressed data: %.2f MB", self.total_compressed_bytes / (1024 * 1024))
            logger.info("Compression ratio: %.2f:1", compression_ratio)
            logger.info("Bandwidth savings: %.1f%%", bandwidth_savings)
        
        logger.info("==============================")


def create_dynamic_rgb_sequence(width: int = 640, height: int = 480, num_frames: int = 60) -> List[npt.NDArray[np.uint8]]:
    """Create a sequence of RGB frames with dynamic content for compression testing.
    
    This creates frames with varying complexity to test compression effectiveness:
    - Simple gradients (high compression)
    - Complex patterns (low compression)
    - Moving objects (temporal compression)
    """
    frames = []
    
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Time-based animation parameter
        t = i / num_frames * 2 * np.pi
        
        # Create different content types based on frame number
        if i < num_frames // 3:
            # Simple gradient (compresses well)
            for y in range(height):
                for x in range(width):
                    frame[y, x, 0] = int((x / width) * 255)  # Red gradient
                    frame[y, x, 1] = int((y / height) * 255)  # Green gradient
                    frame[y, x, 2] = int(128 + 127 * np.sin(t))  # Animated blue
        
        elif i < 2 * num_frames // 3:
            # Complex pattern (compresses poorly)
            noise = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
            frame = noise
        
        else:
            # Moving circle (tests temporal compression)
            center_x = int(width // 2 + width // 4 * np.cos(t))
            center_y = int(height // 2 + height // 4 * np.sin(t))
            radius = 50
            
            # Create circular pattern
            y_grid, x_grid = np.ogrid[:height, :width]
            mask = (x_grid - center_x)**2 + (y_grid - center_y)**2 <= radius**2
            
            frame[mask] = [255, 100, 100]  # Red circle
            frame[~mask] = [50, 50, 200]   # Blue background
        
        frames.append(frame)
    
    return frames


def simulate_rgb_stream(servicer: CompressionDemoServicer, session_id: str = "test_session", device_uid: str = "test_device"):
    """Simulate an RGB stream by sending frames to the servicer."""
    # Create test intrinsics
    intrinsics = Intrinsics(
        focal_length=Vector2(x=500.0, y=500.0),
        principal_point=Vector2(x=320.0, y=240.0),
        resolution=Vector2Int(x=640, y=480),
    )
    
    # Create test device
    device = Device(
        model="TestCamera",
        name="RGB Compression Test",
        uid=device_uid,
    )
    
    # Generate dynamic RGB sequence
    logger.info("Generating test RGB sequence...")
    rgb_frames = create_dynamic_rgb_sequence(640, 480, 90)  # 3 seconds at 30fps
    
    # Convert to ColorFrame objects and send in batches
    batch_size = 30  # Process 30 frames at a time (1 second of video)
    
    for batch_start in range(0, len(rgb_frames), batch_size):
        batch_end = min(batch_start + batch_size, len(rgb_frames))
        batch_frames = rgb_frames[batch_start:batch_end]
        
        color_frames = []
        for i, frame in enumerate(batch_frames):
            # Create XRCpuImage frame data
            frame_data = create_rgb_frame_data(frame, time.time() + i * 0.033)  # 30fps timing
            
            # Create ColorFrame
            color_frame = ColorFrame(
                image=frame_data,
                intrinsics=intrinsics,
            )
            color_frames.append(color_frame)
        
        # Simulate session stream (normally this would come from the actual client)
        if servicer.client_sessions:
            session_stream = list(servicer.client_sessions.values())[0]
            
            logger.info("Processing batch %d-%d (%d frames)", 
                       batch_start + 1, batch_end, len(batch_frames))
            
            # Process the frames
            servicer.on_save_color_frames(color_frames, session_stream, device)
            
            # Small delay to simulate real-time streaming
            time.sleep(0.1)


def main():
    """Main function demonstrating RGB compression."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger.info("ARFlow RGB Compression Demo")
    logger.info("This example demonstrates video compression for RGB streams")
    
    # Test different compression qualities
    qualities = ["low", "medium", "high"]
    
    for quality in qualities:
        logger.info("\n" + "="*60)
        logger.info("TESTING COMPRESSION QUALITY: %s", quality.upper())
        logger.info("="*60)
        
        try:
            # Create servicer with compression enabled
            servicer = CompressionDemoServicer(
                spawn_viewer=False,  # Don't spawn viewer for this demo
                enable_compression=True,
                compression_quality=quality,
            )
            
            # Create a mock session for testing
            from cakelab.arflow_grpc.v1.session_pb2 import Session
            from cakelab.arflow_grpc.v1.session_metadata_pb2 import SessionMetadata
            from cakelab.arflow_grpc.v1.session_uuid_pb2 import SessionUuid
            import rerun as rr
            
            session = Session(
                id=SessionUuid(value=f"test_session_{quality}"),
                metadata=SessionMetadata(name=f"Compression Test {quality}"),
                devices=[],
            )
            
            # Create mock recording stream
            stream = rr.new_recording(
                recording_id=session.id.value,
                application_id="arflow_compression_test",
                spawn=False,
            )
            
            from arflow._session_stream import SessionStream
            session_stream = SessionStream(
                session, stream,
                enable_compression=True,
                compression_quality=quality
            )
            
            servicer.client_sessions[session.id.value] = session_stream
            
            # Simulate RGB streaming
            simulate_rgb_stream(servicer, session.id.value, "test_device")
            
            # Final statistics
            servicer._log_compression_stats()
            
        except ImportError as e:
            logger.error("Required dependencies missing: %s", e)
            logger.error("Please install FFmpeg and OpenCV: pip install arflow[opencv]")
            return
        except Exception as e:
            logger.error("Compression test failed for quality %s: %s", quality, e)
    
    logger.info("\nRGB Compression Demo completed!")
    logger.info("To test with real AR data, run: arflow view --enable-compression --compression-quality medium")


if __name__ == "__main__":
    main() 