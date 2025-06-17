#!/usr/bin/env python3
"""Example demonstrating RGB frame support in ARFlow.

This example shows how to:
1. Create RGB and RGBA frames
2. Convert between different color formats
3. Use OpenCV for YUV to RGB conversion
4. Send RGB frames to ARFlow server
"""

import logging
import time
from typing import Optional

import numpy as np
import numpy.typing as npt

from arflow import ARFlowServicer, ColorFrame, run_server
from arflow._utils import create_rgb_frame_data, create_rgba_frame_data, is_rgb_format
from cakelab.arflow_grpc.v1.device_pb2 import Device
from cakelab.arflow_grpc.v1.intrinsics_pb2 import Intrinsics
from cakelab.arflow_grpc.v1.vector2_pb2 import Vector2
from cakelab.arflow_grpc.v1.vector2_int_pb2 import Vector2Int
from cakelab.arflow_grpc.v1.xr_cpu_image_pb2 import XRCpuImage

# Try to import OpenCV
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    cv2 = None
    OPENCV_AVAILABLE = False

logger = logging.getLogger(__name__)


def create_test_rgb_image(width: int = 640, height: int = 480) -> npt.NDArray[np.uint8]:
    """Create a test RGB image with a gradient pattern."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient pattern
    for y in range(height):
        for x in range(width):
            image[y, x, 0] = (x * 255) // width  # Red gradient
            image[y, x, 1] = (y * 255) // height  # Green gradient
            image[y, x, 2] = 128  # Blue constant
    
    return image


def create_test_rgba_image(width: int = 640, height: int = 480) -> npt.NDArray[np.uint8]:
    """Create a test RGBA image with transparency."""
    image = np.zeros((height, width, 4), dtype=np.uint8)
    
    # Create a pattern with transparency
    for y in range(height):
        for x in range(width):
            image[y, x, 0] = (x * 255) // width  # Red gradient
            image[y, x, 1] = (y * 255) // height  # Green gradient
            image[y, x, 2] = 128  # Blue constant
            image[y, x, 3] = 255 if (x + y) % 2 == 0 else 128  # Checkerboard alpha
    
    return image


def yuv_to_rgb_opencv(yuv_image: npt.NDArray[np.uint8]) -> Optional[npt.NDArray[np.uint8]]:
    """Convert YUV image to RGB using OpenCV if available."""
    if not OPENCV_AVAILABLE:
        logger.warning("OpenCV not available for YUV to RGB conversion")
        return None
    
    try:
        rgb_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB_I420)
        return rgb_image
    except Exception as e:
        logger.error("Failed to convert YUV to RGB: %s", e)
        return None


class RGBDemoServicer(ARFlowServicer):
    """Demo servicer that handles RGB frames."""
    
    def on_save_color_frames(self, frames, session_stream, device):
        """Process incoming color frames and log RGB format information."""
        super().on_save_color_frames(frames, session_stream, device)
        
        for frame in frames:
            format_name = XRCpuImage.Format.Name(frame.image.format)
            is_rgb = is_rgb_format(frame.image.format)
            
            logger.info(
                "Received %s frame (%dx%d) - RGB format: %s",
                format_name,
                frame.image.dimensions.x,
                frame.image.dimensions.y,
                is_rgb
            )
            
            if is_rgb:
                # Extract RGB data
                plane_data = frame.image.planes[0].data
                width = frame.image.dimensions.x
                height = frame.image.dimensions.y
                
                if frame.image.format == XRCpuImage.FORMAT_RGB24:
                    rgb_array = np.frombuffer(plane_data, dtype=np.uint8).reshape((height, width, 3))
                    logger.info("RGB24 frame mean values: R=%.1f, G=%.1f, B=%.1f", 
                               rgb_array[:,:,0].mean(), rgb_array[:,:,1].mean(), rgb_array[:,:,2].mean())
                
                elif frame.image.format == XRCpuImage.FORMAT_RGBA32:
                    rgba_array = np.frombuffer(plane_data, dtype=np.uint8).reshape((height, width, 4))
                    logger.info("RGBA32 frame mean values: R=%.1f, G=%.1f, B=%.1f, A=%.1f", 
                               rgba_array[:,:,0].mean(), rgba_array[:,:,1].mean(), 
                               rgba_array[:,:,2].mean(), rgba_array[:,:,3].mean())


def main():
    """Main function demonstrating RGB frame usage."""
    logging.basicConfig(level=logging.INFO)
    
    logger.info("RGB Frame Support Demo")
    logger.info("OpenCV available: %s", OPENCV_AVAILABLE)
    
    # Create test images
    rgb_image = create_test_rgb_image()
    rgba_image = create_test_rgba_image()
    
    logger.info("Created test images: RGB %s, RGBA %s", rgb_image.shape, rgba_image.shape)
    
    # Create XRCpuImage frames
    rgb_frame_data = create_rgb_frame_data(rgb_image, time.time())
    rgba_frame_data = create_rgba_frame_data(rgba_image, time.time())
    
    logger.info("Created XRCpuImage frames")
    
    # Create color frames with intrinsics
    intrinsics = Intrinsics(
        focal_length=Vector2(x=500.0, y=500.0),
        principal_point=Vector2(x=320.0, y=240.0),
        resolution=Vector2Int(x=640, y=480),
    )
    
    rgb_color_frame = ColorFrame(
        image=rgb_frame_data,
        intrinsics=intrinsics,
    )
    
    rgba_color_frame = ColorFrame(
        image=rgba_frame_data,
        intrinsics=intrinsics,
    )
    
    logger.info("Created ColorFrame objects")
    
    # Demonstrate OpenCV conversion if available
    if OPENCV_AVAILABLE:
        logger.info("Demonstrating OpenCV YUV to RGB conversion...")
        # Create a dummy YUV image for demonstration
        yuv_dummy = np.zeros((480 * 3 // 2, 640), dtype=np.uint8)
        rgb_converted = yuv_to_rgb_opencv(yuv_dummy)
        if rgb_converted is not None:
            logger.info("YUV to RGB conversion successful: %s", rgb_converted.shape)
    
    # Start server with RGB support
    logger.info("Starting ARFlow server with RGB support...")
    try:
        run_server(
            service=RGBDemoServicer,
            spawn_viewer=True,
            port=8500,
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")


if __name__ == "__main__":
    main() 