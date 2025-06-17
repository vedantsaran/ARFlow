# RGB Frame Support in ARFlow

ARFlow now supports RGB and RGBA color frame formats in addition to the existing YUV formats. This document describes the implementation and usage of RGB frame support.

## Supported Formats

The following RGB-based formats are now supported:

- **FORMAT_RGB24**: 24-bit RGB (8 bits per channel, no alpha)
- **FORMAT_RGBA32**: 32-bit RGBA (8 bits per channel including alpha)
- **FORMAT_BGRA32**: 32-bit BGRA (automatically converted to RGBA for Rerun)
- **FORMAT_ARGB32**: 32-bit ARGB (automatically converted to RGBA for Rerun)

## Implementation Details

### Protocol Buffers
- Updated `protos/cakelab/arflow_grpc/v1/xr_cpu_image.proto` to include RGB formats
- Updated generated Python and C# protobuf files to support the new formats

### Python Implementation
- Enhanced `SessionStream.save_color_frames()` to handle RGB formats
- Added color format conversion functions for BGRA→RGBA and ARGB→RGBA
- Added utility functions in `_utils.py` for creating RGB frames
- Optional OpenCV integration for YUV to RGB conversion

### Unity/C# Implementation
- Updated generated C# protobuf files with RGB format enums
- RGB formats are now available for Unity clients

## Usage Examples

### Creating RGB Frames

```python
import numpy as np
from arflow._utils import create_rgb_frame_data, create_rgba_frame_data

# Create RGB image data
rgb_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
rgb_frame = create_rgb_frame_data(rgb_image, timestamp=time.time())

# Create RGBA image data
rgba_image = np.random.randint(0, 255, (480, 640, 4), dtype=np.uint8)
rgba_frame = create_rgba_frame_data(rgba_image, timestamp=time.time())
```

### Checking Format Types

```python
from arflow._utils import is_rgb_format, is_yuv_format
from cakelab.arflow_grpc.v1.xr_cpu_image_pb2 import XRCpuImage

# Check if a format is RGB-based
is_rgb = is_rgb_format(XRCpuImage.FORMAT_RGB24)  # True
is_yuv = is_yuv_format(XRCpuImage.FORMAT_ANDROID_YUV_420_888)  # True
```

### OpenCV Integration

OpenCV support is optional and can be installed with:

```bash
poetry install --extras opencv
```

When OpenCV is available, YUV frames can be converted to RGB:

```python
from arflow._session_stream import _yuv420_to_rgb_cv2

# Convert YUV frame to RGB using OpenCV
rgb_data = _yuv420_to_rgb_cv2(yuv_frame_image)
```

### Custom Servicer with RGB Support

```python
from arflow import ARFlowServicer
from arflow._utils import is_rgb_format

class MyRGBServicer(ARFlowServicer):
    def on_save_color_frames(self, frames, session_stream, device):
        super().on_save_color_frames(frames, session_stream, device)
        
        for frame in frames:
            if is_rgb_format(frame.image.format):
                print(f"Received RGB frame: {frame.image.dimensions.x}x{frame.image.dimensions.y}")
                # Process RGB frame data...
```

## Color Format Conversions

The implementation automatically handles color format conversions:

- **BGRA32** → **RGBA32**: Swaps red and blue channels
- **ARGB32** → **RGBA32**: Moves alpha channel from first to last position
- **YUV420** → **RGB24**: Uses OpenCV conversion (if available)

## Dependencies

### Required
- `numpy`: For array operations and data manipulation
- `rerun-sdk`: For visualization (already included)

### Optional
- `opencv-python`: For YUV to RGB conversion and advanced image processing

## File Changes

### Core Implementation
- `python/arflow/_session_stream.py`: Enhanced color frame processing
- `python/arflow/_utils.py`: Added RGB utility functions
- `python/pyproject.toml`: Added OpenCV as optional dependency

### Protocol Buffers
- `protos/cakelab/arflow_grpc/v1/xr_cpu_image.proto`: Uncommented RGB formats
- `python/cakelab/arflow_grpc/v1/xr_cpu_image_pb2.py`: Updated generated Python code
- `unity/Packages/edu.wpi.cake.arflow/Runtime/Grpc/V1/XrCpuImage.cs`: Updated generated C# code

### Examples and Tests
- `python/examples/rgb_example.py`: Comprehensive RGB usage example
- `python/benchmarks/generate_payload.py`: Added RGB format examples

## Performance Considerations

- RGB formats require more bandwidth than YUV (3-4 bytes per pixel vs YUV's ~1.5 bytes per pixel)
- Color format conversions (BGRA/ARGB to RGBA) add minimal CPU overhead
- OpenCV YUV to RGB conversion is optimized but adds processing time

## Backward Compatibility

The RGB support is fully backward compatible:
- Existing YUV-based code continues to work unchanged
- New RGB formats are additive and don't affect existing functionality
- OpenCV dependency is optional and gracefully handles missing installation

## Testing

Run the RGB example to test the implementation:

```bash
cd python
python examples/rgb_example.py
```

This will start an ARFlow server with RGB support and demonstrate various RGB frame operations. 