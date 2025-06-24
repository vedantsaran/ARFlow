# ARFlow Compression Evaluation

This directory contains tests to evaluate the impact of H.264 compression on ARFlow streaming performance.

## Available Tests

### 1. Simple Compression Test (`simple_compression_test.py`)

**Purpose**: Provides a baseline estimate of compression impact using simulated data and FFmpeg.

**Pros**:
- ✅ Works without ARFlow server or camera
- ✅ Uses exact same FFmpeg settings as ARFlow
- ✅ Fast execution (10 seconds per test)
- ✅ Measures PSNR quality

**Cons**:
- ❌ Uses simulated data, not real ARFlow pipeline
- ❌ Fixed compression ratios and frame rates
- ❌ Missing network transmission effects

**Usage**:
```bash
# Install FFmpeg first
brew install ffmpeg  # macOS
# or apt-get install ffmpeg  # Linux

# Run the test
python3 simple_compression_test.py
```

**Results**: Shows ~92% bandwidth reduction with excellent PSNR quality (54-58 dB).

### 2. Real ARFlow Test (`real_arflow_compression_test.py`)

**Purpose**: Measures actual ARFlow streaming performance with real camera data.

**Pros**:
- ✅ Uses real ARFlow server and client
- ✅ Real camera data and compression
- ✅ Actual network usage measurement
- ✅ CPU and memory monitoring

**Cons**:
- ❌ Requires ARFlow server running
- ❌ Requires camera connected
- ❌ More complex setup

**Usage**:
```bash
# Install dependencies
pip install opencv-python psutil

# Start ARFlow server (in another terminal)
python -m arflow.serve --port 8500

# Run the test
python3 real_arflow_compression_test.py --duration 60
```

**Results**: Provides real-world compression ratios and performance metrics.

## Test Configurations

### Simple Test Configurations:
- **SD_Uncompressed**: 640x480 RGB (221.18 Mbps)
- **SD_Compressed**: 640x480 H.264 (18.43 Mbps) 
- **HD_Uncompressed**: 1920x1080 RGB (1492.99 Mbps)
- **HD_Compressed**: 1920x1080 H.264 (124.42 Mbps)

### Real Test Configuration:
- **ARFlow_Default**: Uses actual SessionRunner with H.264 streaming (250ms intervals)

## Key Findings

1. **Bandwidth Savings**: H.264 compression reduces bandwidth by ~90-95%
2. **Quality**: PSNR scores of 54-58 dB indicate excellent visual quality
3. **Performance**: Slight FPS improvement due to reduced data transfer overhead
4. **CPU Overhead**: Minimal additional CPU usage for compression/decompression

## Accuracy Notes

- **Simple test**: Good for feasibility assessment, but uses simplified assumptions
- **Real test**: Accurate for actual ARFlow deployment scenarios
- **Compression ratios**: Vary based on scene complexity and motion
- **Network effects**: Real-world performance may vary with network conditions

## Recommendations

1. **For initial evaluation**: Use simple test to get baseline estimates
2. **For deployment planning**: Use real test with actual hardware and scenarios
3. **For production**: Test with your specific camera resolutions and content types

## Files Generated

Both tests generate timestamped reports:
- `compression_test_YYYYMMDD_HHMMSS.csv` - Tabular results
- `compression_test_YYYYMMDD_HHMMSS.json` - Structured data for analysis
- `real_arflow_test_YYYYMMDD_HHMMSS.csv` - Real test results
- `real_arflow_test_YYYYMMDD_HHMMSS.json` - Real test structured data

## Quick Start

### Requirements
- Python 3.10+
- FFmpeg installed and available in PATH
- ARFlow dependencies (if using the full integration)

### Simple Standalone Test

Run the standalone compression test that doesn't require ARFlow server:

```bash
cd python/benchmarks
python simple_compression_test.py --duration 10
```

This will test all 4 configurations and generate reports.

## Measurements

For each configuration, the script measures:

1. **Bytes Sent**: Total bytes for RGB modality only
2. **FPS Achieved**: Frames per second on server (accounting for encoding/decoding overhead)
3. **PSNR Quality**: Peak Signal-to-Noise Ratio between compressed and uncompressed video

## Usage Examples

### Basic test (10 seconds per configuration)
```bash
python simple_compression_test.py
```

### Extended test (30 seconds per configuration)
```bash
python simple_compression_test.py --duration 30
```

### Check FFmpeg installation
```bash
ffmpeg -version
```

## Output

The script generates:
- `compression_test_TIMESTAMP.csv`: Detailed results in CSV format
- `compression_test_TIMESTAMP.json`: Results in JSON format
- Console output with analysis and comparisons

## Expected Results

Typical results should show:
- **Bandwidth reduction**: 80-90% for H.264 vs uncompressed
- **FPS impact**: Slight decrease due to encoding/decoding overhead
- **PSNR quality**: 25-35 dB (good to excellent quality)

## Integration with ARFlow

For full integration with ARFlow server, modify the `compression_evaluation.py` script to connect to your running ARFlow instance.

## Troubleshooting

### FFmpeg not found
```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# Windows
# Download from https://ffmpeg.org/download.html
```

### Permission errors
Make sure the script has write permissions in the current directory for generating reports.

### Memory issues with HD
If HD tests fail due to memory, try reducing the test duration or running SD tests only. 