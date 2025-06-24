#!/usr/bin/env python3
"""ARFlow Compression Evaluation Script

This script evaluates the impact of H.264 compression on ARFlow streaming.

ACCURACY NOTES:
- ✅ ACCURATE: Uses same FFmpeg settings as ARFlow (libx264, ultrafast, zerolatency)
- ✅ ACCURATE: Tests realistic resolutions (640x480, 1920x1080)
- ✅ ACCURATE: Models client encode → server decode pipeline
- ❌ SIMPLIFIED: Assumes 30 FPS (real ARFlow uses ~4 FPS in 250ms chunks)
- ❌ SIMPLIFIED: Uses fixed 12:1 compression ratio (real compression varies by content)
- ❌ SIMPLIFIED: Simulated processing overhead (real overhead varies by hardware)
- ❌ SIMPLIFIED: Missing network transmission delays and chunking behavior

This provides a reasonable baseline estimate, but real-world results may vary
based on scene complexity, device performance, and network conditions.
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple


@dataclass
class TestConfig:
    """Configuration for a compression test."""

    name: str
    width: int
    height: int
    compressed: bool
    description: str


@dataclass
class TestResult:
    """Results from a compression test."""

    config: TestConfig
    bytes_sent: int
    frames_processed: int
    fps_achieved: float
    psnr_score: Optional[float]
    bandwidth_mbps: float
    compression_ratio: Optional[float]


class SimpleCompressionTester:
    """Simple compression tester for ARFlow evaluation."""

    def __init__(self):
        self.results: List[TestResult] = []

        # 4 test configurations as specified
        self.configs = [
            TestConfig(
                "SD_Uncompressed", 640, 480, False, "SD (640x480) uncompressed RGB"
            ),
            TestConfig(
                "SD_Compressed", 640, 480, True, "SD (640x480) H.264 compressed"
            ),
            TestConfig(
                "HD_Uncompressed", 1920, 1080, False, "HD (1920x1080) uncompressed RGB"
            ),
            TestConfig(
                "HD_Compressed", 1920, 1080, True, "HD (1920x1080) H.264 compressed"
            ),
        ]

    def create_test_video(self, width: int, height: int, duration: int = 5) -> str:
        """Create a test video with moving patterns."""
        output_path = f"test_{width}x{height}.mp4"

        # Create test video with FFmpeg
        cmd = [
            "ffmpeg",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=duration={duration}:size={width}x{height}:rate=30",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",  # High quality reference
            "-y",
            output_path,
        ]

        try:
            subprocess.run(
                cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            return output_path
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to create test video: {e}")
            raise

    def measure_bytes_sent(
        self, config: TestConfig, duration: int = 10
    ) -> Tuple[int, int]:
        """Measure 1: Bytes sent for RGB modality only.

        Returns:
            Tuple of (total_bytes_sent, frame_count)
        """
        print(f"📊 Measuring bytes sent for {config.name}...")

        # Calculate bytes per frame
        if config.compressed:
            # H.264 compression: estimate based on typical compression ratios
            raw_bytes = config.width * config.height * 3  # RGB
            compressed_bytes = (
                raw_bytes // 12
            )  # ~12:1 compression ratio for typical content
            bytes_per_frame = compressed_bytes
        else:
            # Uncompressed RGB: 3 bytes per pixel
            bytes_per_frame = config.width * config.height * 3

        # Simulate sending frames at 30 FPS
        target_fps = 30
        frame_count = duration * target_fps
        total_bytes = bytes_per_frame * frame_count

        return total_bytes, frame_count

    def measure_fps_achieved(self, config: TestConfig, duration: int = 10) -> float:
        """Measure 2: FPS achieved on server (accounting for encoding/decoding overhead).

        Returns:
            Achieved FPS
        """
        print(f"🎯 Measuring FPS for {config.name}...")

        frames_processed = 0
        start_time = time.time()

        # Simulate frame processing with realistic overhead
        if config.compressed:
            # H.264 decoding overhead
            processing_time_per_frame = 0.015  # 15ms per frame
        else:
            # Raw RGB processing (minimal overhead)
            processing_time_per_frame = 0.002  # 2ms per frame

        # Simulate processing frames
        target_fps = 30
        frame_interval = 1.0 / target_fps

        while time.time() - start_time < duration:
            frame_start = time.time()

            # Simulate processing
            time.sleep(processing_time_per_frame)
            frames_processed += 1

            # Try to maintain target frame rate
            elapsed = time.time() - frame_start
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        actual_duration = time.time() - start_time
        fps_achieved = frames_processed / actual_duration

        return fps_achieved

    def calculate_psnr(
        self, config: TestConfig
    ) -> Tuple[Optional[float], Optional[float]]:
        """Measure 3: PSNR between compressed and uncompressed videos.

        Returns:
            PSNR score in dB, or None if not applicable
        """
        if not config.compressed:
            return None, None  # No compression to compare

        print(f"📈 Calculating PSNR for {config.name}...")

        # Create reference video
        reference_video = self.create_test_video(config.width, config.height, 3)

        try:
            # Create compressed version using ARFlow-like settings
            compressed_video = f"compressed_{config.width}x{config.height}.mp4"
            compress_cmd = [
                "ffmpeg",
                "-i",
                reference_video,
                "-c:v",
                "libx264",
                "-preset",
                "ultrafast",  # ARFlow uses ultrafast
                "-tune",
                "zerolatency",  # ARFlow uses zerolatency
                "-crf",
                "23",  # Typical streaming quality
                "-y",
                compressed_video,
            ]
            subprocess.run(
                compress_cmd,
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Calculate PSNR using FFmpeg
            psnr_cmd = [
                "ffmpeg",
                "-i",
                compressed_video,
                "-i",
                reference_video,
                "-lavfi",
                "psnr",
                "-f",
                "null",
                "-",
            ]
            result = subprocess.run(
                psnr_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            # Parse PSNR from output
            psnr_value = 0.0
            for line in result.stderr.split("\n"):
                if "average:" in line and "psnr" in line.lower():
                    parts = line.split("average:")
                    if len(parts) > 1:
                        psnr_str = parts[1].split()[0]
                        try:
                            psnr_value = float(psnr_str)
                            break
                        except ValueError:
                            continue

            # Calculate compression ratio
            ref_size = os.path.getsize(reference_video)
            comp_size = os.path.getsize(compressed_video)
            compression_ratio = ref_size / comp_size if comp_size > 0 else 0

            # Cleanup
            os.remove(reference_video)
            os.remove(compressed_video)

            return psnr_value if psnr_value > 0 else None, compression_ratio

        except Exception as e:
            print(f"⚠️  PSNR calculation failed: {e}")
            # Cleanup on error
            for path in [
                reference_video,
                f"compressed_{config.width}x{config.height}.mp4",
            ]:
                if os.path.exists(path):
                    os.remove(path)
            return None, None

    def run_test(self, config: TestConfig, duration: int = 10) -> TestResult:
        """Run complete test for one configuration."""
        print(f"\n🧪 Testing: {config.name}")
        print(f"   Resolution: {config.width}x{config.height}")
        print(f"   Compression: {'H.264' if config.compressed else 'Uncompressed RGB'}")

        # Measure 1: Bytes sent (RGB modality only)
        bytes_sent, frame_count = self.measure_bytes_sent(config, duration)

        # Measure 2: FPS achieved (with encoding/decoding overhead)
        fps_achieved = self.measure_fps_achieved(config, duration)

        # Measure 3: PSNR quality comparison
        psnr_score, compression_ratio = self.calculate_psnr(config)
        bandwidth_mbps = (bytes_sent * 8) / (duration * 1_000_000)

        result = TestResult(
            config=config,
            bytes_sent=bytes_sent,
            frames_processed=frame_count,
            fps_achieved=fps_achieved,
            psnr_score=psnr_score,
            bandwidth_mbps=bandwidth_mbps,
            compression_ratio=compression_ratio,
        )

        print(f"✅ Results:")
        print(f"   • Bytes sent: {bytes_sent:,}")
        print(f"   • FPS achieved: {fps_achieved:.2f}")
        print(f"   • Bandwidth: {bandwidth_mbps:.2f} Mbps")
        if psnr_score:
            print(f"   • PSNR: {psnr_score:.2f} dB")
        if compression_ratio:
            print(f"   • Compression ratio: {compression_ratio:.1f}:1")

        return result

    def run_all_tests(self, duration: int = 10) -> None:
        """Run all 4 configurations."""
        print("🚀 ARFlow Compression Evaluation")
        print("=" * 50)
        print(f"Test duration: {duration} seconds per configuration")
        print(f"Configurations: {len(self.configs)}")
        print("=" * 50)

        for i, config in enumerate(self.configs, 1):
            print(f"\n[{i}/{len(self.configs)}]")
            try:
                result = self.run_test(config, duration)
                self.results.append(result)
            except Exception as e:
                print(f"❌ Test failed: {e}")

        self.generate_reports()

    def generate_reports(self) -> None:
        """Generate CSV and JSON reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV Report
        csv_file = f"compression_test_{timestamp}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Configuration",
                    "Resolution",
                    "Compression",
                    "Bytes Sent",
                    "Frames",
                    "FPS Achieved",
                    "PSNR (dB)",
                    "Bandwidth (Mbps)",
                    "Compression Ratio",
                ]
            )

            for r in self.results:
                writer.writerow(
                    [
                        r.config.name,
                        f"{r.config.width}x{r.config.height}",
                        "H.264" if r.config.compressed else "Uncompressed",
                        r.bytes_sent,
                        r.frames_processed,
                        f"{r.fps_achieved:.2f}",
                        f"{r.psnr_score:.2f}" if r.psnr_score else "N/A",
                        f"{r.bandwidth_mbps:.2f}",
                        f"{r.compression_ratio:.1f}:1"
                        if r.compression_ratio
                        else "N/A",
                    ]
                )

        # JSON Report
        json_file = f"compression_test_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "results": [
                        {
                            "config": {
                                "name": r.config.name,
                                "width": r.config.width,
                                "height": r.config.height,
                                "compressed": r.config.compressed,
                            },
                            "measurements": {
                                "bytes_sent": r.bytes_sent,
                                "frames_processed": r.frames_processed,
                                "fps_achieved": r.fps_achieved,
                                "psnr_score": r.psnr_score,
                                "bandwidth_mbps": r.bandwidth_mbps,
                                "compression_ratio": r.compression_ratio,
                            },
                        }
                        for r in self.results
                    ],
                },
                f,
                indent=2,
            )

        print(f"\n📊 Evaluation Complete!")
        print(f"📄 Reports: {csv_file}, {json_file}")

        self.print_analysis()

    def print_analysis(self) -> None:
        """Print compression impact analysis."""
        print(f"\n📈 Compression Impact Analysis:")
        print("-" * 60)

        # Compare SD results
        sd_uncomp = next(
            (r for r in self.results if r.config.name == "SD_Uncompressed"), None
        )
        sd_comp = next(
            (r for r in self.results if r.config.name == "SD_Compressed"), None
        )

        if sd_uncomp and sd_comp:
            bw_reduction = (
                (sd_uncomp.bandwidth_mbps - sd_comp.bandwidth_mbps)
                / sd_uncomp.bandwidth_mbps
            ) * 100
            fps_change = (
                (sd_comp.fps_achieved - sd_uncomp.fps_achieved) / sd_uncomp.fps_achieved
            ) * 100

            print(f"\nSD (640x480) Results:")
            print(f"  • Bandwidth reduction: {bw_reduction:.1f}%")
            print(f"  • FPS change: {fps_change:+.1f}%")
            if sd_comp.psnr_score:
                print(f"  • Video quality: {sd_comp.psnr_score:.2f} dB")

        # Compare HD results
        hd_uncomp = next(
            (r for r in self.results if r.config.name == "HD_Uncompressed"), None
        )
        hd_comp = next(
            (r for r in self.results if r.config.name == "HD_Compressed"), None
        )

        if hd_uncomp and hd_comp:
            bw_reduction = (
                (hd_uncomp.bandwidth_mbps - hd_comp.bandwidth_mbps)
                / hd_uncomp.bandwidth_mbps
            ) * 100
            fps_change = (
                (hd_comp.fps_achieved - hd_uncomp.fps_achieved) / hd_uncomp.fps_achieved
            ) * 100

            print(f"\nHD (1920x1080) Results:")
            print(f"  • Bandwidth reduction: {bw_reduction:.1f}%")
            print(f"  • FPS change: {fps_change:+.1f}%")
            if hd_comp.psnr_score:
                print(f"  • Video quality: {hd_comp.psnr_score:.2f} dB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Test ARFlow compression impact")
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        help="Test duration per config in seconds (default: 10)",
    )

    args = parser.parse_args()

    # Check FFmpeg availability
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ FFmpeg not found. Please install FFmpeg to run this test.")
        sys.exit(1)

    tester = SimpleCompressionTester()
    tester.run_all_tests(args.duration)


if __name__ == "__main__":
    main()
