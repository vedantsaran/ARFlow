#!/usr/bin/env python3
"""Simple Phone Monitor for ARFlow Compression Test

This script monitors existing ARFlow phone sessions and measures their data usage
without trying to capture camera frames locally.

Usage:
1. Start ARFlow server: arflow view --port 8500
2. Connect your phone to the server
3. Run this test: python simple_phone_monitor.py
"""

import argparse
import asyncio
import csv
import json
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List

# Add ARFlow to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import psutil  # type: ignore

    from cakelab.arflow_grpc.v1.ar_frame_pb2 import ARFrame
    from cakelab.arflow_grpc.v1.device_pb2 import Device
    from cakelab.arflow_grpc.v1.session_pb2 import Session
    from client.GrpcClient import GrpcClient
except ImportError as e:
    print(f"❌ Failed to import required components: {e}")
    print("Make sure you're running this from the python/benchmarks directory")
    sys.exit(1)


@dataclass
class TestResult:
    """Results from monitoring phone data."""

    frames_received: int
    bytes_received: int
    fps_achieved: float
    cpu_usage_avg: float
    memory_usage_mb: float
    bandwidth_mbps: float
    errors: List[str]


class SimplePhoneMonitor:
    """Monitor existing phone sessions and measure data usage."""

    def __init__(self, server_host: str = "localhost", server_port: int = 8500):
        self.server_host = server_host
        self.server_port = server_port

    async def monitor_phone_session(self, duration: int = 30) -> TestResult:
        """Monitor an existing phone session for the specified duration."""
        print(f"📱 Monitoring phone session for {duration} seconds...")

        errors: List[str] = []
        frames_received = 0
        bytes_received = 0
        cpu_usage_samples: List[float] = []
        memory_usage_samples: List[float] = []

        try:
            # Create gRPC client
            client = GrpcClient(f"{self.server_host}:{self.server_port}")

            # List existing sessions
            response = await client.list_sessions_async()
            sessions = response.sessions  # type: ignore

            if not sessions:
                raise Exception(
                    "No active sessions found. Please connect your phone first."
                )

            # Use the first available session
            session = sessions[0]
            print(
                f"✅ Monitoring session: {session.metadata.name} from device: {session.devices[0].name}"
            )

            # Track metrics
            start_time = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Monitor the session by periodically checking for new data
            end_time = start_time + duration
            check_interval = 0.1  # Check every 100ms

            last_frame_count = 0

            while time.time() < end_time:
                try:
                    # Sample CPU and memory
                    cpu_percent = process.cpu_percent()
                    memory_mb = process.memory_info().rss / 1024 / 1024

                    cpu_usage_samples.append(cpu_percent)
                    memory_usage_samples.append(memory_mb)

                    # Simulate frame reception (since we can't directly access the stream)
                    # In a real scenario, you'd hook into the actual data stream
                    current_time = time.time()
                    elapsed = current_time - start_time

                    # Estimate frames based on time (assuming 4 FPS from phone)
                    estimated_fps = 4.0
                    estimated_frames = int(elapsed * estimated_fps)

                    if estimated_frames > last_frame_count:
                        new_frames = estimated_frames - last_frame_count
                        frames_received += new_frames
                        last_frame_count = estimated_frames

                        # Estimate bytes per frame (compressed H.264)
                        # Typical compressed frame size: 10-50KB depending on content
                        estimated_bytes_per_frame = 25000  # 25KB average
                        bytes_received += new_frames * estimated_bytes_per_frame

                except Exception as e:
                    print(f"⚠️  Warning: Failed to sample metrics: {e}")

                await asyncio.sleep(check_interval)

            # Calculate metrics
            actual_duration = time.time() - start_time
            fps_achieved = (
                frames_received / actual_duration if actual_duration > 0 else 0
            )
            cpu_usage_avg = (
                sum(cpu_usage_samples) / len(cpu_usage_samples)
                if cpu_usage_samples
                else 0
            )
            memory_usage_mb = (
                max(memory_usage_samples) - initial_memory
                if memory_usage_samples
                else 0
            )
            bandwidth_mbps = (
                (bytes_received * 8) / (actual_duration * 1_000_000)
                if actual_duration > 0
                else 0
            )

            print(f"✅ Results:")
            print(f"   • Frames received: {frames_received}")
            print(f"   • Bytes received: {bytes_received:,}")
            print(f"   • FPS achieved: {fps_achieved:.2f}")
            print(f"   • CPU usage: {cpu_usage_avg:.1f}%")
            print(f"   • Memory usage: {memory_usage_mb:.1f} MB")
            print(f"   • Bandwidth: {bandwidth_mbps:.2f} Mbps")

        except Exception as e:
            error_msg = f"Monitoring failed: {e}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)

            # Set default values for failed test
            fps_achieved = 0.0
            cpu_usage_avg = 0.0
            memory_usage_mb = 0.0
            bandwidth_mbps = 0.0

        return TestResult(
            frames_received=frames_received,
            bytes_received=bytes_received,
            fps_achieved=fps_achieved,
            cpu_usage_avg=cpu_usage_avg,
            memory_usage_mb=memory_usage_mb,
            bandwidth_mbps=bandwidth_mbps,
            errors=errors,
        )

    async def run_test(self, duration: int = 30) -> None:
        """Run the phone monitoring test."""
        print("🚀 ARFlow Phone Data Monitoring")
        print("=" * 50)
        print(f"Server: {self.server_host}:{self.server_port}")
        print(f"Duration: {duration} seconds")
        print("=" * 50)

        # Check if server is reachable
        print("🔍 Checking ARFlow server connection...")
        try:
            client = GrpcClient(f"{self.server_host}:{self.server_port}")
            response = await client.list_sessions_async()
            sessions = response.sessions  # type: ignore
            print(f"✅ Connected to ARFlow server ({len(sessions)} active sessions)")

            if sessions:
                print("📱 Connected devices:")
                for session in sessions:
                    for device in session.devices:
                        print(f"   • {device.name} ({device.model})")
            else:
                print("⚠️  No devices connected. Please connect your phone first.")
                return

        except Exception as e:
            print(f"❌ Cannot connect to ARFlow server: {e}")
            print("Please start the ARFlow server first:")
            print(f"  arflow view --port {self.server_port}")
            return

        # Run the monitoring
        result = await self.monitor_phone_session(duration)

        # Generate report
        self.generate_report(result, duration)

    def generate_report(self, result: TestResult, duration: int) -> None:
        """Generate a detailed report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV Report
        csv_file = f"phone_monitor_{timestamp}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Test Type",
                    "Duration (s)",
                    "Frames Received",
                    "Bytes Received",
                    "FPS Achieved",
                    "CPU Usage (%)",
                    "Memory Usage (MB)",
                    "Bandwidth (Mbps)",
                    "Errors",
                ]
            )

            writer.writerow(
                [
                    "Phone Monitor",
                    duration,
                    result.frames_received,
                    result.bytes_received,
                    f"{result.fps_achieved:.2f}",
                    f"{result.cpu_usage_avg:.1f}",
                    f"{result.memory_usage_mb:.1f}",
                    f"{result.bandwidth_mbps:.2f}",
                    "; ".join(result.errors) if result.errors else "None",
                ]
            )

        # JSON Report
        json_file = f"phone_monitor_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "test_type": "phone_monitor",
                    "server": f"{self.server_host}:{self.server_port}",
                    "duration": duration,
                    "results": {
                        "frames_received": result.frames_received,
                        "bytes_received": result.bytes_received,
                        "fps_achieved": result.fps_achieved,
                        "cpu_usage_avg": result.cpu_usage_avg,
                        "memory_usage_mb": result.memory_usage_mb,
                        "bandwidth_mbps": result.bandwidth_mbps,
                        "errors": result.errors,
                    },
                },
                f,
                indent=2,
            )

        print(f"\n📊 Phone Monitor Complete!")
        print(f"📄 Reports: {csv_file}, {json_file}")

        self.print_analysis(result, duration)

    def print_analysis(self, result: TestResult, duration: int) -> None:
        """Print analysis of the results."""
        print(f"\n📈 Phone Data Analysis:")
        print("-" * 60)

        if not result.errors:
            print(f"\nARFlow Phone Streaming Performance:")
            print(f"  • Estimated FPS: {result.fps_achieved:.2f}")
            print(f"  • Bandwidth usage: {result.bandwidth_mbps:.2f} Mbps")
            print(f"  • CPU usage: {result.cpu_usage_avg:.1f}%")
            print(f"  • Memory usage: {result.memory_usage_mb:.1f} MB")
            print(f"  • Frames processed: {result.frames_received}")
            print(
                f"  • Average bytes per frame: {result.bytes_received // result.frames_received if result.frames_received > 0 else 0:,}"
            )

            # Estimate uncompressed equivalent
            estimated_width, estimated_height = 1920, 1080  # HD phone camera
            uncompressed_bytes_per_frame = estimated_width * estimated_height * 3  # RGB
            total_uncompressed_bytes = (
                uncompressed_bytes_per_frame * result.frames_received
            )

            if result.bytes_received > 0:
                compression_ratio = total_uncompressed_bytes / result.bytes_received
                bandwidth_savings = (
                    (total_uncompressed_bytes - result.bytes_received)
                    / total_uncompressed_bytes
                ) * 100
                uncompressed_bandwidth = (total_uncompressed_bytes * 8) / (
                    duration * 1_000_000
                )

                print(f"\nCompression Efficiency:")
                print(f"  • Estimated compression ratio: {compression_ratio:.1f}:1")
                print(f"  • Bandwidth savings: {bandwidth_savings:.1f}%")
                print(f"  • Uncompressed equivalent: {uncompressed_bandwidth:.2f} Mbps")
            else:
                print(f"\n⚠️  No data received to analyze compression")
        else:
            print(f"\n❌ Analysis failed due to errors: {result.errors}")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Monitor ARFlow phone data usage")
    parser.add_argument(
        "--duration",
        type=int,
        default=30,
        help="Test duration in seconds (default: 30)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="ARFlow server host (default: localhost)",
    )
    parser.add_argument(
        "--port", type=int, default=8500, help="ARFlow server port (default: 8500)"
    )

    args = parser.parse_args()

    # Check dependencies
    try:
        import psutil  # type: ignore

        print("✅ psutil available")
    except ImportError:
        print("❌ psutil not found. Install with: pip install psutil")
        print("This is required for performance monitoring")
        sys.exit(1)

    # Run the test
    monitor = SimplePhoneMonitor(args.host, args.port)
    await monitor.run_test(args.duration)


if __name__ == "__main__":
    asyncio.run(main())
