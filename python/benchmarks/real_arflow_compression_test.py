#!/usr/bin/env python3
"""Real ARFlow Compression Test

This script measures the difference between streaming and non-streaming modes
by monitoring existing ARFlow sessions and measuring their network usage.

Usage:
1. Start ARFlow server: arflow view --port 8500
2. Connect your phone to the server
3. Run this test: python real_arflow_compression_test.py

Note: This test monitors existing sessions from connected devices.
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
    from client.util.GetDeviceInfo import GetDeviceInfo
    from client.util.SessionRunner import SessionRunner
except ImportError as e:
    print(f"❌ Failed to import required components: {e}")
    print("Make sure you're running this from the python/benchmarks directory")
    print("Required: pip install psutil opencv-python")
    sys.exit(1)


@dataclass
class TestConfig:
    """Configuration for a real ARFlow test."""

    name: str
    gathering_interval: int  # milliseconds
    duration: int  # seconds
    description: str


@dataclass
class TestResult:
    """Results from a real ARFlow test."""

    config: TestConfig
    frames_sent: int
    bytes_sent: int
    fps_achieved: float
    cpu_usage_avg: float
    memory_usage_mb: float
    bandwidth_mbps: float
    errors: List[str]


class RealARFlowTester:
    """Test compression impact using real ARFlow components."""

    def __init__(self, server_host: str = "localhost", server_port: int = 8500):
        self.server_host = server_host
        self.server_port = server_port
        self.configs = [
            TestConfig(
                name="ARFlow_Phone_Streaming",
                gathering_interval=250,  # 4 FPS (ARFlow default with streaming)
                duration=30,
                description="ARFlow phone streaming mode (H.264 compressed)",
            ),
        ]
        self.results: List[TestResult] = []

    async def get_existing_session(self) -> tuple[Any, Device]:
        """Get an existing session from a connected phone."""
        print("📱 Looking for existing phone sessions...")

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
            f"✅ Found session: {session.metadata.name} from device: {session.devices[0].name}"
        )

        # Get device info from the session
        device = session.devices[0]

        return session, device

    async def run_test(self, config: TestConfig) -> TestResult:
        """Run a single test configuration."""
        print(f"\n🧪 Testing: {config.name}")
        print(f"   Interval: {config.gathering_interval}ms")
        print(f"   Duration: {config.duration}s")

        errors: List[str] = []
        frames_sent = 0
        bytes_sent = 0
        cpu_usage_samples: List[float] = []
        memory_usage_samples: List[float] = []

        try:
            # Get existing session from phone
            session, device = await self.get_existing_session()

            # Track metrics
            start_time = time.time()
            process = psutil.Process()
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Frame callback to track metrics
            async def on_frame(sess: Any, frame: ARFrame, dev: Device) -> None:
                nonlocal frames_sent, bytes_sent
                frames_sent += 1

                # Count bytes sent (SessionRunner sends compressed H.264 data)
                if frame.color_frame and frame.color_frame.image:
                    # This will be the actual compressed data size
                    bytes_sent += len(frame.color_frame.image.planes[0].data)

            # Create session runner to monitor the existing session
            print("   Monitoring existing phone session...")
            runner = SessionRunner(session, device, on_frame, config.gathering_interval)

            # Start recording
            await runner.start_recording()

            # Monitor performance
            end_time = start_time + config.duration
            sample_interval = 1.0  # Sample every second

            while time.time() < end_time:
                # Sample CPU and memory
                try:
                    cpu_percent = process.cpu_percent()
                    memory_mb = process.memory_info().rss / 1024 / 1024

                    cpu_usage_samples.append(cpu_percent)
                    memory_usage_samples.append(memory_mb)
                except Exception as e:
                    print(f"⚠️  Warning: Failed to sample performance: {e}")

                await asyncio.sleep(sample_interval)

            # Stop the runner
            runner.stopped = True

            # Calculate metrics
            actual_duration = time.time() - start_time
            fps_achieved = frames_sent / actual_duration if actual_duration > 0 else 0
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
                (bytes_sent * 8) / (actual_duration * 1_000_000)
                if actual_duration > 0
                else 0
            )

            print(f"✅ Results:")
            print(f"   • Frames received: {frames_sent}")
            print(f"   • Bytes received: {bytes_sent:,}")
            print(f"   • FPS achieved: {fps_achieved:.2f}")
            print(f"   • CPU usage: {cpu_usage_avg:.1f}%")
            print(f"   • Memory usage: {memory_usage_mb:.1f} MB")
            print(f"   • Bandwidth: {bandwidth_mbps:.2f} Mbps")

        except Exception as e:
            error_msg = f"Test failed: {e}"
            print(f"❌ {error_msg}")
            errors.append(error_msg)

            # Set default values for failed test
            fps_achieved = 0.0
            cpu_usage_avg = 0.0
            memory_usage_mb = 0.0
            bandwidth_mbps = 0.0

        return TestResult(
            config=config,
            frames_sent=frames_sent,
            bytes_sent=bytes_sent,
            fps_achieved=fps_achieved,
            cpu_usage_avg=cpu_usage_avg,
            memory_usage_mb=memory_usage_mb,
            bandwidth_mbps=bandwidth_mbps,
            errors=errors,
        )

    async def run_all_tests(self) -> None:
        """Run all test configurations."""
        print("🚀 Real ARFlow Compression Evaluation")
        print("=" * 50)
        print(f"Server: {self.server_host}:{self.server_port}")
        print(f"Configurations: {len(self.configs)}")
        print("=" * 50)

        # Check if server is reachable
        print("🔍 Checking ARFlow server connection...")
        try:
            client = GrpcClient(f"{self.server_host}:{self.server_port}")
            # Try to list sessions to test connection
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

        # Run tests
        for i, config in enumerate(self.configs, 1):
            print(f"\n[{i}/{len(self.configs)}]")
            result = await self.run_test(config)
            self.results.append(result)

            # Brief pause between tests
            if i < len(self.configs):
                print("⏸️  Pausing between tests...")
                await asyncio.sleep(5)

        self.generate_reports()

    def generate_reports(self) -> None:
        """Generate detailed reports."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # CSV Report
        csv_file = f"real_arflow_test_{timestamp}.csv"
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "Configuration",
                    "Interval (ms)",
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

            for r in self.results:
                writer.writerow(
                    [
                        r.config.name,
                        r.config.gathering_interval,
                        r.config.duration,
                        r.frames_sent,
                        r.bytes_sent,
                        f"{r.fps_achieved:.2f}",
                        f"{r.cpu_usage_avg:.1f}",
                        f"{r.memory_usage_mb:.1f}",
                        f"{r.bandwidth_mbps:.2f}",
                        "; ".join(r.errors) if r.errors else "None",
                    ]
                )

        # JSON Report
        json_file = f"real_arflow_test_{timestamp}.json"
        with open(json_file, "w") as f:
            json.dump(
                {
                    "timestamp": timestamp,
                    "test_type": "real_arflow_compression",
                    "server": f"{self.server_host}:{self.server_port}",
                    "results": [
                        {
                            "config": {
                                "name": r.config.name,
                                "gathering_interval": r.config.gathering_interval,
                                "duration": r.config.duration,
                                "description": r.config.description,
                            },
                            "measurements": {
                                "frames_sent": r.frames_sent,
                                "bytes_sent": r.bytes_sent,
                                "fps_achieved": r.fps_achieved,
                                "cpu_usage_avg": r.cpu_usage_avg,
                                "memory_usage_mb": r.memory_usage_mb,
                                "bandwidth_mbps": r.bandwidth_mbps,
                                "errors": r.errors,
                            },
                        }
                        for r in self.results
                    ],
                },
                f,
                indent=2,
            )

        print(f"\n📊 Real ARFlow Test Complete!")
        print(f"📄 Reports: {csv_file}, {json_file}")

        self.print_analysis()

    def print_analysis(self) -> None:
        """Print analysis of the results."""
        print(f"\n📈 Real ARFlow Performance Analysis:")
        print("-" * 60)

        if self.results and not self.results[0].errors:
            result = self.results[0]

            print(f"\nARFlow Phone Streaming Performance:")
            print(f"  • Actual FPS: {result.fps_achieved:.2f}")
            print(f"  • Bandwidth usage: {result.bandwidth_mbps:.2f} Mbps")
            print(f"  • CPU usage: {result.cpu_usage_avg:.1f}%")
            print(f"  • Memory usage: {result.memory_usage_mb:.1f} MB")
            print(f"  • Frames processed: {result.frames_sent}")
            print(
                f"  • Average bytes per frame: {result.bytes_sent // result.frames_sent if result.frames_sent > 0 else 0:,}"
            )

            # Estimate uncompressed equivalent
            # Assume phone camera resolution (common default)
            estimated_width, estimated_height = 1920, 1080  # HD phone camera
            uncompressed_bytes_per_frame = estimated_width * estimated_height * 3  # RGB
            total_uncompressed_bytes = uncompressed_bytes_per_frame * result.frames_sent
            compression_ratio = (
                total_uncompressed_bytes / result.bytes_sent
                if result.bytes_sent > 0
                else 0
            )

            print(f"\nCompression Efficiency:")
            print(f"  • Estimated compression ratio: {compression_ratio:.1f}:1")
            print(
                f"  • Bandwidth savings: {((total_uncompressed_bytes - result.bytes_sent) / total_uncompressed_bytes) * 100:.1f}%"
            )
            print(
                f"  • Uncompressed equivalent: {(total_uncompressed_bytes * 8) / (result.config.duration * 1_000_000):.2f} Mbps"
            )
        else:
            print("\n⚠️  Could not analyze results due to test failures")


async def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test real ARFlow compression performance"
    )
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
        "--port",
        type=int,
        default=8500,
        help="ARFlow server port (default: 8500)",
    )

    args = parser.parse_args()

    # Check dependencies
    try:
        import cv2  # type: ignore

        print("✅ OpenCV available")
    except ImportError:
        print("❌ OpenCV not found. Install with: pip install opencv-python")
        print("This is required for camera capture in SessionRunner")
        sys.exit(1)

    try:
        import psutil  # type: ignore

        print("✅ psutil available")
    except ImportError:
        print("❌ psutil not found. Install with: pip install psutil")
        print("This is required for performance monitoring")
        sys.exit(1)

    # Update test durations
    tester = RealARFlowTester(args.host, args.port)
    for config in tester.configs:
        config.duration = args.duration

    await tester.run_all_tests()


if __name__ == "__main__":
    asyncio.run(main())
