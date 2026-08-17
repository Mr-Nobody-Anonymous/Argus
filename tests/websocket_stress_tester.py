"""
WebSocket Stress Tester for Argus

Opens N concurrent connections to the live video WebSocket endpoint
(`/api/ws/stream/{camera_id}`) and measures, per connection and in aggregate:

  * frames/second received (binary JPEG messages)
  * detection metadata messages/second (JSON text messages)
  * end-to-end metadata latency (server timestamp -> client receipt)
  * drop / error rate

Usage:
    python tests/websocket_stress_tester.py --connections 10 --duration 15
    python tests/websocket_stress_tester.py --host localhost --port 8000 --cameras 1 2 3
"""
import argparse
import asyncio
import json
import statistics
import time
from typing import Dict, List, Optional

try:
    import websockets
    WS_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on environment
    WS_AVAILABLE = False


DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8000
DEFAULT_CONNECTIONS = 10
DEFAULT_DURATION_S = 15.0


class ConnectionStats:
    """Per-connection counters collected during the stress run."""

    def __init__(self, conn_id: int, camera_id: int):
        self.conn_id = conn_id
        self.camera_id = camera_id
        self.frames = 0
        self.metadata_messages = 0
        self.detections = 0
        self.errors = 0
        self.bytes_received = 0
        self.latencies_ms: List[float] = []
        self.connected = False
        self.failure: Optional[str] = None

    @property
    def avg_latency_ms(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def p95_latency_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        idx = min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))
        return ordered[idx]


async def stress_connection(
    url: str, conn_id: int, camera_id: int, duration_s: float
) -> ConnectionStats:
    """Consume the interleaved binary+JSON stream for `duration_s` seconds."""
    stats = ConnectionStats(conn_id, camera_id)
    deadline = time.time() + duration_s

    try:
        async with websockets.connect(url, max_size=None) as ws:
            stats.connected = True

            while time.time() < deadline:
                remaining = deadline - time.time()
                try:
                    message = await asyncio.wait_for(ws.recv(), timeout=remaining)
                except asyncio.TimeoutError:
                    break

                if isinstance(message, (bytes, bytearray)):
                    # Binary message: JPEG-encoded video frame
                    stats.frames += 1
                    stats.bytes_received += len(message)
                    continue

                # Text message: JSON detection metadata
                stats.metadata_messages += 1
                stats.bytes_received += len(message)
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    stats.errors += 1
                    continue

                if payload.get("error"):
                    stats.errors += 1
                    continue

                stats.detections += len(payload.get("detections", []))

                # The server sends a unix timestamp as a string
                ts = payload.get("timestamp")
                if ts is not None:
                    try:
                        stats.latencies_ms.append((time.time() - float(ts)) * 1000.0)
                    except (TypeError, ValueError):
                        pass

    except Exception as exc:  # noqa: BLE001 - report any connection failure
        stats.failure = f"{type(exc).__name__}: {exc}"

    return stats


def print_report(results: List[ConnectionStats], duration_s: float) -> bool:
    """Print the aggregate stress report. Returns True when the run is healthy."""
    sep = "=" * 66
    print("\n" + sep)
    print("  ARGUS — WEBSOCKET STRESS TEST REPORT")
    print(sep)

    established = [r for r in results if r.connected]
    failed = [r for r in results if not r.connected]

    total_frames = sum(r.frames for r in established)
    total_meta = sum(r.metadata_messages for r in established)
    total_detections = sum(r.detections for r in established)
    total_errors = sum(r.errors for r in established)
    total_bytes = sum(r.bytes_received for r in established)
    all_latencies = [lat for r in established for lat in r.latencies_ms]

    print(f"  Connections requested : {len(results)}")
    print(f"  Connections opened    : {len(established)}")
    print(f"  Connections failed    : {len(failed)}")
    print(f"  Duration              : {duration_s:.1f}s")
    print()
    print(f"  JPEG frames received  : {total_frames}  ({total_frames / duration_s:.1f}/s aggregate)")
    print(f"  Metadata messages     : {total_meta}  ({total_meta / duration_s:.1f}/s aggregate)")
    print(f"  Detections parsed     : {total_detections}")
    print(f"  Payload received      : {total_bytes / 1024:.1f} KiB")
    print(f"  Malformed / errors    : {total_errors}")

    if all_latencies:
        ordered = sorted(all_latencies)
        p95 = ordered[min(len(ordered) - 1, int(round(0.95 * (len(ordered) - 1))))]
        print(f"  Metadata latency      : avg {statistics.mean(all_latencies):.1f} ms | "
              f"p95 {p95:.1f} ms | max {max(all_latencies):.1f} ms")

    if established:
        print("\n  Per-connection breakdown:")
        for r in established:
            print(f"    [{r.conn_id:>3}] cam {r.camera_id} | "
                  f"{r.frames / duration_s:6.1f} fps | "
                  f"{r.metadata_messages / duration_s:6.1f} msg/s | "
                  f"avg {r.avg_latency_ms:6.1f} ms | errors {r.errors}")

    for r in failed:
        print(f"    [{r.conn_id:>3}] cam {r.camera_id} FAILED - {r.failure}")

    print(sep)
    healthy = bool(established) and not failed and total_errors == 0
    print(f"  RESULT: {'PASS' if healthy else 'ISSUES DETECTED'}")
    print(sep)
    return healthy


async def run_stress_test(
    host: str, port: int, cameras: List[int], connections: int, duration_s: float
) -> bool:
    if not WS_AVAILABLE:
        print("ERROR: the 'websockets' package is required for this test.")
        print("Install it with: pip install websockets")
        return False

    tasks = []
    for i in range(connections):
        camera_id = cameras[i % len(cameras)]
        url = f"ws://{host}:{port}/api/ws/stream/{camera_id}"
        tasks.append(stress_connection(url, i + 1, camera_id, duration_s))

    print(f"Opening {connections} concurrent connection(s) to "
          f"ws://{host}:{port}/api/ws/stream/{{camera_id}} for {duration_s:.0f}s...")
    print(f"Cameras under test: {cameras}")

    results = await asyncio.gather(*tasks)
    return print_report(list(results), duration_s)


def main():
    parser = argparse.ArgumentParser(description="Argus WebSocket stress tester")
    parser.add_argument("--host", default=DEFAULT_HOST, help="Backend host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Backend port")
    parser.add_argument("--connections", type=int, default=DEFAULT_CONNECTIONS,
                        help="Number of concurrent WebSocket connections")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                        help="Test duration in seconds")
    parser.add_argument("--cameras", type=int, nargs="+", default=[1],
                        help="Camera IDs to stream from (round-robined across connections)")
    args = parser.parse_args()

    try:
        healthy = asyncio.run(
            run_stress_test(args.host, args.port, args.cameras,
                            args.connections, args.duration)
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
        return

    raise SystemExit(0 if healthy else 1)


if __name__ == "__main__":
    main()
