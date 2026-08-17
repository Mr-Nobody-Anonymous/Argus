#!/usr/bin/env python3
"""
A/B benchmark: swarm pipeline vs. linear baseline.

The README calls Argus an "autonomous swarm agent architecture". That claim is
only defensible with numbers, and this codebase has already demonstrated why:
the consortium broker's fixed 33 ms budget (a 30 FPS GPU assumption) throttled
the detector to 0.218 on CPU, so it skipped 4 of every 5 frames and reported
zero detections — while /health showed model_loaded=true and healthy timings.

This harness runs the SAME frames through both pipelines in-process and reports
throughput, latency, CPU, and — most importantly — detection counts. A faster
pipeline that sees fewer objects is not an optimisation.

Usage:
    python tests/swarm_benchmark.py                 # 100 frames each
    python tests/swarm_benchmark.py --frames 200
    python tests/swarm_benchmark.py --clip data/demo_clip.mp4 --json results.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

try:
    import psutil
except ImportError:
    psutil = None


def load_frames(clip: Path, count: int) -> List[np.ndarray]:
    """Read up to `count` frames, looping the clip if it is shorter."""
    cap = cv2.VideoCapture(str(clip))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open clip: {clip}")

    frames: List[np.ndarray] = []
    while len(frames) < count:
        ok, frame = cap.read()
        if not ok or frame is None:
            if not frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue
        frames.append(frame)
    cap.release()

    if not frames:
        raise SystemExit(f"No frames decoded from {clip}")
    return frames


def percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(int(len(ordered) * pct / 100.0), len(ordered) - 1)
    return ordered[idx]


def run_pipeline(frames: List[np.ndarray], swarm: bool, camera_id: int = 1) -> Dict:
    """
    Run frames through one pipeline variant.

    IMPORTANT: this must be executed in a FRESH process per variant.

    The pipeline is built from module-level singletons (inference engine, deep
    tracker, consortium broker, agent gene vectors) that carry mutable state -
    track IDs, evolved thresholds, inference-time history, broker bids. Running
    both variants in one process lets the first run's state bias the second,
    which produced a misleading 35% "detection regression" on the first attempt
    at this benchmark. `main()` re-executes this script per variant to keep the
    comparison honest.
    """
    os.environ["ARGUS_NO_SWARM"] = "0" if swarm else "1"

    from backend.services.core_engine.processing_coordinator import ProcessingCoordinator

    coordinator = ProcessingCoordinator()
    label = "swarm" if swarm else "linear"
    if coordinator._swarm_enabled != swarm:
        print(f"  ! requested {label} but coordinator reports "
              f"swarm={coordinator._swarm_enabled}")

    from backend.services.vision.image_enhancement import get_image_enhancement
    enhancer = get_image_enhancement()

    latencies: List[float] = []
    detection_counts: List[int] = []
    proc = psutil.Process() if psutil else None
    if proc:
        proc.cpu_percent(None)  # prime the sampler

    started = time.time()
    for frame in frames:
        frame_start = time.perf_counter()
        enhanced = enhancer.enhance_frame(frame, mode="auto")

        if swarm:
            coordinator._swarm_process_frame(camera_id, enhanced, frame, time.time())
        else:
            coordinator._linear_process_frame(camera_id, enhanced, frame, time.time())

        latencies.append((time.perf_counter() - frame_start) * 1000.0)

        analysis = coordinator.get_camera_analysis(camera_id) or {}
        detection_counts.append(len(analysis.get("detections", [])))

    elapsed = time.time() - started
    cpu = proc.cpu_percent(None) if proc else None

    return {
        "mode": label,
        "frames": len(frames),
        "wall_seconds": round(elapsed, 2),
        "fps": round(len(frames) / elapsed, 2) if elapsed > 0 else 0.0,
        "latency_p50_ms": round(statistics.median(latencies), 2) if latencies else 0.0,
        "latency_p95_ms": round(percentile(latencies, 95), 2),
        "latency_mean_ms": round(statistics.fmean(latencies), 2) if latencies else 0.0,
        "detections_total": sum(detection_counts),
        "detections_mean": round(statistics.fmean(detection_counts), 2) if detection_counts else 0.0,
        "frames_with_zero_detections": sum(1 for c in detection_counts if c == 0),
        "cpu_percent": round(cpu, 1) if cpu is not None else None,
    }


def render_table(results: List[Dict]) -> str:
    headers = [
        ("mode", "Mode", 8),
        ("fps", "FPS", 8),
        ("latency_p50_ms", "p50 ms", 9),
        ("latency_p95_ms", "p95 ms", 9),
        ("detections_mean", "Det/frame", 10),
        ("frames_with_zero_detections", "Zero-det", 9),
        ("cpu_percent", "CPU %", 8),
    ]
    line = "  ".join(h[1].ljust(h[2]) for h in headers)
    out = [line, "-" * len(line)]
    for row in results:
        out.append("  ".join(str(row.get(k, "-")).ljust(w) for k, _, w in headers))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark swarm vs linear pipeline")
    parser.add_argument("--clip", default="data/demo_clip.mp4")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--json", help="Write results to this JSON file")
    parser.add_argument("--camera-id", type=int, default=1)
    parser.add_argument("--_worker", choices=("swarm", "linear"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()

    # Worker mode: run ONE variant in this process and emit JSON for the parent.
    if args._worker:
        clip_path = Path(args.clip)
        if not clip_path.is_absolute():
            clip_path = PROJECT_ROOT / clip_path
        worker_frames = load_frames(clip_path, args.frames)
        outcome = run_pipeline(
            worker_frames, swarm=(args._worker == "swarm"), camera_id=args.camera_id
        )
        print("__RESULT__" + json.dumps(outcome))
        return 0

    clip = Path(args.clip)
    if not clip.is_absolute():
        clip = PROJECT_ROOT / clip
    if not clip.exists():
        print(f"Clip not found: {clip}")
        return 1

    print("=" * 74)
    print("Argus swarm A/B benchmark")
    print("=" * 74)
    print(f"Clip:   {clip.name}")
    print(f"Frames: {args.frames}")
    print()

    frames = load_frames(clip, args.frames)
    print(f"Decoded {len(frames)} frames at {frames[0].shape[1]}x{frames[0].shape[0]}\n")

    # Each variant runs in its own subprocess so mutable singleton state
    # (track IDs, evolved gene vectors, broker bid history, inference-time
    # windows) from one variant cannot contaminate the other.
    import subprocess

    results = []
    for swarm in (False, True):
        label = "swarm" if swarm else "linear baseline"
        print(f"Running {label} (isolated process) ...")
        proc = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()),
             "--_worker", "swarm" if swarm else "linear",
             "--clip", str(clip), "--frames", str(args.frames),
             "--camera-id", str(args.camera_id)],
            capture_output=True, text=True, timeout=1800,
        )
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT__"):
                payload = json.loads(line[len("__RESULT__"):])
        if payload is None:
            print(f"  FAILED\n{proc.stdout[-600:]}\n{proc.stderr[-600:]}\n")
            continue
        results.append(payload)
        print(f"  {payload['fps']} FPS, "
              f"p50 {payload['latency_p50_ms']} ms, "
              f"{payload['detections_mean']} det/frame\n")

    if len(results) < 2:
        print("Both variants must run to compare.")
        return 1

    print("=" * 74)
    print(render_table(results))
    print("=" * 74)

    linear, swarm_res = results[0], results[1]

    def delta(key: str) -> str:
        base, new = linear.get(key) or 0, swarm_res.get(key) or 0
        if not base:
            return "n/a"
        return f"{(new - base) / base * 100:+.1f}%"

    print("\nSwarm vs baseline:")
    print(f"  FPS            {delta('fps')}")
    print(f"  p50 latency    {delta('latency_p50_ms')}")
    print(f"  detections     {delta('detections_mean')}")

    print("\nVerdict:")
    if swarm_res["detections_mean"] < linear["detections_mean"] * 0.9:
        print("  REGRESSION - the swarm sees materially fewer objects than the")
        print("  baseline. Throughput gains do not compensate for missed detections.")
    elif swarm_res["fps"] > linear["fps"] * 1.05:
        print("  Swarm improves throughput at comparable detection quality.")
    elif swarm_res["fps"] < linear["fps"] * 0.95:
        print("  Swarm is slower than the baseline with no detection benefit;")
        print("  its coordination overhead is not paying for itself here.")
    else:
        print("  No significant difference on this workload. The swarm's value")
        print("  would need to be demonstrated under multi-camera contention.")

    if args.json:
        out_path = Path(args.json)
        out_path.write_text(json.dumps(
            {"clip": str(clip), "frames": args.frames, "results": results}, indent=2
        ))
        print(f"\nWrote {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
