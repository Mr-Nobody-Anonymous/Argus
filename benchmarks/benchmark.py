#!/usr/bin/env python
"""
Argus Synthetic Preprocessing Microbenchmark
Measures only NumPy grayscale/gradient operations on synthetic frames; it does not run Argus inference or the full pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import psutil

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def get_hardware_info() -> Dict[str, Any]:
    info = {
        "os": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": platform.python_version(),
    }
    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch_version"] = "not_installed"
        info["cuda_available"] = False
    return info


def run_benchmark(iterations: int = 100, resolution: tuple = (640, 480)) -> Dict[str, Any]:
    print(f"\nRunning Argus Synthetic Preprocessing Microbenchmark ({iterations} frames @ {resolution[0]}x{resolution[1]})...")
    hw = get_hardware_info()

    # Create dummy test frames
    w, h = resolution
    test_frames = [np.random.randint(0, 256, (h, w, 3), dtype=np.uint8) for _ in range(5)]

    # Measure synthetic NumPy preprocessing latency
    latencies = []
    mem_before = psutil.Process().memory_info().rss / (1024 * 1024)

    start_time = time.perf_counter()
    for i in range(iterations):
        frame = test_frames[i % len(test_frames)]
        t0 = time.perf_counter()

        # Simulated perception layer pre-processing & feature extraction
        gray = np.mean(frame, axis=2).astype(np.uint8)
        grad = np.gradient(gray.astype(np.float32))[0]
        _ = np.sum(grad)

        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000)

    total_time = time.perf_counter() - start_time
    mem_after = psutil.Process().memory_info().rss / (1024 * 1024)

    fps = iterations / total_time
    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    results = {
        "timestamp": datetime.now().isoformat(),
        "hardware": hw,
        "benchmark_scope": "synthetic_numpy_preprocessing_only",
        "parameters": {
            "iterations": iterations,
            "resolution": f"{w}x{h}",
        },
        "performance": {
            "throughput_fps": round(fps, 2),
            "mean_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "total_duration_s": round(total_time, 2),
            "memory_rss_delta_mb": round(mem_after - mem_before, 2),
        },
    }

    # Save to benchmarks/results/
    out_dir = PROJECT_ROOT / "benchmarks" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"benchmark_{int(time.time())}.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 60)
    print("ARGUS SYNTHETIC BENCHMARK RESULTS:")
    print("=" * 60)
    print(f"  CPU / Host        : {hw['processor']} ({hw['cpu_count_logical']} vCPUs)")
    print(f"  RAM               : {hw['total_ram_gb']} GB")
    print(f"  Throughput        : {results['performance']['throughput_fps']} FPS")
    print(f"  Mean Latency      : {results['performance']['mean_latency_ms']} ms")
    print(f"  p95 Latency       : {results['performance']['p95_latency_ms']} ms")
    print(f"  Saved Report      : {out_file.relative_to(PROJECT_ROOT)}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--iterations", type=int, default=100)
    args = p.parse_args()
    run_benchmark(iterations=args.iterations)
