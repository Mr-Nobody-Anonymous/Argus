# Argus Synthetic Preprocessing Microbenchmark

This directory contains a reproducible NumPy preprocessing microbenchmark. It measures grayscale conversion and a vertical gradient on synthetic frames only; it does not run the detector, tracker, swarm coordinator, or full Argus application. Do not use its FPS as an estimate of camera or model throughput.

---

## Running a Benchmark

To run the standardized benchmark:

```bash
python benchmarks/benchmark.py --iterations 200
```

Results are printed to the console and serialized to a timestamped JSON file with environment metadata. The committed sample records only synthetic preprocessing and is not a full-pipeline performance result.

---

## Benchmark Metrics Captured

- **Throughput (FPS)**: Synthetic NumPy operations per second; not application throughput.
- **Latency (ms)**: Mean and 95th-percentile execution latency per frame.
- **Memory RSS Delta (MB)**: Memory allocated across the benchmark duration.
- **Hardware Profile**: Automatically recorded hardware specs for fair cross-platform comparison.
