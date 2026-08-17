# 🧪 Argus Testing Guide

## The suites that gate correctness

These are the tests that must pass. They run offline, need no external
infrastructure, and are wired into no CI system yet (see
[PRODUCTION_ROADMAP.md](../PRODUCTION_ROADMAP.md) §4).

```bash
pytest tests/test_regression.py tests/test_api_security.py -v
# 44 passed
```

| Suite | Tests | Covers |
|---|---|---|
| `tests/test_regression.py` | 23 | Kalman filter shape/transition correctness, greedy IoU matching, track identity at realistic frame rates, primary-detector starvation, skipped-frame semantics, event deduplication |
| `tests/test_api_security.py` | 21 | Route coverage sweep, token forgery/expiry/replay, privilege escalation, plaintext secrets in config |

### Why these tests are trustworthy

They are **mutation-verified**: reintroducing a real bug makes the relevant test
fail with a diagnostic message. For example, restoring the buggy Kalman
`transitionMatrix` yields:

```
Track IDs changed for stationary objects: [1, 2] -> [9, 10]
```

A test suite that has never been shown to fail is not evidence of anything.

### The route-coverage sweep

`test_api_security.py` walks the **live route table** and asserts that every
`/api` route outside an explicit public allowlist carries an auth dependency.
This means a newly added endpoint cannot silently ship unauthenticated - the
test fails the moment someone forgets.

Public by design: `GET /api/v1/health`, `GET /`, `POST /api/v1/auth/login`,
`POST /api/v1/auth/refresh`, and `GET /metrics`.

---

## Swarm A/B benchmark

```bash
python tests/swarm_benchmark.py --frames 40 --json docs/swarm_benchmark_results.json
```

Compares the swarm pipeline against the linear baseline. Options: `--clip`,
`--frames`, `--json`, `--camera-id`.

**Each variant runs in a separate process.** This is not incidental: module-level
singletons (inference engine, deep tracker, consortium broker, agent gene
vectors) carry mutable state, and running both variants in one process biased
the first benchmark badly enough to invert its conclusion. Do not "simplify"
the harness back to in-process execution.

Latest result (40 frames, 640×480, CPU):

| Mode | FPS | p50 ms | p95 ms | Det/frame |
|---|---|---|---|---|
| linear | 1.27 | 784.20 | 840.84 | 12.8 |
| swarm | 1.61 | 612.78 | 750.53 | 12.8 |

**Always read the detection column next to the FPS column.** The first run of
this benchmark reported +45.2% FPS and −35.5% detections; the "speedup" was the
detector being starved into replaying a stale result.

---

## Manual & diagnostic scripts

> ⚠️ **These predate authentication.** They issue unauthenticated requests and
> will now receive `401`/`403` against a secured API. They are exploratory
> tools, not assertions, and are pending update or retirement (tracked in
> [TODO.md](../TODO.md)). Several also require infrastructure that is not part
> of the default stack.

```bash
python tests/run_all_tests.py                    # legacy aggregate runner
python tests/buffer_monitor.py                   # runs until interrupted - use timeout
python tests/ai_pipeline_test.py                 # YOLO + tracking accuracy
python tests/infrastructure_healthcheck.py       # needs Kafka / Elasticsearch / Qdrant
python tests/stream_simulator.py                 # needs ffmpeg
python tests/websocket_stress_tester.py          # needs a valid token now
bash   tests/resource_monitor.sh                 # Linux/macOS resource logging
```

To exercise them, either run the API with `ARGUS_DISABLE_AUTH=1` (development
only) or update the script to send a bearer token.

---

## Environment notes that will save you time

- **CPU inference is ~1.3–1.6 FPS** with YOLOv8n. Success criteria written
  against a 30 FPS GPU target (below) do not apply to a CPU run.
- **Synthetic frames (drawn rectangles/circles) yield zero YOLO detections.**
  Use `data/demo_clip.mp4` or a real photograph, otherwise you will "discover" a
  detection bug that does not exist.
- **The rules engine deduplicates within a sliding window**, so a repeated event
  will not reappear while the condition persists. Use a fresh process or wait
  out the window when testing event generation.
- **A single deadlocked request can wedge the server** and leave port 8000
  bound. Check with `ss -lntp | grep :8000` and `kill -9` if needed.
- `EventStore` exposes `query_events()` returning `(events, total)` - there is no
  `get_events()`.

---

## Legacy success criteria (GPU-era targets)

Retained for reference. These were written for a GPU deployment and are the
targets the platform aims at, not what a CPU dev box will produce.

### Original per-script criteria

### 1. Video Ingestion Diagnostics (`stream_simulator.py`)
- Simulates RTSP camera stream via FFmpeg
- Creates mock video with moving cars/people
- Outputs sample video if none exists

**SUCCESS CRITERIA:**
- ✓ FFmpeg process starts successfully
- ✓ RTSP URL accepts stream
- ✓ Stream loops continuously

### 2. Frame Buffer Monitor (`buffer_monitor.py`)
- Monitors FPS, drop rate, and queue latency
- Real-time diagnostics with health warnings

**SUCCESS CRITERIA:**
- ✓ FPS ≥ 15 (PASSED if ≥ 25)
- ✓ Drop Rate ≤ 5% (PASSED if = 0%)
- ✓ Latency ≤ 100ms (PASSED if ≤ 50ms)

### 3. AI Pipeline Sanity Test (`ai_pipeline_test.py`)
- Tests YOLOv8 + BoT-SORT tracking
- Verifies OCR/InsightFace crop passing
- Measures inference latency

**SUCCESS CRITERIA:**
- ✓ Avg Latency < 33ms (30 FPS target)
- ✓ ID Switches = 0 (perfect tracking)
- ✓ Successful Crops > 0 (models receiving data)

### 4. Infrastructure Healthcheck (`infrastructure_healthcheck.py`)
- Tests Kafka message publishing
- Verifies Elasticsearch indexing
- Validates Qdrant vector search

**SUCCESS CRITERIA:**
- ✓ All services respond to health checks
- ✓ Vector similarity score > 0.9

### 5. WebSocket Stress Tester (`websocket_stress_tester.py`)
- Floods backend with 100 updates/sec
- Tests map rendering under load

**SUCCESS CRITERIA:**
- ✓ No connection drops
- ✓ Backend handles sustained load

### 6. Resource Monitor (`resource_monitor.sh`)
- Linux/macOS bash script
- Logs CPU, RAM, GPU metrics

**HEALTHY METRICS:**
- CPU: 15% - 60% average
- RAM: < 75% capacity
- GPU VRAM: 40% - 85%
- GPU Util: 30% - 80%

**WARNING THRESHOLDS:**
- CPU > 90%: Processing bottleneck
- RAM > 95%: OOM risk
- GPU = 100%: Consider frame skipping

## Frontend Diagnostics (`leaflet_diagnostic.js`)

Inject into browser console on map page:
```javascript
runFrontendStressDiagnostic()
checkLeafletLayers()
```

**SUCCESS CRITERIA:**
- ✓ FPS ≥ 30
- ✓ No memory leaks (< 500MB heap)

## Test Execution Order

1. Start infrastructure: `docker-compose -f docker-compose.mediamtx.yml up -d`
2. Run buffer monitor: `python tests/buffer_monitor.py`
3. Run AI pipeline test: `python tests/ai_pipeline_test.py`
4. Run healthcheck: `python tests/infrastructure_healthcheck.py`
5. Start stream simulator: `python tests/stream_simulator.py`
6. Run WebSocket stress test: `python tests/websocket_stress_tester.py`
7. Monitor resources: `bash tests/resource_monitor.sh`