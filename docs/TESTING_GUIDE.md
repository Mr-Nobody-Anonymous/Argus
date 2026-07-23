# 🧪 City OS Testing Guide

## Quick Start

```bash
# Run all tests
python tests/run_all_tests.py

# Run individual tests
python tests/buffer_monitor.py      # Frame buffer diagnostics
python tests/ai_pipeline_test.py    # YOLO + tracking accuracy
python tests/infrastructure_healthcheck.py  # Kafka/ES/Qdrant verification
python tests/stream_simulator.py    # RTSP stream simulation
python tests/websocket_stress_tester.py  # WebSocket load test
```

## Test Suite Overview

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