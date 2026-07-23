<div align="center">
<img src="assets/images.png" alt="Argus - The Watchful Guardian" width="400"/>
</div>

<h1 align="center">
   Argus — AI Video Analytics Platform
</h1>

<div align="center">
  <strong>Production-ready multi-camera AI surveillance with YOLOv8, Re-ID tracking, LPR, face recognition, zone alerting, and an autonomous swarm agent architecture.</strong>
</div>

<div align="center">

![GitHub Repo stars](https://img.shields.io/github/stars/Mr-Nobody-Anonymous/Argus?style=social)
![License](https://img.shields.io/github/license/Mr-Nobody-Anonymous/Argus?color=blue)
![Issues](https://img.shields.io/github/issues/Mr-Nobody-Anonymous/Argus?color=red)
![Last Commit](https://img.shields.io/github/last-commit/Mr-Nobody-Anonymous/Argus?color=green)

---

[**Quickstart**](#-quick-start) |
[**Features**](#-features) |
[**Swarm Architecture**](#-swarm-agent-architecture) |
[**Data Pipeline**](#-data-pipeline) |
[**API Reference**](#-api-reference) |
[**Testing**](#-testing) |
[**Folder Structure**](#-folder-structure) |
[**Troubleshooting**](#-troubleshooting)

</div>

---

## 📖 Overview

Argus is a production-ready AI-powered video analytics platform for real-time CCTV monitoring. It ingests up to 4 concurrent RTSP or webcam streams, runs YOLOv8 object detection on each frame, assigns persistent tracking IDs via ByteTrack/DeepSORT/BoT-SORT, and evaluates zone-based rules (intrusion, loitering, speed violation, fall detection) to generate actionable security events.

The system employs a **decentralised multi-agent swarm** where autonomous agents (YOLO, Face, LPR) bid for compute resources through a Consortium Broker. When the swarm is disabled, a classic linear fallback pipeline runs with full backward compatibility.

**Metaphor:** In Greek mythology, Argus Panoptes was a hundred-eyed giant who served as a watchful guardian. This platform embodies that vigilance.

> 📖 **For a detailed file-by-file breakdown, see [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md).**

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+ and Node.js 18+
- 4 GB RAM minimum
- RTSP camera streams *or* a local webcam for testing

### 1. Backend
```bash
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev            # Serves on http://localhost:3000
```

**💡 Windows shortcut:** double-click `run_app.bat`.

### 3. Access
| Service          | URL                          |
|------------------|------------------------------|
| Dashboard        | http://localhost:3000        |
| API Docs         | http://localhost:8000/docs   |
| WebSocket Stream | `ws://localhost:8000/api/ws/stream/{camera_id}` |
| Django Admin     | http://localhost:8000/admin  |

---

## 🌟 Features

### Core Capabilities
- **Multi-Camera Ingestion** – Up to 4 concurrent RTSP/webcam streams with auto-reconnect and exponential backoff.
- **YOLOv8 Detection** – Person, vehicle, bicycle, and object detection with configurable confidence / IoU thresholds.
- **License Plate Recognition** – PaddleOCR-based plate detection and regex validation.
- **Face Recognition** – InsightFace/OpenCV face identification against known reference images.
- **Pose Estimation** – MediaPipe-based human keypoint detection for fall/gesture analysis.
- **Deep Object Tracking** – ByteTrack / DeepSORT / BoT-SORT with Kalman-filtered motion prediction.
- **Zone-Based Rules** – Polygon/rectangle virtual zones with intrusion, loitering, and tripwire alerts.
- **Cross-Camera Re-ID** – Person re-identification across non-overlapping cameras using ResNet feature embeddings and cosine similarity.
- **Event Management** – SQLite-backed event store with 5-second deduplication, snapshot capture, and MQTT publishing.
- **Real-Time WebSocket Streaming** – Interleaved binary (JPEG) + JSON (detection metadata) protocol at ~30 FPS.
- **Image Enhancement** – CLAHE, denoising, sharpening, night vision, deblur, and HDR auto-enhancement.

### 🤖 Swarm Agent Architecture
| Agent | Role | Local Evolution |
|-------|------|----------------|
| **YOLO Detection Agent** | Primary object detection | Optimises `yolo_conf_threshold`, `iou_threshold`, `input_resolution_scale`, `tracker_matching_threshold` via a genetic algorithm. |
| **Face Recognition Agent** | Face matching when persons detected | Evolves `match_distance_threshold`, `min_face_size_px`, `track_timeout`, `frame_skip_cadence`. |
| **LPR Agent** | License plate OCR when vehicles detected | Evolves `segmentation_threshold`, `min_plate_height_px`, `resolution_downscale`, `detection_confidence`, `ocr_beam_width`. |
| **Consortium Broker** | Resource auctioneer | Collects agent bids → resolves allocations → posts context to shared blackboard. |
| **Logic Mutator** | Sandboxed rule gen | Synthesises, tests, and mutates Python detection-filter rules (sandboxed `eval`). |
| **Evolutionary Engine** | Cross-agent optimiser | Runs a DEAP-based genetic algorithm over the entire pipeline parameter space. |

### Infrastructure
- **Docker Compose** – Backend, frontend, Mosquitto MQTT, Qdrant vector DB, Kafka, Elasticsearch + Grafana.
- **MediaMTX** – Optional RTSP/HLS/WebRTC rebroadcast server.
- **MQTT** – Event publishing to `argus/events/{camera_id}/{rule_type}`.
- **SQLite** – Local persistent storage for cameras, zones, events, and behaviour profiles.
- **Qdrant / Kafka** – Optional vector storage and event streaming for external analytics.

---

## 🔄 Data Pipeline

```
┌──────────────┐     ┌─────────────────┐     ┌──────────────────────────────┐
│  RTSP Camera │────▶│ stream_ingestion │────▶│   ProcessingCoordinator      │
│  / Webcam    │     │ (cv2.VideoCapture│     │  (swarm OR fallback loop)    │
└──────────────┘     │  + frame queue)  │     │                              │
                     └─────────────────┘     │  ┌────────────────────────┐  │
                                             │  │  YOLO Agent (primary) │  │
                                             │  │  → detections [dict]   │  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  DeepTracker            │  │
                                             │  │  → Kalman filter        │  │
                                             │  │  → persistent track IDs │  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  LogicMutator           │  │
                                             │  │  → sandboxed rule filter│  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  Consortium Broker      │  │
                                             │  │  → post context         │  │
                                             │  │  → resolve agent bids   │  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  Face Agent (cond.)     │  │
                                             │  │  LPR Agent (cond.)      │  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  PoseEstimator          │  │
                                             │  │  AnomalyDetector        │  │
                                             │  │  SpeedHeightAnalyzer    │  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  RulesEngine             │  │
                                             │  │  → zone checks           │  │
                                             │  │  → event generation      │  │
                                             │  └───────────┬────────────┘  │
                                             └──────────────┼───────────────┘
                                                            ▼
                           ┌─────────────────────────────────────────────┐
                           │         camera_analysis cache               │
                           │  (detections, face, lpr, pose, anomalies)    │
                           └──────────┬──────────────────────┬───────────┘
                                      ▼                      ▼
                           ┌──────────────────┐   ┌──────────────────────┐
                           │  EventStore (DB) │   │  WebSocket Stream    │
                           │  + MQTTPublisher │   │  (JPEG binary + JSON)│
                           └──────────────────┘   └──────────┬───────────┘
                                                             ▼
                                                  ┌──────────────────────┐
                                                  │  LiveVideoPlayer.jsx  │
                                                  │  → canvas overlays    │
                                                  │  → bboxes + labels   │
                                                  │  → zone polygons      │
                                                  └──────────────────────┘
```

### WebSocket Protocol (`ws://localhost:8000/api/ws/stream/{camera_id}`)

1. **Binary message** – JPEG-encoded video frame (~85 % quality)
2. **Text message** – JSON object:

```json
{
  "camera_id": 1,
  "detections": [
    {
      "track_id": 42,
      "class": "person",
      "confidence": 0.95,
      "bbox": { "x1": 120, "y1": 80, "x2": 200, "y2": 360 }
    }
  ],
  "timestamp": "2025-03-20T15:30:00.123456"
}
```

---

## 📊 API Reference

### Camera Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/cameras` | List all cameras |
| `POST` | `/api/v1/cameras` | Add a camera |
| `PUT` | `/api/v1/cameras/{id}` | Update camera |
| `DELETE` | `/api/v1/cameras/{id}` | Remove camera |

### Event Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/events` | Query events (supports `camera_id`, `from_time`, `to_time`, `rule`, `priority`, `status`, pagination) |
| `GET` | `/api/v1/events/{id}` | Get single event |
| `GET` | `/api/v1/events/stats` | Aggregate statistics (24h default) |

### Zone Endpoints
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/zones?camera_id={id}` | List zones (optional filter) |
| `POST` | `/api/v1/zones` | Create zone (polygon / rectangle) |
| `PUT` | `/api/v1/zones/{id}` | Update zone |
| `DELETE` | `/api/v1/zones/{id}` | Remove zone |

### Cross-Camera Tracking
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/cross-camera/tracks` | Active global tracks |
| `GET` | `/api/v1/cross-camera/targets` | Currently targeted persons |
| `POST` | `/api/v1/cross-camera/target` | Start targeted tracking |
| `DELETE` | `/api/v1/cross-camera/target/{id}` | Stop targeted tracking |
| `GET` | `/api/v1/cross-camera/path/{id}` | Movement path for a person |
| `GET` | `/api/v1/cross-camera/predict/{id}` | Predicted trajectory |
| `POST` | `/api/v1/cross-camera/graph` | Set camera adjacency graph |
| `GET` | `/api/v1/cross-camera/graph` | Get camera adjacency graph |
| `POST` | `/api/v1/cross-camera/clear-old` | Prune tracks older than N hours |
| `GET` | `/api/v1/clusters` | DBSCAN trajectory clusters |

### Analytics & Vision
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/analysis/{camera_id}` | Full analysis: speed, height, face, LPR, pose, anomalies |
| `GET` | `/api/v1/lpr` | LPR system status |
| `GET` | `/api/v1/anomalies` | Recent anomaly events |
| `GET` | `/api/v1/trackers` | Deep tracker status + active tracks |
| `GET` | `/api/v1/poses` | Pose estimation statistics |
| `GET` | `/api/v1/faces` | Registered known faces |
| `POST` | `/api/v1/faces/register` | Register a new face |
| `DELETE` | `/api/v1/faces/{id}` | Delete a registered face |
| `GET` | `/api/v1/faces/status` | Face recognition health |

### System & Webcam
| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Full subsystem health check |
| `GET` | `/api/v1/metrics` | CPU, RAM, per-camera FPS + queue depth |
| `GET` | `/api/v1/stats/learning` | Adaptive learning metrics |
| `POST` | `/api/v1/webcam/start` | Start PC webcam (camera_id = index) |
| `GET` | `/api/v1/webcam/status` | Webcam mode status |
| `POST` | `/api/v1/webcam/stop` | Stop webcam |

### Streaming
| Endpoint | Type | Description |
|----------|------|-------------|
| `/api/ws/stream/{camera_id}` | WebSocket | Binary JPEG + JSON detections |
| `/api/mjpeg/stream/{camera_id}` | HTTP | MJPEG fallback stream |
| `/snapshots/{camera_id}/{filename}` | HTTP | Saved event snapshots |

---

## 🧪 Testing

### Quick Webcam Test (no RTSP cameras needed)
```bash
# Option A — standalone script
python backend/scripts/webcam_tester.py --camera 0

# Option B — via API
curl -X POST http://localhost:8000/api/v1/webcam/start

# Option C — automated suite
python tests/run_all_tests.py
python tests/infrastructure_healthcheck.py
```

### Test Simulator (synthetic RTSP streams)
```bash
python tests/stream_simulator.py --count 4
```

### Infrastructure Health Check
```bash
python tests/infrastructure_healthcheck.py
# Outputs a matrix: RTSP ✓ | FastAPI ✓ | SQLite ✓ | Kafka ✗ | ES ✗ | Qdrant ✗
```

---

## ❓ Troubleshooting

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Camera "Offline" | Invalid RTSP URL or network issue | Verify URL; check `docker-compose logs backend` |
| No events generated | Zones not defined, or confidence too high | Define zones; lower `confidence_threshold` in `config.yaml` |
| WebSocket connection failed | Backend not running / wrong port | Ensure uvicorn on `:8000`; check Vite proxy in `vite.config.js` |
| High CPU usage | Heavy YOLO model + no frame skipping | Use `yolov8n.pt`; enable evolutionary engine's frame skipping |
| MQTT not publishing | Broker offline / wrong address | `mosquitto_sub -h localhost -t "argus/#" -v` to verify |

---

## 📁 Project Structure

A condensed tree of the entire codebase:

```
argus/
├── README.md, FOLDER_STRUCTURE.md, LICENSE, TODO.md, INTEGRATION_TODO.md
├── config/config.yaml                    # YAML config (all AI agents, rules, cameras)
├── docker/                               # Dockerfiles
├── docs/                                 # Architecture & testing docs
├── infrastructure/                       # Kafka topics, Qdrant config
├── mediamtx/                             # RTSP server config
├── mosquitto/                            # MQTT broker config
├── assets/                               # Logo images
├── data/                                 # Runtime data (argus.db, snapshots, known_faces)
│
├── backend/
│   ├── api/main.py                       # FastAPI app, CORS, lifespan, 50+ routes
│   ├── api/models.py                     # Pydantic schemas
│   ├── api/stream_routes.py             # Snapshot & MJPEG streaming
│   ├── api/stream_ws.py                 # WebSocket binary+JSON streaming
│   ├── config/config.py                  # Pydantic config parser
│   ├── database/db.py                    # SQLite schema + CRUD
│   ├── services/
│   │   ├── core_engine/                  # AI engine (YOLO, tracking, pipeline)
│   │   ├── vision/                       # Face, LPR, pose, enhancement
│   │   ├── analytics/                    # Cross-camera Re-ID, anomaly, speed/height
│   │   └── management/                   # Rules, zones, events, MQTT, telemetry
│   └── scripts/                          # Webcam tester, DB init, admin
│
├── frontend/
│   ├── src/App.jsx                       # Dark-themed MUI app shell
│   ├── src/components/LiveVideoPlayer.jsx  # Canvas-based bbox/zone overlay
│   ├── src/pages/                        # CameraMgmt, EventFeed, Analytics, Dashboard
│   └── src/services/api.js               # Axios client (cameras, zones, events, etc.)
│
└── tests/                                # Integration + stress + health tests
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
