<div align="center">
<img src="https://raw.githubusercontent.com/Mr-Nobody-Anonymous/Argus/main/images.png" alt="Argus - The Watchful Guardian" width="400">
</div>

<h1 align="center">
   Argus — AI Video Analytics Platform
</h1>

<div align="center">
  <strong>Multi-camera AI surveillance with YOLOv8, Re-ID tracking, LPR, face recognition, zone alerting, and a benchmarked swarm agent architecture.</strong>
</div>

<div align="center">

![GitHub Repo stars](https://img.shields.io/github/stars/Mr-Nobody-Anonymous/Argus?style=social)
![License](https://img.shields.io/github/license/Mr-Nobody-Anonymous/Argus?color=blue)
![Issues](https://img.shields.io/github/issues/Mr-Nobody-Anonymous/Argus?color=red)
![Last Commit](https://img.shields.io/github/last-commit/Mr-Nobody-Anonymous/Argus?color=green)

---

[**Quickstart**](#-quick-start) |
[**Project Status**](#-project-status) |
[**Features**](#-features) |
[**Security**](#-authentication--rbac) |
[**Swarm Architecture**](#-swarm-agent-architecture) |
[**Data Pipeline**](#-data-pipeline) |
[**API Reference**](#-api-reference) |
[**Observability**](#-observability) |
[**Testing**](#-testing) |
[**Troubleshooting**](#-troubleshooting)

</div>

---

## 📖 Overview

Argus is an AI-powered video analytics platform for real-time CCTV monitoring. It ingests concurrent RTSP, video-file, or webcam streams, runs YOLOv8 object detection on each frame, assigns persistent tracking IDs via a Kalman-filtered tracker, and evaluates zone-based rules (intrusion, loitering) to generate actionable security events.

The system employs a **decentralised multi-agent swarm** where autonomous agents (YOLO, Face, LPR) bid for compute resources through a Consortium Broker. The swarm is benchmarked against a linear baseline rather than assumed better — see [Swarm Architecture](#-swarm-agent-architecture) for the measured numbers, and set `ARGUS_NO_SWARM=1` to run the linear path.

**Metaphor:** In Greek mythology, Argus Panoptes was a hundred-eyed giant who served as a watchful guardian. This platform embodies that vigilance.

> 📖 **File-by-file breakdown: [FOLDER_STRUCTURE.md](FOLDER_STRUCTURE.md)** · **Production gap analysis: [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md)**

---

## 📍 Project Status

**Working and verified end-to-end** (live run against a looping demo clip on CPU):

| Area | Status | Evidence |
|---|---|---|
| Detection + tracking pipeline | ✅ Working | 12–15 detections/frame, ~16 stable track IDs for a ~12-person scene |
| JWT auth + 3-role RBAC | ✅ Enforced on all 40 protected API routes | `tests/test_api_security.py` — 21 tests |
| Audit trail | ✅ Writing | Every mutation + auth attempt, including `denied` rows |
| Data retention | ✅ Scheduled | Runs every 6h; surfaced in `/api/v1/health` |
| Prometheus metrics | ✅ Exposed | `GET /metrics` |
| Frontend login + token refresh | ✅ Working | Verified through the Vite proxy |
| Swarm vs linear benchmark | ✅ Measured | +26.8% FPS at identical detection quality |
| Regression suite | ✅ 44 tests green | `tests/test_regression.py`, `tests/test_api_security.py` |

**Not yet production-ready** — these are known gaps, tracked with acceptance criteria in [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md):

| Gap | Impact |
|---|---|
| No TLS termination built in | Tokens travel in cleartext unless you front it with a reverse proxy |
| Face embeddings stored unencrypted | Biometric data at rest needs encryption before real deployment |
| SQLite only | Single-writer; PostgreSQL migration is specified but not built |
| No CI/CD pipeline | Tests exist but nothing runs them automatically |
| Evolutionary engine has no labelled eval set | It optimises against no ground truth, so it can converge on nothing meaningful |

Performance note: on CPU the pipeline runs at **~1.3–1.6 FPS per camera** with YOLOv8n. The 30 FPS figures in older docs assumed GPU inference.

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

# Required: a signing key of at least 32 characters.
# Without it the server generates an ephemeral key, so every restart
# invalidates all issued tokens (fine for a demo, useless in production).
export ARGUS_JWT_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"

python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Create a login

Argus authenticates against the **Django `auth_user` table** — there is no
second user store to drift out of sync. Create the first superuser:

```bash
python backend/scripts/run_admin.py 0.0.0.0:8001   # then visit /admin
# or create one directly:
python -c "
import django, os
os.environ.setdefault('DJANGO_SETTINGS_MODULE','backend.django_admin.settings')
django.setup()
from django.contrib.auth.models import User
User.objects.create_superuser('admin','admin@example.com','changeme')
"
```

**Roles** are derived from Django group membership: a user in the `operator`
group gets the operator role, `viewer` gets viewer, and any superuser or staff
account is an admin. A user in no group defaults to **viewer** (least privilege).

### 3. Frontend
```bash
cd frontend
npm install
npm run dev            # Serves on http://localhost:3000
```

The dashboard opens on a login screen. Tokens are held in `sessionStorage` and
refreshed transparently, so a 30-minute access-token expiry will not sign an
operator out mid-shift.

**💡 Windows shortcut:** double-click `run_app.bat`.

### 4. Access
| Service          | URL                          |
|------------------|------------------------------|
| Dashboard        | http://localhost:3000        |
| API Docs         | http://localhost:8000/docs   |
| Prometheus metrics | http://localhost:8000/metrics |
| WebSocket Stream | `ws://localhost:8000/api/ws/stream/{camera_id}?token=<jwt>` |
| Django Admin     | http://localhost:8001/admin  |

### 5. What a fresh clone does *not* include

`data/` and the model weights are gitignored, so after cloning you will need:

| Missing | How to get it |
|---|---|
| `backend/models/yolov8n.pt` | Downloaded automatically on first run by ultralytics, or fetch it manually |
| `data/argus.db` | Created automatically at startup; add users via Django admin (step 2) |
| `data/demo_clip.mp4` | **Included** — it is the one exception, because the test suite needs it |

Regenerate the demo clip if you ever need to:

```bash
python backend/scripts/make_demo_clip.py --source path/to/street_photo.jpg
```

It reports how many objects YOLO finds in the first frame. Use a real
photograph — the detector returns nothing on drawn shapes, so a synthetic clip
looks identical to a broken pipeline.

### 6. Try it without a camera

No RTSP stream handy? Point a camera row at a local video file — the ingestion
layer detects file sources, paces them to their native FPS, and loops on EOF:

```bash
curl -X POST http://localhost:8000/api/v1/cameras \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"Demo","rtsp_url":"data/demo_clip.mp4","location":"Test"}'
```

---

## 🌟 Features

### Core Capabilities
- **Multi-Camera Ingestion** – Concurrent RTSP / video-file / webcam streams with auto-reconnect, exponential backoff, native-FPS pacing for files, and EOF looping.
- **YOLOv8 Detection** – Person, vehicle, bicycle, and object detection with configurable confidence / IoU thresholds. Ships with `yolov8n.pt`.
- **Object Tracking** – Kalman-filtered motion prediction with two-stage association (IoU, then size-scaled centre distance) so identity survives the ~1s gap between processed frames on CPU.
- **Zone-Based Rules** – Polygon virtual zones with intrusion and loitering detection, keyed on persistent track IDs.
- **Event Management** – SQLite-backed event store with **sliding-window** deduplication (one event per subject per episode, not one per window), snapshot capture, and MQTT publishing.
- **Real-Time WebSocket Streaming** – Interleaved binary (JPEG) + JSON (detection metadata), authenticated at the handshake.
- **Data Retention** – Scheduled purge of events, snapshots, anomalies, plates, and audit rows on independent per-type schedules.
- **Image Enhancement** – CLAHE, denoising, sharpening, night vision, deblur, and HDR auto-enhancement.
- **Cross-Camera Re-ID** – Person re-identification across non-overlapping cameras using feature embeddings and cosine similarity.

**Degrades gracefully when optional models are absent** — the system logs the
fallback at startup rather than failing:

| Feature | With optional dependency | Without it |
|---|---|---|
| License Plate Recognition | PaddleOCR text recognition | Basic OCR heuristics |
| Face Recognition | `opencv-contrib` LBPH matching | Haar-cascade detection + histogram matching |
| Pose Estimation | MediaPipe 33-keypoint skeletons | Geometric fallback pose estimation |

---

## 🔐 Authentication & RBAC

**Every API route requires a bearer token** except four intentional exceptions:
`GET /api/v1/health`, `GET /`, `POST /api/v1/auth/login`, and
`POST /api/v1/auth/refresh`.

### Roles

| Role | Can do |
|---|---|
| **viewer** | Read cameras, zones, events, analysis, anomalies, trackers, poses, metrics; watch live streams |
| **operator** | Everything a viewer can, plus create/update/delete zones, run cross-camera tracking, control the webcam, submit video for processing |
| **admin** | Everything, plus camera CRUD, the face database (register/list/delete), and the audit log |

Roles come from Django group membership and are **re-read from the database on
every request**. A token whose `role` claim has been tampered with is therefore
harmless — the claim is never trusted, only the user ID is.

### Auth endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/v1/auth/login` | Exchange username/password for an access + refresh token pair |
| `POST` | `/api/v1/auth/refresh` | Exchange a refresh token for a new access token |
| `GET` | `/api/v1/auth/me` | Current user identity and role |
| `GET` | `/api/v1/audit` | Query the audit trail (admin only) |

```bash
TOKEN=$(curl -s -X POST http://localhost:8000/api/v1/auth/login \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"changeme"}' | python -c "import sys,json;print(json.load(sys.stdin)['access_token'])")

curl http://localhost:8000/api/v1/cameras -H "Authorization: Bearer $TOKEN"
```

### Hardening that is actually in place

- **Brute-force lockout** — 5 failed attempts locks an account for 300 s; the
  correct password is refused while locked (otherwise the lockout is decorative).
- **Token separation** — a refresh token is rejected where an access token is
  required, so a long-lived credential cannot be replayed as a short-lived one.
- **WebSocket auth before `accept()`** — unauthenticated upgrades are refused at
  the handshake with a 403 rather than accepted and then closed.
- **Static mounts gated** — `/snapshots` is only mounted when auth is disabled,
  because a `StaticFiles` mount bypasses route dependencies entirely.
- **Audit trail** — every mutation and auth attempt is recorded with actor, role,
  client IP, and outcome (including `denied`). Writes never raise; a failed audit
  write must not take down the request path.
- **No plaintext secrets in config** — `config.yaml` supports `${VAR}`
  interpolation and startup refuses to run with unresolved secret placeholders.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `ARGUS_JWT_SECRET` | *(ephemeral)* | HS256 signing key, min 32 chars. **Set this.** |
| `ARGUS_ACCESS_TOKEN_MINUTES` | `30` | Access token lifetime |
| `ARGUS_REFRESH_TOKEN_DAYS` | `7` | Refresh token lifetime |
| `ARGUS_LOGIN_MAX_ATTEMPTS` | `5` | Failures before lockout |
| `ARGUS_LOGIN_LOCKOUT_SECONDS` | `300` | Lockout duration |
| `ARGUS_DISABLE_AUTH` | unset | Disables auth entirely — **local development only** |
| `ARGUS_CORS_ORIGINS` | localhost:3000/5173 | Comma-separated allowlist |
| `ARGUS_NO_SWARM` | unset | `1` forces the linear pipeline |
| `ARGUS_LOG_FORMAT` | `text` | `json` for structured log ingestion |
| `ARGUS_LOG_LEVEL` | `INFO` | Root log level |

---

### 🤖 Swarm Agent Architecture
| Agent | Role | Local Evolution |
|-------|------|----------------|
| **YOLO Detection Agent** | Primary object detection | Optimises `yolo_conf_threshold`, `iou_threshold`, `input_resolution_scale`, `tracker_matching_threshold` via a genetic algorithm. |
| **Face Recognition Agent** | Face matching when persons detected | Evolves `match_distance_threshold`, `min_face_size_px`, `track_timeout`, `frame_skip_cadence`. |
| **LPR Agent** | License plate OCR when vehicles detected | Evolves `segmentation_threshold`, `min_plate_height_px`, `resolution_downscale`, `detection_confidence`, `ocr_beam_width`. |
| **Consortium Broker** | Resource auctioneer | Collects agent bids → resolves allocations → posts context to shared blackboard. |
| **Logic Mutator** | Sandboxed rule gen | Synthesises, tests, and mutates Python detection-filter rules (sandboxed `eval`). |
| **Evolutionary Engine** | Cross-agent optimiser | Runs a DEAP-based genetic algorithm over the entire pipeline parameter space. |

#### Does the swarm actually help? (measured, not asserted)

40 frames of a 640×480 clip, CPU inference, each variant in a **separate
process** so module-level singletons cannot leak state between runs:

| Mode | FPS | p50 latency | p95 latency | Detections/frame | Zero-detection frames |
|---|---|---|---|---|---|
| Linear baseline | 1.27 | 784.20 ms | 840.84 ms | 12.8 | 0 |
| **Swarm** | **1.61** | **612.78 ms** | **750.53 ms** | **12.8** | 0 |

**+26.8% throughput, −21.9% p50 latency, identical detection quality.** The gain
comes from deferring optional enrichment (face, LPR) under load while the
detector still runs on every frame.

Reproduce it yourself:

```bash
python tests/swarm_benchmark.py --frames 40 --json docs/swarm_benchmark_results.json
```

> ⚠️ **The first honest run of this benchmark reported +45.2% FPS and −35.5%
> detections.** The "speedup" was the broker starving the detector until it
> replayed a stale result forever. A performance number without a quality number
> next to it is not a result — see [PRODUCTION_ROADMAP.md §5](PRODUCTION_ROADMAP.md)
> for the full post-mortem.

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

### WebSocket Protocol (`ws://localhost:8000/api/ws/stream/{camera_id}?token=<jwt>`)

The connection is authenticated **before** the upgrade is accepted; an unauthenticated
handshake is refused with HTTP 403. Two message types are interleaved:

1. **Binary message** – JPEG-encoded video frame
2. **Text message** – JSON detection metadata:

```json
{
  "camera_id": 2,
  "detections": [
    {
      "track_id": 1,
      "class": "person",
      "confidence": 0.8251198530197144,
      "bbox": { "x1": 495, "y1": 237, "x2": 590, "y2": 442 }
    }
  ],
  "timestamp": "2026-08-17T16:59:47.691052Z",
  "timestamp_unix": 1786985987.6910653
}
```

Field notes: the class label key is **`class`** (not `class_name`), `bbox` is an
**object** with `x1/y1/x2/y2` (the frontend also accepts a 4-element array), and
`timestamp` is ISO 8601 UTC with `timestamp_unix` alongside for arithmetic.

---

## 📊 API Reference

**48 addressable operations**: 47 in the OpenAPI schema (43 under `/api/v1`,
3 streaming, plus `GET /`) and 1 WebSocket route, with `GET /metrics` served
outside the schema.

Of the 43 `/api/v1` operations, **40 require a token and 3 are public**.
Everything requires `Authorization: Bearer <token>` except the entries marked
*public* below.

The **Role** column is the *minimum* role required.

Interactive docs: http://localhost:8000/docs

### Authentication
| Method | Path | Role | Description |
|--------|------|------|-------------|
| `POST` | `/api/v1/auth/login` | *public* | Username/password → access + refresh tokens |
| `POST` | `/api/v1/auth/refresh` | *public* | Refresh token → new access token |
| `GET` | `/api/v1/auth/me` | viewer | Current identity and role |
| `GET` | `/api/v1/audit` | **admin** | Query the audit trail (filter by user, action, outcome) |

### Cameras
| Method | Path | Role | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/cameras` | viewer | List all cameras |
| `GET` | `/api/v1/cameras/{id}` | viewer | Get one camera |
| `POST` | `/api/v1/cameras` | **admin** | Add a camera |
| `PUT` | `/api/v1/cameras/{id}` | **admin** | Update camera |
| `DELETE` | `/api/v1/cameras/{id}` | **admin** | Remove camera |

### Events
| Method | Path | Role | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/events` | viewer | Query events (`camera_id`, `rule_type`, `priority`, `limit`, `offset`) |
| `GET` | `/api/v1/events/{id}` | viewer | Get single event |
| `GET` | `/api/v1/events/stats` | viewer | Aggregate statistics (24h default) |

> **Note:** the query/response field is **`rule_type`** (`intrusion`, `loitering`),
> not `event_type`.

### Zones
| Method | Path | Role | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/zones?camera_id={id}` | viewer | List zones (optional filter) |
| `POST` | `/api/v1/zones` | operator | Create zone |
| `PUT` | `/api/v1/zones/{id}` | operator | Update zone |
| `DELETE` | `/api/v1/zones/{id}` | operator | Remove zone |

> **Note:** the polygon field is **`coordinates`**, not `polygon`:
> `{"camera_id":2,"name":"Dock","zone_type":"restricted","coordinates":[[0,0],[640,0],[640,480],[0,480]]}`

### Cross-Camera Tracking
| Method | Path | Role | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/cross-camera/tracks` | operator | Active global tracks |
| `GET` | `/api/v1/cross-camera/targets` | operator | Currently targeted persons |
| `POST` | `/api/v1/cross-camera/target` | **admin** | Start targeted tracking |
| `DELETE` | `/api/v1/cross-camera/target/{id}` | **admin** | Stop targeted tracking |
| `GET` | `/api/v1/cross-camera/path/{id}` | operator | Movement path for a person |
| `GET` | `/api/v1/cross-camera/predict/{id}` | operator | Predicted trajectory |
| `GET` | `/api/v1/cross-camera/graph` | operator | Get camera adjacency graph |
| `POST` | `/api/v1/cross-camera/graph` | **admin** | Set camera adjacency graph |
| `POST` | `/api/v1/cross-camera/clear-old` | **admin** | Prune tracks older than N hours |
| `GET` | `/api/v1/clusters` | viewer | DBSCAN trajectory clusters |

### Analytics & Vision
| Method | Path | Role | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/analysis/{camera_id}` | viewer | Full analysis: detections, speed, height, face, LPR, pose, anomalies |
| `GET` | `/api/v1/anomalies` | viewer | Recent anomaly events |
| `GET` | `/api/v1/trackers` | viewer | Deep tracker status + active tracks |
| `GET` | `/api/v1/poses` | viewer | Pose estimation statistics |
| `GET` | `/api/v1/lpr` | operator | LPR system status |
| `POST` | `/api/v1/enhance/analyze` | operator | Image enhancement analysis |
| `POST` | `/api/v1/video/process` | operator | Process a video file through the pipeline |
| `GET` | `/api/v1/faces` | **admin** | Registered known faces |
| `POST` | `/api/v1/faces/register` | **admin** | Register a new face |
| `DELETE` | `/api/v1/faces/{id}` | **admin** | Delete a registered face |
| `GET` | `/api/v1/faces/status` | operator | Face recognition health |

### System & Webcam
| Method | Path | Role | Description |
|--------|------|------|-------------|
| `GET` | `/` | *public* | API banner: name, version, and docs link |
| `GET` | `/api/v1/health` | *public* | Subsystem health, security posture, retention status |
| `GET` | `/metrics` | *public* | **Prometheus** text exposition |
| `GET` | `/api/v1/metrics` | viewer | Dashboard JSON metrics (CPU, RAM, per-camera FPS) |
| `GET` | `/api/v1/stats/learning` | viewer | Adaptive learning metrics |
| `POST` | `/api/v1/webcam/start` | operator | Start PC webcam |
| `GET` | `/api/v1/webcam/status` | viewer | Webcam mode status |
| `POST` | `/api/v1/webcam/stop` | operator | Stop webcam |

### Streaming
| Endpoint | Type | Role | Description |
|----------|------|------|-------------|
| `/api/ws/stream/{camera_id}?token=<jwt>` | WebSocket | viewer | Binary JPEG + JSON detections |
| `/api/stream/{camera_id}` | HTTP | viewer | MJPEG stream |
| `/api/mjpeg/stream/{camera_id}` | HTTP | viewer | MJPEG fallback stream |
| `/api/snapshots/{camera_id}/{filename}` | HTTP | viewer | Saved event snapshots |

Browsers cannot set headers on a WebSocket handshake, so the stream endpoint
takes the token as a **query parameter**. The server validates it *before*
accepting the connection.

---

## 📈 Observability

```bash
curl http://localhost:8000/metrics
```

Prometheus text exposition — no client library dependency. Exposed metrics:

| Metric | Type | Labels |
|---|---|---|
| `argus_uptime_seconds` | gauge | — |
| `argus_process_cpu_percent` | gauge | — |
| `argus_process_memory_bytes` | gauge | — |
| `argus_camera_fps` | gauge | `camera_id`, `name` |
| `argus_camera_up` | gauge | `camera_id`, `name` |
| `argus_camera_queue_depth` | gauge | `camera_id`, `name` |
| `argus_cameras_total` | gauge | `status` |
| `argus_inference_latency_ms` | gauge | — |
| `argus_model_loaded` | gauge | — |
| `argus_detections_current` | gauge | `camera_id` |
| `argus_active_tracks` | gauge | — |
| `argus_events_24h_total` | gauge | `rule` |
| `argus_auth_enabled` | gauge | — |
| `argus_ephemeral_jwt_secret` | gauge | — |

`/metrics` is unauthenticated by design so a scraper needs no credentials — it
carries operational counters only, no frames, identities, or event contents.
Restrict it at the network layer if your threat model requires it.

**Structured logging:** set `ARGUS_LOG_FORMAT=json` to emit one JSON object per
log line, with any `extra={...}` fields merged in — so logs can be filtered by
`camera_id` instead of grepped.

```json
{"timestamp": "2026-08-17T16:44:05", "level": "INFO", "service": "argus.ingestion", "message": "camera online", "camera_id": 2, "fps": 14.8}
```

> These metrics are not decoration. Within a minute of `/metrics` going live it
> exposed two real bugs — 74 active tracks for a 12-person scene, and 377
> loitering events from a single camera — that code review had missed entirely.

---

## 🧪 Testing

### The suites that gate correctness

```bash
pytest tests/test_regression.py tests/test_api_security.py -v
```

**44 tests, all passing.**

| Suite | Tests | Guards against |
|---|---|---|
| `tests/test_regression.py` | 23 | Pipeline defects: Kalman shape/transition errors, track identity churn at realistic frame rates, primary-detector starvation, skipped frames reported as empty, event-dedup storms |
| `tests/test_api_security.py` | 21 | Unauthenticated routes, forged/expired/foreign-signed tokens, refresh-as-access replay, privilege escalation via a tampered `role` claim, plaintext secrets in config |

These are **mutation-verified** — deliberately reintroducing a bug (e.g. the
buggy Kalman `transitionMatrix`) makes the relevant test fail with a readable
message, so the suite is proven to detect the regression it claims to cover.

`test_api_security.py` includes a static sweep that walks the live route table
and asserts every `/api` route outside the public allowlist carries an auth
dependency — so a **newly added endpoint cannot silently ship unauthenticated**.

### Swarm A/B benchmark

```bash
python tests/swarm_benchmark.py --frames 40 --json docs/swarm_benchmark_results.json
```

Runs each pipeline variant in a **separate process** (module-level singletons
would otherwise leak state between runs and bias the result). Options:
`--clip`, `--frames`, `--json`, `--camera-id`.

### Manual / diagnostic scripts

These are exploratory tools, not assertions — they predate the pytest suites and
several require infrastructure that is not part of the default stack:

```bash
python backend/scripts/webcam_tester.py --camera 0   # needs a webcam
python tests/infrastructure_healthcheck.py           # needs Kafka/ES/Qdrant
python tests/stream_simulator.py --count 4           # needs ffmpeg
python tests/buffer_monitor.py                       # runs until interrupted
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
| **401 on every API call** | Auth is enabled and no token was sent | Log in first; the dashboard does this for you. For curl, see [Authentication](#-authentication--rbac) |
| **403 on a WebSocket** | Token missing from the handshake | Use `?token=<jwt>` — browsers cannot set headers on a WS upgrade |
| **Tokens stop working after restart** | `ARGUS_JWT_SECRET` unset, so an ephemeral key was generated | Set `ARGUS_JWT_SECRET` to a fixed value ≥32 chars |
| **429 with "Try again in N seconds"** | 5 failed logins locked the account | Wait out the lockout, or tune `ARGUS_LOGIN_LOCKOUT_SECONDS` |
| Login fails for a known-good user | Password not hashed by Django, or user inactive | Create/reset the user through Django admin |
| **`Address already in use` on :8000** | A previous uvicorn is still holding the port | `ss -lntp \| grep :8000` then `kill -9 <pid>` |
| Detections appear but `track_id` is always 0 | Tracker silently failed and the exception was swallowed | Check logs for `Error updating tracker`; run `pytest tests/test_regression.py -k Track` |
| Event feed floods with duplicates | Old build without sliding-window dedup | Update; dedup now keys on track ID and slides while a condition persists |
| Zone POST returns 422 | Wrong field name | The polygon field is `coordinates`, not `polygon` |
| Frontend shows nothing but the login box | Backend unreachable through the Vite proxy | Confirm uvicorn on `:8000`; check `vite.config.js` proxy |
| `npm run dev`/`vite build` dies with "Killed" or "The service was stopped" | Out of memory — esbuild and a torch-loaded backend on the same small machine | Build with the backend stopped, or give the machine more RAM (~2 GB is not enough for both) |

---

## 📁 Project Structure

```
argus/
├── README.md, FOLDER_STRUCTURE.md, PRODUCTION_ROADMAP.md, TODO.md,
│   INTEGRATION_TODO.md, LICENSE
├── requirements.txt                      # Core deps (incl. PyJWT)
├── requirements-optional.txt             # Heavy/optional extras (django, kafka, qdrant…)
├── .env.example                          # Every ARGUS_* variable, documented
├── config/config.yaml                    # Agents, rules, retention; supports ${VAR}
├── docker/, docker-compose.yml           # Container builds
├── infrastructure/, mediamtx/, mosquitto/
├── data/                                 # argus.db, snapshots, known_faces, demo_clip.mp4
│
├── backend/
│   ├── api/
│   │   ├── main.py                       # FastAPI app, all routes, RBAC, audit middleware
│   │   ├── auth.py                       # JWT, roles, Django password verification, lockout
│   │   ├── observability.py              # Prometheus exposition + JSON log formatter
│   │   ├── models.py                     # Pydantic schemas
│   │   ├── stream_routes.py              # Snapshot & MJPEG (auth-gated)
│   │   └── stream_ws.py                  # WebSocket streaming (auth before accept)
│   ├── config/config.py                  # Config parser, ${VAR} interpolation, secret redaction
│   ├── database/db.py                    # SQLite schema + CRUD
│   ├── django_admin/                     # Django admin over the same DB (auth_user lives here)
│   ├── models/yolov8n.pt                 # Detection weights
│   ├── services/
│   │   ├── core_engine/                  # inference_engine, deep_tracker, yolo_detection_agent,
│   │   │                                 #   consortium_broker, processing_coordinator,
│   │   │                                 #   evolutionary_engine, logic_mutator
│   │   ├── vision/                       # face_recognition, lpr, pose, image_enhancement
│   │   ├── analytics/                    # person_reid, cross_camera_tracker, anomaly_detector
│   │   └── management/                   # rules_engine, zone_manager, event_store, retention,
│   │                                     #   audit_log, mqtt_publisher, stream_ingestion
│   └── scripts/                          # webcam_tester, init_db, run_admin, create_test_events
│
├── frontend/
│   ├── src/App.jsx                       # Shell + auth gate (login vs dashboard)
│   ├── src/pages/Login.jsx               # Login screen
│   ├── src/pages/                        # SurveillanceDashboard, CameraManagement,
│   │                                     #   EventFeed, AnalyticsDashboard, AdaptiveLearning
│   ├── src/components/LiveVideoPlayer.jsx  # Canvas bbox/zone overlays over the WS stream
│   └── src/services/api.js               # Axios client, token store, transparent refresh
│
├── tests/
│   ├── test_regression.py                # 23 pipeline-correctness tests
│   ├── test_api_security.py              # 21 auth/RBAC/secret tests
│   ├── swarm_benchmark.py                # Swarm vs linear A/B harness
│   └── (diagnostic scripts: buffer_monitor, stream_simulator, …)
│
└── docs/                                 # Architecture, testing guide, research notes,
                                          #   swarm_benchmark_results.json
```

---

## 📄 License

MIT License — see [LICENSE](LICENSE).
