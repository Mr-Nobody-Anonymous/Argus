<div align="center">
<img src="https://raw.githubusercontent.com/Mr-Nobody-Anonymous/Argus/images.png" alt="Argus - The Watchful Guardian" width="400">
</div>

<h1 align="center">
   Argus — AI Video Analytics Platform
</h1>

<div align="center">
  <strong>Multi-camera AI surveillance with YOLOv8, Re-ID tracking, LPR, face recognition, zone alerting, and a benchmarked swarm agent architecture.</strong>
</div>

<div align="center">

[![CI](https://github.com/Mr-Nobody-Anonymous/Argus/actions/workflows/ci.yml/badge.svg)](https://github.com/Mr-Nobody-Anonymous/Argus/actions/workflows/ci.yml)
[![Security Scan](https://github.com/Mr-Nobody-Anonymous/Argus/actions/workflows/security.yml/badge.svg)](https://github.com/Mr-Nobody-Anonymous/Argus/actions/workflows/security.yml)
[![Docker](https://github.com/Mr-Nobody-Anonymous/Argus/actions/workflows/docker.yml/badge.svg)](https://github.com/Mr-Nobody-Anonymous/Argus/actions/workflows/docker.yml)
![License](https://img.shields.io/github/license/Mr-Nobody-Anonymous/Argus?color=blue)
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

**Current Development Status:** **Beta / Hardened Edge Prototype**

| Architecture Component | Status | Implementation Details & Evidence |
|---|---|---|
| **Detection & Tracking Pipeline** | ✅ Working | YOLOv8n + DeepTracker (Kalman filter + Hungarian matching). CPU baseline: ~1.3–1.6 FPS/cam; GPU: ~30 FPS/cam. |
| **Authentication & RBAC** | ✅ Enforced | JWT + 3-role RBAC (`admin`, `operator`, `viewer`). Sliding-window rate limiter protects auth routes against brute-force. |
| **Face Recognition & Biometrics** | 🔒 Opt-in | Disabled by default. When enabled, embeddings are AES-256-GCM encrypted; new enrollments do not save source face crops. Older plaintext image files require explicit cleanup. |
| **Cryptographic Audit Trail** | ✅ Tamper-Evident | Append-only SHA-256 hash chaining with integrity verification (`GET /api/v1/audit/integrity`). |
| **Database Architecture** | ✅ Dual-Tier | SQLite (default zero-install edge) + PostgreSQL adapter (`backend/database/postgres.py`) with migration tooling. |
| **CI & Security Workflows** | ⚙️ Configured | GitHub Actions cover Python 3.11/3.13, frontend builds, security scans, and Docker image checks; badges above link to the latest workflow runs, when available. |
| **Scientific Evaluation Set** | ⚠️ Not published | The evaluation harness exists, but this repository does not include a public labeled ground-truth dataset yet. |
| **Swarm Agent Benchmark** | 📊 One recorded run | On a bundled 40-frame CPU sample: 1.27 FPS linear vs 1.61 FPS swarm, with the same detection count; no ground-truth accuracy comparison or hardware metadata. |
| **TLS / Production Exposure** | ⚠️ Reverse Proxy Required | Cleartext HTTP is development-only. Production requires fronting with Caddy/Nginx (recipes in `infrastructure/`). |

> 📖 **Security Policy: [SECURITY.md](SECURITY.md)** · **Database Migration: [docs/DATABASE_MIGRATION.md](docs/DATABASE_MIGRATION.md)** · **Roadmap: [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md)**

---

## 🚀 Quick Start

### One command, any OS

```bash
python argus.py start
```

That is the whole install. It works the same on **Windows, macOS and Linux**,
and it is safe to re-run — everything below is skipped once it is already done:

1. generates a strong `ARGUS_JWT_SECRET` into `.env` (first run only)
2. picks a runtime — **Docker** if the daemon is responding, otherwise a local
   `.venv` (force either with `--native` / `--docker`)
3. installs dependencies, using **CPU PyTorch wheels** so the download is
   ~200 MB instead of ~2.5 GB of unusable CUDA payload
4. creates the database and seeds the `admin` user
5. builds the dashboard and serves it from the API on **one port**
6. waits for `/api/v1/health`, then opens your browser

```
  Argus is running
    Dashboard   http://localhost:8000
    API docs    http://localhost:8000/docs
    Login       admin (set with: python argus.py create-admin)

    Stop it     python argus.py stop
```

**Prefer not to use a terminal?** Double-click `start.bat` (Windows) or
`start.command` (macOS/Linux). `stop.bat` / `stop.command` shut it down.

**Prefer `make`?** `make start` and `make stop` wrap the same launcher, and
`make docker-start` / `make docker-stop` run it in a container (generating the
required signing key on first use). `make` on its own lists every target. The
Makefile is a thin wrapper, never a second implementation — Windows has no
`make`, so the logic stays in `argus.py`.

| Command | What it does |
|---|---|
| `python argus.py start` | Set everything up and run it |
| `python argus.py stop` | Stop everything (graceful, then forced) |
| `python argus.py status` | Show mode, port, PID and health |
| `python argus.py create-admin` | Create or update admin credentials (interactive prompt) |
| `python argus.py doctor` | Check this machine *before* installing |
| `python argus.py reset` | Delete the venv/build (`--all` also drops the DB) |

Useful flags: `--port 9000`, `--native`, `--docker`, `--rebuild`,
`--reinstall`, `--no-browser`, `--host 127.0.0.1`.

If something goes wrong, `start` prints the tail of `.argus/backend.log` and
tells you where the full log is. Run `python argus.py doctor` first if you want
to check Python, Node, Docker, disk space and port availability up front.

> The launcher itself imports **only the Python standard library** — it has to
> run before any dependency exists. A test enforces that.

---

### Manual setup (for development)

Prefer hot-reload and separate processes? The original flow still works.

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

python -m uvicorn backend.api.main:app --reload --host 127.0.0.1 --port 8000
```

### 2. Create a login

Argus authenticates against the **Django `auth_user` table** — there is no
second user store to drift out of sync. Create the first superuser:

```bash
# Create the Django auth tables and initialize a secure superuser:
python backend/scripts/run_admin.py --setup-only

# Set a password interactively before first login
python argus.py create-admin --username admin
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

**💡 Shortcut:** `python argus.py start` does all of the above in one step and serves the built dashboard on port 8000 instead.

### 4. Access
| Service          | URL                          |
|------------------|------------------------------|
| Dashboard (`argus.py start`) | http://localhost:8000 |
| Dashboard (`npm run dev`)    | http://localhost:3000 |
| ├ Cameras / Events / Analytics | `/` · `/events` · `/analytics` |
| └ Adaptive Learning | `/learning` |
| API banner (JSON) | http://localhost:8000/api |
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
- **Zone-Based Rules** – Polygon virtual zones (intrusion, loitering) and line
  tripwires (`line_crossing`), keyed on persistent track IDs. Scene-scoped rules
  (`speed_violation`, `fall_detection`, `abandoned_object`) need no zone and run
  on every camera.
- **Rule Honesty** – `GET /api/v1/rules/status` reports, per rule, whether it is
  configured, implemented, and *actually able to fire here*. A rule whose
  prerequisite is missing (no ground-plane calibration, no pose keypoints)
  reports its blocker instead of quietly producing nothing — or worse, producing
  a fabricated number.
- **Observations Become Events** – Perception findings (dwell, pacing,
  abandonment, occupancy anomalies, disappearances, scene changes) are promoted
  into the operator event feed with the evidence that justified them. Only
  grounded, actionable observations are promoted; bookkeeping kinds stay out.
- **Event Management** – SQLite-backed event store with **sliding-window**
  deduplication (one event per subject per episode, not one per window),
  snapshot capture, and MQTT publishing. Events follow an enforced lifecycle —
  `detected → open → acknowledged → resolved`, or `false_positive` — recording
  who acted and when; an illegal transition is rejected, not silently written.
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
| **Logic Mutator** | AST-constrained rule generation | Generates and evaluates internal Python filter rules with AST checks and restricted globals. This is not a sandbox for hostile code. |
| **Evolutionary Engine** | Cross-agent optimiser | Runs a self-contained genetic algorithm (elitism, crossover, Gaussian mutation) over the pipeline parameter space. No external GA library is used. |

#### What one recorded swarm run measured

40 frames of a 640×480 clip, CPU inference, each variant in a **separate
process** so module-level singletons cannot leak state between runs:

| Mode | FPS | p50 latency | p95 latency | Detections/frame | Zero-detection frames |
|---|---|---|---|---|---|
| Linear baseline | 1.27 | 784.20 ms | 840.84 ms | 12.8 | 0 |
| **Swarm** | **1.61** | **612.78 ms** | **750.53 ms** | **12.8** | 0 |

**In this recorded run only:** the swarm processed 1.61 vs 1.27 FPS (+26.8%) and p50 latency was 612.78 vs 784.20 ms (−21.9%). Both modes returned the same mean count (12.8 detections/frame), but this benchmark has no ground-truth labels, so it does not establish equal detection accuracy. Hardware metadata was not captured; treat these figures as one sample, not a general performance claim.

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
│  RTSP Camera │───▶│ stream_ingestion │───▶│   ProcessingCoordinator      │
│  / Webcam    │     │ (cv2.VideoCapture│    │  (swarm OR fallback loop)    │
└──────────────┘     │  + frame queue)  │    │                              │
                     └─────────────────┘     │  ┌────────────────────────┐  │
                                             │  │  YOLO Agent (primary)  │  │
                                             │  │  → detections [dict]   │  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  DeepTracker           │  │
                                             │  │  → Kalman filter       │  │
                                             │  │  → persistent track IDs│  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  LogicMutator          │  │
                                             │  │  →AST-constrained rule filter│  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  Consortium Broker     │  │
                                             │  │  → post context        │  │
                                             │  │  → resolve agent bids  │  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  Face Agent (cond.)    │  │
                                             │  │  LPR Agent (cond.)     │  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  PoseEstimator         │  │
                                             │  │  AnomalyDetector       │  │
                                             │  │  SpeedHeightAnalyzer   │  │
                                             │  └───────────┬────────────┘  │
                                             │              ▼               │
                                             │  ┌────────────────────────┐  │
                                             │  │  RulesEngine           │  │
                                             │  │  → zone checks         │  │
                                             │  │  → event generation    │  │
                                             │  └───────────┬────────────┘  │
                                             └──────────────┼───────────────┘
                                                            ▼
                           ┌─────────────────────────────────────────────┐
                           │         camera_analysis cache               │
                           │  (detections, face, lpr, pose, anomalies)   │
                           └──────────┬──────────────────────┬───────────┘
                                      ▼                      ▼
                           ┌──────────────────┐   ┌──────────────────────┐
                           │  EventStore (DB) │   │  WebSocket Stream    │
                           │  + MQTTPublisher │   │  (JPEG binary + JSON)│
                           └──────────────────┘   └──────────┬───────────┘
                                                             ▼
                                                  ┌──────────────────────┐
                                                  │  LiveVideoPlayer.jsx │
                                                  │  → canvas overlays   │
                                                  │  → bboxes + labels   │
                                                  │  → zone polygons     │
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

## 🖥️ Operator dashboard

The default screen is a **command center**: a live video wall with detection
overlays, an alert feed, and — deliberately — the state of the things that
silently fail.

* **Video wall.** Frames arrive over the documented WebSocket protocol and
  detections are drawn client-side as corner brackets with track ID and
  confidence. Brackets occlude far less of a subject than full rectangles,
  which matters when the operator is trying to identify the person inside the
  box. Overlay geometry accounts for letterboxing: the frame is rendered with
  `object-fit: contain`, so scaling by the element size would smear every box
  across the black bars.
* **Liveness is measured, not assumed.** A stalled camera holds its socket open
  indefinitely, so an open connection proves nothing. Each tile runs its own
  frame clock and degrades `LIVE → STALE → NO SIGNAL`, rather than showing an
  old picture labelled live.
* **Failure never renders as zero.** When a poll fails the panel keeps its last
  known value and marks itself stale. A surveillance dashboard that prints
  "0 events" after a failed request is actively dangerous — the operator reads
  it as "nothing happened".
* **Delivery state is on screen.** If no alert channel can deliver, a banner
  says so. Argus spent its whole history recording events and sending them
  nowhere; that failure mode is now impossible to miss.
* **Evidence is fetched with credentials.** Snapshots and clips are
  role-protected, and neither `<img src>` nor `<video src>` can carry a bearer
  token, so both are fetched through the authenticated client and played from
  object URLs.

Run it with `python argus.py start` (builds the UI and serves it from the API
on port 8000), or `npm run dev` in `frontend/` for hot reload on port 3000.

---

## 📊 API Reference

**98 registered routes**: 92 operations are under `/api/v1`; the rest are
system, streaming, WebSocket, and documentation endpoints. `GET /` is excluded:
it serves the dashboard when `frontend/dist` exists and a build hint when it
does not, so it is not part of the API surface.

Of the 92 `/api/v1` operations, **89 require a token and 3 are public**.
Everything requires `Authorization: Bearer <token>` except the entries marked
*public* below.

The **Role** column is the *minimum* role required.

Interactive docs: http://localhost:8000/docs

### CityOS Intersection Intelligence
Geometry-only traffic layer (no biometrics enter it): digital twin,
road-user classification and trajectories, wrong-way / near-miss / VRU
safety analytics, traffic-flow statistics and signal-optimiser control.

| Method | Path | Role | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/cityos/status` | viewer | Intersections overview + privacy posture |
| `GET` | `/api/v1/cityos/twin` | viewer | Digital-twin snapshots for all intersections |
| `GET` | `/api/v1/cityos/twin/{camera_id}` | viewer | Digital twin for one camera's intersection |
| `GET` | `/api/v1/cityos/alerts` | viewer | Merged safety alert feed (`kind` filter) |
| `GET` | `/api/v1/cityos/flow/{camera_id}` | viewer | Volume series, turning matrix, speed summary |
| `GET` | `/api/v1/cityos/signal/{camera_id}` | viewer | Signal phase state + adaptive recommendation |
| `POST` | `/api/v1/cityos/signal/{camera_id}/mode` | operator | Set mode: fixed / adaptive / manual |
| `POST` | `/api/v1/cityos/signal/{camera_id}/phase` | operator | Manual override: force NS or EW green |
| `POST` | `/api/v1/cityos/bind` | admin | Bind a camera to a named intersection |

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

### Perception
| Method | Path | Role | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/perception/tracks` | viewer | Every tracked entity in the world model, with its accumulated attributes |
| `GET` | `/api/v1/perception/tracks/{track_id}` | viewer | One entity: attributes, trajectory, relationships, observations |
| `GET` | `/api/v1/perception/tracks/{track_id}/explain` | viewer | **Evidence chain** — measurements and inferences under separate keys |
| `GET` | `/api/v1/perception/capabilities` | viewer | What can run here, measured cost, and *why* anything is unavailable |
| `GET` | `/api/v1/perception/changes` | viewer | Per-camera change baselines and whether they are mature enough to judge |
| `GET` | `/api/v1/perception/stats` | viewer | Throughput and measured per-stage cost |

`/explain` is the endpoint to reach for before acting on an alert. It never
returns a single confident narrative: `measured` holds only direct
observations, `inferred` and `relationships` hold conclusions drawn from them,
and each carries `grounded` (is there a measurement under this?) and
`actionable` (grounded **and** above the confidence floor). A conclusion that
rests only on other conclusions is reported as not actionable however
confident it looks.

`/capabilities` is deliberately blunt about what this host cannot do. On a
machine with no CUDA and no OCR engine it reports 12 of 18 capabilities
available, each unavailable one naming the reason and the remedy — for example
`ocr` explains that `tesseract` is missing *and* that text regions are still
being detected without it.

### Rules, Coverage & Event Lifecycle
| Method | Path | Role | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/rules/status` | viewer | Per rule: configured, implemented, and whether it **can actually fire here** — with the specific blocker when it cannot |
| `GET` | `/api/v1/rules/calibration` | viewer | Per-camera ground-plane calibration, and what its absence disables |
| `GET` | `/api/v1/observations/promotion` | viewer | How many perception observations became events, and which kinds are deliberately not promoted |
| `GET` | `/api/v1/events/lifecycle` | viewer | The permitted event state machine, so clients need not hardcode it |
| `PATCH` | `/api/v1/events/{event_id}/status` | operator | Advance an event; records the acting user. Illegal transitions return `400` |
| `GET` | `/api/v1/notifications/status` | viewer | Which alert channels can **actually deliver right now**, plus policy and suppression counters |
| `POST` | `/api/v1/notifications/test` | admin | Send a synthetic alert through every channel and report the real per-transport outcome |
| `GET` | `/api/v1/evidence/status` | viewer | Pre-event ring-buffer occupancy per camera and clip counters |
| `GET` | `/api/v1/events/{event_id}/clip` | viewer | Download the pre-event video clip; `404` with a reason if none, `410` once retention has removed it |

### Alerts are delivered, not just recorded

`MQTTPublisher.publish_event()` was fully written and `mqtt.enabled` was `true`
in `config.yaml`, but **nothing in the codebase ever called it**. Every event
Argus produced was stored in SQLite and sent nowhere, and no endpoint revealed
that. `EventStore.create_event` — the one choke point every event passes
through — now dispatches through a notification service with MQTT and webhook
transports, an alert policy (priority floor, allow/deny, midnight-wrapping
quiet hours, per-camera rate limit) and counters for everything it drops.

Channel availability is **probed live**, never inferred from the config flag:
with `mqtt.enabled: true` and no broker running, `notifications/status`
reports the channel as unavailable and gives the reason. A delivery that was
never attempted is never reported as sent, and a suppressed alert is counted
rather than swallowed. Delivery is best-effort and cannot break event
recording — a transport that raises is logged and reported, and the event is
still persisted.

### Pre-event video evidence

A snapshot shows the instant a rule fired, not the approach. Argus keeps a
per-camera ring buffer of recent frames and exports an mp4 when a high-value
rule fires (`evidence_clips.clip_rules`); the path is attached to the event
metadata and served from `/api/v1/events/{event_id}/clip`.

Frames are buffered **JPEG-encoded rather than raw**, measured on 480p:

| | per frame | 10 s @ 10 fps, one camera |
|---|---|---|
| raw BGR | 0.88 MB | **88 MB** |
| JPEG q=80 | 0.005–0.20 MB | 0.5–20 MB |

Raw buffering costs ~352 MB across four cameras — fatal beside the detection
model. Encoding costs ~2 ms/frame against a ~108 ms/frame YOLO budget. The
buffer is bounded **in bytes as well as frames**, because frame size varies
~40× with scene content, so a frame count alone is not a memory guarantee.
Clips expire under `retention.clips_days` with a `clips_max_mb` ceiling.

A rule declared `enabled: true` in `config.yaml` used to mean nothing on its
own — `speed_violation`, `fall_detection` and `abandoned_object` were all
advertised as enabled while no code evaluated them. They are implemented now,
and `rules/status` exists so the difference between *configured* and *working*
can never again be invisible.

Two of them refuse to fire rather than guess:

* **`speed_violation`** needs per-camera `camera_calibration`. Multiplying pixel
  displacement by a global 0.05 m/px guess yields a number that looks like a
  measurement and is not one. Uncalibrated cameras report the blocker instead.
* **`fall_detection`** needs real pose keypoints. The bounding-box aspect-ratio
  fallback cannot distinguish a fall from crouching or lying down, and a
  high-priority medical alert must not rest on that.

### Memory & Search
| Method | Path | Role | Description |
|--------|------|------|-------------|
| `GET` | `/api/v1/memory/recall` | viewer | *"What happened near the loading bay yesterday?"* — free text over observations and their evidence, filterable by `camera_id`, `kind`, `when`, `min_confidence` |
| `GET` | `/api/v1/memory/summary` | viewer | Plain-language digest of a period — the shift-handover answer |
| `GET` | `/api/v1/memory/appearances/{camera_id}/{track_id}/similar` | viewer | *"Where else has this person been?"* — ranked by appearance, annotated with travel-time plausibility |
| `GET` | `/api/v1/memory/stats` | viewer | What is stored, which vector backend is live, and its measured limits |

`when` accepts `today`, `yesterday`, `last_hour`, `last_week`, `24h`.

**Appearance matches are candidates for review, never identifications.** The
descriptor compares clothing colour layout in three bands, so two people
dressed alike match strongly. Every result carries `is_identification: false`
and the caveat travels in the payload rather than living only in these docs.
Matches are also checked for physical plausibility — two cameras seeing
matching clothes *at the same moment* is evidence of two people, and that is
reported rather than hidden.

Vector search uses exact brute-force cosine in SQLite (measured: 5 ms at 1k
vectors, 37 ms at 10k, 207 ms at 50k). Qdrant is used automatically **if a
server actually answers** — never because `config.yaml` says `enabled: true`.
Past ~25 000 vectors `/memory/stats` warns you to run it.

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

## 🚢 Deployment

Argus runs anywhere that can run a container. Full guide:
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**.

```bash
cp .env.example .env
echo "ARGUS_JWT_SECRET=$(openssl rand -base64 48)" >> .env
echo "ARGUS_ADMIN_PASSWORD=choose-a-strong-one"    >> .env
docker compose -f docker-compose.prod.yml up -d
```

| Target | Config | Hosts |
|---|---|---|
| Docker / VPS / on-prem | `docker-compose.prod.yml` | Everything |
| Render | `render.yaml` | Everything |
| Fly.io | `fly.toml` | Everything |
| Railway | `railway.json` | Everything |
| GHCR image | `.github/workflows/deploy.yml` | Everything |
| Vercel | `vercel.json` | **Dashboard only** |

The container reads the platform's `$PORT`, creates the schema on first boot,
sets the admin password from `ARGUS_ADMIN_PASSWORD` without ever baking a
default into the image, and refuses to start without `ARGUS_JWT_SECRET` rather
than signing tokens with an ephemeral key that logs everyone out on restart.

**Two things worth knowing before choosing a host.**

*Argus needs a disk.* All mutable state — the SQLite database, snapshots,
evidence clips — lives under `ARGUS_DATA_DIR`. On a container platform that
must be a mounted volume. Mounting one at any other path persists nothing while
appearing to work until the first restart.

*Vercel cannot host the backend.* Not a configuration problem: the dependency
set is ~955 MB against a 500 MB function limit, serverless functions cannot
hold WebSocket connections open for video, there is no persistent process for
the retention thread, and the filesystem is ephemeral. `vercel.json` therefore
deploys the **dashboard only**; point it at an API hosted elsewhere by setting
`VITE_API_ORIGIN` at build time. Serving the UI from the API container is
simpler and has no CORS surface.

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
pytest -q     # collects maintained test_*.py suites under tests/
```

Check the latest GitHub Actions run for current pass status; suite counts and results change as tests evolve.

Every push and pull request runs these on Python 3.11 and 3.13 via
[`.github/workflows/ci.yml`](.github/workflows/ci.yml), which additionally
boots the real server to assert `/health` responds, an unauthenticated request
is refused with `401`, and `/metrics` emits `argus_` lines — plus a secret
scan, a gitignore-hygiene check, and a frontend build. See
[CONTRIBUTING.md](CONTRIBUTING.md).

| Suite group | Files | Coverage |
|---|---|---|
| Maintained pytest suites | tests/test_*.py | Pipeline regression, API authorization, security hardening, CityOS, and sensor behavior. pytest.ini defines collection. |

The regression suite contains targeted cases for previously observed failures. The latest workflow run is the source of truth for whether the current tree passes.

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
│   INTEGRATION_TODO.md, CONTRIBUTING.md, CHANGELOG.md, LICENSE
├── .github/workflows/ci.yml              # Lint, secret scan, tests (3.11/3.13), frontend build
├── pytest.ini                            # Test collection + markers
├── .dockerignore                         # Keeps .git/node_modules/data out of build context
├── requirements.txt                      # Core deps (incl. PyJWT)
├── requirements-optional.txt             # Heavy/optional extras (django, kafka, qdrant…)
├── .env.example                          # Every ARGUS_* variable, documented
├── config/config.yaml                    # Agents, rules, retention; supports ${VAR}
├── Dockerfile                            # Production image (API + built dashboard)
├── docker/, docker-compose.yml           # Development container builds
├── docker-compose.prod.yml               # Production stack (single image + volume)
├── Makefile                              # make start / make stop shortcuts
├── render.yaml, fly.toml, railway.json   # One-click platform blueprints
├── vercel.json                           # Static dashboard only - see docs/DEPLOYMENT.md
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
│   ├── src/theme.js                      # Control-room design system; severity colours
│   ├── src/pages/CommandCenter.jsx       # Video wall, alert feed, delivery + evidence state
│   ├── src/pages/Login.jsx               # Login screen
│   ├── src/pages/                        # CameraManagement, EventFeed, AnalyticsDashboard,
│   │                                     #   AdaptiveLearningDashboard, MemoryExplorer
│   ├── src/components/CameraTile.jsx     # Live tile: WS frames + letterbox-correct overlays
│   ├── src/components/ClipPlayer.jsx     # Authenticated pre-event clip playback
│   ├── src/components/EvidenceImage.jsx  # Authenticated snapshot fetch
│   ├── src/components/ui.jsx             # Panels, metrics, status dots
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
