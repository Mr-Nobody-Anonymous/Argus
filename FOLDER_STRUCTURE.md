# Argus — Folder Structure & Architecture Guide

This document provides a comprehensive overview of the Argus AI Video Analytics Platform's directory layout, explaining what each folder and file contains, its purpose, and how it integrates into the pipeline.

---

## 📂 Root Directory

| File | Description |
|------|-------------|
| `README.md` | Project overview, quickstart, feature list, API reference, and data pipeline diagram |
| `FOLDER_STRUCTURE.md` | **This file** — detailed directory documentation, file-by-file |
| `TODO.md` | Original project restructuring checklist (fully completed) |
| `INTEGRATION_TODO.md` | Integration fixes applied: WebSocket route conflict resolution, bbox overlay fixes, zone_alerts dict unification, import verification |
| `LICENSE` | MIT License |
| `requirements.txt` | Python backend dependencies (FastAPI, YOLO, OpenCV, PaddleOCR, InsightFace, MediaPipe, scikit-learn, etc.) |
| `package.json` | Root-level workspace pointer (frontend is in `frontend/`) |
| `package-lock.json` | Root dependency lock |
| `run_app.bat` | Windows launcher — starts uvicorn backend + Vite frontend concurrently |
| `run_webcam.bat` | Windows launcher — runs webcam tester for PC camera (standalone) |
| `docker-compose.yml` | Main Docker Compose — backend, frontend, Mosquitto MQTT, Qdrant, Kafka |
| `docker-compose.mediamtx.yml` | Optional Docker Compose — MediaMTX RTSP server + Kafka + Elasticsearch + Grafana |
| `images.png` | Project logo (local reference) |
| `Sentinel Sight Assignment.pdf` | Original project brief document |
| `.gitignore` | Git ignore rules for Python, Node, Docker, and OS files |

---

## 🐍 `backend/` — Python Backend Application

The backend is a **FastAPI** application (`backend/api/main.py`) that orchestrates:
- Camera stream ingestion (`stream_ingestion.py`)
- YOLOv8 AI inference and deep tracking (`core_engine/`)
- Vision services — face recognition, LPR, pose estimation, image enhancement (`vision/`)
- Analytics — cross-camera Re-ID, anomaly detection, speed/height analysis (`analytics/`)
- Management — zone rules, event store, MQTT publishing, telemetry, state recovery (`management/`)
- A **decentralised multi-agent swarm** for autonomous resource bidding and parameter evolution

### `backend/__init__.py`
Package marker. Imports services and makes them available via `backend.services.*`.

---

### `backend/api/` — FastAPI Application Layer

The HTTP/WebSocket serving layer. All routes are registered in `main.py` and available at `http://localhost:8000/docs`.

#### `api/__init__.py`
Package marker.

#### `api/main.py`
**Main FastAPI application entry point.** This file:
- Creates the `FastAPI` app with title "Argus API" and version "2.1.0"
- Sets up CORS middleware for `http://localhost:5173` (Vite dev server)
- Mounts static file serving for `/snapshots/` (event snapshot images)
- Imports and registers all route modules:
  - `stream_routes.py` — snapshot retrieval at `/snapshots/{camera_id}/{filename}` and MJPEG fallback at `/api/mjpeg/stream/{camera_id}`
  - `stream_ws.py` — WebSocket streaming at `/api/ws/stream/{camera_id}` (included via `include_router`)
- Defines a `lifespan` context manager that:
  - On startup: initializes DB, starts camera processing pipelines, initializes LPR/anomaly/pose/tracker services
  - On shutdown: stops all camera processing, disconnects MQTT, closes DB
- **Defines 50+ API endpoints** across 10 categories:
  1. **Cameras** — GET/POST/PUT/DELETE `/api/v1/cameras`
  2. **Zones** — GET/POST/PUT/DELETE `/api/v1/zones`
  3. **Events** — GET `/api/v1/events`, `/api/v1/events/{id}`, `/api/v1/events/stats`
  4. **Analysis** — GET `/api/v1/analysis/{camera_id}` (returns full speed/height/face/LPR/pose/anomaly data)
  5. **LPR** — GET `/api/v1/lpr` (system status)
  6. **Anomalies** — GET `/api/v1/anomalies` (recent anomaly events)
  7. **Tracker** — GET `/api/v1/trackers` (deep tracker status + active tracks)
  8. **Pose** — GET `/api/v1/poses` (pose statistics)
  9. **Enhancement** — POST `/api/v1/enhance/analyze` (image quality)
  10. **Face Recognition** — GET/POST/DELETE `/api/v1/faces/*`
  11. **Webcam** — POST/GET `webcam/start`, `webcam/status`, `webcam/stop`
  12. **Cross-Camera Tracking** — GET/POST/DELETE `cross-camera/*` (12 endpoints)
  13. **Video Testing** — POST `/api/v1/video/process`
  14. **Clusters** — GET `/api/v1/clusters`
  15. **System** — GET `/api/v1/health`, `/api/v1/metrics`, `/api/v1/stats/learning`

#### `api/models.py`
**Pydantic models** for API request/response schemas:
- `CameraBase`, `CameraCreate`, `CameraUpdate`, `Camera` — name, rtsp_url, location_tag, status, fps
- `ZoneBase`, `ZoneCreate`, `Zone` — camera_id, name, type (polygon/rectangle), coordinates [[x,y], ...]
- `EventBase`, `Event` — camera_id, timestamp, rule_type, object_type, confidence, bbox, snapshot_path, priority, status, metadata
- `HealthResponse` — status, subsystems dict, uptime_seconds
- `MetricsResponse` — cameras list, system dict (CPU %, memory MB, disk %)

#### `api/stream_routes.py`
**HTTP stream routes** (no WebSocket — consolidated in `stream_ws.py` to avoid route conflicts):
- `GET /api/mjpeg/stream/{camera_id}` — MJPEG streaming as fallback for browsers that don't support WebSocket
- `GET /snapshots/{camera_id}/{filename}` — Retrieve saved snapshot images from `data/snapshots/`
- Path traversal protection via `Path(filename).name`

#### `api/stream_ws.py`
**WebSocket endpoint** for real-time video streaming with AI overlay data.
- `WebSocket /api/ws/stream/{camera_id}` — Accepts concurrent frontend connections
- **Protocol** (interleaved):
  1. **Binary message** (Blob) — JPEG-encoded video frame at ~85% quality
  2. **Text message** (JSON) — Detection metadata: `{camera_id, detections: [{track_id, class, confidence, bbox: {x1,y1,x2,y2}}], timestamp}`
- Supports both legacy `Detection` dataclass objects AND dict-based detections from the swarm pipeline
- Registers/unregisters active streams with `UserAttentionTracker` for priority boosting
- MJPEG fallback at `/api/mjpeg/stream/{camera_id}` (same route prefix as `stream_routes.py` uses `/mjpeg/` to avoid collisions)

---

### `backend/config/` — Configuration Management

#### `config/__init__.py`
Package marker.

#### `config/config.py`
**YAML configuration loader** using Pydantic models. Defines 30+ config sections:
- `SystemConfig` — fps_target (15), max_cameras (4), snapshot_retention_days (30)
- `InferenceConfig` — model (yolov8n.pt), confidence_threshold (0.5), device (cpu), classes ([0,2] = person, car)
- `EnhancementConfig` — enabled, auto_enhance, low_light, denoise, sharpen, night_vision, deblur, hdr
- `FaceRecognitionConfig` — enabled (False by default), confidence_threshold, model_path, max_faces_per_frame
- `LicensePlateConfig` — enabled, region (us), min_confidence
- `AnomalyConfig` — enabled, sensitivity (0.7), window_size (100)
- `ReIDConfig` — enabled, similarity_threshold (0.7), feature_dim (2048)
- `TrackerConfig` — algorithm (bytetrack), track_buffer (30), match_threshold (0.6)
- `PoseConfig` — enabled, min_detection_confidence (0.5)
- `CrossCameraTrackerConfig` — similarity_threshold (0.5), max_track_age_hours (24), trajectory_prediction
- `SpeedAnalysisConfig` — calibration_factor (0.05), reference_distance_m (10.0)
- `HeightAnalysisConfig` — avg_person_height_m (1.7)
- `MQTTConfig` — broker (localhost), port (1883), topic_prefix (argus), qos (1)
- `DatabaseConfig` — url (sqlite:///../data/argus.db)
- `QdrantConfig` / `KafkaConfig` — optional infrastructure
- **Agent configs**: `YoloAgentConfig`, `FaceAgentConfig`, `LprAgentConfig` — each with `LocalEvolutionConfig` and gene vector bounds
- **Swarm configs**: `ConsortiumConfig`, `UserAttentionConfig`, `LogicMutationConfig`, `StateRecoveryConfig`, `TelemetryConfig`
- **Evolution**: `EvolutionaryEngineConfig` with `SynthesizerConfig`, `EvaluatorConfig`, `MutationConfig`
- **Rule types**: `RuleConfig` per rule (intrusion, loitering, speed_violation, fall_detection, abandoned_object)
- `get_config()` — singleton accessor; recursively tries `config/config.yaml`, `../config/config.yaml`

---

### `backend/database/` — Database Layer

#### `database/__init__.py`
Package marker.

#### `database/db.py`
**SQLite database setup** with schema management:
- `Database` class with thread-safe `execute()`/`fetchone()`/`fetchall()` methods
- Tables created on init:
  - `cameras` — id, name, location_tag, rtsp_url (UNIQUE), status, fps, timestamps
  - `zones` — id, camera_id (FK → cameras ON DELETE CASCADE), name, type, coordinates (JSON string)
  - `events` — id, camera_id (FK), timestamp, rule_type, object_type, confidence, bbox (JSON), snapshot_path, priority, status, metadata (JSON), created_at
  - `behavior_profiles` — id, person_id (UNIQUE), patterns (JSON), timestamps
- Indexes on events: camera_id, timestamp DESC, rule_type, priority, status, (camera_id, timestamp DESC)
- `get_db()` — singleton with path from config
- `close_db()` — graceful shutdown

---

### `backend/django_admin/` — Django Admin Interface

#### `settings.py`
Django settings — SQLite database, installed apps (django.contrib.*, rest_framework), MIDDLEWARE, TEMPLATES, static files.

#### `models.py`
Django ORM models mirroring the SQLite schema: `Cameras`, `Zones`, `Events`.

#### `admin.py`
Django admin registration for Camera, Zone, and Event models.

#### `urls.py`
Django URL routing — includes admin site and REST framework URLs.

---

### `backend/models/` — AI Model Files

| File | Description |
|------|-------------|
| `yolov8n.pt` | YOLOv8 nano model (~6.3M params) — lightweight, good for CPU |
| `yolov8m.pt` | YOLOv8 medium model (~25.9M params) — more accurate, GPU recommended |

---

### `backend/scripts/` — Utility Scripts

| File | Description |
|------|-------------|
| `__init__.py` | Package marker |
| `create_test_events.py` | Generates synthetic test events (intrusion, loitering) for development/testing |
| `init_db.py` | Initializes SQLite database schema and inserts default camera/zone data |
| `run_admin.py` | Launches Django admin interface (`python manage.py runserver`) |
| `webcam_tester.py` | Standalone webcam testing script with CLI args: `--camera 0`, `--model yolov8n.pt`. Tests face recognition, emotion detection, pose estimation, image enhancement, speed/height analysis |

---

### `backend/services/` — Core Service Modules

This is the heart of the platform. All services are instantiated as singletons via `get_*()` functions and wired together through `ProcessingCoordinator`.

#### `services/__init__.py`
Re-exports all service singleton accessors:
```python
get_camera_manager, get_zone_manager, get_event_store,
get_processing_coordinator, get_inference_engine,
get_mqtt_publisher, get_image_enhancement, get_face_recognition,
get_speed_height_analyzer, get_license_plate_recognition,
get_anomaly_detector, get_pose_estimator, get_deep_tracker,
get_person_reid, get_adaptive_learning_engine, get_cross_camera_tracker,
get_telemetry_monitor, get_user_attention_tracker,
get_evolutionary_engine, get_consortium_broker, get_logic_mutator,
get_state_recovery_manager,
get_yolo_detection_agent, get_face_recognition_agent, get_lpr_agent
```

---

#### `backend/services/core_engine/` — Core AI Engine

| # | File | Description |
|---|------|-------------|
| 1 | `inference_engine.py` | **YOLOv8 object detection.** Loads model from `backend/models/`, warms up with dummy frame, runs `model()` with configurable `conf`/`classes`. Returns list of `{class_id, class_name, confidence, bbox}`. Tracks inference times for performance monitoring (`get_avg_inference_time()`). |
| 2 | `yolo_detection_agent.py` | **Autonomous YOLO agent.** Wraps `InferenceEngine` as a swarm agent with its own evolutionary gene vector (`YoloGeneVector`). Optimizes `yolo_conf_threshold`, `iou_threshold`, `input_resolution_scale`, `tracker_matching_threshold` via local genetic mutation. Submits bids to `ConsortiumBroker` based on urgency/compute cost/detection rate. Adapts parameters based on broker allocation (throttle reduces resolution/raises threshold). |
| 3 | `deep_tracker.py` | **Multi-algorithm deep tracker** (ByteTrack/DeepSORT/BoT-SORT). Maintains persistent track IDs across frames using Kalman filter prediction + IoU matching. `update(detections, frame)` → returns detections with `track_id` assigned. Tracks active tracks, manages expired tracks (configurable `track_buffer`). |
| 4 | `yolo_tracker.py` | **Legacy YOLO tracker** — standalone `YOLOTracker` class with basic feature extraction (color histogram) and simple track assignment. Used as fallback if `deep_tracker` is disabled. |
| 5 | `multistream_pipeline.py` | **Multi-stream ingestion engine.** `FFmpegCapture` uses FFmpeg subprocess for zero-copy frame reading from RTSP (more efficient than OpenCV). `MultiStreamPipeline` manages up to 8 workers with adaptive frame skipping based on inference latency. Supports MediaMTX-rebroadcast URLs. |
| 6 | `video_pipeline.py` | **Optimized video pipeline** with separate decode and stream threads per camera. Adaptive frame skipping based on `inference_times` history. Dual-queue architecture: one queue for inference processing, one for WebSocket streaming (JPEG-encoded). |
| 7 | `processing_coordinator.py` | **Central orchestrator.** The `ProcessingCoordinator` class ties everything together. Runs a per-camera processing loop in daemon threads. Two modes: **swarm mode** (default, asymmetric event-driven) and **linear mode** (classic pipeline fallback). In swarm mode: YOLO agent runs → posts context to broker → broker resolves bids → face/LPR agents run conditionally → pose/anomaly/speed analysis → rules engine → stores in `camera_analysis` cache. Provides `get_latest_frame(camera_id)` for WebSocket streaming (returns `(frame_np, detections_list)` tuple). |
| 8 | `object_detection_tracker.py` | Legacy integrated detection + tracking pipeline (YOLO + ByteTrack/DeepSORT). |
| 9 | `object_detection_tracker_refactored.py` | Refactored version with improved architecture, used as reference for swarm migration. |
| 10 | `consortium_broker.py` | **Decentralised resource auctioneer.** Implements `AgentBid` (urgency, compute_cost, contextual_relevance, current_load) and `ResourceAllocation` (throttle_factor, priority_boost, should_process). Agents post context to a shared blackboard (`post_context()`/`read_context()`). `resolve_cycle()` computes allocations using proportional bidding strategy. |
| 11 | `evolutionary_engine.py` | **DEAP-based genetic algorithm** that synthesises new detection rules, evaluates fitness (inference speed, tracking accuracy, FP ratio, rule precision), and mutates pipeline parameters. `record_frame_metrics()` collects per-frame telemetry. `get_optimization_vector()` returns best-known gene vector. |
| 12 | `logic_mutator.py` | **Self-referential logic mutation engine.** Generates Python one-liner filter rules in a sandboxed `eval()` environment (restricted builtins, only `math` imports). Tests rules against cached frames, prunes low-fitness variants. |
| 13 | `stream_ws.py` | **Duplicate of api/stream_ws.py** (legacy, kept for backward compatibility). |

---

#### `backend/services/vision/` — Vision AI Services

| # | File | Description |
|---|------|-------------|
| 1 | `face_recognition.py` | **Face detection and recognition** using InsightFace/OpenCV. `recognize_faces(frame, detect_emotions=True)` → returns list of `{bbox, person_name, is_known, confidence, emotion}`. `register_face()` stores embeddings of known persons. Loads reference images from `data/known_faces/`. Tracks faces across frames with timeout. |
| 2 | `face_recognition_agent.py` | **Autonomous face recognition agent.** Wraps `FaceRecognition` service with local evolution of `match_distance_threshold`, `min_face_size_px`, `track_timeout_seconds`, `frame_skip_cadence`. Reads context from broker (`human_detected`, `crowd_detected`) to adjust urgency. |
| 3 | `lpr_agent.py` | **Autonomous LPR agent.** Wraps `LicensePlateRecognition` service with local evolution of `segmentation_threshold`, `min_plate_height_px`, `resolution_downscale`, `detection_confidence`, `ocr_beam_width`. Activates only when broker context shows `vehicle_detected`. |
| 4 | `license_plate_recognition.py` | **LPR pipeline** using PaddleOCR. `detect_plates(frame, detections)` → returns list of `{plate_text, confidence, bbox}`. Validates plate text against regex patterns (US/EU formats). |
| 5 | `image_enhancement.py` | **Image enhancement pipeline.** `enhance_frame(frame, mode="auto")` applies CLAHE, denoising (Non-local Means), sharpening, night vision, deblur, and HDR based on detected quality issues. `detect_quality_issues()` analyses brightness, contrast, blur, noise. |
| 6 | `pose_estimator.py` | **Human pose estimation** using MediaPipe Pose. `estimate_pose(frame, bbox)` → returns keypoints (nose, eyes, shoulders, hips, etc.) and detected actions (standing, sitting, lying/fall). `get_pose_statistics()` returns aggregated metrics. |

---

#### `backend/services/analytics/` — Analytics & Advanced AI

| # | File | Description |
|---|------|-------------|
| 1 | `person_reid.py` | **Person re-identification** using ResNet50 feature embeddings. `extract_features(person_img)` → 2048-dim normalized feature vector. `compute_similarity(feat1, feat2)` → cosine similarity [0,1]. `find_match(query_features, camera_id, timestamp)` → finds same person across cameras. Uses torchvision ResNet with removed FC layer. |
| 2 | `cross_camera_tracker.py` | **Cross-camera person tracking.** Maintains global identity across all cameras. `set_target(person_id, camera_id, reason)` starts targeted tracking with path recording. `add_track_point(person_id, camera_id, bbox, features)` updates position. `find_cross_camera_match(features, current_camera)` uses cosine similarity. `predict_next_camera(person_id)` predicts transition based on camera adjacency graph. `cluster_trajectories(threshold)` uses DBSCAN. `process_video_file()` simulates cross-camera tracking for testing. |
| 3 | `anomaly_detector.py` | **Anomaly detection** using multiple approaches: motion-based (frame differencing + contour analysis), behavior-based (speed anomalies, trajectory anomalies, loitering), abandoned object detection, criminal activity detection (erratic movement, casing, suspicious grouping, close person contact), crowd density analysis, and behavior pattern prediction. `detect_suspicious_behavior()` identifies potential security threats. |
| 4 | `speed_height_analysis.py` | **Velocity and height estimation.** `analyze_object()` returns `speed_mps, speed_kmh, speed_category (stationary/walking/jogging/running/sprinting), direction (8 compass points), height_m, height_category (child/short/average/tall/very_tall), bbox_area, aspect_ratio, track_duration`. Uses Kalman-filter smoothed velocity, perspective-adjusted calibration, and class-specific reference heights. |
| 5 | `adaptive_learning.py` | **Adaptive learning engine.** Tracks AI performance evolution over time, updates behavior profiles based on detection patterns, computes emotion baselines for known persons. Provides dashboard statistics. |

---

#### `backend/services/management/` — Management & Infrastructure

| # | File | Description |
|---|------|-------------|
| 1 | `camera_manager.py` | **Camera CRUD.** `create_camera()` → INSERT into `cameras` table. `get_all_cameras()` → SELECT all. `update_camera()` → update fields (name, rtsp_url, status, fps). `update_status()` → set online/offline/error + FPS. `delete_camera()` → DELETE cascade. |
| 2 | `zone_manager.py` | **Zone CRUD + geometry checking.** `create_zone()` validates polygon (≥3 points) / rectangle (2 points). Uses Shapely for `is_point_in_zone()` — point-in-polygon via `Polygon.contains(Point)`, rectangle via min/max bounds. Coordinates stored as JSON. |
| 3 | `zone_alerts.py` | **Virtual tripwire and geofence monitoring.** `check_zone_crossings()` runs each detection against loaded zones. Supports line (tripwire crossing), polygon (enter/exit), and intrusion (dwell time >30s) zone types. **Now fully compatible with both legacy `Detection` dataclass objects and swarm dict format** via 5 helper functions: `_get_center_from_dict_or_obj()`, `_get_track_id_from_dict_or_obj()`, `_get_class_name_from_dict_or_obj()`, `_get_confidence_from_dict_or_obj()`, `_get_bbox_from_dict_or_obj()`. Uses ray casting for `_point_in_polygon()` and cross-product for `_line_intersection()`. |
| 4 | `rules_engine.py` | **Zone-based event generation.** `process_detections()` checks each detection against zones for: **intrusion** (object enters restricted zone → high priority event + snapshot), **loitering** (person stays >30s threshold → medium priority event). Uses grid-based object tracking (`center//50` grid cells). 5-second deduplication window. Saves snapshots with bbox overlay + timestamp to `data/snapshots/`. |
| 5 | `event_store.py` | **Event storage and querying.** `create_event()` → INSERT + return with ID. `query_events()` supports filtering by camera_id, time range, rule_type, priority, status, with pagination. `get_event_stats()` returns counts by rule type and priority for last N hours. 30-day retention policy via `delete_old_events()`. |
| 6 | `mqtt_publisher.py` | **MQTT event publishing.** Uses `paho-mqtt` with async loop. `publish_event(event)` → JSON payload to `argus/events/{camera_id}/{rule_type}`. `publish_camera_status()` → status updates on `argus/status/{camera_id}`. Automatic reconnect with logging. |
| 7 | `stream_ingestion.py` | **RTSP/webcam stream ingestion.** `_capture_loop()` runs in daemon thread per camera with cv2.VideoCapture. Exponential backoff on failure (up to 10 retries, max 60s wait). Frame queue with maxsize=100 — drops oldest frame if full (prevents memory leak). FPS tracking from last 30 frames. Supports `webcam://{index}` URLs for local camera testing. |
| 8 | `telemetry_monitor.py` | **Real-time hardware telemetry.** Monitors CPU/GPU/RAM/VRAM usage with configurable warning/critical thresholds. Stress multiplier curve adjusts processing intensity based on system load. Cache-based sampling at 500ms intervals. |
| 9 | `user_attention_tracker.py` | **Viewport attention matrix.** Tracks which cameras are actively viewed by users (`register_active_stream()`/`unregister_active_stream()`). Boosts processing priority for attentively watched cameras (1.5x multiplier for active view, 2.0x for click interaction). Decays unviewed camera priority to 0.3x. |
| 10 | `state_recovery_manager.py` | **Byzantine fault-tolerant state recovery.** Heartbeat monitoring at 500ms intervals. Consecutive error tracking with configurable threshold (10 errors → recovery mode). Auto-rollback of pipeline parameters to last known good state. Ledger history preserves last 5 stable config snapshots. |
| 11 | `model_optimizer.py` | **Model optimization service.** ONNX export, FP16 quantization, and TensorRT conversion for faster inference. Manages model versioning and fallback. |

---

### `backend/services/legacy/` — Legacy Code

Empty directory — placeholder for legacy code migration from previous versions.

---

## ⚛️ `frontend/` — React Dashboard

A modern React + Vite + Material UI dashboard for real-time video surveillance monitoring.

| File/Directory | Description |
|----------------|-------------|
| `frontend/index.html` | HTML entry point with `<div id="root">` and module script |
| `frontend/vite.config.js` | Vite configuration — dev server on port 3000, proxy `/api` → `localhost:8000`, `/snapshots` → `localhost:8000` |
| `frontend/package.json` | Dependencies: React 18, MUI 5, Axios, Recharts, Framer Motion, React Router 6 |

### `frontend/src/` — Source Code

#### `src/main.jsx`
React entry point — renders `<App />` into DOM.

#### `src/App.jsx`
**Main application shell** with:
- Dark theme (MUI `createTheme` with black background, cyan accent, gradient app bar)
- Permanent sidebar drawer (240px) with navigation:
  - `Radar` icon → Dashboard (`/dashboard`)
  - `Videocam` icon → Cameras (`/`)
  - `Event` icon → Events (`/events`)
  - `Analytics` icon → Analytics (`/analytics`)
- React Router v6 routing
- Status chip showing "AI Video Analytics"

#### `src/services/api.js`
**Axios API client** with typed endpoint exports:
- `cameraAPI` — getAll, getById, create, update, delete
- `zoneAPI` — getAll (filtered by camera_id), create, update, delete
- `eventAPI` — getAll (with params: camera_id, from_time, to_time, rule, priority, status, limit, offset), getById, getStats
- `systemAPI` — health, metrics
- `crossCameraAPI` — 11 endpoints (getTracks, getTargets, setTarget, deleteTarget, getPath, predictTrajectory, getGraph, setGraph, clearOldTracks)
- `learningAPI` — getStats
- `videoAPI` — processVideo (multipart)
- `clusterAPI` — getClusters
- Base URL: `/api/v1`

#### `src/components/LiveVideoPlayer.jsx`
**Real-time video player with AI overlay canvas.**
- Connects to `ws://localhost:8000/api/ws/stream/{cameraId}` (WebSocket)
- **Binary message** (Blob) → loads JPEG frame onto `<img>` element via `URL.createObjectURL()`
- **Text message** (JSON) → parses detection metadata → draws on `<canvas>` overlay:
  - Bounding boxes with class-specific colors (person=green, car=red, truck=orange, etc.)
  - Label with class name, confidence (0.95), and track ID
  - Zone polygons (from `zones` prop) with dashed lines and labels
  - Handles both `{x1,y1,x2,y2}` object and `[x1,y1,x2,y2]` array bbox formats
- Disconnected badge overlay when connection drops
- Auto-reconnect after 3 seconds
- Loading spinner on initial connection

#### `src/pages/`

| File | Description |
|------|-------------|
| `SurveillanceDashboard.jsx` | Command center with live camera feeds grid, real-time stats cards (detections, events, FPS), recent alerts panel |
| `CameraManagement.jsx` | Camera CRUD interface — add/edit/delete cameras, configure RTSP URLs, view status and FPS |
| `EventFeed.jsx` | Event feed with filtering by type, priority, camera; severity indicators (high=red, medium=yellow, low=green); timestamp display |
| `AnalyticsDashboard.jsx` | Charts and trends — event counts over time, detection class distribution, FPS timeline, heatmap |
| `AdaptiveLearningDashboard.jsx` | AI evolution metrics — fitness over generations, gene vector history, agent status |

### `frontend/public/` — Static Assets

| File | Description |
|------|-------------|
| `leaflet_diagnostic.js` | Leaflet map diagnostic script (browser-based geolocation test) |
| `assets/argus-logo.png` | Platform logo (PNG) |
| `assets/argus-logo.svg` | Platform logo (SVG) |

---

## ⚙️ `config/` — Configuration

| File | Description |
|------|-------------|
| `config.yaml` | **Master configuration file** (~300 lines). Defines: camera definitions with RTSP URLs and zones, YOLO inference params, all enhancement modes, face recognition model/threshold, LPR region/confidence, anomaly sensitivity, ReID similarity threshold, tracker algorithm/buffer, pose confidence, cross-camera tracker settings, speed/height calibration, MQTT broker/port/topics, database URL, Qdrant/Kafka connection, consortium broker bidding, user attention multipliers, logic mutation sandbox, state recovery thresholds, telemetry sampling/warning/critical thresholds, evolutionary engine synthesizer/evaluator/mutation config, all three agent configs with gene vector bounds and compute costs, and all rule types with enabled/priority/threshold settings. |

---

## 🐳 `docker/` — Docker Build Files

| File | Description |
|------|-------------|
| `Dockerfile.backend` | Multi-stage build: installs Python deps (pip install -r requirements.txt), copies backend code, exposes port 8000, runs `uvicorn backend.api.main:app --host 0.0.0.0 --port 8000` |
| `Dockerfile.frontend` | Builds React app with Vite (`npm run build`), serves with nginx on port 80, copies `dist/` to nginx html |

---

## 📚 `docs/` — Documentation

| File | Description |
|------|-------------|
| `ARCHITECTURE_BLUEPRINT.md` | System architecture, component diagram, design patterns (singleton, observer, strategy), data flow, deployment topology |
| `ARCHITECTURE_CITYOS.md` | CityOS smart city integration architecture (Kafka topics, Qdrant collections, Grafana dashboards) |
| `DEMO_SCRIPT.md` | Step-by-step demo: 1) Add cameras 2) Create zones 3) Generate events 4) Cross-camera tracking 5) WebSocket stream |
| `INTEGRATION_SUMMARY.md` | Integration points: MQTT, Kafka, Qdrant, Elasticsearch, Grafana, Django Admin, MediaMTX |
| `RESEARCH_NOTES.md` | Model benchmarks (YOLOv8 vs YOLOv5, ByteTrack vs DeepSORT), optimal configs, edge cases |
| `TESTING_GUIDE.md` | Manual + automated test procedures, expected outputs, troubleshooting |

---

## 🔧 `infrastructure/` — Infrastructure Configuration

| File | Description |
|------|-------------|
| `kafka_topics.json` | Kafka topic definitions: `argus-events` (security alerts), `argus-telemetry` (hardware metrics), `argus-video-frames` (frame metadata) |
| `vectordb_config.py` | Qdrant vector database client setup, collection creation (face-embeddings, person-reid-features, anomaly-signatures), point insertion/search |

---

## 📡 `mediamtx/` — RTSP Server

| File | Description |
|------|-------------|
| `config.yml` | MediaMTX configuration: RTSP port :8554, HLS, WebRTC, authentication, paths for stream publishing/reading, recording settings |

---

## 📨 `mosquitto/` — MQTT Broker

| File | Description |
|------|-------------|
| `config/mosquitto.conf` | Mosquitto configuration: listener on port 1883, allow_anonymous true, persistence, log settings |

---

## 🧪 `tests/` — Testing

| File | Description |
|------|-------------|
| `ai_pipeline_test.py` | End-to-end AI pipeline test: injects test frames → verifies YOLO detections → checks tracker IDs → validates rule engine events |
| `buffer_monitor.py` | Frame buffer memory monitoring test — runs capture loop and measures queue growth, memory usage, frame drop rates |
| `infrastructure_healthcheck.py` | Health check matrix: RTSP port (:8554) ✓/✗, FastAPI HTTP (:8000) ✓/✗, SQLite DB ✓/✗, Kafka (:9092) ✓/✗, Elasticsearch (:9200) ✓/✗, Qdrant (:6333) ✓/✗. Outputs formatted test matrix. |
| `resource_monitor.sh` | Shell script for monitoring system resources (CPU, RAM, disk I/O) during load tests |
| `run_all_tests.py` | Test runner that executes all test scripts and aggregates results |
| `stream_simulator.py` | Multi-threaded RTSP stream simulator. Creates synthetic video files with moving objects (car, pedestrian, bicycle) for up to 4 cameras. Publishes via FFmpeg to MediaMTX endpoints. Supports webcam fallback. CLI args: `--count 4`, `--webcam`, `--urls ...` |
| `websocket_stress_tester.py` | WebSocket stress test — opens N concurrent connections, measures frames/second received, detection latency, drop rate |
| `manual/` | Manual testing directory (empty — for ad-hoc test scripts) |

---

## 🎨 `assets/` — Static Assets

| File | Description |
|------|-------------|
| `argus-logo.png` | Argus logo (PNG, 400×100) |
| `argus-logo.svg` | Argus logo (SVG vector) |

---

## 💾 `data/` — Runtime Data

| Directory | Description |
|-----------|-------------|
| `data/snapshots/` | Event snapshot images captured by RulesEngine (JPEG, with bbox overlay + timestamp) |
| `data/argus.db` | SQLite database (cameras, zones, events, behavior_profiles) |
| `data/known_faces/` | Reference face images for face recognition (PNG/JPG, one per known person) |
| `data/qdrant/` | Qdrant vector database persistent storage |
| `data/kafka/` | Kafka log data (when running Kafka without Docker) |
| `data/streams/` | MediaMTX stream cache (HLS segments, recordings) |

---

## 🏗️ Architecture Overview

```
┌──────────────────────────────────────────────────────────────────┐
│                        Argus Architecture                         │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐│
│  │   Cameras    │───▶│ RTSP Streams │───▶│   Stream Ingestion   ││
│  │ (RTSP/IP)    │    │ (MediaMTX)   │    │ (threaded cv2 loop)  ││
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘│
│                                                      │            │
│  ┌──────────────────────────────────────────────────┐│           │
│  │         Processing Coordinator                    ││           │
│  │  ┌─────────────────┐  ┌──────────────────────┐  ││           │
│  │  │  SWARM MODE      │  │  LINEAR MODE (fall.) │  ││           │
│  │  │  YOLO Agent ─────┼─▶│  YOLO Inference      │  ││           │
│  │  │  ConsortiumBroker│  │  Deep Tracker        │  ││           │
│  │  │  Face Agent      │  │  Face Recognition    │  ││           │
│  │  │  LPR Agent       │  │  LPR Detection       │  ││           │
│  │  │  Logic Mutator   │  │  Pose Estimation     │  ││           │
│  │  │  EvolutionaryEng │  │  Anomaly Detection   │  ││           │
│  │  └─────────────────┘  └──────────────────────┘  ││           │
│  │                                                   ││           │
│  │  ┌─────────────────────────────────────────────┐  ││           │
│  │  │  Vision Services: Face, LPR, Pose, Enhance  │  ││           │
│  │  │  Analytics: Re-ID, Speed/Height, Cross-Cam   │  ││           │
│  │  │  Management: Rules, Zones, Events, MQTT      │  ││           │
│  │  └─────────────────────────────────────────────┘  ││           │
│  └───────────────────────────────────────────────────┘│           │
│                                                        │           │
│  ┌────────────────────────────────────────────────────▼───┐       │
│  │              FastAPI Backend (:8000)                    │       │
│  │  REST API (/api/v1/*) + WebSocket (/api/ws/stream/*)    │       │
│  │  + Django Admin (/admin)                                 │       │
│  └─────────────────────────────────────────────────────────┘       │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              React Frontend (:3000)                            │ │
│  │  Dashboard · Camera Management · Event Feed · Analytics       │ │
│  │  LiveVideoPlayer.jsx (canvas bbox overlays via WebSocket)      │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                    │
│  ┌────────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │   MQTT     │  │  Qdrant  │  │  Kafka   │  │  ES / Grafana  │  │
│  │ (Mosquitto)│  │ (Vector) │  │ (Events) │  │  (Analytics)   │  │
│  └────────────┘  └──────────┘  └──────────┘  └────────────────┘  │
│                                                                    │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow (Detailed)

```
1. Camera Stream Ingest
   RTSP URL → stream_ingestion.py (cv2.VideoCapture in daemon thread)
   → Frame queue (maxsize=100, drops oldest if full)

2. Frame Processing (per camera, daemon thread)
   → ImageEnhancement.enhance_frame() → CLAHE, denoise, sharpen
   
   SWARM MODE (enabled):
     → YOLO Agent (yolo_detection_agent.py)
       → InferenceEngine.detect_objects() → List[Dict {class_name, confidence, bbox}]
       → DeepTracker.update() → Kalman filter → adds track_id
       → LogicMutator.apply_filter() → sandboxed rule filter
     → ConsortiumBroker.post_context() → "human_detected", "vehicle_detected"
     → Agent bids → broker resolves → face/lpr agents may process
     → Face Agent / LPR Agent / PoseEstimator (conditional)
     → AnomalyDetector (motion + behavior)
     → SpeedHeightAnalyzer (velocity, direction, height)
   
   LINEAR MODE (fallback):
     → InferenceEngine → DeepTracker → FaceRecognition → LPR → Pose → Anomaly → SpeedHeight

3. Rules Engine
   → RulesEngine.process_detections()
     → ZoneManager.is_point_in_zone() (Shapely polygon check)
     → ZoneAlerts (tripwire crossing, intrusion, loitering)
     → EventStore.create_event() → SQLite INSERT
     → MQTTPublisher.publish_event() → MQTT JSON payload
     → Screenshot capture to data/snapshots/

4. API Layer
   → camera_analysis cache updated with all results
   → WebSocket stream sends JPEG binary + JSON metadata at ~30 FPS
   → REST API serves query endpoints (/events, /analysis, /health)

5. Frontend
   → LiveVideoPlayer.jsx receives binary frames → renders on <img>
   → Receives JSON detection metadata → draws bbox overlays on <canvas>
   → Class-specific colors, confidence labels, track IDs, zone polygons
   → MUI dashboard pages for management and monitoring
```

---

## 🤝 Integration Points

| System | Protocol | Data |I already updated the README.md. Now let me update the FOLDER_STRUCTURE.md with the same level of comprehensive detail reflecting the current state.

<read_file>
<path>c:/Users/hp/Desktop/Argus/FOLDER_STRUCTURE.md</path>
</read_file>
