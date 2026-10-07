# Argus Integration Status

> **Status: all items below are verified working.** The "Next Steps to Verify"
> at the foot of this file have been executed and their results recorded.
> For remaining production gaps see [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md).

## Fixes Applied

### 1. WebSocket Route Conflicts (resolved)
- `stream_ws.py` - Consolidates all WebSocket streaming at `/api/ws/stream/{camera_id}`
- `stream_routes.py` - Rewritten to use `/snapshots/{camera_id}/{filename}` and `/api/mjpeg/stream/{camera_id}` — no collision with `stream_ws.py`
- `main.py` - Registers only `stream_router` (from `stream_routes.py`) under prefix `/api`

### 2. Bounding Box Overlays in Frontend (fixed)
- `LiveVideoPlayer.jsx` - Now properly handles the interleaved binary+JSON WebSocket protocol:
  - Binary messages → render as JPEG frames on `<img>`
  - Text messages → parse JSON detection metadata → draw canvas overlays
  - Supports both `{x1,y1,x2,y2}` object and `[x1,y1,x2,y2]` array bbox formats
  - Renders zone polygons with dashed lines + labels
  - Tracks detection colors by class name

### 3. Zone Alerts Dict Support (unified)
- `zone_alerts.py` - Rewritten with helper functions that accept both legacy Detection objects and swarm dict format
- `check_zone_crossings()` - Handles mixed detection formats transparently
- `from_dict_detection()` - Class method on `ZoneEvent` for swarm pipeline integration

### 4. Import Paths (verified)
- `analytics/person_reid.py` - Uses `from backend.config...` and `from backend.database...` — correct
- `analytics/anomaly_detector.py` - Uses `from backend.config...` — correct
- `analytics/cross_camera_tracker.py` - Uses `from backend.config...` — correct

### 5. Authentication layer (added after the original integration)

- `backend/api/auth.py` - JWT issuing/validation against the Django `auth_user`
  table, three roles, brute-force lockout.
- `main.py` - RBAC dependency on all 40 protected routes; `audit_mutations` middleware.
- `stream_ws.py` - authenticates **before** `websocket.accept()`, so an
  unauthenticated upgrade is refused at the handshake rather than accepted and
  then closed.
- `stream_routes.py` - MJPEG and snapshot routes are viewer-gated. The
  `/snapshots` **static mount** is only registered when auth is disabled,
  because a `StaticFiles` mount bypasses route dependencies entirely.
- `frontend/src/services/api.js` - bearer-token interceptor, transparent refresh
  on 401, and `buildStreamUrl()` which appends `?token=` for the WebSocket
  (browsers cannot set headers on a WS handshake).
- `frontend/src/pages/Login.jsx` + `App.jsx` - login gate and session handling.

### 6. Data Pipeline Flow
```
Camera (RTSP/webcam)
  → stream_ingestion.py (cv2.VideoCapture + frame queue)
    → processing_coordinator.py (swarm/fallback loop)
      → yolo_agent → detections (dict format)
      → deep_tracker → persistent track IDs
      → logic_mutator → AST-constrained filter
      → consortium_broker → agent allocation
      → face_agent / lpr_agent (if allocated)
      → pose_estimator (person keypoints)
      → anomaly_detector (motion + behavior)
      → speed_height_analyzer → speed/height/direction
      → rules_engine → zone-based alerts
        → event_store → SQLite
        → mqtt_publisher → MQTT broker
      → camera_analysis cache → WebSocket/FastAPI
        → stream_ws.py → JPEG frames + JSON metadata
          → LiveVideoPlayer.jsx → bbox overlays on canvas
```

### Verification Results

All four checks executed against a live instance streaming a looping demo clip.

| # | Check | Result |
|---|---|---|
| 1 | `uvicorn backend.api.main:app --host 0.0.0.0 --port 8000` | ✅ Starts clean; 43 `/api` operations registered (+ `/` and `/metrics`); YOLO model loads and warms up |
| 2 | Frontend at `http://localhost:3000` | ✅ Login screen renders; after sign-in the dashboard loads through the Vite proxy (`401` without a token, `200` with one) |
| 3 | `http://localhost:8000/docs` | ✅ OpenAPI schema generates; 41 documented paths (the WebSocket route is not part of an OpenAPI schema) |
| 4 | `ws://localhost:8000/api/ws/stream/{camera_id}` | ✅ No token → **403 at handshake**. With `?token=<jwt>` → interleaved binary JPEG frames + JSON detection metadata |

Additional verification performed since:

| Check | Result |
|---|---|
| RBAC matrix across 16 endpoints × 3 roles | ✅ 0 mismatches against the documented role table |
| Audit trail | ✅ Login, successful mutation, and a viewer's **denied** mutation all recorded with actor, role, IP, outcome |
| Brute-force lockout | ✅ 5 × 401 then 429; the correct password is also refused while locked |
| Fixture output | 12.8 detections/frame in one recorded sample; this is a count, not an accuracy measurement. |
| Automated pytest suites | Regression, API security, CityOS, and sensor suites are collected by CI; check the latest Actions run for pass status. |
| Swarm A/B sample | One 40-frame run showed +26.8% FPS and equal detection counts; no ground-truth accuracy comparison was run. |

### Known deviations from the original integration notes

- **Bbox format**: the WebSocket sends `bbox` as an **object**
  (`{x1,y1,x2,y2}`); `LiveVideoPlayer.jsx` accepts both that and the 4-element
  array form.
- **Timestamps**: the stream previously emitted a stringified Unix float while
  the documented protocol promised ISO 8601. It now sends ISO 8601 UTC in
  `timestamp` with the numeric value in `timestamp_unix`. The REST API and audit
  log still use their own formats - unifying them is a roadmap item.
- **Detection dict key**: the class label is sent as `class`, not `class_name`
  (which is the internal field name).
