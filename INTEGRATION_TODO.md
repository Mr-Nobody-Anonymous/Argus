# Argus Integration Status
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

### 5. Data Pipeline Flow
```
Camera (RTSP/webcam)
  → stream_ingestion.py (cv2.VideoCapture + frame queue)
    → processing_coordinator.py (swarm/fallback loop)
      → yolo_agent → detections (dict format)
      → deep_tracker → persistent track IDs
      → logic_mutator → sandboxed filter
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

### Next Steps to Verify
1. Run `python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000`
2. Open `http://localhost:3000` (frontend via `npm run dev`)
3. Check `http://localhost:8000/docs` for API endpoints
4. Verify WebSocket at `ws://localhost:8000/api/ws/stream/{camera_id}`
