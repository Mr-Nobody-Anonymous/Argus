# 🏙️ Argus City OS - Tactical Command Center Architecture

## Current State Audit - What Breaks at 100 Cameras

**Files That MUST Be Deleted/Replaced:**
- `backend/services/camera_manager.py` - Uses OpenCV VideoCapture synchronously
- `backend/services/stream_ingestion.py` - Single-threaded frame capture
- `backend/api/main.py` - Monolithic API handling video streams directly

**Issues Identified:**
- OpenCV `VideoCapture` opens 100+ RTSP connections directly = connection failures
- Single-threaded frame processing = bottleneck at ~20 cameras
- No backpressure handling = memory exhaustion
- Direct face/OCR in FastAPI = UI freezes
- No GPU memory management = OOM crashes

---

## New Directory Structure

```
Argus/
├── docker-compose.cityos.yml
├── ingestion/
│   ├── mediamtx/config.yml
│   └── ffmpeg_workers/
├── ai_workers/
│   ├── detection/worker.py
│   ├── classification/
│   ├── action/
│   └── common/gpu_pool.py
├── backend/
│   ├── api/websocket.py
│   └── services/entity_registry.py
├── frontend/
│   └── src/components/layout/CommandCenter.jsx
└── ARCHITECTURE_CITYOS.md
```

---

## Docker Compose Orchestration

```yaml
version: '3.8'
services:
  mediamtx:
    image: aler9/rtsp-simple-server:latest
    ports: ["8554:8554", "8080:8080"]
    
  kafka:
    image: bitnami/kafka:3.6
    
  qdrant:
    image: qdrant/qdrant:v1.9.0
    
  elasticsearch:
    image: docker.elastic.co/elasticsearch/elasticsearch:8.12.0
    
  worker-detection:
    build: .
    deploy:
      resources:
        reservations:
          devices: [{driver: nvidia, capabilities: [gpu]}]
```

---

## Entity Tracker Logic

```python
# backend/services/entity_registry.py
class EntityTracker:
    def register_face(self, track_id, camera_id, frame, bbox):
        embedding = self._get_insightface_embedding(frame[bbox])
        matches = self._search_similar_faces(embedding, threshold=0.45)
        if matches:
            return self._trigger_entity_card(matches[0]['person_id'], camera_id, track_id)
```

---

## Frontend Component Architecture

```jsx
// CommandCenter.jsx - Dark tactical theme
// VideoCanvas.jsx - HTML5 canvas overlay
// LeftMapPanel.jsx - OpenLayers map
```

---

## Kafka Schema Designs

- `raw_frames` - Binary frames from FFmpeg
- `ai_detections` - YOLOv8 + BoT-SORT results with track IDs
- `ai_biometrics` - Face embeddings, Re-ID vectors
- `ai_events` - Final triggers for Elasticsearch

---

## GPU VRAM Management Strategy

- Load only essential models per worker
- Lazy unload when VRAM > 80%
- Keep models hot if used within 5 minutes

---

## Execution Plan (25 Days)

1. Days 1-3: Infrastructure (docker-compose, MediaMTX, FFmpeg workers)
2. Days 4-7: Detection workers (YOLOv8 + BoT-SORT)
3. Days 8-9: Storage layer (Qdrant, Elasticsearch)
4. Days 10-12: API layer (WebSocket streaming)
5. Days 13-15: UI foundation (tactical layout, video grid)
6. Days 16-20: Advanced features (face, Re-ID, action recognition)
7. Days 21-25: Optimization (FAISS, Real-ESRGAN, Grafana)