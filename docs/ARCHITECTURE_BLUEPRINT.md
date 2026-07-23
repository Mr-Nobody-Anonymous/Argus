# 🏗️ AI-Powered CCTV Surveillance Architecture Blueprint

## Overview
This document provides a comprehensive architectural blueprint for integrating FFmpeg MediaMTX, YOLOv8, Kafka, Qdrant, and other components into a production-ready surveillance platform.

---

## 1. MULTI-STREAM INGESTION & PIPELINE ENGINE (FFmpeg + MediaMTX + OpenCV + FastAPI)

### Architecture Flow
```
IP Cameras/RSTP Streams → MediaMTX (RTSP Server) → FFmpeg Capture → Frame Queue → YOLO Inference → Kafka Events
```

### Components Created:
- `docker-compose.mediamtx.yml` - Docker orchestration for all services
- `mediamtx/config.yml` - RTSP/WebRTC/HLS server configuration  
- `backend/services/multistream_pipeline.py` - FFmpeg-based frame capture with adaptive skipping

### Key Features:
- Zero-copy frame extraction via FFmpeg subprocess
- Adaptive frame skipping when inference lags (>100ms per frame)
- Separate streams for inference (15 FPS) and live preview (30 FPS)
- Auto-reconnect with exponential backoff

### Usage:
```bash
# Start all infrastructure
docker-compose -f docker-compose.mediamtx.yml up -d

# Add camera stream to MediaMTX
ffmpeg -i rtsp://camera/stream -c:v copy -f rtsp rtsp://localhost:8554/cameras/cam001

# Run FastAPI backend
uvicorn backend.api.main:app --reload
```

---

## 2. AI MODEL INTEGRATION (YOLOv8 + ByteTrack + ANPR)

### Component: `backend/services/object_detection_tracker.py`

### YOLOv8 Configuration:
```python
# Recommended models:
# - yolov8s.pt (small) - 37.5M params, 12ms inference on GPU
# - yolov8m.pt (medium) - 50.6M params, balanced
# - yolov8n-seg.pt (nano-seg) - with segmentation masks

# Vehicle detection filter (lines 52-54)
VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
```

### Inference Optimization:
```python
# Run inference with optimizations (yolov8s, 640x640, FP16)
results = self.model(
    frame,
    imgsz=640,          # Optimal size for real-time
    conf=0.25,          # Lower threshold for surveillance
    iou=0.45,           # Standard IoU threshold
    max_det=100,        # Max detections per frame
    half=True,          # FP16 for faster inference
    device=self.device
)
```

### ANPR Integration (PaddleOCR):
```python
# Automatic license plate reading (lines 193-214)
plate_text, plate_conf = self._read_license_plate(vehicle_roi)
if plate_text:
    track.license_plate = plate_text
    track.license_confidence = plate_conf
```

---

## 3. LOW-LATENCY LIVE STREAMING (WebSocket + Canvas Overlay)

### Backend: `backend/api/stream_ws.py`
- WebSocket endpoint: `ws://localhost:8000/ws/stream/{camera_id}`
- Binary JPEG streaming for minimal latency
- JSON metadata for bounding boxes

### Frontend: `frontend/src/components/LiveVideoPlayer.jsx`
- React component with canvas overlay
- WebSocket connection with auto-reconnect
- Color-coded bounding boxes by object class

---

## 4. SECURITY, ANOMALY & RE-IDENTIFICATION

### Components Needed:
- **InsightFace**: Face recognition and embedding extraction
- **Deep-Person-Reid (fast-reid)**: Appearance feature extraction
- **Anomalib**: Unsupervised anomaly detection

### Integration Plan:
```python
# Face Recognition with InsightFace
# Store embeddings in Qdrant vector database

# Re-ID Features
# Extract person appearance features for cross-camera tracking
# Store in Qdrant/FAISS with track_id

# Anomaly Detection
# Pass frame backgrounds to Anomalib
# Flag unusual patterns (crowd anomalies, unusual movements)
```

---

## 5. REAL-TIME EVENT STREAMING (Kafka + Elasticsearch + Leaflet)

### Kafka Producer Integration:
```python
# backend/services/kafka_producer.py
from kafka import KafkaProducer
import json

producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

def send_detection_event(camera_id, track, plate=None):
    event = {
        'camera_id': camera_id,
        'track_id': track.track_id,
        'class': track.class_name,
        'confidence': track.confidence,
        'bbox': track.bbox,
        'license_plate': plate,
        'timestamp': track.timestamp
    }
    producer.send('surveillance-events', event)
```

### Frontend Mapping (Leaflet):
```jsx
// frontend/src/components/MapTracker.jsx
import { MapContainer, TileLayer, Marker, Popup } from 'react-leaflet'

// Real-time asset tracking on map
// Connect to WebSocket for coordinate updates
```

---

## 6. PERFORMANCE & SCALING OPTIMIZATION

### TensorRT Conversion Commands:
```bash
# Install ONNX Runtime
pip install onnxruntime onnxruntime-gpu

# Export YOLO to ONNX
python -c "
from ultralytics import YOLO
model = YOLO('yolov8s.pt')
model.export(format='onnx', half=True, simplify=True)
"

# Convert to TensorRT (NVIDIA GPU)
trtexec --onnx=yolov8s.onnx --fp16 --saveEngine=yolov8s.trt
```

### Hardware-Specific Optimizations:

| Hardware | Optimization | Expected FPS |
|----------|-------------|--------------|
| NVIDIA T4/V100 | TensorRT FP16 | 50-100 FPS |
| NVIDIA Jetson | ONNX INT8 | 20-40 FPS |
| CPU (Intel) | OpenVINO | 10-20 FPS |
| CPU (AMD) | ONNX Runtime | 5-15 FPS |

---

## Deployment Commands

```bash
# 1. Start infrastructure
docker-compose -f docker-compose.mediamtx.yml up -d

# 2. Install Python dependencies
pip install -r backend/requirements.txt

# 3. Run backend
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

# 4. Run frontend
cd frontend && npm install && npm run dev

# 5. Access Django admin
python run_admin.py
# http://localhost:8001/admin/
```

---

## File Structure
```
Argus/
├── docker-compose.mediamtx.yml     # Infrastructure orchestration
├── mediamtx/config.yml            # RTSP/WebRTC server config
├── backend/
│   ├── services/
│   │   ├── multistream_pipeline.py   # FFmpeg capture + frame queue
│   │   ├── object_detection_tracker.py # YOLO + tracking + ANPR
│   │   ├── zone_alerts.py           # Tripwire/geofence logic
│   │   └── model_optimizer.py       # TensorRT/ONNX optimization
│   └── api/
│       └── stream_ws.py            # WebSocket streaming
└── frontend/
    └── src/components/
        └── LiveVideoPlayer.jsx     # Live video with overlays