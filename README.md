<p align="center">
  <img src="https://raw.githubusercontent.com/Mr-Nobody-Anonymous/Argus/main/Stylized%20Multifaceted%20Eye%20Logo%20for%20Argus.png?v=3" alt="Argus - The Watchful Guardian" width="120"/>
</p>

<h1 align="center">
  👁️ Argus - AI Video Analytics Platform
</h1>

<p align="center">
  <a href="https://github.com/Mr-Nobody-Anonymous/Argus/stargazers">![GitHub Repo stars](https://img.shields.io/github/stars/Mr-Nobody-Anonymous/Argus?style=social)</a>
  <a href="https://github.com/Mr-Nobody-Anonymous/Argus/blob/main/LICENSE">![License](https://img.shields.io/github/license/Mr-Nobody-Anonymous/Argus?color=blue)</a>
  <a href="https://github.com/Mr-Nobody-Anonymous/Argus/issues">![Issues](https://img.shields.io/github/issues/Mr-Nobody-Anonymous/Argus?color=red)</a>
  <a href="https://github.com/Mr-Nobody-Anonymous/Argus/commits/main">![Last Commit](https://img.shields.io/github/last-commit/Mr-Nobody-Anonymous/Argus?color=green)</a>
</p>

<p align="center">
  <a href="#quick-start">Quickstart</a> | 
  <a href="#features">Features</a> | 
  <a href="#api-reference">API</a> | 
  <a href="#testing">Testing</a> | 
  <a href="#troubleshooting">FAQ</a>
</p>

**Metaphorical meaning:** Because the giant had a hundred eyes and was always watching, the term "Argus" in English is also used today to describe a watchful guardian or a highly observant person.

A production-ready AI-powered video analytics platform for real-time CCTV monitoring with object detection, zone-based rules, cross-camera tracking, and advanced criminal activity prediction.

## 🌟 Features

<!-- cspell:ignore venv uvicorn wowzaec streamlock yolov cuda mosquitto argus Ultralytics Avigilon Saimax -->

### Core Capabilities

- **Multi-Camera Support**: Manage up to 4 concurrent RTSP camera streams
- **Real-Time Object Detection**: YOLOv8-powered person and vehicle detection
- **License Plate Recognition (LPR)**: Automatic license plate detection and OCR using PaddleOCR
- **Zone-Based Rules**: Define polygon/rectangle zones with custom rules
- **Event Generation**: Intrusion, loitering, speed violation, and fall detection
- **Cross-Camera Tracking**: Track persons across multiple camera views with trajectory prediction
- **Web Dashboard**: Modern React UI with real-time updates
- **Analytics**: Event trends, heatmaps, and performance metrics
- **MQTT Integration**: Publish events for smart home/automation systems

### 🎯 Advanced AI Features

- **Face Recognition**: Identify known persons using InsightFace/OpenCV
- **Pose Estimation**: Human pose and gesture analysis with MediaPipe (fall detection)
- **Anomaly Detection**: Unusual behavior detection (speed, trajectory, loitering)
- **Deep Object Tracking**: Persistent object IDs using ByteTrack/DeepSORT/BoT-SORT
- **Speed Analysis**: Velocity estimation and categorization (km/h, direction)
- **Height Analysis**: Real-world height estimation and classification
- **Image Enhancement**: CLAHE, denoise, sharpen, night vision, deblur, HDR
- **Criminal Activity Prediction**: Proactive security alerts based on behavior patterns

### 🔧 Infrastructure Features

- **Qdrant Vector DB**: Optional vector storage for face embeddings and features
- **Kafka Integration**: Optional event streaming for analytics pipelines

### Technical Highlights

- ✅ Auto-reconnect with exponential backoff
- ✅ Frame queue management (prevents memory overflow)
- ✅ Event deduplication (5-second window)
- ✅ Performance monitoring (FPS, latency, queue depth)
- ✅ Cross-camera person matching with ReID features
- ✅ Trajectory clustering and prediction
- ✅ Privacy-first (local processing, no cloud)
- ✅ GDPR-friendly (events + snapshots only, configurable retention)
- ✅ Docker deployment with health checks

---

## 🚀 Quick Start

### Prerequisites

- Docker & Docker Compose
- 4GB RAM minimum
- RTSP camera streams or test videos

### Installation

#### Option 1: Local Setup (Recommended for Testing)
**Prerequisites**: Python 3.9+, Node.js 18+

1. **Backend**
```bash
# Setup Virtual Env
python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

# Install Deps
pip install -r backend/requirements.txt

# Run
python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Frontend**
```bash
cd frontend
npm install
npm run dev
```

**💡 Windows Shortcut:**
Just double-click `run_app.bat` to start everything!

3. **Access**
- Dashboard: http://localhost:3000

#### Option 2: Docker Deployment (Bonus)
1. **Start Services**
```bash
docker-compose up -d
```
2. **Access**
- Dashboard: http://localhost:3000
- API: http://localhost:8000/docs

### First-Time Setup

The YOLO model will be downloaded automatically on first run. This may take a few minutes.

---

## 📊 API Reference

### Camera Endpoints

- `GET /api/v1/cameras` - List all cameras
- `POST /api/v1/cameras` - Add camera
- `PUT /api/v1/cameras/{id}` - Update camera
- `DELETE /api/v1/cameras/{id}` - Delete camera

### Event Endpoints

- `GET /api/v1/events` - Query events (supports filtering)
- `GET /api/v1/events/{id}` - Get event details
- `GET /api/v1/events/stats` - Get statistics

### Zone Endpoints

- `GET /api/v1/zones?camera_id={id}` - List zones
- `POST /api/v1/zones` - Create zone
- `PUT /api/v1/zones/{id}` - Update zone
- `DELETE /api/v1/zones/{id}` - Delete zone

### Cross-Camera Tracking Endpoints

- `GET /api/v1/cross-camera/tracks` - Get all active cross-camera tracks
- `GET /api/v1/cross-camera/targets` - Get targeted persons
- `POST /api/v1/cross-camera/target` - Start targeted tracking
- `DELETE /api/v1/cross-camera/target/{id}` - Stop targeted tracking
- `GET /api/v1/cross-camera/path/{id}` - Get tracking path
- `GET /api/v1/cross-camera/predict/{id}` - Predict trajectory
- `GET /api/v1/clusters` - Get trajectory clustering

### System Endpoints

- `GET /api/v1/health` - System health check
- `GET /api/v1/metrics` - Performance metrics

Full API documentation: http://localhost:8000/docs

---

## 🧪 Testing

### Webcam Testing (PC Camera)

For quick testing without RTSP cameras, use the built-in webcam tester:

**Option A: Quick Test (Standalone Script)**
```bash
# Activate virtual environment
venv\Scripts\activate.bat

# Run webcam tester directly
python backend/webcam_tester.py
```

**Option B: Via API**
```bash
# Start webcam capture
curl -X POST http://localhost:8000/api/v1/webcam/start

# Check webcam status
curl http://localhost:8000/api/v1/webcam/status

# Get analysis from webcam (camera ID 999)
curl http://localhost:8000/api/v1/analysis/999

# Stop webcam
curl -X POST http://localhost:8000/api/v1/webcam/stop
```

**Features available in webcam mode:**
- Object Detection (person, car, animals, etc.)
- Face Recognition with known face registration
- **Emotion Detection** (happy, sad, angry, surprised, etc.)
- Speed & Height Analysis
- Image Quality Assessment & Enhancement
- Cross-Camera Tracking Simulation

---

## 🐛 Troubleshooting

### Camera shows "Offline"

- Verify RTSP URL is correct
- Check network connectivity to camera
- Review logs: `docker-compose logs backend`

### No events generated

- Ensure zones are defined for the camera
- Check if objects (person/vehicle) are in frame
- Verify confidence threshold (default: 0.5)
- Check logs for inference errors

### High CPU usage

- Reduce `fps_target` in config
- Use smaller YOLO model (yolov8n.pt)
- Limit number of concurrent cameras

### MQTT not working

- Ensure Mosquitto container is running
- Check MQTT broker address in config
- Test with: `mosquitto_sub -h localhost -t "argus/#" -v`

---

## 📁 Project Structure

```
argus/
├── backend/
│   ├── api/              # FastAPI application
│   ├── services/         # Core services (including cross-camera tracker)
│   ├── database/         # Database layer
│   └── config/           # Configuration
├── frontend/
│   └── src/
│       ├── pages/        # React pages
│       ├── components/   # React components
│       └── services/     # API client
├── config/              # YAML configuration
├── data/              # Database & snapshots
├── assets/            # Logo and static assets
│   └── argus-logo.png # Argus watchful eye logo
├── docker-compose.yml
└── README.md
```

---

## 📄 License

MIT License - See LICENSE file for details

---

**Built for Saimax Tech Solutions AI Job Assignment**