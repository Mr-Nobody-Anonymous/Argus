# 👁️ Argus - AI Video Analytics Platform

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

## 📹 Adding Cameras

### Via Web UI

1. Navigate to http://localhost:3000
2. Click "Add Camera"
3. Enter:
   - **Name**: Front Entrance
   - <!-- cspell:ignore RTSP -->
   - **RTSP URL**: `rtsp://your-camera-url`
   - **Location Tag**: Building A - Floor 1 (optional)
4. Click "Add"

### Via API

```bash
curl -X POST http://localhost:8000/api/v1/cameras \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Front Entrance",
    "rtsp_url": "rtsp://admin:pass@192.168.1.100:554/stream",
    "location_tag": "Building A - Floor 1"
  }'
```

### Using Test Streams

**Option A: Local Video File (Easiest)**
1. Place a video file (e.g., `people_walking.mp4`) in the project root folder.
2. When adding a camera, simply enter the **Filename** or **Absolute Path** as the URL:
   - **URL**: `people_walking.mp4`
   *(The system will play the video and auto-restart when it ends)*

**Option B: Public RTSP Stream**
```
rtsp://wowzaec2demo.streamlock.net/vod/mp4:BigBuckBunny_115k.mp4
```

**Option C: FFmpeg Simulation (Advanced)**
```bash
ffmpeg -re -stream_loop -1 -i test_video.mp4 -f rtsp rtsp://localhost:8554/stream
```

---

## 🎯 Defining Zones

Zones are areas within a camera's view where rules are applied.

### Via Web UI

1. Go to Camera Management
2. Click "Add Zone" on a camera card
3. Enter:
   - **Zone Name**: Restricted Area
   - **Coordinates**: `[[100,100],[200,100],[200,200],[100,200]]`
4. Click "Add Zone"

### Coordinate Format

- **Polygon**: Array of [x, y] points (minimum 3 points)
  ```json
  [[100,100], [200,100], [200,200], [100,200]]
  ```
- **Rectangle**: Two points (top-left, bottom-right)
  ```json
  [[0,0], [640,480]]
  ```

---

## ⚙️ Configuration

Edit `config/config.yaml`:

```yaml
system:
  fps_target: 15
  max_cameras: 4
  snapshot_retention_days: 30
  log_level: INFO

inference:
  model: yolov8n.pt
  confidence_threshold: 0.5
  device: cpu  # or cuda for GPU

rules:
  intrusion:
    enabled: true
    priority: high
  loitering:
    enabled: true
    threshold_seconds: 30
    priority: medium

mqtt:
  enabled: true
  broker: localhost
  port: 1883
```

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

### Manual Testing

1. Add a test camera (use public RTSP stream)
2. Define a zone covering part of the video
3. Wait for objects (people/vehicles) to enter the zone
4. Check Events page for generated alerts
5. View Analytics dashboard for statistics

### API Testing

```bash
# Health check
curl http://localhost:8000/api/v1/health

# List cameras
curl http://localhost:8000/api/v1/cameras

# Query events
curl "http://localhost:8000/api/v1/events?limit=10"

# Get metrics
curl http://localhost:8000/api/v1/metrics

# Cross-camera tracking
curl http://localhost:8000/api/v1/cross-camera/tracks
curl http://localhost:8000/api/v1/clusters
```

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

## 🔒 Security & Privacy

### Data Minimization

- Only events + snapshots stored (no full video)
- Configurable retention period (default: 30 days)
- Automatic cleanup of old events

### Local Processing

- 100% local inference (no cloud uploads)
- All data stays on-premise
- MQTT events are opt-in

### GDPR Compliance

- ✅ Data minimization
- ✅ Local processing
- ✅ Retention policy
- ⚠️ User consent management (not implemented)
- ⚠️ Right to be forgotten (manual)

---

## 🎯 Argus Prediction Engine

### Advanced Prediction Features

- **Trajectory Prediction**: Predicts future positions of moving objects using linear extrapolation
- **Suspicious Behavior Detection**: Identifies potential criminal activity patterns:
  - Erratic movement (rapid direction changes)
  - Suspicious speed variations (irregular acceleration/deceleration)
  - Casing behavior (systematic area scanning/recce)
- **Group Behavior Analysis**: Detects coordinated suspicious activities between multiple persons
- **Violence Prevention Patterns**: Fall detection and abnormal pose patterns
- **Cross-Camera Prediction**: Predicts next camera a person will appear in

### Accessing Prediction Features

```bash
# Get trajectory predictions for an object
curl "http://localhost:8000/api/v1/predictions/trajectory?object_id=cam1_obj_0"

# Get suspicious behavior analysis
curl "http://localhost:8000/api/v1/predictions/suspicious?camera_id=1"

# Get group behavior anomalies
curl "http://localhost:8000/api/v1/predictions/group-behavior?camera_id=1"

# Get cross-camera predictions
curl "http://localhost:8000/api/v1/cross-camera/predict/{person_id}"
```

---

## 📹 Camera Setup Guide

### Hikvision Cameras

1. **Find RTSP URL** in camera web interface:
   - Navigate to Configuration → Network → Advanced Settings
   - RTSP port: Usually `554`
   - Main stream path: `/Streaming/Channels/101`
   - Sub stream path: `/Streaming/Channels/102` (recommended for better performance)

2. **RTSP URL Format**:
   ```
   rtsp://username:password@192.168.1.100:554/Streaming/Channels/101
   ```

3. **Common Issues**:
   - Enable RTSP in camera settings
   - Check firewall allows port 554
   - Use sub-stream for multiple cameras

### Dahua Cameras

```
rtsp://username:password@192.168.1.100:554/cam/realmonitor?channel=1&subtype=0
```

### Generic IP Cameras

```
rtsp://[username]:[password]@[ip]:554/[path]
# Common paths: /stream1, /live, /video, /h264
```

### USB/Webcam Access

For local testing without IP cameras:
```bash
# Via API
curl -X POST http://localhost:8000/api/v1/webcam/start

# Or run the standalone tester
python backend/webcam_tester.py --camera 0
```

---

## 🖥️ User Interface Guide

### Dashboard Overview

Access at http://localhost:3000

1. **Camera Grid**: Live camera feeds with detection overlays
2. **Events Panel**: Real-time event alerts with filtering
3. **Analytics**: Charts showing detection trends, speeds, and patterns
4. **Zone Editor**: Define restricted/perimeter zones for rules

### Navigation

- **Cameras**: Add/edit/remove camera feeds
- **Zones**: Define virtual boundaries for alerts
- **Events**: Review and manage security alerts
- **Analytics**: View system performance and statistics
- **Cross-Camera Tracking**: Monitor persons across multiple cameras
- **Faces**: Register known persons for face recognition
- **Settings**: Configure system parameters

### Event Types

| Type | Description | Priority |
|------|-------------|----------|
| intrusion | Person/vehicle enters restricted zone | high |
| loitering | Object stays in zone > 30 seconds | medium |
| speed_violation | Vehicle exceeds speed threshold | high |
| fall_detection | Person fall detected via pose | high |
| abandoned_object | Stationary object detected | medium |
| large_motion | Unusual motion pattern | medium |
| erratic_movement | Suspicious erratic behavior | high |
| suspicious_grouping | Multiple persons acting together | medium |
| cross_camera_movement | Person tracked across cameras | medium |

---

## 🚀 Running on a Server

### Production Deployment (Docker)

```bash
# On Linux server
sudo docker-compose up -d

# Check status
sudo docker-compose ps

# View logs
sudo docker-compose logs -f backend

# Stop services
sudo docker-compose down
```

### Headless Mode (No Display)

```bash
# Run backend only
python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000

# Run frontend only if needed
cd frontend && npm run build
```

### Systemd Service (Linux)

```bash
# Create service file
sudo nano /etc/systemd/system/argus.service

# Add content:
[Unit]
Description=Argus AI Video Analytics
After=docker.service
Requires=docker.service

[Service]
Type=simple
ExecStart=/usr/bin/docker-compose -f /opt/argus/docker-compose.yml up
WorkingDirectory=/opt/argus
Restart=always

[Install]
WantedBy=multi-user.target

# Enable service
sudo systemctl enable argus
sudo systemctl start argus
```

---

## 🚀 Running on Your PC

### Quick Start (Windows)

1. **Double-click** `run_app.bat` - This starts both backend and frontend automatically

### Manual Setup

```bash
# 1. Clone and setup
git clone <repo-url>
cd Argus

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install dependencies
pip install -r backend/requirements.txt

# 4. Start backend
python -m uvicorn backend.api.main:app --reload --host 0.0.0.0 --port 8000

# 5. In another terminal, start frontend
cd frontend
npm install
npm run dev
```

### Testing with Webcam

```bash
# Start webcam tester (shows live feed with detections)
python backend/webcam_tester.py

# Or via API endpoints
curl -X POST http://localhost:8000/api/v1/webcam/start
curl http://localhost:8000/api/v1/analysis/999
curl -X POST http://localhost:8000/api/v1/webcam/stop
```

---

## 🚧 Known Limitations

1. **No Authentication**: Single-user mode (no login required)
2. **Limited Scalability**: Max 4 cameras on CPU inference
3. **Basic Tracking**: Simple centroid tracking (no persistent IDs)
4. **Zone Editor**: JSON-based (no GUI polygon drawing)
5. **No Video Recording**: Events + snapshots only

---

## 🗺️ Roadmap

### Week 1: Performance & Scalability

- [ ] PostgreSQL migration
- [ ] GPU acceleration (TensorRT/OpenVINO)
- [ ] WebSocket for real-time updates
- [ ] Multi-model support (face recognition, LPR)

### Week 2: Security & Features

- [ ] JWT authentication
- [ ] Role-based access control
- [ ] Clip recording (pre/post event buffers)
- [ ] GUI zone editor
- [ ] Email/SMS notifications

---

## 📄 License

MIT License - See LICENSE file for details

---

## 🙏 Acknowledgments

Built using:
- **YOLOv8** (Ultralytics) for object detection
- **FastAPI** for backend API
- **React** + **Material-UI** for frontend
- **OpenCV** for video processing
- **Recharts** for analytics visualization

Inspired by: Milestone VMS, BriefCam, Avigilon, Frigate NVR

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section above
2. Review logs: `docker-compose logs`
3. Check API health: http://localhost:8000/api/v1/health

---

**Built for Saimax Tech Solutions AI Job Assignment**