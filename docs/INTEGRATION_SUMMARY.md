# Argus Forked Repositories Integration Summary

> **Status note (current).** This document records which upstream projects
> inspired or supplied each capability. Two things have changed since it was
> written:
>
> 1. **File paths.** Services moved into subpackages: `backend/services/vision/`
>    (face, LPR, pose, enhancement), `backend/services/core_engine/` (inference,
>    tracking, agents), `backend/services/analytics/` (Re-ID, anomaly, speed),
>    `backend/services/management/` (rules, zones, events, retention, audit).
>    So `backend/services/deep_tracker.py` is now
>    `backend/services/core_engine/deep_tracker.py`, and so on.
> 2. **Heavy dependencies are optional and mostly absent.** The system runs
>    without PaddleOCR, MediaPipe, InsightFace, and `opencv-contrib`, falling
>    back to lighter implementations and logging which fallback is active at
>    startup. See the compatibility table below.
>
> For what is actually verified working, see the Project Status table in the
> [README](../README.md).

## Current dependency reality

| Capability | Ideal dependency | Installed by default | Fallback in use |
|---|---|---|---|
| License plate OCR | PaddleOCR | ❌ | Basic OCR heuristics |
| Face matching | `opencv-contrib` (LBPH) / InsightFace | ❌ | Haar cascade + histogram matching |
| Pose estimation | MediaPipe (33 keypoints) | ❌ | Geometric fallback |
| Object detection | Ultralytics YOLOv8 | ✅ | — (`yolov8n.pt` ships in `backend/models/`) |
| Tracking | ByteTrack/BoT-SORT/DeepSORT | ✅ (own Kalman implementation) | — |
| Vector DB / streaming | Qdrant, Kafka | ❌ (optional) | Not required for core operation |

`requirements-optional.txt` holds the heavy extras; `requirements.txt` is the
minimum needed to run the platform.

## Overview
This document summarizes the integration of forked repositories into the Argus AI Video Analytics Platform.

## Integrated Repositories & Features

### 1. License Plate Recognition (LPR)
**Repositories Integrated:**
- PaddleOCR - Primary OCR engine
- License-Plate-Recognition-System
- Automatic-Number-Plate-Recognition-ANPR-Facial-Recognition-System-FRS

**Implementation:**
- Created `backend/services/license_plate_recognition.py`
- Vehicle detection → license plate region detection → OCR
- Region-specific plate formatting (US, EU, UK)

### 2. Enhanced Face Recognition
**Repositories Integrated:**
- insightface - State-of-the-art 2D/3D face analysis (already in requirements)
- face_recognition - Alternative library

**Status:** Already implemented in existing `face_recognition.py`

### 3. Anomaly Detection
**Repositories Integrated:**
- anomalib - Anomaly detection algorithms
- AnomalyDetectionCVPR2018-Pytorch - Video anomaly detection

**Implementation:**
- Created `backend/services/anomaly_detector.py`
- Motion-based anomalies (large moving regions)
- Behavior-based anomalies (speed, trajectory, loitering)
- Abandoned object detection

### 4. Person Re-identification
**Repositories Integrated:**
- deep-person-reid - Person re-identification
- fast-reid - Alternative re-id system

**Implementation:**
- Created `backend/services/person_reid.py`
- Feature extraction using ResNet backbone
- Cross-camera person matching
- Trajectory reconstruction

### 5. Advanced Object Tracking
**Repositories Integrated:**
- deep_sort - Deep SORT tracker
- BoT-SORT - Boosted tracktor
- ByteTrack - Multi-object tracker

**Implementation:**
- Created `backend/services/deep_tracker.py`
- Kalman filter for motion prediction
- IoU-based matching
- Persistent track IDs

### 6. Pose Estimation
**Repositories Integrated:**
- mmpose - OpenMMLab pose estimation
- openpose - CMU's OpenPose
- MediaPipe - Cross-platform ML solutions

**Implementation:**
- Created `backend/services/pose_estimator.py`
- MediaPipe Pose for 33 keypoints
- Pose classification (standing, sitting, lying)
- Fall detection capability

### 7. Video Understanding
**Repositories Integrated:**
- mmaction2 - Video action recognition
- pytorchvideo - Video understanding

**Status:** Dependencies added to requirements

### 8. Image Enhancement
**Repositories Integrated:**
- Real-ESRGAN - Super-resolution
- BasicSR - Image restoration
- MiDaS - Depth estimation

**Status:** Already implemented in existing `image_enhancement.py`

### 9. Infrastructure
**Repositories Integrated:**
- Qdrant - Vector database (added to docker-compose)
- Kafka - Event streaming (added to docker-compose)

## New Service Files Created

| File (current path) | Description |
|------|-------------|
| `backend/services/vision/license_plate_recognition.py` | License plate detection & OCR |
| `backend/services/analytics/anomaly_detector.py` | Motion & behavior anomaly detection |
| `backend/services/analytics/person_reid.py` | Cross-camera person re-identification |
| `backend/services/core_engine/deep_tracker.py` | Object tracking with Kalman filters + two-stage association |
| `backend/services/vision/pose_estimator.py` | Human pose estimation & fall detection |

## Updated Files

| File | Changes |
|------|---------|
| `requirements.txt` / `requirements-optional.txt` | Core deps split from heavy optional extras (paddleocr, mediapipe, anomalib, pytorchvideo, django, kafka, qdrant) |
| `docker-compose.yml` | Added Qdrant and Kafka services |
| `config/config.yaml` | Added LPR, anomaly, reid, tracker, pose configurations |
| `backend/api/main.py` | Added LPR, anomaly, and tracker endpoints |
| `processing_coordinator.py` | Integrated all new services into processing pipeline |
| `README.md` | Updated features and documentation |

## New API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/api/v1/lpr` | License plate recognition status |
| `/api/v1/anomalies` | Recent anomaly detection results |
| `/api/v1/trackers` | Deep tracker status |
| `/api/v1/poses` | Pose estimation status |
| `/api/v1/analysis/{camera_id}` | Enhanced with LPR, pose, and anomaly data |
| `/api/v1/health` | Enhanced with subsystem status for all new services |

## Architecture Integration

The services are integrated into the processing pipeline in this order:

1. **Frame Ingestion** → `stream_ingestion`
2. **Image Enhancement** → `image_enhancement` (auto-enhance)
3. **Object Detection** → `inference_engine` (YOLOv8)
4. **Deep Tracking** → `deep_tracker` (persistent IDs)
5. **Zone Rules** → `rules_engine` (intrusion, loitering)
6. **Face Recognition** → `face_recognition` (persons)
7. **License Plate Recognition** → `license_plate_recognition` (vehicles)
8. **Pose Estimation** → `pose_estimator` (persons)
9. **Speed/Height Analysis** → `speed_height_analyzer`
10. **Anomaly Detection** → `anomaly_detector`

## Configuration

New configuration options in `config/config.yaml`:

```yaml
license_plate:
  enabled: true
  region: us
  min_confidence: 0.6

anomaly:
  enabled: true
  sensitivity: 0.7
  window_size: 100

tracker:
  enabled: true
  algorithm: bytetrack
  track_buffer: 30

pose:
  enabled: true
  min_detection_confidence: 0.5
```

## Running the Integrated System

### Docker Deployment (Recommended)
```bash
docker-compose up -d
# Services: backend, frontend, mqtt, qdrant, kafka
```

### Local Development
```bash
# Backend
pip install -r backend/requirements.txt
python -m uvicorn backend.api.main:app --reload
```

## Benefits of Integration

1. **Enhanced Security Monitoring**
   - License plate tracking for parking/entry control
   - Fall detection for elderly care facilities
   - Anomaly detection for suspicious activities

2. **Improved Tracking**
   - Persistent object IDs across frames
   - Better trajectory analysis
   - Cross-camera person re-identification

3. **Scalable Infrastructure**
   - Vector database for feature storage
   - Event streaming for analytics
   - Modular service architecture

4. **Advanced Analytics**
   - Pose-based activity recognition
   - Behavior anomaly patterns
   - Speed/height trend analysis