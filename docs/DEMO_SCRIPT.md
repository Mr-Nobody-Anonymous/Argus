# Argus - Demo Script (3-5 Minutes)

## Preparation (Before Demo)

1. **Set a signing key and start the backend**:
   ```bash
   export ARGUS_JWT_SECRET="$(python -c 'import secrets;print(secrets.token_urlsafe(48))')"
   python -m uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
   # or: docker-compose up -d
   ```

2. **Wait for services** (~30 s; the log prints
   `YOLO model loaded and warmed up successfully` when ready)

3. **Have a login ready.** The dashboard now opens on a login screen. Create a
   user through Django admin, or reuse an existing one. Demo with an **admin**
   account so every panel is reachable — a viewer account will (correctly) get
   403s on camera creation, which is confusing mid-demo.

4. **Prepare a stream**: `data/demo_clip.mp4` ships with the repo and loops
   automatically, which is more reliable than a public RTSP endpoint.

> **Timing note:** on CPU the pipeline runs at ~1.3–1.6 FPS per camera. The live
> view updates visibly but is not smooth video — say so up front rather than
> letting the audience assume something is broken. On GPU it is real-time.

---

## Demo Flow

### **Minute 0:00 - 0:30: Introduction & Sign-in**

**Say**: "Argus is an AI-powered video analytics platform that monitors CCTV cameras in real-time, detects intrusions and loitering, and generates intelligent alerts."

**Show**:
- Navigate to http://localhost:3000
- **Sign in.** Mention that every API route requires a token and that roles come
  from the Django user table — there is no separate credential store.
- Point out the username/role chip in the top bar, then the navigation sidebar

---

### **Minute 0:30 - 1:30: Add Camera**

**Say**: "Let's add a camera to the system."

**Do**:
1. Click "Add Camera" button
2. Fill in:
   - Name: "Front Entrance"
   - RTSP URL: `people_walking.mp4` (or absolute path if needed)
   - Location: "Building A - Floor 1"
3. Click "Add"

**Note**: Using the local video ensures better accuracy and no network lag during the demo!

**Show**:
- Camera appears in grid with "Online" status
- FPS counter updates in real-time
- Status indicator turns green

**Say**: "The system automatically connects to the stream, starts processing frames, and displays real-time performance metrics."

---

### **Minute 1:30 - 2:30: Define Zone**

**Say**: "Now let's define a restricted zone where we want to detect intrusions."

**Do**:
1. Click "Add Zone" on the camera card
2. Enter:
   - Zone Name: "Restricted Area"
   - Coordinates: `[[200,200],[400,200],[400,400],[200,400]]`
3. Click "Add Zone"

**Show**:
- Zone successfully created confirmation

**Say**: "Zones are defined using polygon coordinates. In production, this would have a visual editor, but for the MVP, JSON input demonstrates the underlying flexibility."

---

### **Minute 2:30 - 3:30: View Events**

**Say**: "Let's check if any events have been generated."

**Do**:
1. Navigate to "Events" page
2. Show event list with color-coded priorities
3. Click on an event to view details

**Show**:
- Event feed with real-time updates
- Filters for camera, rule type, and priority
- Event details modal with:
  - Snapshot with bounding box
  - Metadata (zone name, confidence, timestamp)
  - Object type and detection confidence

**Say**: "Events are generated when objects enter restricted zones. Each event includes a snapshot with the detected object highlighted, along with full metadata for investigation."

---

### **Minute 3:30 - 4:30: Analytics Dashboard**

**Say**: "The analytics dashboard provides insights into system performance and event trends."

**Do**:
1. Navigate to "Analytics" page
2. Scroll through the dashboard

**Show**:
- Summary cards (total events, active cameras, CPU/memory usage)
- Event distribution charts (by rule type, by priority)
- Camera performance metrics (FPS, queue depth)

**Say**: "This gives operators a bird's-eye view of the entire system, helping identify patterns and optimize camera placement."

---

### **Minute 4:30 - 5:00: System Health & Wrap-up**

**Say**: "Let's verify system health."

**Do**:
1. Open http://localhost:8000/api/v1/health in browser
2. Show JSON response with subsystem status

**Show**:
```json
{
  "status": "healthy",
  "subsystems": {
    "database": "ok",
    "mqtt": "ok",
    "cameras": {"total": 1, "online": 1},
    "inference": {"model_loaded": true, "avg_inference_time_ms": 45}
  }
}
```

**Say**: "The system provides comprehensive health monitoring and metrics, essential for production deployments."

---

## Key Points to Emphasize

1. **Real-time Processing**: Live FPS updates, instant event generation
2. **Privacy-First**: All processing happens locally, no cloud uploads
3. **Scalability**: Designed to handle multiple cameras with queue management
4. **Secured by default**: JWT auth + three-role RBAC on all 40 protected routes, audit
   trail on every mutation, scheduled data retention
5. **Measured, not asserted**: the swarm architecture is benchmarked against a
   linear baseline (+26.8% FPS at identical detection quality), and 44
   mutation-verified tests guard the pipeline
6. **Extensibility**: MQTT integration, API-first design, modular architecture

---

## Backup Talking Points

If extra time or questions:

- **MQTT Integration**: "Events are published to MQTT topics for smart home integration"
- **Event Deduplication**: "Alerts are deduplicated per tracked subject on a
  sliding window, so someone standing in a zone produces one event rather than
  one every few seconds. It re-arms only after they actually leave."
- **Audit Trail**: "Every mutation and login is recorded with actor, role, IP,
  and outcome — including denials, which are the interesting ones"
- **Auto-Reconnect**: "If a camera stream fails, the system automatically retries with exponential backoff"
- **GDPR Compliance**: "Only events and snapshots are stored, with configurable retention policies"

---

## Common Questions & Answers

**Q: How many cameras can it handle?**
A: On CPU each camera runs at ~1.3–1.6 FPS, so a handful is realistic. GPU
inference is where the design targets real-time on many streams. Honest answer:
it has been verified end-to-end at small scale, not load-tested at 100 cameras —
the scaling plan is written up in `docs/ARCHITECTURE_CITYOS.md`.

**Q: Is it production-ready?**
A: Not yet, and the gaps are documented rather than hidden. Auth, RBAC, audit,
retention, and metrics are in place. Missing: TLS termination, encryption of
face embeddings at rest, PostgreSQL, and CI. See `PRODUCTION_ROADMAP.md`.

**Q: What about privacy concerns?**
A: All processing is local. No data leaves your network. Only events + snapshots are stored, not full video.

**Q: Can it detect other objects?**
A: Yes! The YOLO model supports 80 object classes. Currently configured for person and vehicle, but easily extensible.

**Q: How accurate is the detection?**
A: The shipped model is YOLOv8n (the smallest variant), with a configurable
confidence threshold defaulting to 50%. Quoting a single accuracy figure would
be misleading — accuracy depends on the model variant, the scene, and the
threshold. There is no labelled evaluation set for this deployment yet, which is
an open roadmap item.

---

## Demo Environment Checklist

- [ ] `ARGUS_JWT_SECRET` exported before starting the backend
- [ ] Backend running; log shows `YOLO model loaded and warmed up successfully`
- [ ] **Admin credentials to hand** (a viewer account will hit 403s on camera creation)
- [ ] Frontend running (`npm run dev`) and the login screen renders
- [ ] `data/demo_clip.mp4` present, or a reachable RTSP stream
- [ ] Browser tabs pre-opened: dashboard, `/docs`, `/metrics`
- [ ] Example zone coordinates ready to paste (field name is `coordinates`, not `polygon`)
- [ ] Know your numbers: ~1.3–1.6 FPS on CPU, 12–15 detections/frame on the demo clip
