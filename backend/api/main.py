"""
Main FastAPI application for Argus
"""
import logging
import os
import sys
import time
from pathlib import Path
from fastapi import (
    FastAPI, HTTPException, Query, UploadFile, File, Form, Depends, Request, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, List
import psutil
import cv2
import numpy as np

# Ensure the project root is importable so `backend.*` absolute imports resolve
# whether the app is launched from the repo root or from inside backend/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.api.models import (
    Camera, CameraCreate, CameraUpdate,
    Zone, ZoneCreate,
    Event,
    HealthResponse, MetricsResponse,
    LoginRequest, RefreshRequest
)
from backend.services.management.camera_manager import get_camera_manager
from backend.services.management.zone_manager import get_zone_manager
from backend.services.management.event_store import get_event_store
from backend.services.core_engine.processing_coordinator import get_processing_coordinator
from backend.services.core_engine.inference_engine import get_inference_engine
from backend.services.management.mqtt_publisher import get_mqtt_publisher
from backend.services.vision.image_enhancement import get_image_enhancement
from backend.services.vision.face_recognition import get_face_recognition
from backend.services.vision.license_plate_recognition import get_license_plate_recognition
from backend.services.analytics.anomaly_detector import get_anomaly_detector
from backend.services.vision.pose_estimator import get_pose_estimator
from backend.services.core_engine.deep_tracker import get_deep_tracker
from backend.services.analytics.cross_camera_tracker import get_cross_camera_tracker, SKLEARN_AVAILABLE
from backend.database.db import get_db, close_db
from backend.config.config import get_config, resolve_path
from backend.api.stream_routes import router as stream_router
from backend.api.stream_ws import router as ws_stream_router
from backend.api.auth import (
    AuthUser, authenticate_user, create_token, token_to_user,
    get_current_user, require_role, auth_enabled,
    ROLE_ADMIN, ROLE_OPERATOR, ROLE_VIEWER,
    is_locked_out, record_failed_attempt, clear_failed_attempts, client_key,
    EPHEMERAL_SECRET_IN_USE,
)
from backend.services.management.audit_log import get_audit_log
from backend.services.management.retention import get_retention_scheduler
from backend.api.observability import (
    get_metrics_registry, configure_logging,
)
from backend.config.config import find_unresolved_secrets

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Apply ARGUS_LOG_FORMAT/ARGUS_LOG_LEVEL (json logging for production ingestion)
configure_logging()

# Startup time for uptime calculation
startup_time = time.time()


def _validate_startup_config():
    """
    Fail fast on configuration problems instead of discovering them mid-run.

    Currently checks for unresolved ${VAR} secret references. Anything found is
    reported by dotted path so the operator knows exactly which variable to set.
    """
    try:
        import yaml
        from backend.config.config import PROJECT_ROOT
        config_file = PROJECT_ROOT / "config" / "config.yaml"
        if not config_file.exists():
            return
        with open(config_file) as fh:
            raw = yaml.safe_load(fh)
        unresolved = find_unresolved_secrets(raw)
        if unresolved:
            logger.error("Configuration has unresolved secret references:")
            for item in unresolved:
                logger.error(f"  - {item}")
            logger.error("Set these environment variables (see .env.example).")
    except Exception as exc:  # noqa: BLE001 - validation must not block startup
        logger.warning(f"Config validation skipped: {exc}")


def _start_retention_scheduler():
    try:
        get_retention_scheduler().start()
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Could not start retention scheduler: {exc}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle management"""
    logger.info("Starting Argus API...")

    # ── Security posture banner ──
    # Surfaced loudly at startup so an insecure configuration can never be in
    # effect without being obvious in the logs.
    if not auth_enabled():
        logger.warning("=" * 72)
        logger.warning("AUTHENTICATION IS DISABLED (ARGUS_DISABLE_AUTH). Development use only.")
        logger.warning("Every endpoint, including camera and identity management, is OPEN.")
        logger.warning("=" * 72)
    else:
        logger.info("Authentication ENABLED (JWT, roles: admin/operator/viewer)")
        if EPHEMERAL_SECRET_IN_USE:
            logger.warning(
                "ARGUS_JWT_SECRET unset - using an ephemeral key; tokens die on restart."
            )

    # Fail fast on unresolved ${VAR} secret references.
    _validate_startup_config()

    # Initialize database
    get_db()

    # Audit log lives in the same database
    get_audit_log()

    # Initialize new services
    get_license_plate_recognition().init_database()
    get_anomaly_detector().init_database()
    get_pose_estimator()
    get_deep_tracker()

    # Start the retention/purge scheduler (privacy: data must not accumulate
    # forever). Runs in a daemon thread; interval and policy come from config.
    _start_retention_scheduler()
    
    # Start processing for existing cameras
    coordinator = get_processing_coordinator()
    coordinator.start_all_cameras()
    
    yield
    
    # Shutdown
    logger.info("Shutting down Argus API...")
    coordinator.stop_all_cameras()
    get_mqtt_publisher().disconnect()
    close_db()


# Create FastAPI app
app = FastAPI(
    title="Argus API",
    description="AI Video Analytics Platform - The Watchful Guardian. Features: Cross-Camera Tracking, Image Enhancement, Face Recognition, Speed & Height Analysis, LPR, Anomaly Detection, Pose Estimation",
    version="2.1.0",
    lifespan=lifespan
)

# ── CORS ─────────────────────────────────────────────────────────────────────
# Browsers reject `Access-Control-Allow-Origin: *` together with credentials,
# so an explicit allowlist is required.
#
# Origins come from ARGUS_CORS_ORIGINS (comma-separated). The permissive
# catch-all regex that previously matched ANY http(s) host was a sandbox-preview
# convenience and is NOT enabled by default - it effectively disabled origin
# checking. Set ARGUS_CORS_ORIGIN_REGEX explicitly if you need it (e.g. for a
# tunnelled preview host).
_default_cors_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]
_cors_env = os.environ.get("ARGUS_CORS_ORIGINS", "").strip()
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env else _default_cors_origins
)
_cors_origin_regex = os.environ.get("ARGUS_CORS_ORIGIN_REGEX", "").strip() or None

if _cors_origin_regex:
    logger.warning(
        "ARGUS_CORS_ORIGIN_REGEX is set (%s) - ensure it is not overly permissive.",
        _cors_origin_regex,
    )

app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_origin_regex=_cors_origin_regex,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Paths audited by the login handler itself (which knows the attempted
# username even when authentication fails), so the middleware skips them.
_AUDIT_EXEMPT_PATHS = {"/api/v1/auth/login", "/api/v1/auth/refresh"}


@app.middleware("http")
async def audit_mutations(request: Request, call_next):
    """
    Audit every state-changing request.

    Implemented as middleware rather than per-handler decoration so a new
    mutating endpoint is covered automatically - an audit trail with gaps is
    worse than none, because it implies completeness it does not have.
    """
    if request.method not in ("POST", "PUT", "PATCH", "DELETE"):
        return await call_next(request)

    path = request.url.path
    if path in _AUDIT_EXEMPT_PATHS:
        return await call_next(request)

    response = await call_next(request)

    try:
        # Resolve the caller from the bearer token. Unauthenticated attempts on
        # protected routes are still worth recording.
        principal, role, user_id = "anonymous", None, None
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token_user = token_to_user(header[7:].strip())
            if token_user:
                principal, role, user_id = token_user.username, token_user.role, token_user.id
        elif not auth_enabled():
            principal, role = "dev-auth-disabled", ROLE_ADMIN

        # Derive resource + id from /api/v1/<resource>/<id>
        parts = [p for p in path.split("/") if p]
        resource = parts[2] if len(parts) > 2 else path
        resource_id = parts[3] if len(parts) > 3 else None

        if response.status_code < 400:
            outcome = "success"
        elif response.status_code in (401, 403):
            outcome = "denied"
        else:
            outcome = "error"

        get_audit_log().record(
            username=principal,
            user_id=user_id,
            role=role,
            action=request.method.lower(),
            resource=resource,
            resource_id=resource_id,
            outcome=outcome,
            client_ip=request.client.host if request.client else None,
            detail=f"{request.method} {path} -> {response.status_code}",
        )
    except Exception as exc:  # noqa: BLE001 - auditing must never break a request
        logger.error(f"Audit middleware error: {exc}")

    return response

# Mount static files for snapshots
snapshot_dir = resolve_path(get_config().system.snapshot_dir)
snapshot_dir.mkdir(parents=True, exist_ok=True)
# NOTE: snapshots are event evidence containing identifiable people, so the
# bare StaticFiles mount is only enabled when auth is explicitly disabled for
# local development. Authenticated access goes through
# GET /api/snapshots/{camera_id}/{filename}, which enforces the viewer role and
# validates the path against traversal.
if not auth_enabled():
    app.mount("/snapshots", StaticFiles(directory=str(snapshot_dir)), name="snapshots")
    logger.warning(
        "Auth disabled: /snapshots is mounted WITHOUT access control (development only)."
    )

# Register WebSocket and streaming endpoints.
# stream_routes -> /api/snapshots/... and /api/mjpeg/stream/{camera_id}
# stream_ws     -> /api/ws/stream/{camera_id}  (documented WebSocket protocol)
app.include_router(stream_router, prefix="/api")
app.include_router(ws_stream_router, prefix="/api")


# ==================== Authentication Endpoints ====================

@app.post("/api/v1/auth/login", response_model=dict)
async def login(request: Request, credentials: LoginRequest):
    """
    Authenticate against the Django `auth_user` table and issue JWTs.

    Failed attempts are throttled per (client IP, username) to blunt credential
    stuffing, and both outcomes are written to the audit log.
    """
    audit = get_audit_log()
    throttle_key = client_key(request, credentials.username)
    client_ip = request.client.host if request.client else None

    locked, retry_after = is_locked_out(throttle_key)
    if locked:
        audit.record(
            username=credentials.username, action="login", resource="auth",
            outcome="locked_out", client_ip=client_ip,
            detail=f"Locked for another {retry_after}s",
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Too many failed attempts. Try again in {retry_after} seconds.",
            headers={"Retry-After": str(retry_after)},
        )

    user = authenticate_user(credentials.username, credentials.password)
    if user is None:
        record_failed_attempt(throttle_key)
        audit.record(
            username=credentials.username, action="login", resource="auth",
            outcome="failure", client_ip=client_ip,
        )
        # Identical message for unknown user and wrong password - do not leak
        # which usernames exist.
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    clear_failed_attempts(throttle_key)
    audit.record(
        username=user.username, user_id=user.id, role=user.role,
        action="login", resource="auth", outcome="success", client_ip=client_ip,
    )
    return {
        "access_token": create_token(user, "access"),
        "refresh_token": create_token(user, "refresh"),
        "token_type": "bearer",
        "user": {"id": user.id, "username": user.username, "role": user.role},
    }


@app.post("/api/v1/auth/refresh", response_model=dict)
async def refresh_token(payload: RefreshRequest):
    """Exchange a valid refresh token for a new access token."""
    user = token_to_user(payload.refresh_token, expected_type="refresh")
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {
        "access_token": create_token(user, "access"),
        "token_type": "bearer",
        "user": {"id": user.id, "username": user.username, "role": user.role},
    }


@app.get("/api/v1/auth/me", response_model=dict)
async def whoami(user: AuthUser = Depends(get_current_user)):
    """Return the authenticated principal and its effective role."""
    return {
        "id": user.id,
        "username": user.username,
        "role": user.role,
        "is_superuser": user.is_superuser,
        "email": user.email,
    }


@app.get("/api/v1/audit", response_model=dict)
async def get_audit_entries(
    username: Optional[str] = Query(None),
    resource: Optional[str] = Query(None),
    action: Optional[str] = Query(None),
    outcome: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0),
    _user: AuthUser = Depends(require_role(ROLE_ADMIN)),
):
    """Read the audit trail. Admin only."""
    entries, total = get_audit_log().query(
        username=username, resource=resource, action=action,
        outcome=outcome, limit=limit, offset=offset,
    )
    return {"entries": entries, "total": total, "limit": limit, "offset": offset}


# ==================== Camera Endpoints ====================

@app.get("/api/v1/cameras", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_cameras():
    """Get all cameras"""
    try:
        camera_manager = get_camera_manager()
        cameras = camera_manager.get_all_cameras()
        return {"cameras": cameras, "count": len(cameras)}
    except Exception as e:
        logger.error(f"Error getting cameras: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cameras", response_model=dict, dependencies=[Depends(require_role(ROLE_ADMIN))])
async def create_camera(camera: CameraCreate):
    """Create a new camera"""
    try:
        camera_manager = get_camera_manager()
        
        # Check for duplicate URL
        existing = camera_manager.get_camera_by_url(camera.rtsp_url)
        if existing:
            raise HTTPException(status_code=400, detail="Camera with this RTSP URL already exists")
        
        # Create camera
        new_camera = camera_manager.create_camera(
            name=camera.name,
            rtsp_url=camera.rtsp_url,
            location_tag=camera.location_tag
        )
        
        # Start processing
        coordinator = get_processing_coordinator()
        coordinator.start_camera_processing(new_camera['id'])
        
        return {"camera": new_camera, "status": "created"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cameras/{camera_id}", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_camera(camera_id: int):
    """Get camera by ID"""
    try:
        camera_manager = get_camera_manager()
        camera = camera_manager.get_camera(camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        return {"camera": camera}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/cameras/{camera_id}", response_model=dict, dependencies=[Depends(require_role(ROLE_ADMIN))])
async def update_camera(camera_id: int, camera: CameraUpdate):
    """Update camera"""
    try:
        camera_manager = get_camera_manager()
        
        # Check if camera exists
        existing = camera_manager.get_camera(camera_id)
        if not existing:
            raise HTTPException(status_code=404, detail="Camera not found")
        
        # Update camera
        updated_camera = camera_manager.update_camera(
            camera_id,
            **camera.model_dump(exclude_unset=True)
        )
        
        return {"camera": updated_camera, "status": "updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/cameras/{camera_id}", dependencies=[Depends(require_role(ROLE_ADMIN))])
async def delete_camera(camera_id: int):
    """Delete camera"""
    try:
        camera_manager = get_camera_manager()
        coordinator = get_processing_coordinator()

        if not camera_manager.get_camera(camera_id):
            raise HTTPException(status_code=404, detail="Camera not found")
        
        # Stop processing
        coordinator.stop_camera_processing(camera_id)
        
        # Delete camera
        camera_manager.delete_camera(camera_id)
        
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting camera: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Zone Endpoints ====================

@app.get("/api/v1/zones", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_zones(camera_id: Optional[int] = Query(None)):
    """Get zones, optionally filtered by camera"""
    try:
        zone_manager = get_zone_manager()
        if camera_id:
            zones = zone_manager.get_zones_by_camera(camera_id)
        else:
            zones = zone_manager.get_all_zones()
        return {"zones": zones, "count": len(zones)}
    except Exception as e:
        logger.error(f"Error getting zones: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/zones", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def create_zone(zone: ZoneCreate):
    """Create a new zone"""
    try:
        zone_manager = get_zone_manager()
        new_zone = zone_manager.create_zone(
            camera_id=zone.camera_id,
            name=zone.name,
            zone_type=zone.type,
            coordinates=zone.coordinates
        )
        return {"zone": new_zone, "status": "created"}
    except Exception as e:
        logger.error(f"Error creating zone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/zones/{zone_id}", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def update_zone(zone_id: int, zone: ZoneCreate):
    """Update zone"""
    try:
        zone_manager = get_zone_manager()
        updated_zone = zone_manager.update_zone(
            zone_id,
            name=zone.name,
            type=zone.type,
            coordinates=zone.coordinates
        )
        if not updated_zone:
            raise HTTPException(status_code=404, detail="Zone not found")
        return {"zone": updated_zone, "status": "updated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating zone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/zones/{zone_id}", dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def delete_zone(zone_id: int):
    """Delete zone"""
    try:
        zone_manager = get_zone_manager()
        if not zone_manager.get_zone(zone_id):
            raise HTTPException(status_code=404, detail="Zone not found")
        zone_manager.delete_zone(zone_id)
        return {"status": "deleted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting zone: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Event Endpoints ====================

@app.get("/api/v1/events", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_events(
    camera_id: Optional[int] = Query(None),
    from_time: Optional[str] = Query(None),
    to_time: Optional[str] = Query(None),
    rule: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, le=500),
    offset: int = Query(0, ge=0)
):
    """Query events with filters"""
    try:
        event_store = get_event_store()
        
        # Parse datetime strings
        from_dt = datetime.fromisoformat(from_time) if from_time else None
        to_dt = datetime.fromisoformat(to_time) if to_time else None
        
        events, total = event_store.query_events(
            camera_id=camera_id,
            from_time=from_dt,
            to_time=to_dt,
            rule_type=rule,
            priority=priority,
            status=status,
            limit=limit,
            offset=offset
        )
        
        return {
            "events": events,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        logger.error(f"Error querying events: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/events/stats", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_event_stats(
    camera_id: Optional[int] = None,
    hours: int = 24
):
    """Get event statistics"""
    try:
        event_store = get_event_store()
        stats = event_store.get_event_stats(camera_id=camera_id, hours=hours)
        return stats
    except Exception as e:
        logger.error(f"Error getting event stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/events/{event_id}", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_event(event_id: int):
    """Get event by ID"""
    try:
        event_store = get_event_store()
        event = event_store.get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        return {"event": event}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting event: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Analysis Endpoints ====================

@app.get("/api/v1/analysis/{camera_id}", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_camera_analysis(camera_id: int):
    """Get detailed analysis for a camera including speed, height, LPR, pose, and anomalies"""
    try:
        coordinator = get_processing_coordinator()
        analysis = coordinator.get_camera_analysis(camera_id)
        if not analysis:
            return {"status": "no_data", "message": "No analysis data available for this camera"}
        
        return {
            "camera_id": camera_id,
            "timestamp": analysis.get('timestamp'),
            "detections": analysis.get('detections', []),
            "face_results": analysis.get('face_results', []),
            "lpr_results": analysis.get('lpr_results', []),
            "pose_results": analysis.get('pose_results', []),
            "anomalies": analysis.get('anomalies', []),
            "analysis_results": [
                {
                    'object_id': r.get('object_id'),
                    'track_id': r.get('track_id'),
                    'class_name': r.get('class_name'),
                    'speed_kmh': r.get('speed_kmh', 0),
                    'speed_category': r.get('speed_category', 'unknown'),
                    'height_m': r.get('height_m', 0),
                    'height_category': r.get('height_category', 'unknown'),
                    'direction': r.get('direction', 'unknown'),
                    'bbox_area': r.get('bbox_area', 0),
                    'track_duration_s': r.get('track_duration_s', 0),
                    'face_recognition': r.get('face_recognition')
                }
                for r in analysis.get('analysis_results', [])
            ],
            "image_quality": analysis.get('image_quality', {})
        }
    except Exception as e:
        logger.error(f"Error getting analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== LPR Endpoints ====================

@app.get("/api/v1/lpr", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def get_lpr_status():
    """Get license plate recognition status"""
    try:
        lpr = get_license_plate_recognition()
        return {
            "enabled": lpr.enabled,
            "initialized": lpr._initialized,
            "region": lpr.region
        }
    except Exception as e:
        logger.error(f"Error getting LPR status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Anomaly Endpoints ====================

@app.get("/api/v1/anomalies", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_recent_anomalies(
    camera_id: Optional[int] = Query(None),
    limit: int = Query(100, le=500)
):
    """Get recent anomalies detected across cameras"""
    try:
        coordinator = get_processing_coordinator()
        
        anomalies = []
        if camera_id:
            analysis = coordinator.get_camera_analysis(camera_id)
            if analysis and analysis.get('anomalies'):
                anomalies = analysis.get('anomalies', [])[:limit]
        
        return {
            "anomalies": anomalies,
            "count": len(anomalies)
        }
    except Exception as e:
        logger.error(f"Error getting anomalies: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Tracker Endpoints ====================

@app.get("/api/v1/trackers", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_tracker_status():
    """Get deep tracker status and active tracks"""
    try:
        tracker = get_deep_tracker()
        return {
            "enabled": tracker.enabled,
            "algorithm": tracker.algorithm,
            "active_tracks": tracker.get_active_tracks()
        }
    except Exception as e:
        logger.error(f"Error getting tracker status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Pose Endpoints ====================

@app.get("/api/v1/poses", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_pose_status():
    """Get pose estimation status"""
    try:
        pose = get_pose_estimator()
        return pose.get_pose_statistics()
    except Exception as e:
        logger.error(f"Error getting pose status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Enhancement Endpoints ====================

@app.post("/api/v1/enhance/analyze", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def analyze_image_quality(file: UploadFile = File(...)):
    """Upload an image/frame to analyze its quality"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        enhancer = get_image_enhancement()
        quality = enhancer.detect_quality_issues(frame)
        
        # Show enhanced version comparison
        enhanced = enhancer.enhance_frame(frame, mode="auto")
        
        return {
            "filename": file.filename,
            "original_quality": quality,
            "enhancement_applied": len(quality.get('issues', [])) > 0,
            "issues_found": quality.get('issues', []),
            "recommended_mode": "auto"
        }
    except Exception as e:
        logger.error(f"Error analyzing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Face Recognition Endpoints ====================

@app.get("/api/v1/faces", response_model=dict, dependencies=[Depends(require_role(ROLE_ADMIN))])
async def get_known_faces():
    """Get list of registered known faces"""
    try:
        face_recognition = get_face_recognition()
        faces = face_recognition.get_known_faces_list()
        return {"faces": faces, "count": len(faces)}
    except Exception as e:
        logger.error(f"Error getting known faces: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/faces/register", response_model=dict, dependencies=[Depends(require_role(ROLE_ADMIN))])
async def register_face(name: str = Form(...), file: UploadFile = File(...)):
    """Register a new face for recognition"""
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        face_recognition = get_face_recognition()
        success = face_recognition.register_face(name, gray)
        
        if success:
            return {"status": "registered", "name": name}
        else:
            raise HTTPException(status_code=500, detail="Failed to register face")
    except Exception as e:
        logger.error(f"Error registering face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/faces/{face_id}", dependencies=[Depends(require_role(ROLE_ADMIN))])
async def delete_face(face_id: int):
    """Delete a registered face"""
    try:
        face_recognition = get_face_recognition()
        face_recognition.delete_face(face_id)
        return {"status": "deleted"}
    except Exception as e:
        logger.error(f"Error deleting face: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/faces/status", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def get_face_recognition_status():
    """Get face recognition system status"""
    try:
        face_recognition = get_face_recognition()
        return {
            "enabled": face_recognition.enabled,
            "initialized": face_recognition.is_initialized(),
            "known_faces_count": len(face_recognition.get_known_faces_list()),
            "emotion_detection_available": face_recognition.get_emotion_detection_status()
        }
    except Exception as e:
        logger.error(f"Error getting face status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Webcam Test Endpoints ====================

@app.post("/api/v1/webcam/start", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def start_webcam(camera_id: int = 0):
    """Start webcam capture for testing (camera_id is the PC webcam index)"""
    try:
        coordinator = get_processing_coordinator()
        
        if coordinator.is_webcam_mode():
            return {"status": "already_running", "message": "Webcam mode already active"}
        
        success = coordinator.enable_webcam_mode(camera_id)
        if success:
            coordinator.start_camera_processing(999)  # Use camera ID 999 for webcam
            return {"status": "started", "webcam_id": camera_id}
        else:
            raise HTTPException(status_code=500, detail="Could not open webcam")
    except Exception as e:
        logger.error(f"Error starting webcam: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/webcam/stop", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def stop_webcam():
    """Stop webcam capture"""
    try:
        coordinator = get_processing_coordinator()
        coordinator.disable_webcam_mode()
        coordinator.stop_camera_processing(999)
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Error stopping webcam: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/webcam/status", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_webcam_status():
    """Get webcam status"""
    try:
        coordinator = get_processing_coordinator()
        return {
            "webcam_mode": coordinator.is_webcam_mode(),
            "webcam_id": coordinator.webcam_id if coordinator.is_webcam_mode() else None
        }
    except Exception as e:
        logger.error(f"Error getting webcam status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Cross-Camera Tracker Endpoints ====================

@app.get("/api/v1/cross-camera/tracks", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def get_cross_camera_tracks():
    """Get all active cross-camera tracks"""
    try:
        tracker = get_cross_camera_tracker()
        return tracker.get_tracking_summary()
    except Exception as e:
        logger.error(f"Error getting cross-camera tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cross-camera/targets", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def get_cross_camera_targets():
    """Get all currently targeted persons"""
    try:
        tracker = get_cross_camera_tracker()
        return {"targets": tracker.get_targeted_persons()}
    except Exception as e:
        logger.error(f"Error getting cross-camera targets: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cross-camera/target", response_model=dict, dependencies=[Depends(require_role(ROLE_ADMIN))])
async def create_cross_camera_target(person_id: str = Form(...), camera_id: int = Form(...), reason: str = Form("")):
    """Start targeted tracking for a person across cameras"""
    try:
        tracker = get_cross_camera_tracker()
        global_track_id = tracker.set_target(person_id, camera_id, reason)
        return {"global_track_id": global_track_id, "status": "started"}
    except Exception as e:
        logger.error(f"Error creating cross-camera target: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/v1/cross-camera/target/{person_id}", dependencies=[Depends(require_role(ROLE_ADMIN))])
async def delete_cross_camera_target(person_id: str):
    """Stop targeted tracking for a person"""
    try:
        tracker = get_cross_camera_tracker()
        tracker.stop_target(person_id)
        return {"status": "stopped"}
    except Exception as e:
        logger.error(f"Error deleting cross-camera target: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cross-camera/path/{person_id}", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def get_cross_camera_path(person_id: str):
    """Get tracking path for a specific person"""
    try:
        tracker = get_cross_camera_tracker()
        path = tracker.get_tracking_path(person_id)
        return {"person_id": person_id, "path": path}
    except Exception as e:
        logger.error(f"Error getting cross-camera path: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cross-camera/predict/{person_id}", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def predict_person_trajectory(person_id: str, horizon_seconds: int = Query(30)):
    """Predict future trajectory for a tracked person"""
    try:
        tracker = get_cross_camera_tracker()
        prediction = tracker.get_trajectory_prediction(person_id, horizon_seconds)
        if prediction:
            return prediction
        return {"person_id": person_id, "prediction": None, "message": "Insufficient data for prediction"}
    except Exception as e:
        logger.error(f"Error predicting trajectory: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cross-camera/graph", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def get_camera_graph():
    """Get camera adjacency graph"""
    try:
        tracker = get_cross_camera_tracker()
        return {"camera_graph": tracker.camera_graph}
    except Exception as e:
        logger.error(f"Error getting camera graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cross-camera/graph", response_model=dict, dependencies=[Depends(require_role(ROLE_ADMIN))])
async def set_camera_graph(graph: dict):
    """Set camera adjacency graph"""
    try:
        tracker = get_cross_camera_tracker()
        tracker.set_camera_graph(graph)
        return {"status": "updated", "camera_count": len(graph)}
    except Exception as e:
        logger.error(f"Error setting camera graph: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cross-camera/clear-old", response_model=dict, dependencies=[Depends(require_role(ROLE_ADMIN))])
async def clear_old_tracks(max_age_hours: int = Query(24)):
    """Clear tracks older than specified hours"""
    try:
        tracker = get_cross_camera_tracker()
        count = tracker.clear_old_tracks(max_age_hours)
        return {"status": "cleared", "removed_tracks": count}
    except Exception as e:
        logger.error(f"Error clearing old tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/stats/learning", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_learning_stats():
    """Get learning statistics including behavior profiles and track analysis"""
    try:
        tracker = get_cross_camera_tracker()
        anomaly = get_anomaly_detector()
        
        # Get cross-camera tracking stats
        tracker_stats = tracker.get_tracker_statistics()
        
        stats = {
            "total_behavior_profiles": tracker_stats.get('total_tracked', 0),
            "total_emotion_baselines": len(get_face_recognition().get_known_faces_list()),
            "learning_buffer_size": tracker_stats.get('active_tracks', 0),
            "sklearn_available": SKLEARN_AVAILABLE,
            "cross_camera_stats": tracker_stats,
            "anomaly_stats": {
                "enabled": anomaly.enabled,
                "pattern_history_size": len(anomaly.pattern_history) if hasattr(anomaly, 'pattern_history') else 0
            },
            "features": {
                "behavior_pattern_learning": True,
                "emotion_recognition": True,
                "trajectory_prediction": True,
                "cross_camera_tracking": True,
                "clustering": True
            }
        }
        
        return stats
    except Exception as e:
        logger.error(f"Error getting learning stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Video Testing Endpoints ====================

@app.post("/api/v1/video/process", response_model=dict, dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def process_test_video(
    video_path: str = Form(...),
    camera_ids: str = Form(...),
    duration_seconds: int = Form(60)
):
    """Process a video file for cross-camera tracking testing"""
    try:
        tracker = get_cross_camera_tracker()
        
        # Parse camera IDs from comma-separated string
        camera_id_list = [int(cid.strip()) for cid in camera_ids.split(',') if cid.strip().isdigit()]
        
        if not camera_id_list:
            raise HTTPException(status_code=400, detail="Invalid camera IDs provided")
        
        result = tracker.process_video_file(video_path, camera_id_list, duration_seconds)
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing test video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/clusters", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_trajectory_clusters():
    """Get trajectory clustering analysis for tracked persons"""
    try:
        tracker = get_cross_camera_tracker()
        clusters = tracker.get_clusters()
        return {"clusters": clusters, "count": len(clusters)}
    except Exception as e:
        logger.error(f"Error getting clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Observability ====================

@app.get("/metrics", include_in_schema=False)
async def prometheus_metrics():
    """
    Prometheus text exposition.

    Unauthenticated by design so a scraper needs no credentials, matching the
    convention for /health. It exposes operational counters only - no frames,
    no identities, no event contents. Restrict at the network layer if needed.
    """
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(
        get_metrics_registry().collect(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )


# ==================== System Endpoints ====================

@app.get("/api/v1/health", response_model=dict)
async def health_check():
    """System health check with enhanced subsystems"""
    try:
        camera_manager = get_camera_manager()
        inference_engine = get_inference_engine()
        mqtt_publisher = get_mqtt_publisher()
        face_recognition = get_face_recognition()
        image_enhancement = get_image_enhancement()
        lpr = get_license_plate_recognition()
        anomaly = get_anomaly_detector()
        pose = get_pose_estimator()
        tracker = get_deep_tracker()
        coordinator = get_processing_coordinator()
        
        cameras = camera_manager.get_all_cameras()
        online_cameras = [c for c in cameras if c['status'] == 'online']
        speed_stats = coordinator.get_speed_stats()
        
        health = {
            "status": "healthy",
            "version": "2.1.0",
            "subsystems": {
                "database": "ok",
                # Security posture is reported so a misconfigured deployment is
                # visible from monitoring, not just from the startup logs.
                "security": {
                    "auth_enabled": auth_enabled(),
                    "ephemeral_jwt_secret": EPHEMERAL_SECRET_IN_USE,
                    "cors_origins": _cors_origins,
                    "cors_origin_regex": _cors_origin_regex,
                },
                "retention": get_retention_scheduler().status(),
                "mqtt": "ok" if mqtt_publisher.is_connected() else "disconnected",
                "cameras": {
                    "total": len(cameras),
                    "online": len(online_cameras),
                    "offline": len(cameras) - len(online_cameras)
                },
                "inference": {
                    "model_loaded": inference_engine.is_model_loaded(),
                    "avg_inference_time_ms": round(inference_engine.get_avg_inference_time(), 2)
                },
                "image_enhancement": {
                    "enabled": True
                },
                "face_recognition": {
                    "enabled": face_recognition.enabled,
                    "initialized": face_recognition.is_initialized(),
                    "known_faces": len(face_recognition.get_known_faces_list())
                },
                "speed_analysis": {
                    "tracked_objects": speed_stats.get('total_tracked', 0)
                },
                "license_plate_recognition": {
                    "enabled": lpr.enabled,
                    "initialized": lpr._initialized
                },
                "anomaly_detection": {
                    "enabled": anomaly.enabled
                },
                "pose_estimation": {
                    "enabled": pose.enabled,
                    "initialized": pose._initialized
                },
                "deep_tracking": {
                    "enabled": tracker.enabled
                }
            },
            "uptime_seconds": round(time.time() - startup_time, 2)
        }
        
        return health
    except Exception as e:
        logger.error(f"Error in health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.get("/api/v1/metrics", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_metrics():
    """Get system metrics"""
    try:
        camera_manager = get_camera_manager()
        inference_engine = get_inference_engine()
        coordinator = get_processing_coordinator()
        
        cameras = camera_manager.get_all_cameras()
        processing_status = coordinator.get_processing_status()
        
        camera_metrics = []
        for camera in cameras:
            camera_id = camera['id']
            status = processing_status.get(camera_id, {})
            
            camera_metrics.append({
                "id": camera_id,
                "name": camera['name'],
                "status": camera['status'],
                "fps": round(camera.get('fps', 0), 2),
                "queue_depth": status.get('queue_depth', 0),
                "inference_time_ms": round(inference_engine.get_avg_inference_time(), 2),
                "analysis": status.get('latest_analysis', {})
            })
        
        # System metrics
        system_metrics = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "memory_mb": round(psutil.Process().memory_info().rss / 1024 / 1024, 2),
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        return {
            "cameras": camera_metrics,
            "system": system_metrics
        }
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "Argus API",
        "description": "The Watchful Guardian - AI Video Analytics Platform",
        "version": "2.1.0",
        "status": "running",
        "features": [
            "Object Detection (YOLOv8)",
            "License Plate Recognition (PaddleOCR)",
            "Image Enhancement (CLAHE, Denoise, Sharpen, Night Vision, Deblur, HDR)",
            "Face Recognition (OpenCV/InsightFace)",
            "Pose Estimation (MediaPipe)",
            "Speed Analysis (m/s, km/h, direction)",
            "Height Analysis (meters, categories)",
            "Anomaly Detection (behavior, motion, loitering)",
            "Deep Object Tracking (ByteTrack/DeepSORT/BoT-SORT)",
            "Cross-Camera Tracking (trajectory prediction, clustering)",
            "Zone-based Rules (Intrusion, Loitering, Speed Violation)",
            "MQTT Integration",
            "Qdrant Vector Database (optional)",
            "Kafka Event Streaming (optional)"
        ]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
