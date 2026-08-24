"""
Main FastAPI application for Argus
"""
import logging
import os
import sys
import time
import json
from pathlib import Path
from fastapi import (
    FastAPI, HTTPException, Query, UploadFile, File, Form, Depends, Request,
    Body, status
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
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


# NOTE: this literal route MUST be registered before /events/{event_id}.
# FastAPI matches in declaration order, so with the parameterised route first
# a request for /events/lifecycle is parsed as event_id="lifecycle" and fails
# with a 422 instead of ever reaching this handler.
@app.get("/api/v1/events/lifecycle", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_event_lifecycle():
    """The permitted event state machine, so clients need not hardcode it."""
    from backend.services.management.event_store import EventStore

    return {
        "statuses": sorted(EventStore.ALLOWED_TRANSITIONS),
        "transitions": {
            k: sorted(v) for k, v in EventStore.ALLOWED_TRANSITIONS.items()
        },
        "terminal": sorted(
            k for k, v in EventStore.ALLOWED_TRANSITIONS.items() if not v
        ),
    }


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


@app.patch("/api/v1/events/{event_id}/status", response_model=dict,
           dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def update_event_status(
    event_id: int,
    payload: dict = Body(...),
    user: AuthUser = Depends(get_current_user),
):
    """Move an event through its lifecycle.

    detected -> open -> acknowledged -> resolved, or dismissed as
    false_positive from any live state. The transition is enforced by the
    event store; an illegal one is a 400 rather than a silent write, and the
    acting user is recorded so a review can be proven after the fact.
    """
    status_value = (payload or {}).get("status")
    if not status_value:
        raise HTTPException(status_code=400, detail="status is required")
    try:
        event_store = get_event_store()
        event = event_store.update_event_status(
            event_id, status_value, actor=getattr(user, "username", None)
        )
        if event is None:
            raise HTTPException(status_code=404, detail="Event not found")
        # The audit middleware already records every mutating request with the
        # principal, path and outcome, so no explicit audit call is needed here.
        return {"event": event}
    except ValueError as e:
        # Unknown status or illegal transition: the caller's fault, not ours.
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating event status: {e}")
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
        
        # Report the learning engine's OWN counters. This endpoint previously
        # relabelled cross-camera tracker numbers as "behavior profiles" and
        # "learning buffer size" - the adaptive learning engine was never
        # consulted, and in fact was never called by anything, so the figures
        # described a subsystem that had learned nothing. Borrowed numbers
        # under someone else's name are indistinguishable from working.
        from backend.services.analytics.adaptive_learning import (
            get_adaptive_learning_engine,
        )
        learner = get_adaptive_learning_engine()
        learning_stats = learner.get_learning_stats()

        stats = {
            **learning_stats,
            "cross_camera_stats": tracker_stats,
            "anomaly_stats": {
                "enabled": anomaly.enabled,
                "pattern_history_size": len(anomaly.pattern_history) if hasattr(anomaly, 'pattern_history') else 0
            },
            # Availability is probed, not asserted. Several of these depend on
            # optional packages that are absent on a CPU-only host.
            "features": {
                "behavior_pattern_learning": True,
                "emotion_recognition": get_face_recognition().enabled,
                "trajectory_prediction": bool(tracker_stats),
                "cross_camera_tracking": True,
                "clustering": SKLEARN_AVAILABLE,
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


# ==================== Perception ====================
# The accumulated world model: what Argus knows about each tracked entity over
# time, and what it can currently run on this hardware.

@app.get("/api/v1/perception/tracks", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_perception_tracks():
    """Every active track with its attributes, relationships and inferences."""
    try:
        from backend.services.perception import get_pipeline
        pipeline = get_pipeline()
        return {"tracks": pipeline.active_tracks(), "stats": pipeline.stats()}
    except Exception as e:
        logger.error(f"Error reading perception tracks: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/perception/tracks/{track_id}", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_perception_track(track_id: int):
    """Everything reliably known about one entity.

    Measurement and inference are kept in separate keys so a reader can always
    tell evidence from conclusion.
    """
    try:
        from backend.services.perception import get_pipeline
        summary = get_pipeline().describe_track(track_id)
        if summary is None:
            raise HTTPException(status_code=404, detail="Track not found")
        return summary
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error reading track {track_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/perception/capabilities", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_perception_capabilities():
    """What Argus can run here, what it costs, and why anything is unavailable.

    Availability is probed by importing each backend, never assumed from
    config: a package can be declared and still fail to load.
    """
    try:
        from backend.services.perception import get_registry
        return get_registry().report()
    except Exception as e:
        logger.error(f"Error reading capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/perception/tracks/{track_id}/explain", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def explain_perception_track(track_id: int):
    """Why Argus believes what it believes about one entity.

    Returns measurements and inferences under separate keys rather than one
    confident narrative. An operator about to act on an alert needs to know
    which parts were observed and which were concluded - a system that blurs
    the two teaches people either to over-trust it or to ignore it.
    """
    try:
        from backend.services.perception import get_pipeline
        explained = get_pipeline().explain(track_id)
        if explained is None:
            raise HTTPException(status_code=404,
                                detail=f"No track {track_id} in the world model")
        return explained
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error explaining track {track_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/perception/changes", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_perception_changes():
    """Per-camera change-detection baselines and their maturity.

    A baseline that has not seen enough samples reports `ready: false` and
    judges nothing, so this endpoint also answers "is anomaly detection
    actually working on this camera yet?".
    """
    try:
        from backend.services.perception import get_pipeline
        return get_pipeline().change.report()
    except Exception as e:
        logger.error(f"Error reading change baselines: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/perception/stats", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_perception_stats():
    """Pipeline throughput and measured per-stage cost.

    Stage costs are measured on this host, not estimated, so the numbers here
    are what scheduling decisions should be based on.
    """
    try:
        from backend.services.perception import get_pipeline
        return get_pipeline().stats()
    except Exception as e:
        logger.error(f"Error reading perception stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memory/recall", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def memory_recall(text: Optional[str] = None,
                        camera_id: Optional[int] = None,
                        when: Optional[str] = None,
                        kind: Optional[str] = None,
                        min_confidence: float = 0.0,
                        limit: int = 100):
    """"What happened near the loading bay yesterday?"

    Free text searches observation summaries and their evidence; `when`
    accepts today / yesterday / last_hour / last_week / 24h. Every result
    carries the evidence behind it, and the response reports how many are
    grounded so an empty-evidence claim cannot pass as a finding.
    """
    try:
        from backend.services.perception import recall
        kinds = [k.strip() for k in kind.split(",")] if kind else None
        return recall(text=text, camera_id=camera_id, when=when, kinds=kinds,
                      min_confidence=min_confidence, limit=min(int(limit), 500))
    except Exception as e:
        logger.error(f"Memory recall failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memory/summary", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def memory_summary(camera_id: Optional[int] = None,
                         when: str = "today"):
    """A plain-language digest of a period - the shift-handover answer."""
    try:
        from backend.services.perception import summarise_period
        return summarise_period(camera_id=camera_id, when=when)
    except Exception as e:
        logger.error(f"Memory summary failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memory/appearances/{camera_id}/{track_id}/similar",
         response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def memory_find_across_cameras(camera_id: int, track_id: int,
                                     limit: int = 10):
    """"Where else has this person been?"

    Ranked by appearance similarity, annotated with whether the journey was
    physically possible in the time available. Results are **candidates for
    review, never identifications**: the descriptor compares clothing colour
    layout, so two people dressed alike match strongly. The caveat travels
    with the payload rather than living only in the docs.
    """
    try:
        from backend.services.perception import find_across_cameras
        result = find_across_cameras(camera_id, track_id,
                                     limit=min(int(limit), 50))
        if result.get("error"):
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cross-camera search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/memory/stats", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def memory_stats():
    """What is stored, which vector backend is live, and its measured limits.

    Reports the SQLite brute-force backend honestly, including a warning once
    the descriptor table grows past the point where a scan exceeds ~100 ms.
    """
    try:
        from backend.services.perception import get_memory
        return get_memory().report()
    except Exception as e:
        logger.error(f"Memory stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/evidence/status", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_evidence_status():
    """Pre-event buffer occupancy and clip counters, per camera."""
    try:
        from backend.services.management.evidence_clips import get_evidence_service

        return get_evidence_service().status()
    except Exception as e:
        logger.error(f"Evidence status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/events/{event_id}/clip",
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_event_clip(event_id: int):
    """Download the pre-event clip for an event, if one was written."""
    try:
        event = get_event_store().get_event(event_id)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        meta = event.get("metadata") or {}
        if isinstance(meta, str):
            try:
                meta = json.loads(meta)
            except (ValueError, TypeError):
                meta = {}
        clip = meta.get("clip") or {}
        if not clip.get("written") or not clip.get("path"):
            raise HTTPException(
                status_code=404,
                detail=(
                    "No clip for this event: "
                    + (clip.get("reason") or "clips not enabled for this rule")
                ),
            )
        path = Path(clip["path"])
        if not path.exists():
            raise HTTPException(
                status_code=410,
                detail="Clip has been removed by the retention policy",
            )
        return FileResponse(str(path), media_type="video/mp4", filename=path.name)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Clip fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/notifications/status", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_notification_status():
    """Which alert channels can actually deliver, and what policy is filtering.

    Events are always recorded. This reports whether any of them are being
    *sent* anywhere - a distinction that was invisible while MQTT was enabled
    in config and no code ever called the publisher.
    """
    try:
        from backend.services.management.notifications import (
            get_notification_service,
        )

        return get_notification_service().status()
    except Exception as e:
        logger.error(f"Notification status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/notifications/test", response_model=dict,
          dependencies=[Depends(require_role(ROLE_ADMIN))])
async def send_test_notification():
    """Send a synthetic alert through every configured channel.

    Delivery is reported per transport with the real outcome, so a
    misconfigured webhook is discovered here rather than during an incident.
    Bypasses policy deliberately: this tests transport, not filtering.
    """
    try:
        from backend.services.management.notifications import (
            get_notification_service,
        )

        service = get_notification_service()
        probe = {
            "id": 0,
            "camera_id": 0,
            "rule_type": "test_notification",
            "object_type": None,
            "confidence": 1.0,
            "priority": "low",
            "status": "detected",
            "timestamp": datetime.now(),
            "metadata": {
                "source": "manual_test",
                "note": "Synthetic event from /notifications/test",
            },
        }
        results = service._deliver(probe)
        delivered = [r.to_dict() for r in results]
        return {
            "sent": bool(delivered),
            "results": delivered,
            "all_delivered": bool(delivered) and all(
                r["delivered"] for r in delivered
            ),
            "note": (
                "No channels are configured." if not delivered else
                "Policy was bypassed for this test; real events are filtered."
            ),
        }
    except Exception as e:
        logger.error(f"Test notification failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rules/status", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_rules_status():
    """Which rules are configured, implemented, and actually able to fire.

    config.yaml can declare a rule `enabled: true` while no code evaluates it -
    which was true of speed_violation, fall_detection and abandoned_object for
    the life of this project. This endpoint makes that difference visible
    instead of leaving an operator to assume coverage they do not have.
    """
    try:
        from backend.services.management.rules_engine import get_rules_engine

        engine = get_rules_engine()
        status = engine.rule_status()
        return {
            "rules": status,
            "summary": {
                "configured": len(status),
                "implemented": sum(1 for v in status.values() if v["implemented"]),
                "can_fire_now": sum(1 for v in status.values() if v["can_fire"]),
                "blocked": [k for k, v in status.items() if v["blockers"]],
            },
        }
    except Exception as e:
        logger.error(f"Rule status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/rules/calibration", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_calibration_status():
    """Per-camera ground-plane calibration, and what it gates."""
    try:
        from backend.services.management.calibration import get_calibration_registry
        from backend.database.db import get_db

        registry = get_calibration_registry()
        db = get_db()
        cameras = db.execute("SELECT id, name FROM cameras")
        out = []
        for row in cameras:
            cid = row["id"] if isinstance(row, dict) else row[0]
            name = row["name"] if isinstance(row, dict) else row[1]
            cal = registry.get(cid)
            out.append({
                "camera_id": cid,
                "name": name,
                "calibrated": cal.is_calibrated,
                "meters_per_pixel": cal.meters_per_pixel,
                "source": cal.source,
                "reason": cal.reason(),
            })
        return {"cameras": out, **registry.report()}
    except Exception as e:
        logger.error(f"Calibration status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/observations/promotion", response_model=dict, dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_promotion_stats():
    """How many perception observations became operator-visible events."""
    try:
        from backend.services.management.observation_events import (
            KIND_POLICY,
            get_observation_bridge,
        )

        bridge = get_observation_bridge()
        return {
            "stats": bridge.stats(),
            "policy": {
                kind: {
                    "promoted_to_events": p.promote,
                    "priority": p.priority,
                    "cooldown_s": p.cooldown_s,
                    "description": p.description,
                }
                for kind, p in sorted(KIND_POLICY.items())
            },
            "note": (
                "Observations that are not promoted remain fully queryable "
                "through /api/v1/memory/recall - suppressing an alert never "
                "discards the record."
            ),
        }
    except Exception as e:
        logger.error(f"Promotion stats failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/api")
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


# ==================== CityOS Intersection Intelligence ====================
# Geometry-only traffic layer: digital twin, road-user classification,
# trajectories, wrong-way / near-miss / VRU safety, flow analytics and
# signal-optimiser recommendations. No biometrics enter this layer.

@app.get("/api/v1/cityos/status", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_status():
    """CityOS overview: intersections, object counts, privacy posture."""
    try:
        from backend.services.cityos import get_cityos_engine
        return get_cityos_engine().status()
    except Exception as e:
        logger.error(f"CityOS status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/twin", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_twin_all():
    """Digital-twin snapshots for every known intersection."""
    try:
        from backend.services.cityos import get_cityos_engine
        return get_cityos_engine().twin()
    except Exception as e:
        logger.error(f"CityOS twin failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/twin/{camera_id}", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_twin(camera_id: int):
    """Digital-twin snapshot for the intersection a camera covers."""
    try:
        from backend.services.cityos import get_cityos_engine
        twin = get_cityos_engine().twin(camera_id)
        if twin.get("error"):
            raise HTTPException(status_code=404, detail=twin["error"])
        return twin
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS twin failed for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/alerts", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_alerts(
    kind: Optional[str] = Query(None),
    limit: int = Query(50, le=200),
):
    """Merged real-time safety alert feed (wrong-way, near-miss, VRU)."""
    try:
        from backend.services.cityos import get_cityos_engine
        alerts = get_cityos_engine().alerts(limit=limit, kind=kind)
        return {"alerts": alerts, "count": len(alerts)}
    except Exception as e:
        logger.error(f"CityOS alerts failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/flow/{camera_id}", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_flow(camera_id: int, minutes: int = Query(30, le=120)):
    """Traffic-flow analytics for one camera's intersection."""
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        with engine._lock:
            iid = engine.camera_to_intersection.get(camera_id)
        if iid is None:
            raise HTTPException(status_code=404,
                                detail=f"no intersection bound to camera {camera_id}")
        inter = engine.get_intersection(iid)
        return {
            "intersection_id": iid,
            "volume_series": inter.flow.volume_series(minutes=minutes),
            "turning_matrix": inter.flow.turning_matrix(),
            "speed_summary": inter.flow.speed_summary(),
            "stats": inter.flow.stats(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS flow failed for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/signal/{camera_id}", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_signal_status(camera_id: int):
    """Signal phase state + adaptive recommendation for an intersection."""
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        with engine._lock:
            iid = engine.camera_to_intersection.get(camera_id)
        if iid is None:
            raise HTTPException(status_code=404,
                                detail=f"no intersection bound to camera {camera_id}")
        inter = engine.get_intersection(iid)
        users = inter.perception.active_users()
        demand = inter.flow.demand_by_approach(users)
        return {
            "intersection_id": iid,
            "status": inter.signal.status(),
            "recommendation": inter.signal.recommend(demand),
            "demand_by_approach": demand,
            "recent_commands": inter.signal.recent_commands(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS signal status failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SignalModeRequest(BaseModel):
    mode: str


@app.post("/api/v1/cityos/signal/{camera_id}/mode", response_model=dict,
          dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def cityos_set_signal_mode(camera_id: int, payload: SignalModeRequest):
    """Set signal mode: fixed | adaptive | manual."""
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        with engine._lock:
            iid = engine.camera_to_intersection.get(camera_id)
        if iid is None:
            raise HTTPException(status_code=404,
                                detail=f"no intersection bound to camera {camera_id}")
        inter = engine.get_intersection(iid)
        try:
            status = inter.signal.set_mode(payload.mode)
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        return {"intersection_id": iid, "signal": status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS set signal mode failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class SignalPhaseRequest(BaseModel):
    phase: str


@app.post("/api/v1/cityos/signal/{camera_id}/phase", response_model=dict,
          dependencies=[Depends(require_role(ROLE_OPERATOR))])
async def cityos_force_signal_phase(camera_id: int, payload: SignalPhaseRequest):
    """Manual operator override: force NS or EW green (audit-logged)."""
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        with engine._lock:
            iid = engine.camera_to_intersection.get(camera_id)
        if iid is None:
            raise HTTPException(status_code=404,
                                detail=f"no intersection bound to camera {camera_id}")
        inter = engine.get_intersection(iid)
        try:
            status = inter.signal.force_phase(payload.phase.upper())
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        return {"intersection_id": iid, "signal": status}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS force phase failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cityos/bind", response_model=dict,
          dependencies=[Depends(require_role(ROLE_ADMIN))])
async def cityos_bind_camera(
    camera_id: int = Body(...),
    intersection_id: str = Body(...),
):
    """Bind a camera to a named intersection (scales 1 -> N intersections)."""
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        engine.bind_camera(int(camera_id), intersection_id.strip())
        inter = engine.get_intersection(intersection_id.strip())
        return {"bound": True, "intersection": inter.summary()}
    except Exception as e:
        logger.error(f"CityOS bind failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/map/{camera_id}", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_map(camera_id: int):
    """Lane-level intersection geometry + calibration for a camera."""
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        with engine._lock:
            iid = engine.camera_to_intersection.get(camera_id)
        if iid is None:
            raise HTTPException(status_code=404,
                                detail=f"no intersection bound to camera {camera_id}")
        inter = engine.get_intersection(iid)
        return {
            "intersection_id": iid,
            "calibration": inter.calibration.to_dict(),
            "map": inter.map.to_dict(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS map failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/api/v1/cityos/map/{camera_id}", response_model=dict,
         dependencies=[Depends(require_role(ROLE_ADMIN))])
async def cityos_set_map(camera_id: int, payload: dict = Body(...)):
    """Configure calibration and/or lane geometry for an intersection.

    Body keys (all optional): calibration {view_width_m, view_height_m,
    yaw_deg, centre}, stop_lines {approach: coord}, bike_lanes bool.
    """
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        cal = payload.get("calibration")
        if cal:
            engine.set_calibration(camera_id, cal)
        map_cfg = {k: v for k, v in payload.items()
                   if k in ("stop_lines", "bike_lanes", "crosswalks")}
        if map_cfg or cal:
            # Rebuild the map so stop-line changes take effect.
            current = {}
            with engine._lock:
                iid = engine.camera_to_intersection.get(camera_id)
            if iid:
                inter = engine.get_intersection(iid)
                current["crosswalks"] = inter.map.crosswalks
                current.update(map_cfg)
                engine.set_map(camera_id, current)
        return await cityos_map(camera_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS set map failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/queue/{camera_id}", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_queue(camera_id: int):
    """Queue depth/length/growth per approach."""
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        with engine._lock:
            iid = engine.camera_to_intersection.get(camera_id)
        if iid is None:
            raise HTTPException(status_code=404,
                                detail=f"no intersection bound to camera {camera_id}")
        inter = engine.get_intersection(iid)
        return {"intersection_id": iid,
                "queues": inter.flow.queue_status()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS queue failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/pedestrian/{camera_id}", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_pedestrian(camera_id: int):
    """Pedestrian signal phases and waiting-pedestrian estimate."""
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        with engine._lock:
            iid = engine.camera_to_intersection.get(camera_id)
        if iid is None:
            raise HTTPException(status_code=404,
                                detail=f"no intersection bound to camera {camera_id}")
        inter = engine.get_intersection(iid)
        users = inter.perception.active_users()
        waiting = [u for u in users
                   if u["category"] == "pedestrian"
                   and not u.get("in_crosswalk")
                   and abs(u.get("distance_to_stop_line_m") or 999) < 15.0]
        return {
            "intersection_id": iid,
            "ped_signal": inter.signal.ped_states(),
            "waiting_pedestrians": len(waiting),
            "waiting_objects": [
                {"track_id": u["track_id"],
                 "position": u["position"],
                 "approach": u.get("approach")} for u in waiting[:20]
            ],
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS pedestrian failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/corridors", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_corridors():
    """Corridor links, pending handoffs and travel-time statistics."""
    try:
        from backend.services.cityos import get_cityos_engine
        return get_cityos_engine().corridor.summary()
    except Exception as e:
        logger.error(f"CityOS corridors failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/cityos/corridors", response_model=dict,
          dependencies=[Depends(require_role(ROLE_ADMIN))])
async def cityos_add_corridor(payload: dict = Body(...)):
    """Define a corridor link between two intersections.

    Body: link_id, from_intersection, to_intersection, exit_approach,
    entry_approach, min_travel_s, max_travel_s.
    """
    required = ("link_id", "from_intersection", "to_intersection",
                "exit_approach", "entry_approach")
    missing = [k for k in required if not payload.get(k)]
    if missing:
        raise HTTPException(status_code=400,
                            detail=f"missing fields: {missing}")
    try:
        from backend.services.cityos import get_cityos_engine
        from backend.services.cityos.corridor import CorridorLink
        link = CorridorLink(
            link_id=str(payload["link_id"]),
            from_iid=str(payload["from_intersection"]),
            to_iid=str(payload["to_intersection"]),
            exit_approach=str(payload["exit_approach"]),
            entry_approach=str(payload["entry_approach"]),
            min_travel_s=float(payload.get("min_travel_s", 20)),
            max_travel_s=float(payload.get("max_travel_s", 300)),
        )
        get_cityos_engine().add_corridor_link(link)
        return {"added": True, "link": link.to_dict()}
    except Exception as e:
        logger.error(f"CityOS add corridor failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/replay/{camera_id}", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_replay(camera_id: int, seconds_ago: float = Query(30, ge=0, le=1800)):
    """Deterministic replay: the recorded twin snapshot nearest `seconds_ago`."""
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        with engine._lock:
            iid = engine.camera_to_intersection.get(camera_id)
        if iid is None:
            raise HTTPException(status_code=404,
                                detail=f"no intersection bound to camera {camera_id}")
        snap = engine.get_intersection(iid).replay_at(seconds_ago)
        if snap is None:
            return {"available": False,
                    "note": "no snapshots recorded yet; they accumulate every 5 s"}
        return {"available": True, **snap}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS replay failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/cityos/health/{camera_id}", response_model=dict,
         dependencies=[Depends(require_role(ROLE_VIEWER))])
async def cityos_sensor_health(camera_id: int):
    """Deep sensor-health diagnostics for one intersection's sensor."""
    try:
        from backend.services.cityos import get_cityos_engine
        engine = get_cityos_engine()
        with engine._lock:
            iid = engine.camera_to_intersection.get(camera_id)
        if iid is None:
            raise HTTPException(status_code=404,
                                detail=f"no intersection bound to camera {camera_id}")
        return {"intersection_id": iid,
                "sensor_health": engine.get_intersection(iid).sensor_health()}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"CityOS sensor health failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ==================== Single-port web UI ====================
# Serving the built React app from FastAPI means `argus start` exposes ONE url
# (http://localhost:8000) with no Node runtime, no second port and no proxy.
# This is mounted last, after every API router, so it can never shadow an API
# route: StaticFiles only ever sees paths that did not match anything above.
#
# When frontend/dist is absent (a source checkout that has not been built yet)
# the mount is skipped and "/" serves a short message telling the user how to
# build it, instead of a confusing 404.

_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"


def _ui_is_built() -> bool:
    return (_FRONTEND_DIST / "index.html").is_file()


if _ui_is_built():
    from starlette.exceptions import HTTPException as StarletteHTTPException

    class _SpaStaticFiles(StaticFiles):
        """StaticFiles that falls back to index.html for client-side routes.

        The dashboard uses browser-side routing (/events, /analytics, ...).
        Those paths have no file on disk, so a plain StaticFiles mount would
        404 whenever a user refreshes the page or opens a deep link.
        """

        # Paths that must keep returning a real 404 instead of the SPA shell.
        # An unknown /api/... route is a client error and has to stay JSON:
        # serving index.html there would make every typo look like a success
        # and break error handling in API consumers.
        _NEVER_SPA = ("api/", "docs", "redoc", "openapi.json", "metrics", "snapshots/")

        async def get_response(self, path: str, scope):
            # Starlette RAISES HTTPException(404) for a missing file rather
            # than returning a 404 response, so both paths must be handled or
            # deep links silently break on refresh.
            def _is_api(p: str) -> bool:
                p = p.lstrip("/")
                return any(p == n or p.startswith(n) for n in self._NEVER_SPA)

            try:
                response = await super().get_response(path, scope)
            except StarletteHTTPException as exc:
                if exc.status_code != 404 or _is_api(path):
                    raise
                return await super().get_response("index.html", scope)
            if response.status_code == 404 and not _is_api(path):
                return await super().get_response("index.html", scope)
            return response

    app.mount("/", _SpaStaticFiles(directory=str(_FRONTEND_DIST), html=True), name="ui")
    logger.info(f"Serving dashboard from {_FRONTEND_DIST}")
else:
    @app.get("/", include_in_schema=False)
    async def _ui_not_built():
        return {
            "name": "Argus API",
            "status": "running",
            "dashboard": "not built",
            "hint": "Run 'python argus.py start' (it builds the UI automatically), "
                    "or build it manually with: cd frontend && npm install && npm run build",
            "api_docs": "/docs",
            "api_root": "/api",
        }

    logger.info("frontend/dist not found - dashboard not served (API only)")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
