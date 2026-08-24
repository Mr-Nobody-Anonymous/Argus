"""
WebSocket endpoint for real-time video streaming with AI overlays
Provides smooth video feed with bounding box overlays to frontend
"""
import asyncio
import json
import base64
import logging
import time
from datetime import datetime, timezone
import cv2
import numpy as np
from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import Optional

from backend.api.auth import authenticate_websocket, require_role, ROLE_VIEWER

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    """Current UTC time as an ISO 8601 string with a trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


router = APIRouter()


@router.websocket("/ws/stream/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: int):
    """
    WebSocket endpoint for real-time video streaming with AI overlays.

    Sends binary frame data with overlay JSON metadata.
    Protocol:
    - Binary frame: JPEG bytes
    - JSON metadata: {bbox: [...], tracks: [...], timestamp: ...}

    Authentication: a valid access token is required BEFORE the socket is
    accepted, so an unauthorised client never receives a single video frame.
    Browsers cannot set headers on a WebSocket handshake, so pass the token as
    a query parameter:

        ws://host:8000/api/ws/stream/2?token=<access_token>
    """
    user = await authenticate_websocket(websocket, required=ROLE_VIEWER)
    if user is None:
        # authenticate_websocket already closed the socket with code 1008.
        return

    await websocket.accept()
    logger.info(
        f"WebSocket stream opened for camera {camera_id} by "
        f"{user.username} (role={user.role})"
    )

    attention_tracker = None
    try:
        from backend.services.core_engine.processing_coordinator import get_processing_coordinator
        from backend.services.management.user_attention_tracker import get_user_attention_tracker

        # Boost processing priority for cameras a user is actively watching.
        try:
            attention_tracker = get_user_attention_tracker()
            attention_tracker.register_active_stream(camera_id)
        except Exception as exc:  # noqa: BLE001 - attention tracking is best-effort
            logger.warning(f"Could not register active stream for camera {camera_id}: {exc}")
            attention_tracker = None

        loop = asyncio.get_running_loop()

        while True:
            try:
                # Get current frame with detection results.
                # `get_latest_frame` performs a blocking queue read plus a lock
                # acquisition, so run it in the default executor to avoid
                # stalling the event loop (which throttles every other
                # connection sharing this worker).
                coordinator = get_processing_coordinator()
                frame_data = await loop.run_in_executor(
                    None, coordinator.get_latest_frame, camera_id
                )

                if frame_data:
                    frame, detections = frame_data

                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    frame_bytes = buffer.tobytes()

                    # Send as binary message
                    await websocket.send_bytes(frame_bytes)

                    # Send detection metadata as JSON (handle both dict and object detections)
                    detection_list = []
                    for d in detections:
                        if hasattr(d, 'track_id'):
                            detection_list.append({
                                "track_id": d.track_id,
                                "class": d.class_name,
                                "confidence": d.confidence,
                                "bbox": {
                                    "x1": d.bbox[0],
                                    "y1": d.bbox[1],
                                    "x2": d.bbox[2],
                                    "y2": d.bbox[3]
                                }
                            })
                        elif isinstance(d, dict):
                            bbox = d.get('bbox', [0, 0, 0, 0])
                            detection_list.append({
                                "track_id": d.get('track_id', 0),
                                "class": d.get('class_name', 'unknown'),
                                "confidence": d.get('confidence', 0.0),
                                "bbox": {
                                    "x1": bbox[0] if isinstance(bbox, (list, tuple)) else 0,
                                    "y1": bbox[1] if isinstance(bbox, (list, tuple)) else 0,
                                    "x2": bbox[2] if isinstance(bbox, (list, tuple)) else 0,
                                    "y2": bbox[3] if isinstance(bbox, (list, tuple)) else 0
                                }
                            })

                    await websocket.send_json({
                        "camera_id": camera_id,
                        "detections": detection_list,
                        # ISO 8601 UTC, matching the documented protocol and the
                        # REST API. A stringified Unix float forced every client
                        # to special-case this one endpoint.
                        "timestamp": _utc_now_iso(),
                        "timestamp_unix": time.time(),
                    })
                else:
                    await websocket.send_json({
                        "camera_id": camera_id,
                        "detections": [],
                        "timestamp": _utc_now_iso(),
                        "timestamp_unix": time.time(),
                    })

                await asyncio.sleep(1/30)  # 30 FPS stream

            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.warning(f"Stream error for camera {camera_id}: {e}")
                try:
                    await websocket.send_json(
                        {"error": str(e), "camera_id": camera_id, "detections": []}
                    )
                except Exception:
                    # Peer is gone - stop the loop instead of spinning forever.
                    break
                await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"WebSocket stream failed for camera {camera_id}: {e}")
    finally:
        # Always release the attention slot, otherwise the camera keeps its
        # priority boost forever after the viewer disconnects.
        if attention_tracker is not None:
            try:
                attention_tracker.unregister_active_stream(camera_id)
            except Exception:
                pass
        try:
            await websocket.close()
        except Exception:
            pass


@router.get("/stream/{camera_id}", dependencies=[Depends(require_role(ROLE_VIEWER))])
async def mjpeg_stream(camera_id: int):
    """
    MJPEG streaming endpoint as fallback for older browsers.
    """
    async def generate():
        try:
            from backend.services.core_engine.processing_coordinator import get_processing_coordinator

            crlf = b'\r\n'
            while True:
                coordinator = get_processing_coordinator()
                frame_data = coordinator.get_latest_frame(camera_id)

                if frame_data:
                    frame = frame_data[0]
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()

                    yield (b'--frame' + crlf
                           + b'Content-Type: image/jpeg' + crlf + crlf
                           + frame_bytes + crlf)

                await asyncio.sleep(1/30)

        except Exception:
            pass

    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")
