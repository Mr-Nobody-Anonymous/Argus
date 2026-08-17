"""
Stream routes for snapshot retrieval and MJPEG fallback.
WebSocket streaming is consolidated in stream_ws.py to avoid endpoint conflicts.
"""
import asyncio
import logging
import cv2
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter()

# Path to snapshots directory (anchored to the project root, not the CWD)
from backend.config.config import get_config, resolve_path
from backend.api.auth import require_role, ROLE_VIEWER

SNAPSHOT_DIR = resolve_path(get_config().system.snapshot_dir)


@router.get("/snapshots/{camera_id}/{filename}", dependencies=[Depends(require_role(ROLE_VIEWER))])
async def get_snapshot(camera_id: int, filename: str):
    """
    Retrieve a saved snapshot image for a given camera and event.
    Files are stored as: cam{camera_id}_{rule_type}_{timestamp}.jpg
    """
    # Sanitize filename to prevent path traversal
    safe_filename = Path(filename).name
    filepath = SNAPSHOT_DIR / safe_filename

    if not filepath.exists():
        raise HTTPException(status_code=404, detail="Snapshot not found")

    return FileResponse(str(filepath), media_type="image/jpeg")


@router.get("/mjpeg/stream/{camera_id}", dependencies=[Depends(require_role(ROLE_VIEWER))])
async def mjpeg_stream(camera_id: int):
    """
    MJPEG streaming endpoint as a fallback for environments where
    WebSocket is unavailable.
    """
    async def generate():
        try:
            from backend.services.core_engine.processing_coordinator import get_processing_coordinator

            loop = asyncio.get_running_loop()

            while True:
                coordinator = get_processing_coordinator()
                # Frame retrieval and JPEG encoding are blocking calls - keep
                # them off the event loop so other requests are not stalled.
                frame_data = await loop.run_in_executor(
                    None, coordinator.get_latest_frame, camera_id
                )

                if frame_data:
                    frame = frame_data[0]
                    ok, buffer = await loop.run_in_executor(
                        None, lambda: cv2.imencode('.jpg', frame)
                    )
                    if ok:
                        frame_bytes = buffer.tobytes()
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                await asyncio.sleep(1/30)

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning(f"MJPEG stream ended for camera {camera_id}: {exc}")

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


def register_routes(app):
    """Register all stream routes with FastAPI app."""
    app.include_router(router)
