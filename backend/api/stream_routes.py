"""
Stream routes for snapshot retrieval and MJPEG fallback.
WebSocket streaming is consolidated in stream_ws.py to avoid endpoint conflicts.
"""
import asyncio
import cv2
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

router = APIRouter()

# Path to snapshots directory
SNAPSHOT_DIR = Path("data/snapshots")


@router.get("/snapshots/{camera_id}/{filename}")
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


@router.get("/mjpeg/stream/{camera_id}")
async def mjpeg_stream(camera_id: int):
    """
    MJPEG streaming endpoint as a fallback for environments where
    WebSocket is unavailable.
    """
    async def generate():
        try:
            from services.core_engine.processing_coordinator import get_processing_coordinator

            while True:
                coordinator = get_processing_coordinator()
                frame_data = coordinator.get_latest_frame(camera_id)

                if frame_data:
                    frame = frame_data[0]
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()

                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                await asyncio.sleep(1/30)

        except Exception:
            pass

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


def register_routes(app):
    """Register all stream routes with FastAPI app."""
    app.include_router(router)
