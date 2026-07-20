"""
WebSocket streaming routes for City OS
Integrates with FastAPI main application
"""
import asyncio
import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

router = APIRouter()

# Track active connections
active_connections: dict[int, list] = {}

@router.websocket("/ws/stream/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: int):
    """
    WebSocket endpoint for real-time video streaming with AI overlays.
    Sends binary JPEG frames + JSON detection metadata.
    """
    await websocket.accept()
    
    if camera_id not in active_connections:
        active_connections[camera_id] = []
    active_connections[camera_id].append(websocket)
    
    try:
        while True:
            # Get frame from processing coordinator
            from backend.services.processing_coordinator import get_processing_coordinator
            
            coordinator = get_processing_coordinator()
            frame_data = coordinator.get_latest_frame(camera_id)
            
            if frame_data:
                frame, detections = frame_data
                
                # Encode as JPEG (fast binary transmission)
                import cv2
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMMEDIATE_QUALITY, 85])
                frame_bytes = buffer.tobytes()
                
                # Send binary frame
                await websocket.send_bytes(frame_bytes)
                
                # Send detection metadata
                await websocket.send_json({
                    "camera_id": camera_id,
                    "detections": [
                        {
                            "track_id": d.track_id if hasattr(d, 'track_id') else i,
                            "class": d.class_name if hasattr(d, 'class_name') else 'unknown',
                            "confidence": float(d.confidence) if hasattr(d, 'confidence') else 0,
                            "bbox": getattr(d, 'bbox', [0, 0, 0, 0])
                        }
                        for i, d in enumerate(detections) if d
                    ],
                    "timestamp": frame_data.get("timestamp")
                })
            else:
                # Send empty frame to keep connection alive
                await websocket.send_json({"camera_id": camera_id, "detections": []})
            
            await asyncio.sleep(1/30)  # 30 FPS stream
            
    except WebSocketDisconnect:
        active_connections[camera_id].remove(websocket)
    except Exception as e:
        if camera_id in active_connections:
            active_connections[camera_id].remove(websocket)
        await websocket.close()

def register_routes(app):
    """Register WebSocket routes with FastAPI app."""
    app.include_router(router)