"""
WebSocket streaming routes for City OS
Integrates with FastAPI main application
"""
import asyncio
import json
import cv2
import os
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from backend.services.processing_coordinator import get_processing_coordinator

router = APIRouter()

# Configuration from environment
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

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
            coordinator = get_processing_coordinator()
            frame_data = coordinator.get_latest_frame(camera_id)
            
            if frame_data:
                frame, detections = frame_data
                
                # Encode as JPEG (fast binary transmission)
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMMEDIATE_QUALITY, JPEG_QUALITY])
                frame_bytes = buffer.tobytes()
                
                # Send binary frame
                await websocket.send_bytes(frame_bytes)
                
                # Process detections efficiently
                detection_list = []
                for i, d in enumerate(detections):
                    if d:
                        detection_list.append({
                            "track_id": getattr(d, 'track_id', i),
                            "class": getattr(d, 'class_name', 'unknown'),
                            "confidence": float(getattr(d, 'confidence', 0)),
                            "bbox": getattr(d, 'bbox', [0, 0, 0, 0])
                        })
                
                # Send detection metadata
                await websocket.send_json({
                    "camera_id": camera_id,
                    "detections": detection_list,
                    "timestamp": frame_data.get("timestamp")
                })
            else:
                # Send empty frame to keep connection alive
                await websocket.send_json({"camera_id": camera_id, "detections": []})
            
            await asyncio.sleep(1/30)  # 30 FPS stream
            
    except WebSocketDisconnect:
        pass
    except Exception as e:
        pass
    finally:
        # Guaranteed cleanup regardless of how loop exits
        if camera_id in active_connections:
            if websocket in active_connections[camera_id]:
                active_connections[camera_id].remove(websocket)
            if not active_connections[camera_id]:
                del active_connections[camera_id]


def register_routes(app):
    """Register WebSocket routes with FastAPI app."""
    app.include_router(router)