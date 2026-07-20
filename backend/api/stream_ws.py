"""
WebSocket endpoint for real-time video streaming with AI overlays
Provides smooth video feed with bounding box overlays to frontend
"""
import json
import base64
import cv2
import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from typing import Optional

router = APIRouter()


@router.websocket("/ws/stream/{camera_id}")
async def websocket_stream(websocket: WebSocket, camera_id: int):
    """
    WebSocket endpoint for real-time video streaming with AI overlays.
    
    Sends binary frame data with overlay JSON metadata.
    Protocol: 
    - Binary frame: JPEG bytes
    - JSON metadata: {bbox: [...], tracks: [...], timestamp: ...}
    """
    await websocket.accept()
    
    try:
        from backend.services.processing_coordinator import get_processing_coordinator
        
        while True:
            try:
                # Get current frame with detection results
                coordinator = get_processing_coordinator()
                frame_data = coordinator.get_latest_frame(camera_id)
                
                if frame_data:
                    frame, detections = frame_data
                    
                    # Encode frame as JPEG
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMMEDIATE_QUALITY, 85])
                    frame_bytes = buffer.tobytes()
                    
                    # Send as binary message
                    await websocket.send_bytes(frame_bytes)
                    
                    # Send detection metadata as JSON
                    detection_data = {
                        "camera_id": camera_id,
                        "detections": [
                            {
                                "track_id": d.track_id,
                                "class": d.class_name,
                                "confidence": d.confidence,
                                "bbox": {
                                    "x1": d.bbox[0],
                                    "y1": d.bbox[1],
                                    "x2": d.bbox[2],
                                    "y2": d.bbox[3]
                                }
                            }
                            for d in detections
                        ],
                        "timestamp": frame_data.get("timestamp")
                    }
                    await websocket.send_json(detection_data)
                
                await asyncio.sleep(1/30)  # 30 FPS stream
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                await asyncio.sleep(0.1)
                
    except Exception as e:
        await websocket.close()


@router.get("/stream/{camera_id}")
async def mjpeg_stream(camera_id: int):
    """
    MJPEG streaming endpoint as fallback for older browsers.
    """
    async def generate():
        try:
            from backend.services.processing_coordinator import get_processing_coordinator
            
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
    
    return StreamingResponse(generate(), media_type="multipart/x-mixed-replace;boundary=frame")


import asyncio  # Add at top after imports