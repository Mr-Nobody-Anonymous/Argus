"""
Multi-Stream Ingestion Engine (FFmpeg + MediaMTX + OpenCV + FastAPI)
Production-grade frame processing with adaptive frame skipping
"""
import cv2
import subprocess
import threading
import queue
import time
import logging
import os
from pathlib import Path
from typing import Dict, Optional, List, Tuple
import numpy as np

logger = logging.getLogger(__name__)


class FFmpegCapture:
    """
    FFmpeg subprocess wrapper for zero-copy frame reading.
    More efficient than OpenCV VideoCapture for RTSP streams.
    """
    
    def __init__(self, rtsp_url: str, camera_id: int):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self.pipe = None
        self.frame_queue = queue.Queue(maxsize=5)
        self.stop_event = threading.Event()
        self.thread = None
        self.fps = 0
        
    def start(self):
        """Launch FFmpeg subprocess"""
        # FFmpeg command: decode RTSP to raw frames
        ffmpeg_cmd = [
            'ffmpeg',
            '-fflags', 'nobuffer',
            '-flags', 'low_delay',
            '-strict', 'experimental',
            '-analyzeduration', '100',
            '-probesize', '100',
            '-rtsp_transport', 'tcp',
            '-i', self.rtsp_url,
            '-vf', 'scale=640:360',  # Downscale for faster processing
            '-pix_fmt', 'bgr24',
            '-vcodec', 'rawvideo',
            '-an', '-sn',  # Disable audio/subtitle
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-'
        ]
        
        self.pipe = subprocess.Popen(
            ffmpeg_cmd,
            stdout=subprocess.PIPE,
            bufsize=10**6,
            stderr=subprocess.DEVNULL
        )
        
        self.thread = threading.Thread(target=self._read_frames, daemon=True)
        self.thread.start()
        logger.info(f"Started FFmpeg for camera {self.camera_id}")

    def _read_frames(self):
        """Read raw frames from FFmpeg subprocess"""
        frame_width, frame_height = 640, 360
        frame_size = frame_width * frame_height * 3
        
        while not self.stop_event.is_set():
            try:
                raw_frame = self.pipe.stdout.read(frame_size)
                if len(raw_frame) != frame_size:
                    logger.warning(f"Camera {self.camera_id}: Incomplete frame")
                    continue
                    
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((frame_height, frame_width, 3))
                
                # Put in queue, drop old if full
                try:
                    self.frame_queue.put_nowait((frame, time.time()))
                except queue.Full:
                    self.frame_queue.get_nowait()
                    self.frame_queue.put_nowait((frame, time.time()))
                    
            except Exception as e:
                logger.error(f"FFmpeg read error: {e}")
                break

    def read(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, float]]:
        """Get next frame with timestamp"""
        try:
            return self.frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        """Stop FFmpeg subprocess"""
        self.stop_event.set()
        if self.pipe:
            self.pipe.terminate()
            self.pipe.wait()


class MultiStreamPipeline:
    """
    High-performance multi-camera processing pipeline.
    Integrates with MediaMTX RTSP server and FastAPI backend.
    """
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.captures: Dict[int, FFmpegCapture] = {}
        self.processors: Dict[int, 'FrameProcessor'] = {}
        self.frame_skip: Dict[int, int] = {}  # Adaptive skip rates
        self.inference_times: Dict[int, List[float]] = {}
        
    def add_camera(self, camera_id: int, rtsp_url: str, mediamtx_url: Optional[str] = None):
        """
        Add camera to pipeline.
        If mediamtx_url provided, prefer that over direct RTSP.
        """
        # Use MediaMTX-rebroadcast URL if available
        if mediamtx_url:
            stream_url = mediamtx_url
        else:
            stream_url = rtsp_url
            
        capture = FFmpegCapture(stream_url, camera_id)
        capture.start()
        self.captures[camera_id] = capture
        
        self.frame_skip[camera_id] = 0
        self.inference_times[camera_id] = []
        
    def process_frames(self, camera_id: int, processor):
        """
        Process frames from camera with adaptive frame skipping.
        Skips frames when inference is running behind.
        """
        frame_count = 0
        
        while camera_id in self.captures:
            # Calculate adaptive skip rate
            if self.inference_times.get(camera_id):
                avg_time = sum(self.inference_times[camera_id][-30:]) / len(self.inference_times[camera_id][-30:])
                if avg_time > 100:  # Inference taking >100ms
                    self.frame_skip[camera_id] = min(self.frame_skip[camera_id] + 1, 5)
                elif avg_time < 50:
                    self.frame_skip[camera_id] = max(self.frame_skip[camera_id] - 1, 0)
            
            # Skip frames based on performance
            if frame_count < self.frame_skip[camera_id]:
                self.captures[camera_id].read()
                frame_count += 1
                continue
                
            frame_data = self.captures[camera_id].read()
            if frame_data:
                frame, timestamp = frame_data
                start_time = time.time()
                
                # Process frame
                results = processor.process(frame)
                
                # Record inference time
                inference_ms = (time.time() - start_time) * 1000
                self.inference_times[camera_id].append(inference_ms)
                
            frame_count += 1

    def remove_camera(self, camera_id: int):
        """Remove camera from pipeline"""
        if camera_id in self.captures:
            self.captures[camera_id].stop()
            del self.captures[camera_id]


class FrameProcessor:
    """
    Base frame processor interface.
    Extend for YOLO, OCR, face recognition, etc.
    """
    
    def __init__(self):
        self.frame_count = 0
        
    def process(self, frame: np.ndarray):
        """Process frame and return results"""
        raise NotImplementedError


# MediaMTX Integration
def publish_to_mediamtx(rtsp_url: str, stream_path: str):
    """
    Publish external RTSP to MediaMTX for rebroadcast.
    Useful for creating WebRTC/WebSocket streams.
    """
    command = [
        'ffmpeg',
        '-i', rtsp_url,
        '-c:v', 'copy',
        '-f', 'rtsp',
        f'rtsp://localhost:8554/{stream_path}'
    ]
    return subprocess.Popen(command)


# Connection manager
_pipeline = None

def get_multistream_pipeline() -> MultiStreamPipeline:
    global _pipeline
    if _pipeline is None:
        _pipeline = MultiStreamPipeline()
    return _pipeline