"""
RTSP stream ingestion service with auto-reconnect and frame queue management
"""
import cv2
import os
import threading
import queue
import time
import logging
from datetime import datetime
from typing import Dict, Optional
from backend.services.management.camera_manager import get_camera_manager

logger = logging.getLogger(__name__)


class StreamIngestion:
    def __init__(self, max_queue_size: int = 100):
        self.max_queue_size = max_queue_size
        self.frame_queues: Dict[int, queue.Queue] = {}
        self.capture_threads: Dict[int, threading.Thread] = {}
        self.stop_flags: Dict[int, threading.Event] = {}
        self.camera_manager = get_camera_manager()
        self.fps_trackers: Dict[int, list] = {}

    def start_camera(self, camera_id: int, rtsp_url: str):
        """Start capturing frames from a camera"""
        if camera_id in self.capture_threads and self.capture_threads[camera_id].is_alive():
            logger.warning(f"Camera {camera_id} already running")
            return

        # Create frame queue and stop flag
        self.frame_queues[camera_id] = queue.Queue(maxsize=self.max_queue_size)
        self.stop_flags[camera_id] = threading.Event()
        self.fps_trackers[camera_id] = []

        # Start capture thread
        thread = threading.Thread(
            target=self._capture_loop,
            args=(camera_id, rtsp_url),
            daemon=True
        )
        self.capture_threads[camera_id] = thread
        thread.start()
        logger.info(f"Started camera {camera_id}: {rtsp_url}")

    def stop_camera(self, camera_id: int):
        """Stop capturing frames from a camera"""
        if camera_id in self.stop_flags:
            self.stop_flags[camera_id].set()
            logger.info(f"Stopping camera {camera_id}")

    def get_frame(self, camera_id: int, timeout: float = 1.0) -> Optional[tuple]:
        """Get next frame from camera queue"""
        if camera_id not in self.frame_queues:
            return None

        try:
            frame_data = self.frame_queues[camera_id].get(timeout=timeout)
            return frame_data
        except queue.Empty:
            return None

    def _capture_loop(self, camera_id: int, rtsp_url: str):
        """Main capture loop with auto-reconnect"""
        retry_count = 0
        max_retries = 10
        cap = None
        # Local video files are a valid source (used for demos/tests). They differ
        # from live streams in two ways that must be handled explicitly:
        #   1. They hit EOF, which is normal termination, not a stream failure.
        #   2. They decode as fast as the CPU allows, so playback must be paced
        #      to the file's own FPS instead of spinning at ~500 fps.
        is_file_source = (
            not rtsp_url.startswith("webcam://")
            and "://" not in rtsp_url
            and os.path.exists(rtsp_url.split("?")[0])
        )
        source_frame_interval = 0.0
        last_status_update = 0.0

        while not self.stop_flags[camera_id].is_set():
            try:
                # Open video capture
                # Handle webcam URLs (webcam://0)
                if rtsp_url.startswith("webcam://"):
                    webcam_idx = int(rtsp_url.replace("webcam://", ""))
                    if cap is None or not cap.isOpened():
                        logger.info(f"Camera {camera_id}: Opening webcam {webcam_idx}")
                        cap = cv2.VideoCapture(webcam_idx)
                        if not cap.isOpened():
                            raise Exception(f"Failed to open webcam {webcam_idx}")
                    # Webcam doesn't need retry status updates
                    if retry_count > 0:
                        retry_count = 0
                        logger.info(f"Camera {camera_id}: Webcam opened successfully")
                else:
                    if cap is None or not cap.isOpened():
                        logger.info(f"Camera {camera_id}: Connecting to {rtsp_url}")
                        # Strip any query string for local files (e.g. clip.mp4?x=2)
                        cap = cv2.VideoCapture(
                            rtsp_url.split("?")[0] if is_file_source else rtsp_url
                        )

                        if not cap.isOpened():
                            raise Exception("Failed to open stream")

                        if is_file_source:
                            src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
                            if src_fps > 0:
                                source_frame_interval = 1.0 / src_fps
                            logger.info(
                                f"Camera {camera_id}: File source opened "
                                f"({src_fps:.2f} fps, looping on EOF)"
                            )
                        
                        # Update camera status to online
                        self.camera_manager.update_status(camera_id, 'online')
                        retry_count = 0
                        logger.info(f"Camera {camera_id}: Connected successfully")

                # Read frame
                ret, frame = cap.read()

                if not ret or frame is None:
                    if is_file_source:
                        # EOF on a file is not an error — rewind and keep playing
                        # so the demo clip behaves like a continuous feed.
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            raise Exception("Failed to read frame after rewind")
                    else:
                        raise Exception("Failed to read frame")

                # Track FPS
                current_time = time.time()
                self.fps_trackers[camera_id].append(current_time)
                
                # Keep only last 30 frames for FPS calculation
                if len(self.fps_trackers[camera_id]) > 30:
                    self.fps_trackers[camera_id] = self.fps_trackers[camera_id][-30:]
                
                # Calculate FPS
                if len(self.fps_trackers[camera_id]) >= 2:
                    time_diff = self.fps_trackers[camera_id][-1] - self.fps_trackers[camera_id][0]
                    fps = (len(self.fps_trackers[camera_id]) - 1) / time_diff if time_diff > 0 else 0
                    # Throttle DB writes to ~1/s. Writing on every frame issued
                    # hundreds of UPDATEs per second and flooded the log.
                    if current_time - last_status_update >= 1.0:
                        self.camera_manager.update_status(camera_id, 'online', fps=fps)
                        last_status_update = current_time

                # Put frame in queue (drop oldest if full)
                try:
                    self.frame_queues[camera_id].put_nowait((frame, datetime.now()))
                except queue.Full:
                    # Drop oldest frame
                    try:
                        self.frame_queues[camera_id].get_nowait()
                        self.frame_queues[camera_id].put_nowait((frame, datetime.now()))
                        logger.debug(f"Camera {camera_id}: Dropped frame (queue full)")
                    except:
                        pass

                # Pace file playback to the clip's native frame rate so a demo
                # video behaves like a real-time feed instead of decoding at
                # several hundred fps and saturating the queue.
                if is_file_source and source_frame_interval > 0:
                    elapsed = time.time() - current_time
                    remaining = source_frame_interval - elapsed
                    if remaining > 0:
                        time.sleep(remaining)

            except Exception as e:
                logger.error(f"Camera {camera_id}: Error - {e}")
                
                # Clean up capture
                if cap:
                    cap.release()
                    cap = None
                
                # Update status to error (only for RTSP cameras)
                if not rtsp_url.startswith("webcam://"):
                    self.camera_manager.update_status(camera_id, 'error')
                
                # Exponential backoff
                retry_count += 1
                if retry_count > max_retries:
                    logger.error(f"Camera {camera_id}: Max retries exceeded, pausing")
                    self.camera_manager.update_status(camera_id, 'offline')
                    break
                
                wait_time = min(2 ** retry_count, 60)
                logger.info(f"Camera {camera_id}: Retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)

        # Cleanup
        if cap:
            cap.release()
        self.camera_manager.update_status(camera_id, 'offline')
        logger.info(f"Camera {camera_id}: Capture loop stopped")

    def get_queue_depth(self, camera_id: int) -> int:
        """Get current queue depth for a camera"""
        if camera_id in self.frame_queues:
            return self.frame_queues[camera_id].qsize()
        return 0

    def stop_all(self):
        """Stop all cameras"""
        for camera_id in list(self.stop_flags.keys()):
            self.stop_camera(camera_id)


# Global stream ingestion instance
_stream_ingestion = None


def get_stream_ingestion() -> StreamIngestion:
    """Get global stream ingestion instance"""
    global _stream_ingestion
    if _stream_ingestion is None:
        _stream_ingestion = StreamIngestion()
    return _stream_ingestion
