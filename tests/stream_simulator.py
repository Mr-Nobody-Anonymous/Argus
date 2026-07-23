"""
Multi-Threaded RTSP Stream Simulator for Argus Testing
Pipes synthetic video (from file or webcam fallback) concurrently to multiple MediaMTX endpoints.
Handles graceful looping so video files restart seamlessly without breaking the RTSP pipeline.
"""
import subprocess
import os
import sys
import re
import time
import threading
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

# ── Configuration ──────────────────────────────────────────────────────────────

DEFAULT_RTSP_URLS = [
    "rtsp://localhost:8554/cam1",
    "rtsp://localhost:8554/cam2",
    "rtsp://localhost:8554/cam3",
    "rtsp://localhost:8554/cam4",
]
STREAM_BITRATE = os.getenv("STREAM_BITRATE", "2M")
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "640"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "480"))
FPS = int(os.getenv("STREAM_FPS", "30"))
CAMERA_COUNT = int(os.getenv("CAMERA_COUNT", "4"))
SAMPLE_VIDEO_DIR = Path(os.getenv("SAMPLE_VIDEO_DIR", "data/streams"))
SAMPLE_VIDEO_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("StreamSimulator")


# ── Security Validation ────────────────────────────────────────────────────────

_PATH_SAFE = re.compile(r"^[\w\-. /\\]+$")
_RTSP_SAFE = re.compile(r"^rtsp://[\w\-.:]+(/[\w\-.]+)*$")


def validate_path(path: str) -> bool:
    """Ensure the path contains only safe characters (no injection)."""
    return bool(_PATH_SAFE.match(path))


def validate_rtsp_url(url: str) -> bool:
    """Ensure the RTSP URL matches the expected pattern."""
    return bool(_RTSP_SAFE.match(url))


# ── Synthetic Video Generator ──────────────────────────────────────────────────

def create_synthetic_video(
    output_path: str,
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    duration_sec: int = 10,
    fps: int = FPS,
    variant: int = 1,
) -> None:
    """
    Create a synthetic test video with moving objects and camera ID overlay.
    Multiple variants (1-4) produce different movement patterns for variety.
    """
    if not validate_path(output_path):
        logger.error("Invalid output path for synthetic video")
        return

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    total_frames = duration_sec * fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        logger.error("Failed to open VideoWriter for %s", output_path)
        return

    for i in range(total_frames):
        # Dark background with subtle gradient
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        for y in range(height):
            shade = int(15 + (y / height) * 20)
            frame[y, :] = (shade, shade, shade)

        # ── Camera ID label ──
        cv2.putText(
            frame,
            f"CAM {variant}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (100, 180, 255),
            2,
        )

        # ── Moving objects ──
        # Car (red rectangle) — horizontal motion
        car_x = int((i * 3 * variant) % (width - 120))
        cv2.rectangle(frame, (car_x, height - 80), (car_x + 100, height - 40), (0, 0, 200), -1)
        cv2.putText(frame, "CAR", (car_x + 15, height - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Person (green rectangle) — walking from opposite direction
        person_x = int((width - 80) - (i * 2 * variant) % (width - 60))
        cv2.rectangle(frame, (person_x, height - 180), (person_x + 25, height - 60), (0, 200, 0), -1)
        cv2.putText(frame, "PED", (person_x + 2, height - 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Cyclist / motorbike (blue) — alternate lane
        if variant >= 3:
            bike_x = int((width // 2) + (i * 2.5 * (variant - 2)) % (width // 2))
            cv2.rectangle(frame, (bike_x, height - 130), (bike_x + 40, height - 60), (200, 100, 0), -1)
            cv2.putText(frame, "BIKE", (bike_x + 2, height - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Timestamp overlay
        ts = time.strftime("%H:%M:%S")
        cv2.putText(frame, ts, (width - 120, height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

        writer.write(frame)

    writer.release()
    logger.info("Created synthetic video: %s (variant %d)", output_path, variant)


# ── Multi-Publisher (FFmpeg subprocess) ────────────────────────────────────────

class RTSPPublisher:
    """
    Manages a single FFmpeg subprocess that reads a video file in a continuous
    loop and publishes it to an RTSP endpoint.
    """

    def __init__(self, video_path: str, rtsp_url: str, bitrate: str = STREAM_BITRATE):
        self.video_path = video_path
        self.rtsp_url = rtsp_url
        self.bitrate = bitrate
        self._process: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Launch the FFmpeg subprocess in a daemon thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Publisher already running for %s", self.rtsp_url)
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Started RTSP publisher: %s", self.rtsp_url)

    def stop(self, timeout: float = 5.0) -> None:
        """Signal stop and terminate the subprocess."""
        self._stop_event.set()
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        if self._thread:
            self._thread.join(timeout=timeout)
        logger.info("Stopped RTSP publisher: %s", self.rtsp_url)

    def _run(self) -> None:
        """Continuous FFmpeg pipeline with auto-restart on EOF."""
        while not self._stop_event.is_set():
            if not os.path.isfile(self.video_path):
                logger.error("Video file missing: %s — recreating", self.video_path)
                # Infer variant index from RTSP URL
                variant = 1
                for idx, url in enumerate(DEFAULT_RTSP_URLS, start=1):
                    if url == self.rtsp_url:
                        variant = idx
                        break
                create_synthetic_video(self.video_path, variant=variant)

            cmd = [
                "ffmpeg",
                "-re",                         # Native frame rate
                "-stream_loop", "-1",          # Infinite loop
                "-i", self.video_path,
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-tune", "zerolatency",
                "-b:v", self.bitrate,
                "-max_delay", "100000",
                "-bufsize", "1M",
                "-f", "rtsp",
                "-rtsp_transport", "tcp",
                self.rtsp_url,
            ]

            try:
                self._process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self._process.wait()
            except Exception as exc:
                logger.error("FFmpeg error for %s: %s", self.rtsp_url, exc)

            if not self._stop_event.is_set():
                logger.info("Restarting RTSP publisher: %s", self.rtsp_url)
                time.sleep(1.0)


# ── Webcam Fallback Publisher ──────────────────────────────────────────────────

class WebcamPublisher:
    """
    Alternative publisher that streams from the local machine's webcam via
    FFmpeg when no video file is available. Useful for quick Windows testing.
    """

    def __init__(self, rtsp_url: str, camera_id: int = 0):
        self.rtsp_url = rtsp_url
        self.camera_id = camera_id
        self._process: Optional[subprocess.Popen] = None
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Started webcam publisher: %s (camera_id=%d)", self.rtsp_url, self.camera_id)

    def stop(self, timeout: float = 5.0) -> None:
        self._stop_event.set()
        if self._process:
            self._process.terminate()
            try:
                self._process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self._process.kill()
        if self._thread:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        cmd = [
            "ffmpeg",
            "-f", "dshow" if sys.platform == "win32" else "v4l2",
            "-i", f"video={self.camera_id}" if sys.platform == "win32" else f"/dev/video{self.camera_id}",
            "-c:v", "libx264",
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-b:v", "1M",
            "-max_delay", "100000",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_url,
        ]

        try:
            self._process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self._process.wait()
        except Exception as exc:
            logger.error("Webcam FFmpeg error for %s: %s", self.rtsp_url, exc)


# ── Multi-Stream Orchestrator ──────────────────────────────────────────────────

class MultiStreamSimulator:
    """
    Orchestrates N concurrent RTSP publishers.
    Generates synthetic video files on first run if none exist.
    Falls back to webcam if a video file cannot be created.
    """

    def __init__(
        self,
        rtsp_urls: Optional[List[str]] = None,
        video_dir: Path = SAMPLE_VIDEO_DIR,
        use_webcam_fallback: bool = False,
    ):
        self.rtsp_urls = rtsp_urls or DEFAULT_RTSP_URLS[:CAMERA_COUNT]
        self.video_dir = video_dir
        self.use_webcam_fallback = use_webcam_fallback
        self._publishers: List[object] = []

    def ensure_video_assets(self) -> bool:
        """Create synthetic video files for each camera if missing."""
        video_dir = self.video_dir
        video_dir.mkdir(parents=True, exist_ok=True)
        all_exist = True

        for idx, url in enumerate(self.rtsp_urls, start=1):
            video_path = video_dir / f"cam{idx}.mp4"
            if not video_path.exists():
                logger.info("Generating synthetic video: %s", video_path)
                create_synthetic_video(str(video_path), variant=idx)
            if not video_path.exists():
                all_exist = False

        return all_exist

    def start_all(self) -> None:
        """Start all RTSP publishers concurrently."""
        videos_ready = self.ensure_video_assets()

        for idx, url in enumerate(self.rtsp_urls, start=1):
            video_path = self.video_dir / f"cam{idx}.mp4"

            if video_path.exists():
                pub = RTSPPublisher(str(video_path), url)
            elif self.use_webcam_fallback:
                logger.warning("Video missing for %s, falling back to webcam", url)
                pub = WebcamPublisher(url, camera_id=idx - 1)
            else:
                logger.error("No video source for %s — skipping", url)
                continue

            pub.start()
            self._publishers.append(pub)

        logger.info(
            "MultiStreamSimulator running with %d publishers. Press Ctrl+C to stop.",
            len(self._publishers),
        )

    def stop_all(self, timeout: float = 5.0) -> None:
        """Gracefully stop all publishers."""
        for pub in self._publishers:
            pub.stop(timeout=timeout)
        self._publishers.clear()
        logger.info("All publishers stopped.")

    def wait_until_interrupted(self) -> None:
        """Block until KeyboardInterrupt, then clean up."""
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received interrupt — shutting down…")
        finally:
            self.stop_all()


# ── CLI Entry Point ────────────────────────────────────────────────────────────

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Argus RTSP Stream Simulator")
    parser.add_argument(
        "--video-dir",
        default=str(SAMPLE_VIDEO_DIR),
        help="Directory for synthetic video assets",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=CAMERA_COUNT,
        choices=range(1, 5),
        help="Number of concurrent RTSP streams (1-4)",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Fall back to webcam when video file is missing",
    )
    parser.add_argument(
        "--urls",
        nargs="*",
        default=None,
        help="Custom RTSP URLs (overrides defaults)",
    )
    args = parser.parse_args()

    urls = args.urls or DEFAULT_RTSP_URLS[:args.count]
    simulator = MultiStreamSimulator(
        rtsp_urls=urls,
        video_dir=Path(args.video_dir),
        use_webcam_fallback=args.webcam,
    )
    simulator.start_all()
    simulator.wait_until_interrupted()


if __name__ == "__main__":
    main()
</｜｜DSML｜｜parameter>
</create_file>
