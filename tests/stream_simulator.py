"""
RTSP Stream Simulator for City OS Testing
Uses FFmpeg to stream a mock video file in a continuous loop to MediaMTX
"""
import subprocess
import os
import sys
import re
from pathlib import Path
import cv2
import numpy as np

# Configuration from environment
DEFAULT_RTSP_URL = os.getenv("DEFAULT_RTSP_URL", "rtsp://localhost:8554/live/camera1")
STREAM_BITRATE = os.getenv("STREAM_BITRATE", "2M")
VIDEO_WIDTH = int(os.getenv("VIDEO_WIDTH", "640"))
VIDEO_HEIGHT = int(os.getenv("VIDEO_HEIGHT", "480"))

def validate_path(path: str) -> bool:
    """Validate path contains only safe characters."""
    return bool(re.match(r'^[\w\-. /\\]+$', path))

def validate_rtsp_url(url: str) -> bool:
    """Validate RTSP URL format."""
    return bool(re.match(r'^rtsp://[\w\-.:]+(/[\w\-.]+)*$', url))

def run_rtsp_simulator(video_path: str, rtsp_url: str):
    """Simulate a live RTSP stream using a video file."""
    
    # Security validation
    if not validate_path(video_path):
        print(f"ERROR: Invalid video path (security violation)")
        return
    
    if not validate_rtsp_url(rtsp_url):
        print(f"ERROR: Invalid RTSP URL (security violation)")
        return
    
    # Create sample video if doesn't exist
    if not os.path.exists(video_path):
        print(f"Mock video file '{video_path}' not found. Creating sample...")
        create_sample_video(video_path)
    
    if not os.path.exists(video_path):
        print(f"CRITICAL: Failed to create video file")
        return
    
    # FFmpeg command optimized for low-latency live streaming loop
    ffmpeg_cmd = [
        'ffmpeg',
        '-re',                         # Read input at native frame rate
        '-stream_loop', '-1',          # Loop the input video infinitely
        '-i', video_path,              # Input file path (validated)
        '-c:v', 'libx264',             # Encode to H.264
        '-preset', 'veryfast',         # Low overhead encoding preset
        '-tune', 'zerolatency',        # Optimize for real-time streaming
        '-b:v', STREAM_BITRATE,          # Configurable bitrate
        '-f', 'rtsp',                  # Output format
        '-rtsp_transport', 'tcp',      # Use TCP transport for stability
        rtsp_url  # URL validated above
    ]
    
    print(f"Starting RTSP stream simulation: {rtsp_url}")
    
    try:
        process = subprocess.Popen(
            ffmpeg_cmd, 
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.PIPE, 
            text=True
        )
        # Monitor stderr for initialization issues
        import time
        for _ in range(10):
            time.sleep(0.5)
            if process.poll() is not None:
                _, err = process.communicate()
                print(f"FFmpeg failed to start. Error:\n{err}")
                return
        
        print("Stream simulation running. Press Ctrl+C to stop.")
        process.wait()
    except KeyboardInterrupt:
        print("\nStopping RTSP stream simulation...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    except Exception as e:
        print(f"Error: {e}")
        if 'process' in locals():
            process.terminate()

def create_sample_video(output_path: str):
    """Create a sample test video using OpenCV."""
    
    if not validate_path(output_path):
        print(f"ERROR: Invalid output path")
        return
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create 10-second test video with moving objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (VIDEO_WIDTH, VIDEO_HEIGHT))
    
    if not out.isOpened():
        print(f"ERROR: Failed to create video writer")
        return
    
    for i in range(300):  # 10 seconds at 30 FPS
        frame = np.zeros((VIDEO_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
        frame[:, :] = (20, 20, 30)  # Dark background
        
        # Draw moving "car" (red rectangle)
        car_x = int(50 + (i * 2)) % (VIDEO_WIDTH - 100)
        cv2.rectangle(frame, (car_x, 400, car_x + 100, 430), (0, 0, 255), -1)
        cv2.putText(frame, 'CAR', (car_x + 10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw moving "person" (green rectangle)
        person_x = int(100 + (i * 1.5)) % (VIDEO_WIDTH - 30)
        cv2.rectangle(frame, (person_x, 300, person_x + 30, 400), (0, 255, 0), -1)
        cv2.putText(frame, 'PED', (person_x + 5, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"Created sample video: {output_path}")

if __name__ == "__main__":
    MOCK_VIDEO = "sample_cctv.mp4"
    TARGET_RTSP = DEFAULT_RTSP_URL
    run_rtsp_simulator(MOCK_VIDEO, TARGET_RTSP)