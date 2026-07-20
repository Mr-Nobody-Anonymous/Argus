"""
RTSP Stream Simulator for City OS Testing
Uses FFmpeg to stream a mock video file in a continuous loop to MediaMTX
"""
import subprocess
import os
import sys
from pathlib import Path

def run_rtsp_simulator(video_path: str, rtsp_url: str):
    """Simulate a live RTSP stream using a video file."""
    
    # Create sample video if doesn't exist
    if not os.path.exists(video_path):
        print(f"Mock video file '{video_path}' not found. Creating sample...")
        create_sample_video(video_path)
    
    # FFmpeg command optimized for low-latency live streaming loop
    ffmpeg_cmd = [
        'ffmpeg',
        '-re',                         # Read input at native frame rate
        '-stream_loop', '-1',          # Loop the input video infinitely
        '-i', video_path,              # Input file path
        '-c:v', 'libx264',             # Encode to H.264
        '-preset', 'veryfast',         # Low overhead encoding preset
        '-tune', 'zerolatency',        # Optimize for real-time streaming
        '-b:v', '2M',                 # 2Mbps bitrate
        '-f', 'rtsp',                   # Output format
        '-rtsp_transport', 'tcp',      # Use TCP transport for stability
        rtsp_url
    ]
    
    print(f"Starting RTSP stream simulation: {rtsp_url}")
    print(f"Command: {' '.join(ffmpeg_cmd)}")
    
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
        process.wait()
    except Exception as e:
        print(f"Error: {e}")

def create_sample_video(output_path: str):
    """Create a sample test video using OpenCV."""
    import cv2
    import numpy as np
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Create 10-second test video with moving objects
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    for i in range(300):  # 10 seconds at 30 FPS
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Draw moving "car" (red rectangle)
        car_x = int(50 + (i * 2))
        cv2.rectangle(frame, (car_x, 400, car_x + 100, 430), (0, 0, 255), -1)
        cv2.putText(frame, 'CAR', (car_x + 10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw moving "person" (green rectangle)
        person_x = int(100 + (i * 1.5))
        cv2.rectangle(frame, (person_x, 300, person_x + 30, 400), (0, 255, 0), -1)
        cv2.putText(frame, 'PERSON', (person_x + 5, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"Created sample video: {output_path}")

if __name__ == "__main__":
    MOCK_VIDEO = "sample_cctv.mp4"
    TARGET_RTSP = "rtsp://localhost:8554/live/camera1"
    run_rtsp_simulator(MOCK_VIDEO, TARGET_RTSP)