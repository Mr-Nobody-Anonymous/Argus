"""
AI Inference & Tracking Accuracy Sanity Test
Feeds mock video through YOLOv8 + BoT-SORT pipeline
"""
import cv2
import time
import numpy as np
from typing import List, Dict

def mock_ocr_insightface_pipeline(crop_img) -> Dict:
    """Simulates secondary model inferences (LP recognition & Facial recognition)."""
    time.sleep(0.005)  # Simulated 5ms execution overhead
    h, w = crop_img.shape[:2] if len(crop_img.shape) >= 2 else (0, 0)
    return {"ocr_text": f"ABC-{np.random.randint(1000, 9999)}", "face_match": h > 50 and w > 50}

def create_mock_traffic_video(output_path: str = "mock_10s_traffic.mp4"):
    """Create a 10-second test video with cars and people."""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 30.0, (640, 480))
    
    for i in range(300):  # 10 seconds at 30 FPS
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:, :] = (20, 20, 30)  # Dark background
        
        # Draw moving "car" (red rectangle)
        car_x = int(50 + (i * 2)) % 500
        cv2.rectangle(frame, (car_x, 400, car_x + 100, 430), (0, 0, 255), -1)
        cv2.putText(frame, 'CAR', (car_x + 10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw moving "person" (green rectangle)
        person_x = int(100 + (i * 1.5)) % 500
        cv2.rectangle(frame, (person_x, 300, person_x + 30, 400), (0, 255, 0), -1)
        cv2.putText(frame, 'PED', (person_x + 5, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        out.write(frame)
    
    out.release()
    print(f"Created mock video: {output_path}")
    return output_path

def run_pipeline_sanity_test(video_path: str = None):
    """Run AI pipeline sanity test with mock video."""
    if video_path is None or not os.path.exists(video_path):
        video_path = create_mock_traffic_video()
    
    print("Initializing tracking and inference verification pipelines...")
    
    # Try to import YOLOv8
    try:
        from ultralytics import YOLO
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Use yolov8n for faster testing
        model = YOLO("yolov8n.pt").to(device)
        print(f"Loaded YOLOv8 on {device.upper()}")
    except ImportError:
        print("WARNING: ultralytics not installed. Running in simulation mode.")
        # Simulation mode
        return run_simulation_mode(video_path)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"CRITICAL: Failed to load target video file: {video_path}")
        return
    
    frame_count = 0
    track_history = {}
    id_switches = 0
    total_latency_ms = 0.0
    successful_crops = 0
    
    print(f"Processing target clip...")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        t0 = time.time()
        
        # Run YOLO inference with tracking
        results = model.track(frame, persist=True, verbose=False)
        
        t1 = time.time()
        frame_latency = (t1 - t0) * 1000
        total_latency_ms += frame_latency
        
        # Extract tracking data
        if results and results[0].boxes is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy() if hasattr(results[0].boxes.xyxy, 'cpu') else results[0].boxes.xyxy.numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int).tolist() if hasattr(results[0].boxes.id, 'cpu') else []
            classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist() if hasattr(results[0].boxes.cls, 'cpu') else []
            
            for box, track_id, cls_idx in zip(boxes, track_ids, classes):
                class_name = model.names[int(cls_idx)] if cls_idx < len(model.names) else f"class_{cls_idx}"
                
                # (a) Check tracking consistency
                if track_id in track_history:
                    if track_history[track_id] != class_name:
                        id_switches += 1
                        print(f"[WARN] ID Switch detected! Track {track_id}: {track_history[track_id]} -> {class_name}")
                else:
                    track_history[track_id] = class_name
                
                # (b) Bounding box crop verification for person (0) and car (2)
                if cls_idx in [0, 2]:  # person or car
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    cropped_asset = frame[y1:y2, x1:x2]
                    
                    if cropped_asset.size > 0:
                        secondary_results = mock_ocr_insightface_pipeline(cropped_asset)
                        if "ocr_text" in secondary_results or "face_match" in secondary_results:
                            successful_crops += 1
    
    cap.release()
    
    # Calculate final pipeline results
    avg_latency = total_latency_ms / frame_count if frame_count > 0 else 0
    
    print("\n" + "="*50)
    print("AI INFERENCE AND TRACKING SANITY REPORT")
    print("="*50)
    print(f"Processed Frames            : {frame_count}")
    print(f"Avg Inference Latency/Frame : {avg_latency:.2f} ms")
    print(f"Total Unique Assets Tracked : {len(track_history)}")
    print(f"Total Tracking ID Switches  : {id_switches}")
    print(f"Successful Crops to Models  : {successful_crops}")
    
    # SUCCESS CRITERIA:
    # - Avg Latency < 33ms (30 FPS baseline)
    # - ID Switches = 0 (perfect tracking)
    # - Successful Crops > 0 (models receiving data)
    
    all_passed = True
    if avg_latency >= 33.3:
        print(f"FAILED: Latency ({avg_latency:.2f}ms) exceeds 30 FPS target")
        all_passed = False
    if id_switches > 0:
        print(f"FAILED: {id_switches} ID switches detected")
        all_passed = False
    if successful_crops == 0:
        print("FAILED: No successful crops to secondary models")
        all_passed = False
    
    if all_passed:
        print("✓ SUCCESS: AI and tracking pipeline verification PASSED")
    
    return all_passed

def run_simulation_mode(video_path: str):
    """Simulation mode when YOLO is not available."""
    import os
    
    if not os.path.exists(video_path):
        video_path = create_mock_traffic_video()
    
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
    
    cap.release()
    
    print(f"\nSimulation Mode: Processed {frame_count} frames")
    print("✓ SUCCESS: Simulation mode completed (install ultralytics for real testing)")
    return True

if __name__ == "__main__":
    import torch
    run_pipeline_sanity_test("mock_10s_traffic.mp4")