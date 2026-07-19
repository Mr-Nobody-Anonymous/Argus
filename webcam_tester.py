"""
Webcam testing script for Argus backend.
Allows running the backend with local PC camera for testing face recognition,
emotion detection, and other features.
"""
import cv2
import numpy as np
import time
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.inference_engine import get_inference_engine
from backend.services.image_enhancement import get_image_enhancement
from backend.services.speed_height_analysis import get_speed_height_analyzer
from backend.config.config import get_config

# Try to import optional emotion detection
try:
    from deepface import DeepFace
    EMOTION_DETECTION_AVAILABLE = True
except ImportError:
    EMOTION_DETECTION_AVAILABLE = False
    print("Warning: deepface not installed. Run: pip install deepface for emotion detection")


class WebcamTester:
    """Test Argus features using PC webcam"""
    
    def __init__(self, camera_id: int = 0):
        self.camera_id = camera_id
        self.cap = None
        self.inference_engine = get_inference_engine()
        self.image_enhancement = get_image_enhancement()
        self.speed_analyzer = get_speed_height_analyzer()
        self.config = get_config()
        
        # Stats
        self.frame_count = 0
        self.start_time = time.time()
        self.fps = 0
        
    def initialize(self) -> bool:
        """Initialize webcam and models"""
        print("Initializing webcam tester...")
        
        # Initialize webcam
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
        
        # Set resolution (optional)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print(f"Webcam {self.camera_id} opened successfully")
        
        # Initialize inference engine
        if not self.inference_engine.is_model_loaded():
            print("Loading YOLOv8 model...")
            self.inference_engine.load_model()
        
        print("All systems ready!")
        return True
    
    def process_frame(self, frame: np.ndarray) -> dict:
        """Process a single frame through all analysis pipelines"""
        self.frame_count += 1
        
        # Step 1: Image Enhancement
        enhanced_frame = self.image_enhancement.enhance_frame(frame, mode="auto")
        
        # Step 2: Object Detection
        detections = self.inference_engine.detect_objects(enhanced_frame)
        
        # Step 3: Face Detection & Recognition
        face_results = []
        try:
            from backend.services.face_recognition import get_face_recognition
            face_recognition = get_face_recognition()
            if face_recognition.enabled and face_recognition.is_initialized():
                face_results = face_recognition.recognize_faces(enhanced_frame, detect_emotions=True)
        except Exception as e:
            print(f"Face recognition error: {e}")
        
        # Step 4: Speed/Height Analysis
        analysis_results = []
        frame_time = time.time()
        
        for detection in detections:
            obj_id = f"webcam_{self.speed_analyzer.get_next_object_id()}"
            
            analysis = self.speed_analyzer.analyze_object(
                object_id=obj_id,
                bbox=detection['bbox'],
                class_name=detection['class_name'],
                frame_time=frame_time,
                frame_shape=frame.shape[:2]
            )
            analysis_results.append(analysis)
        
        # Calculate FPS
        elapsed = time.time() - self.start_time
        if elapsed > 0:
            self.fps = self.frame_count / elapsed
        
        return {
            'detections': detections,
            'face_results': face_results,
            'analysis_results': analysis_results,
            'fps': round(self.fps, 2),
            'image_quality': self.image_enhancement.detect_quality_issues(frame)
        }
    
    def draw_results(self, frame: np.ndarray, results: dict) -> np.ndarray:
        """Draw detection and analysis results on frame"""
        display = frame.copy()
        
        # Draw detections
        for det in results['detections']:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Draw face results
        for face in results['face_results']:
            x1, y1, x2, y2 = face['bbox']
            name = face.get('person_name', 'unknown')
            conf = face.get('confidence', 0)
            
            color = (0, 255, 0) if face.get('is_known') else (0, 165, 255)
            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
            
            label = f"{name} {conf:.2f}"
            if 'dominant_emotion' in face:
                label += f" [{face['dominant_emotion']}]"
            
            cv2.putText(display, label, (x1, y1 - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Draw FPS
        cv2.putText(display, f"FPS: {results['fps']}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        return display
    
    def run(self, show_window: bool = True, save_output: bool = False):
        """Run the webcam tester loop"""
        if not self.initialize():
            return
        
        print("\n=== Argus Webcam Tester ===")
        print("Press 'q' to quit, 's' to save frame, 'e' for enhancement cycle")
        print("Showing: Object Detection, Face Recognition, Emotion Detection (if available)")
        print("=" * 40)
        
        enhancement_modes = ["auto", "low_light", "denoise", "sharpen", "night", "deblur", "hdr"]
        mode_idx = 0
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v') if save_output else None
        out = None
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Process frame with current enhancement mode
            enhanced = self.image_enhancement.enhance_frame(frame, mode=enhancement_modes[mode_idx])
            detections = self.inference_engine.detect_objects(enhanced)
            
            # Get face recognition results
            try:
                from backend.services.face_recognition import get_face_recognition
                face_recognition = get_face_recognition()
                face_results = []
                if face_recognition.enabled and face_recognition.is_initialized():
                    face_results = face_recognition.recognize_faces(enhanced, detect_emotions=True)
            except Exception as e:
                face_results = []
            
            # Draw results
            display = self.draw_results(frame, {
                'detections': detections,
                'face_results': face_results,
                'fps': self.fps
            })
            
            # Show active enhancement mode
            cv2.putText(display, f"Mode: {enhancement_modes[mode_idx]}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if show_window:
                cv2.imshow('Argus Webcam Tester', display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    # Save frame
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"data/webcam_test_{timestamp}.jpg", frame)
                    print(f"Saved frame to data/webcam_test_{timestamp}.jpg")
                elif key == ord('e'):
                    mode_idx = (mode_idx + 1) % len(enhancement_modes)
                    print(f"Changed enhancement mode to: {enhancement_modes[mode_idx]}")
            
            if save_output:
                if out is None:
                    h, w = display.shape[:2]
                    out = cv2.VideoWriter('webcam_output.mp4', fourcc, 20.0, (w, h))
                out.write(display)
        
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\nWebcam tester stopped.")


def main():
    """Main entry point for webcam testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Argus Webcam Tester')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--no-display', action='store_true', help='Run without display window')
    parser.add_argument('--save-video', action='store_true', help='Save output to webcam_output.mp4')
    
    args = parser.parse_args()
    
    tester = WebcamTester(camera_id=args.camera)
    tester.run(show_window=not args.no_display, save_output=args.save_video)


if __name__ == "__main__":
    main()