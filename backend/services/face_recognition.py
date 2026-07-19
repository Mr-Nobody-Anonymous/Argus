"""
Facial recognition service for identifying known persons in video streams
"""
import logging
import cv2
import numpy as np
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from backend.config.config import get_config, section_to_dict
from backend.database.db import get_db

logger = logging.getLogger(__name__)


class FaceRecognition:
    """
    Facial recognition using lightweight face detection + feature extraction.
    
    Features:
    - Face detection using Haar Cascades or OpenCV DNN
    - Face encoding/feature extraction
    - Face matching against known faces database
    - Registration of new faces
    - Confidence scoring for matches
    - Face tracking across frames (avoid re-identification)
    """

    def __init__(self):
        self.config = get_config()
        face_config = section_to_dict(getattr(self.config, 'face_recognition', {}))
        self.enabled = face_config.get('enabled', False)
        self.confidence_threshold = face_config.get('confidence_threshold', 0.6)
        self.model_path = face_config.get('model_path', 'models/face_recognition')
        
        self.db = get_db()
        self.face_detector = None
        self.face_recognizer = None
        self._initialized = False
        
        # Face tracking to avoid repeated identification
        self.tracked_faces: Dict[str, dict] = {}
        self.track_timeout = 5.0  # seconds before re-checking a face
        
        # Known faces database (loaded from DB)
        self.known_faces: List[dict] = []
        
        # Known faces directory for storing images
        self.faces_dir = Path("data/known_faces")
        self.faces_dir.mkdir(parents=True, exist_ok=True)
        
        if self.enabled:
            self._initialize()

    def _initialize(self):
        """Initialize face detection and recognition models"""
        try:
            # Use OpenCV's DNN face detector (more accurate than Haar)
            model_file = str(Path(self.model_path) / "opencv_face_detector_uint8.pb")
            config_file = str(Path(self.model_path) / "opencv_face_detector.pbtxt")
            
            # Try DNN first, fall back to Haar cascade
            dnn_available = os.path.exists(model_file) and os.path.exists(config_file)
            
            if dnn_available:
                try:
                    self.face_detector = cv2.dnn.readNetFromTensorFlow(model_file, config_file)
                    logger.info("Loaded OpenCV DNN face detector")
                except Exception:
                    dnn_available = False
            
            if not dnn_available:
                # Use Haar cascade as fallback
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                if os.path.exists(cascade_path):
                    self.face_detector = cv2.CascadeClassifier(cascade_path)
                    logger.info("Loaded Haar cascade face detector")
                else:
                    # Download is handled by first detection
                    self.face_detector = cv2.CascadeClassifier()
                    if not self.face_detector.load(cascade_path):
                        # Try to create one from OpenCV's built-in data
                        logger.warning("Face detector cascade not found, detection may fail")
                        self.face_detector = None
            
            # Use LBPH face recognizer for identification
            self.face_recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # Load known faces from database
            self._load_known_faces()
            
            self._initialized = True
            logger.info("Face recognition initialized")
        
        except Exception as e:
            logger.error(f"Error initializing face recognition: {e}")
            self._initialized = False

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect faces in a frame.
        
        Returns:
            List of face detections: [{bbox, confidence, face_image}]
        """
        if not self.enabled or not self._initialized:
            return []
        
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = frame.shape[:2]
            
            faces = []
            
            if isinstance(self.face_detector, cv2.dnn.Net):
                # DNN-based detection
                blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123], False, False)
                self.face_detector.setInput(blob)
                detections = self.face_detector.forward()
                
                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > self.confidence_threshold:
                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        x1, y1, x2, y2 = box.astype(int)
                        
                        # Ensure within bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(w, x2), min(h, y2)
                        
                        if x2 > x1 and y2 > y1:
                            face_img = gray[y1:y2, x1:x2]
                            faces.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': float(confidence),
                                'face_image': face_img
                            })
            
            elif isinstance(self.face_detector, cv2.CascadeClassifier):
                # Haar cascade detection
                detected = self.face_detector.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                )
                
                for (x, y, fw, fh) in detected:
                    face_img = gray[y:y+fh, x:x+fw]
                    faces.append({
                        'bbox': [x, y, x+fw, y+fh],
                        'confidence': 0.9,  # Haar doesn't give confidence, use default
                        'face_image': face_img
                    })
            
            return faces
        
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []

    def recognize_faces(self, frame: np.ndarray, detect_emotions: bool = False) -> List[Dict]:
        """
        Detect and recognize faces in a frame.
        
        Args:
            frame: Input video frame
            detect_emotions: Whether to detect emotions using deepface (optional)
        
        Returns:
            List of recognized faces: [{bbox, person_name, confidence, is_known, dominant_emotion}]
        """
        if not self.enabled or not self._initialized:
            return []
        
        detected_faces = self.detect_faces(frame)
        results = []
        
        for face in detected_faces:
            bbox = face['bbox']
            face_img = face['face_image']
            
            # Skip tracking for faces already identified recently
            track_key = f"{bbox[0]//50}_{bbox[1]//50}"  # Grid-based tracking
            if track_key in self.tracked_faces:
                tracked = self.tracked_faces[track_key]
                if (datetime.now() - tracked['last_seen']).total_seconds() < self.track_timeout:
                    results.append({
                        'bbox': bbox,
                        'person_name': tracked['person_name'],
                        'confidence': tracked['confidence'],
                        'is_known': tracked['is_known']
                    })
                    continue
            
            # Match against known faces
            match_result = self._match_face(face_img)
            
            if match_result:
                person_name, match_confidence = match_result
                result = {
                    'bbox': bbox,
                    'person_name': person_name,
                    'confidence': match_confidence,
                    'is_known': True
                }
            else:
                result = {
                    'bbox': bbox,
                    'person_name': 'unknown',
                    'confidence': face['confidence'],
                    'is_known': False
                }
            
            # Emotion detection (if deepface available and requested)
            if detect_emotions and result['is_known']:
                try:
                    from deepface import DeepFace
                    # Convert grayscale to BGR for deepface
                    if len(face_img.shape) == 2:
                        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_GRAY2BGR)
                    else:
                        face_bgr = face_img
                    emotion_result = DeepFace.analyze(face_bgr, actions=['emotion'], enforce_detection=False)
                    if emotion_result and len(emotion_result) > 0:
                        result['dominant_emotion'] = emotion_result[0].get('dominant_emotion', 'unknown')
                except Exception as e:
                    logger.debug(f"Emotion detection failed: {e}")
            
            results.append(result)
            
            # Update tracking
            self.tracked_faces[track_key] = {
                'person_name': result['person_name'],
                'confidence': result['confidence'],
                'is_known': result['is_known'],
                'last_seen': datetime.now()
            }
        
        # Cleanup old tracking data
        self._cleanup_tracking()
        
        return results

    def _match_face(self, face_img: np.ndarray) -> Optional[Tuple[str, float]]:
        """Match a face against known faces database"""
        if not self.known_faces or face_img.size == 0:
            return None
        
        try:
            # Resize face to standard size
            face_resized = cv2.resize(face_img, (100, 100))
            
            best_match = None
            best_confidence = 0
            
            for known in self.known_faces:
                known_encoding = np.array(known['encoding']).reshape(100, 100)
                
                # Simple similarity matching using correlation
                # In production, use a proper face recognition model
                similarity = cv2.matchTemplate(
                    face_resized.astype(np.float32),
                    known_encoding.astype(np.float32),
                    cv2.TM_CCOEFF_NORMED
                )[0][0]
                
                if similarity > best_confidence:
                    best_confidence = similarity
                    best_match = known['name']
            
            if best_match and best_confidence > self.confidence_threshold:
                return (best_match, float(best_confidence))
            
            return None
        
        except Exception as e:
            logger.error(f"Error matching face: {e}")
            return None

    def register_face(self, name: str, face_image: np.ndarray) -> bool:
        """
        Register a new face in the database.
        
        Args:
            name: Person's name
            face_image: Grayscale face image
        
        Returns:
            True if registered successfully
        """
        try:
            # Resize to standard size
            face_resized = cv2.resize(face_image, (100, 100))
            
            # Save face image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{name}_{timestamp}.jpg"
            filepath = self.faces_dir / filename
            cv2.imwrite(str(filepath), face_resized)
            
            # Store encoding in database
            encoding_list = face_resized.flatten().tolist()
            
            self.db.execute(
                """
                INSERT INTO known_faces (name, encoding, image_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (name, json.dumps(encoding_list), str(filepath), datetime.now())
            )
            
            # Reload known faces
            self._load_known_faces()
            
            logger.info(f"Registered face for {name}")
            return True
        
        except Exception as e:
            logger.error(f"Error registering face: {e}")
            return False

    def _load_known_faces(self):
        """Load known faces from database"""
        try:
            # Check if table exists
            tables = self.db.fetchall(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='known_faces'"
            )
            
            if not tables:
                # Create table
                self.db.execute("""
                    CREATE TABLE IF NOT EXISTS known_faces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT NOT NULL,
                        encoding TEXT NOT NULL,
                        image_path TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                self.known_faces = []
                return
            
            rows = self.db.fetchall("SELECT * FROM known_faces")
            self.known_faces = []
            
            for row in rows:
                encoding = json.loads(row['encoding'])
                self.known_faces.append({
                    'id': row['id'],
                    'name': row['name'],
                    'encoding': encoding,
                    'image_path': row['image_path']
                })
            
            logger.info(f"Loaded {len(self.known_faces)} known faces")
        
        except Exception as e:
            logger.error(f"Error loading known faces: {e}")
            self.known_faces = []

    def get_known_faces_list(self) -> List[Dict]:
        """Get list of all registered faces"""
        try:
            rows = self.db.fetchall("SELECT id, name, image_path, created_at FROM known_faces")
            return [dict(row) for row in rows]
        except Exception as e:
            logger.error(f"Error getting known faces: {e}")
            return []

    def delete_face(self, face_id: int) -> bool:
        """Delete a known face"""
        try:
            self.db.execute("DELETE FROM known_faces WHERE id = ?", (face_id,))
            self._load_known_faces()
            return True
        except Exception as e:
            logger.error(f"Error deleting face: {e}")
            return False

    def _cleanup_tracking(self):
        """Remove expired tracking entries"""
        current_time = datetime.now()
        expired = [
            k for k, v in self.tracked_faces.items()
            if (current_time - v['last_seen']).total_seconds() > self.track_timeout
        ]
        for key in expired:
            del self.tracked_faces[key]

    def is_initialized(self) -> bool:
        """Check if face recognition is initialized"""
        return self._initialized

    def get_emotion_detection_status(self) -> bool:
        """Check if emotion detection is available (requires deepface)"""
        try:
            import deepface
            return True
        except ImportError:
            return False


# Global instance
_face_recognition = None


def get_face_recognition() -> FaceRecognition:
    """Get global face recognition instance"""
    global _face_recognition
    if _face_recognition is None:
        _face_recognition = FaceRecognition()
    return _face_recognition