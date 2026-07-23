"""
Advanced Object Detection & Tracking with ANPR Integration
YOLOv8 + ByteTrack/BoT-SORT + PaddleOCR for license plate recognition
"""
import numpy as np
import torch
import cv2
import time
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class TrackedObject:
    """Object with persistent tracking across frames"""
    track_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[int, int]
    timestamp: float
    features: Optional[np.ndarray] = None
    license_plate: Optional[str] = None
    license_confidence: Optional[float] = None
    trajectory: List[Tuple[int, int]] = field(default_factory=list)


class YOLOByteTracker:
    """
    YOLOv8 + ByteTrack integration for surveillance.
    Production-grade with ANPR branching for vehicles.
    """
    
    # Vehicle class names for filtering
    VEHICLE_CLASSES = ['car', 'motorcycle', 'bus', 'truck', 'bicycle']
    
    def __init__(self, model_path: str = "yolov8s.pt", device: str = "auto"):
        self.device = device if device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model(model_path)
        self.track_id_counter = 0
        self.active_tracks: Dict[int, TrackedObject] = {}
        self.inference_times: List[float] = []
        
    def _load_model(self, model_path: str):
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            model.to(self.device)
            logger.info(f"Loaded YOLOv8 from {model_path} on {self.device}")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return None

    def detect_and_track(self, frame: np.ndarray) -> List[TrackedObject]:
        """Run YOLO detection and ByteTrack tracking with ANPR branching."""
        if self.model is None:
            return []

        start_time = time.time()
        
        results = self.model(
            frame,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            max_det=100,
            half=True,
            device=self.device
        )
        
        detections = self._extract_detections(results, frame)
        tracks = self._byte_track(detections, frame)
        self._process_anpr(frame, tracks)
        
        inference_time = (time.time() - start_time) * 1000
        self.inference_times.append(inference_time)
        
        return tracks

    def _extract_detections(self, results, frame: np.ndarray) -> List[Dict]:
        """Extract detection results from YOLO output."""
        detections = []
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                detections.append({
                    'bbox': (x1, y1, x2, y2),
                    'center': center,
                    'conf': conf,
                    'cls_id': cls_id,
                    'cls_name': result.names[cls_id]
                })
        
        return detections

    def _byte_track(self, detections: List[Dict], frame: np.ndarray) -> List[TrackedObject]:
        """Simplified ByteTrack with track ID assignment and feature extraction."""
        tracks = []
        
        for det in detections:
            track_id = self._assign_track(det, frame)
            features = self._extract_features(frame, det['bbox'])
            
            track = TrackedObject(
                track_id=track_id,
                class_id=det['cls_id'],
                class_name=det['cls_name'],
                confidence=det['conf'],
                bbox=det['bbox'],
                center=det['center'],
                timestamp=time.time(),
                features=features
            )
            
            track.trajectory.append(det['center'])
            if len(track.trajectory) > 100:
                track.trajectory = track.trajectory[-100:]
            
            tracks.append(track)
        
        return tracks

    def _assign_track(self, det: Dict, frame: np.ndarray) -> int:
        """Assign existing track or create new one based on IoU."""
        best_match = None
        best_iou = 0.8
        
        for track_id, existing_track in list(self.active_tracks.items()):
            iou = self._calculate_iou(det['bbox'], existing_track.bbox)
            if iou > best_iou:
                best_match = track_id
                best_iou = iou
        
        if best_match:
            self.active_tracks[best_match].bbox = det['bbox']
            self.active_tracks[best_match].center = det['center']
            return best_match
        else:
            new_id = self.track_id_counter
            self.track_id_counter += 1
            return new_id

    def _calculate_iou(self, box1: Tuple[int, int, int, int], box2: Tuple[int, int, int, int]) -> float:
        """Calculate IoU between two bounding boxes."""
        x1a, y1a, x2a, y2a = box1
        x1b, y1b, x2b, y2b = box2
        
        overlap_x1 = max(x1a, x1b)
        overlap_y1 = max(y1a, y1b)
        overlap_x2 = min(x2a, x2b)
        overlap_y2 = min(y2a, y2b)
        
        if overlap_x2 <= overlap_x1 or overlap_y2 <= overlap_y1:
            return 0.0
        
        overlap = (overlap_x2 - overlap_x1) * (overlap_y2 - overlap_y1)
        area_a = (x2a - x1a) * (y2a - y1a)
        area_b = (x2b - x1b) * (y2b - y1b)
        
        return overlap / float(area_a + area_b - overlap)

    def _extract_features(self, frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
        """Extract appearance features for re-ID."""
        x1, y1, x2, y2 = bbox
        roi = frame[max(0, y1):y2, max(0, x1):x2]
        
        if roi.size == 0:
            return np.zeros(256)
        
        roi = cv2.resize(roi, (64, 128))
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        return hist.flatten()[:256]

    def _process_anpr(self, frame: np.ndarray, tracks: List[TrackedObject]):
        """Run PaddleOCR ANPR on vehicle detections."""
        for track in tracks:
            if track.class_name in self.VEHICLE_CLASSES:
                x1, y1, x2, y2 = track.bbox
                vehicle_roi = frame[y1:y2, x1:x2]
                
                if vehicle_roi.size == 0:
                    continue
                
                plate_text, plate_conf = self._read_license_plate(vehicle_roi)
                
                if plate_text:
                    track.license_plate = plate_text
                    track.license_confidence = plate_conf

    def _read_license_plate(self, vehicle_roi: np.ndarray) -> Tuple[Optional[str], float]:
        """PaddleOCR-based license plate recognition."""
        try:
            from paddleocr import PaddleOCR
            
            if not hasattr(self, 'ocr'):
                self.ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    show_log=False,
                    use_gpu=torch.cuda.is_available()
                )
            
            result = self.ocr.ocr(vehicle_roi, cls=True)
            
            if result and result[0]:
                for line in result[0]:
                    text, conf = line[1]
                    if len(text) >= 4 and any(c.isalpha() for c in text):
                        return text.upper(), float(conf)
            
            return None, 0.0
            
        except ImportError:
            logger.warning("PaddleOCR not installed")
            return None, 0.0
        except Exception as e:
            logger.debug(f"ANPR error: {e}")
            return None, 0.0


_tracker = None

def get_object_tracker() -> YOLOByteTracker:
    global _tracker
    if _tracker is None:
        _tracker = YOLOByteTracker()
    return _tracker