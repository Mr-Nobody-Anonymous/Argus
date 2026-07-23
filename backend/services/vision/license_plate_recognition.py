"""
License Plate Recognition (LPR) service for vehicle license plate detection and OCR.
Integrates PaddleOCR and custom detection models for ANPR functionality.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from datetime import datetime
import json
from backend.config.config import get_config, section_to_dict
from backend.database.db import get_db

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class LicensePlateRecognition:
    """
    License Plate Recognition using multiple approaches:
    - Vehicle detection with YOLO
    - License plate detection with specialized models
    - OCR with PaddleOCR for text extraction
    - Region-based recognition (different countries)
    """

    def __init__(self):
        self.config = get_config()
        lpr_config = section_to_dict(getattr(self.config, 'license_plate', {}))
        self.enabled = lpr_config.get('enabled', True)
        self.region = lpr_config.get('region', 'us')  # us, eu, uk, etc.
        self.min_confidence = lpr_config.get('min_confidence', 0.6)

        self.db = get_db()
        self._initialized = False
        self.ocr_engine = None
        
        # License plate patterns by region
        self.plate_patterns = {
            'us': r'^[A-Z]{1,3}\d{1,4}[A-Z]{0,2}$',
            'eu': r'^[A-Z]{1,2}\d{1,3}[A-Z]{1,2}\d{1,3}$',
            'uk': r'^[A-Z]{2}\d{2,3}[A-Z]{3}$',
        }

        if self.enabled and cv2 is not None and np is not None:
            self._initialize()
        elif self.enabled:
            self.enabled = False
            logger.warning("LPR dependencies unavailable, LPR disabled")

    def _initialize(self):
        """Initialize OCR engine and detection models"""
        try:
            # Try to use PaddleOCR if available
            try:
                from paddleocr import PaddleOCR
                self.ocr_engine = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=False,  # Set to True if CUDA available
                    show_log=False
                )
                logger.info("PaddleOCR initialized for license plate recognition")
            except ImportError:
                logger.warning("PaddleOCR not available, using basic OCR")
                self.ocr_engine = None

            self._initialized = True

        except Exception as e:
            logger.error(f"Error initializing license plate recognition: {e}")
            self._initialized = False

    def detect_plates(self, frame: np.ndarray, detections: List[Dict] = None) -> List[Dict]:
        """
        Detect and recognize license plates in a frame.
        
        Args:
            frame: Input video frame
            detections: Optional pre-detected vehicles (from YOLO)
        
        Returns:
            List of license plate detections with text and confidence
        """
        if not self.enabled:
            return []

        if cv2 is None or np is None:
            return []

        plates = []

        try:
            # If no detections provided, run basic vehicle detection
            if detections is None:
                detections = []

            # Process vehicle detections for license plates
            for det in detections:
                if det.get('class_name') in ['car', 'truck', 'bus', 'motorcycle']:
                    x1, y1, x2, y2 = det['bbox']
                    
                    # Extract vehicle region
                    vehicle_img = frame[max(0, y1):y2, max(0, x1):x2]
                    
                    if vehicle_img.size == 0:
                        continue

                    # Detect license plate within vehicle
                    plate_result = self._detect_single_plate(vehicle_img)
                    
                    if plate_result:
                        plate_result.update({
                            'vehicle_bbox': det['bbox'],
                            'vehicle_class': det['class_name'],
                            'vehicle_confidence': det['confidence']
                        })
                        plates.append(plate_result)

            return plates

        except Exception as e:
            logger.error(f"Error detecting license plates: {e}")
            return []

    def _detect_single_plate(self, vehicle_img: np.ndarray) -> Optional[Dict]:
        """Detect and OCR a single license plate"""
        try:
            if cv2 is None or np is None:
                return None
            # Preprocess for plate detection
            gray = cv2.cvtColor(vehicle_img, cv2.COLOR_BGR2GRAY)
            
            # Apply enhancement for better OCR
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            
            # Edge detection to find plate-like regions
            edges = cv2.Canny(enhanced, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Find rectangular contours (potential plates)
            plate_regions = []
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = w / h if h > 0 else 0
                
                # License plates typically have aspect ratio 2:1 to 5:1
                if 2.0 < aspect_ratio < 6.0 and w > 50 and h > 15:
                    plate_regions.append((x, y, x + w, y + h))
            
            # If no plate found, try full image OCR
            if not plate_regions and self.ocr_engine:
                ocr_result = self._ocr_plate(vehicle_img)
                if ocr_result:
                    return ocr_result
                return None

            # Try OCR on each candidate region
            for x1, y1, x2, y2 in plate_regions:
                plate_img = vehicle_img[max(0, y1):y2, max(0, x1):x2]
                
                if self.ocr_engine:
                    ocr_result = self._ocr_plate(plate_img)
                    if ocr_result:
                        ocr_result['localization'] = [x1, y1, x2, y2]
                        return ocr_result

            return None

        except Exception as e:
            logger.error(f"Error in single plate detection: {e}")
            return None

    def _ocr_plate(self, plate_img: np.ndarray) -> Optional[Dict]:
        """Perform OCR on a license plate image"""
        if self.ocr_engine is None:
            return None

        if cv2 is None or np is None:
            return None

        try:
            # PaddleOCR expects RGB
            rgb_img = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            
            result = self.ocr_engine.ocr(rgb_img, cls=True)
            
            if not result or not result[0]:
                return None

            # Get best OCR result
            best_text = ""
            best_conf = 0
            
            for line in result:
                if line:
                    for text, score in line:
                        if score > self.min_confidence and score > best_conf:
                            best_text = text
                            best_conf = score

            if best_text:
                return {
                    'plate_text': self._format_plate_text(best_text),
                    'confidence': float(best_conf),
                    'raw_text': best_text
                }

            return None

        except Exception as e:
            logger.error(f"Error in OCR: {e}")
            return None

    def _format_plate_text(self, text: str) -> str:
        """Format and clean license plate text"""
        # Remove non-alphanumeric characters and convert to uppercase
        cleaned = ''.join(c for c in text if c.isalnum()).upper()
        
        # Apply region-specific formatting
        if self.region == 'us' and len(cleaned) >= 5:
            # US format: ABC1234 or similar
            return cleaned[:3] + '-' + cleaned[3:] if len(cleaned) > 3 else cleaned
        
        return cleaned

    def register_detection(self, plate_text: str, frame_id: int, vehicle_bbox: List[int]) -> int:
        """Register a license plate detection in the database"""
        try:
            self.db.execute(
                """
                INSERT INTO license_plates (plate_text, frame_id, vehicle_bbox, detected_at)
                VALUES (?, ?, ?, ?)
                """,
                (plate_text, frame_id, json.dumps(vehicle_bbox), datetime.now())
            )
            return self.db.cursor.lastrowid
        except Exception as e:
            logger.error(f"Error registering plate detection: {e}")
            return -1

    def init_database(self):
        """Initialize the license plates table"""
        try:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS license_plates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plate_text TEXT NOT NULL,
                    frame_id INTEGER,
                    vehicle_bbox TEXT,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("License plates database initialized")
        except Exception as e:
            logger.error(f"Error initializing LPR database: {e}")


# Global instance
_license_plate_recognition = None


def get_license_plate_recognition() -> LicensePlateRecognition:
    """Get global license plate recognition instance"""
    global _license_plate_recognition
    if _license_plate_recognition is None:
        _license_plate_recognition = LicensePlateRecognition()
    return _license_plate_recognition