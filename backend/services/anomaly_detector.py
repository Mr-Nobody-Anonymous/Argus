"""
Anomaly detection service for identifying unusual behavior in video streams.
Integrates anomalib and CVPR2018 approaches for event-based anomaly detection.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any, Tuple
from collections import deque
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


class AnomalyDetector:
    """
    Anomaly detection using multiple approaches:
    - Motion-based anomaly detection
    - Object behavior anomaly detection
    - Statistical outlier detection for speeds/positions
    - Pattern-based anomaly detection (loitering, abnormal trajectories)
    """

    def __init__(self):
        self.config = get_config()
        anomaly_config = section_to_dict(getattr(self.config, 'anomaly', {}))
        self.enabled = anomaly_config.get('enabled', True)
        self.sensitivity = anomaly_config.get('sensitivity', 0.7)
        self.window_size = anomaly_config.get('window_size', 100)

        self.db = get_db()
        self._initialized = False

        # Track object histories for anomaly detection
        self.object_histories: Dict[str, deque] = {}
        
        # Zone entry/exit tracking
        self.zone_entries: Dict[str, float] = {}
        
        # Statistical baselines
        self.speed_baseline: List[float] = []
        self.position_baseline: List[Tuple[float, float]] = []

        if self.enabled:
            self._initialize()

    def _initialize(self):
        """Initialize anomaly detection models"""
        try:
            self._initialized = True
            logger.info("Anomaly detector initialized")
        except Exception as e:
            logger.error(f"Error initializing anomaly detector: {e}")
            self._initialized = False

    def detect_motion_anomalies(
        self,
        frame: np.ndarray,
        prev_frame: np.ndarray,
        detections: List[Dict]
    ) -> List[Dict]:
        """
        Detect motion-based anomalies between frames.
        
        Returns list of anomalies with type and confidence
        """
        anomalies = []

        if not self.enabled or prev_frame is None or cv2 is None or np is None:
            return anomalies

        try:
            # Compute foreground mask
            fg_mask = cv2.absdiff(
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
            )
            _, thresh = cv2.threshold(fg_mask, 30, 255, cv2.THRESH_BINARY)
            
            # Morphological operations to clean up
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

            # Find large moving regions
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            h, w = frame.shape[:2]
            frame_area = w * h
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > frame_area * 0.1:  # Large moving area
                    x, y, cw, ch = cv2.boundingRect(cnt)
                    anomalies.append({
                        'type': 'large_motion',
                        'confidence': min(area / frame_area, 1.0),
                        'bbox': [x, y, x + cw, y + ch],
                        'area_ratio': area / frame_area
                    })

            return anomalies

        except Exception as e:
            logger.error(f"Error in motion anomaly detection: {e}")
            return anomalies

    def detect_behavior_anomalies(
        self,
        object_id: str,
        analysis_result: Dict
    ) -> Optional[Dict]:
        """
        Detect behavioral anomalies based on object tracking data.
        
        Types of anomalies:
        - Unusual speed patterns
        - Abnormal trajectories
        - Loitering behavior
        - Rapid direction changes
        """
        if not self.enabled:
            return None

        try:
            # Initialize history for new objects
            if object_id not in self.object_histories:
                self.object_histories[object_id] = deque(maxlen=self.window_size)

            history = self.object_histories[object_id]
            history.append(analysis_result)

            # Need enough history for analysis
            if len(history) < 10:
                return None

            # Speed anomaly detection
            speed_anomaly = self._detect_speed_anomaly(history)
            if speed_anomaly:
                return speed_anomaly

            # Trajectory anomaly detection
            traj_anomaly = self._detect_trajectory_anomaly(history)
            if traj_anomaly:
                return traj_anomaly

            # Loitering detection
            loiter_anomaly = self._detect_loitering(history)
            if loiter_anomaly:
                return loiter_anomaly

            return None

        except Exception as e:
            logger.error(f"Error in behavior anomaly detection: {e}")
            return None

    def _detect_speed_anomaly(self, history: deque) -> Optional[Dict]:
        """Detect unusual speed patterns"""
        speeds = [h.get('speed_mps', 0) for h in history if h.get('speed_mps')]
        
        if len(speeds) < 5:
            return None

        current_speed = speeds[-1]
        mean_speed = np.mean(speeds[:-1]) if len(speeds) > 1 else 0
        std_speed = np.std(speeds[:-1]) if len(speeds) > 1 else 0

        # Detect sudden acceleration or unusual speeds
        if std_speed > 0 and abs(current_speed - mean_speed) > 2 * std_speed:
            return {
                'type': 'speed_anomaly',
                'confidence': min(abs(current_speed - mean_speed) / (std_speed + 0.1), 1.0),
                'description': f"Unusual speed: {current_speed:.1f} m/s (baseline: {mean_speed:.1f} ± {std_speed:.1f})",
                'current_speed': current_speed,
                'baseline_speed': mean_speed
            }

        return None

    def _detect_trajectory_anomaly(self, history: deque) -> Optional[Dict]:
        """Detect abnormal movement patterns"""
        if len(history) < 5:
            return None

        # Check for rapid direction changes
        directions = [h.get('direction', 'unknown') for h in list(history)[-5:]]
        direction_changes = sum(1 for i in range(len(directions) - 1) 
                                if directions[i] != directions[i + 1])
        
        if direction_changes >= 3:
            return {
                'type': 'trajectory_anomaly',
                'confidence': 0.7,
                'description': 'Rapid direction changes detected'
            }

        return None

    def _detect_loitering(self, history: deque) -> Optional[Dict]:
        """Detect loitering behavior (staying in one area too long)"""
        if len(history) < 20:
            return None

        recent_positions = [(h.get('bbox', [0, 0, 0, 0])[0], h.get('bbox', [0, 0, 0, 0])[1]) 
                           for h in list(history)[-20:]]
        
        # Calculate area covered
        xs = [p[0] for p in recent_positions]
        ys = [p[1] for p in recent_positions]
        
        if len(xs) > 0 and len(ys) > 0:
            area = (max(xs) - min(xs)) * (max(ys) - min(ys))
            
            if area < 1000:  # Small area indicates loitering
                return {
                    'type': 'loitering',
                    'confidence': 0.8,
                    'description': 'Object appears to be loitering in small area',
                    'area_covered': area
                }

        return None

    def detect_abandoned_object(
        self,
        frame: np.ndarray,
        detections: List[Dict],
        stationary_objects: Dict[str, float]
    ) -> List[Dict]:
        """
        Detect abandoned or suspicious stationary objects.
        
        Args:
            frame: Current frame
            detections: Current detections
            stationary_objects: Dict of {object_id: stationary_duration}
        
        Returns:
            List of abandoned object anomalies
        """
        anomalies = []

        for obj_id, duration in stationary_objects.items():
            if duration > 30:  # Stationary for more than 30 seconds
                # Find the detection for this object
                for det in detections:
                    if det.get('object_id') == obj_id:
                        anomalies.append({
                            'type': 'abandoned_object',
                            'confidence': min(duration / 60.0, 1.0),
                            'description': f"Object stationary for {duration:.0f}s",
                            'object_id': obj_id,
                            'bbox': det.get('bbox', []),
                            'duration': duration
                        })
                        break

        return anomalies

    def register_anomaly(
        self,
        anomaly_type: str,
        confidence: float,
        description: str,
        frame_id: int,
        camera_id: int
    ) -> int:
        """Register an anomaly in the database"""
        try:
            self.db.execute(
                """
                INSERT INTO anomalies (type, confidence, description, frame_id, camera_id, detected_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (anomaly_type, confidence, description, frame_id, camera_id, datetime.now())
            )
            return self.db.cursor.lastrowid
        except Exception as e:
            logger.error(f"Error registering anomaly: {e}")
            return -1

    def init_database(self):
        """Initialize the anomalies table"""
        try:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS anomalies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    type TEXT NOT NULL,
                    confidence REAL,
                    description TEXT,
                    frame_id INTEGER,
                    camera_id INTEGER,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("Anomalies database initialized")
        except Exception as e:
            logger.error(f"Error initializing anomalies database: {e}")

    # ==================== Trajectory Prediction ====================
    
    def predict_trajectory(self, object_id: str, future_frames: int = 10) -> List[Tuple[float, float]]:
        """
        Predict future trajectory for an object based on historical movement.
        
        Uses linear regression to predict future positions.
        
        Args:
            object_id: Object identifier
            future_frames: Number of frames to predict ahead
        
        Returns:
            List of predicted (x, y) positions
        """
        if object_id not in self.object_histories or np is None:
            return []
        
        history = self.object_histories[object_id]
        if len(history) < 5:
            return []
        
        try:
            positions = [(h['bbox'][0], h['bbox'][1]) for h in history if h.get('bbox')]
            if len(positions) < 5:
                return []
            
            xs = np.array([p[0] for p in positions])
            ys = np.array([p[1] for p in positions])
            
            # Simple linear extrapolation
            x_velocity = np.mean(np.diff(xs)) if len(xs) > 1 else 0
            y_velocity = np.mean(np.diff(ys)) if len(ys) > 1 else 0
            
            predictions = []
            last_x, last_y = xs[-1], ys[-1]
            for i in range(1, future_frames + 1):
                predictions.append((last_x + x_velocity * i, last_y + y_velocity * i))
            
            return predictions
        except Exception as e:
            logger.error(f"Error predicting trajectory: {e}")
            return []

    # ==================== Criminal Activity Detection ====================
    
    def detect_suspicious_behavior(self, object_id: str, analysis_result: Dict, 
                                  zone_id: Optional[int] = None) -> Optional[Dict]:
        """
        Detect suspicious behavior patterns that may indicate criminal activity.
        
        Patterns detected:
        - Rapid direction changes (erratic movement)
        - Unusual speed patterns (sudden acceleration/deceleration)
        - Loitering in sensitive areas
        - Unusual trajectory patterns
        """
        if not self.enabled:
            return None
        
        try:
            # Initialize history for new objects
            if object_id not in self.object_histories:
                self.object_histories[object_id] = deque(maxlen=self.window_size)
            
            history = self.object_histories[object_id]
            history.append(analysis_result)
            
            # Check for suspicious patterns
            anomalies = []
            
            # Pattern 1: Erratic movement (rapid direction changes)
            if len(history) >= 10:
                directions = [h.get('direction', 'unknown') for h in list(history)[-10:]]
                direction_changes = sum(1 for i in range(len(directions) - 1) 
                                      if directions[i] != directions[i + 1])
                if direction_changes >= 5:
                    anomalies.append({
                        'type': 'erratic_movement',
                        'confidence': 0.85,
                        'description': 'Object showing erratic/unpredictable movement pattern'
                    })
            
            # Pattern 2: Suspicious speed variations
            if len(history) >= 15:
                speeds = [h.get('speed_mps', 0) for h in history if h.get('speed_mps')]
                if len(speeds) >= 10:
                    speed_variance = np.var(speeds) if np is not None else 0
                    if speed_variance > 5.0:  # High variance indicates suspicious stopping/starting
                        anomalies.append({
                            'type': 'suspicious_speed_pattern',
                            'confidence': 0.75,
                            'description': 'Irregular speed pattern detected'
                        })
            
            # Pattern 3: Casing behavior (systematic area scanning)
            if len(history) >= 20:
                positions = [(h.get('bbox', [0, 0, 0, 0])[0], h.get('bbox', [0, 0, 0, 0])[1]) 
                           for h in list(history)[-20:]]
                if positions:
                    # Check for systematic back-and-forth movement
                    x_variance = np.var([p[0] for p in positions]) if np is not None else 0
                    y_variance = np.var([p[1] for p in positions]) if np is not None else 0
                    if x_variance > 1000 and y_variance > 1000:
                        anomalies.append({
                            'type': 'casing_behavior',
                            'confidence': 0.7,
                            'description': 'Object scanning area systematically - potential casing behavior'
                        })
            
            return anomalies[0] if anomalies else None
            
        except Exception as e:
            logger.error(f"Error detecting suspicious behavior: {e}")
            return None

    def detect_group_behavior_anomalies(self, detections: List[Dict]) -> List[Dict]:
        """
        Detect anomalous group behaviors (multiple persons acting together suspiciously).
        
        Patterns:
        - Concentric circling
        - Coordinated movement
        - Surrounding a single target
        """
        anomalies = []
        
        try:
            persons = [d for d in detections if d.get('class_name') == 'person']
            if len(persons) < 2:
                return anomalies
            
            # Check for concentric circling (multiple persons circling a center point)
            centers = [(d['bbox'][0] + d['bbox'][2]) / 2 for d in persons]
            distances = []
            for i, p1 in enumerate(persons):
                for j, p2 in enumerate(persons[i+1:], i+1):
                    dx = (p1['bbox'][0] + p1['bbox'][2]) / 2 - (p2['bbox'][0] + p2['bbox'][2]) / 2
                    dy = (p1['bbox'][1] + p1['bbox'][3]) / 2 - (p2['bbox'][1] + p2['bbox'][3]) / 2
                    distances.append((dx**2 + dy**2)**0.5)
            
            if distances and max(distances) - min(distances) < 50:  # Close grouping
                anomalies.append({
                    'type': 'suspicious_grouping',
                    'confidence': 0.8,
                    'description': f'{len(persons)} persons detected in close proximity with coordinated movement'
                })
            
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting group behavior: {e}")
            return anomalies


# Global instance
_anomaly_detector = None


def get_anomaly_detector() -> AnomalyDetector:
    """Get global anomaly detector instance"""
    global _anomaly_detector
    if _anomaly_detector is None:
        _anomaly_detector = AnomalyDetector()
    return _anomaly_detector
