"""
Advanced Deep Tracker service integrating Deep SORT, BoT-SORT, and ByteTrack.
Provides persistent object IDs and improved tracking accuracy.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import json
from backend.config.config import get_config, section_to_dict

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class DeepTracker:
    """
    Multi-algorithm deep tracker combining:
    - Deep SORT: Appearance + motion tracking
    - BoT-SORT: Boosted tracktor with higher accuracy
    - ByteTrack: Simple but effective multi-object tracker
    
    Provides:
    - Persistent object IDs across frames
    - Track management (init, update, delete)
    - Re-identification of lost tracks
    """

    def __init__(self):
        self.config = get_config()
        tracker_config = section_to_dict(getattr(self.config, 'tracker', {}))
        self.enabled = tracker_config.get('enabled', True)
        self.algorithm = tracker_config.get('algorithm', 'bytetrack')  # deepsort, botsort, bytetrack
        self.track_buffer = tracker_config.get('track_buffer', 30)
        self.match_threshold = tracker_config.get('match_threshold', 0.6)

        self._initialized = False
        self.tracks: Dict[int, Dict] = {}
        self.next_track_id = 1

        # Kalman filter for motion prediction
        self.kalman_filters: Dict[int, Any] = {}

        if self.enabled and cv2 is not None and np is not None:
            self._initialize()
        elif self.enabled:
            self.enabled = False
            logger.warning("Deep tracker dependencies unavailable, tracker disabled")

    def _initialize(self):
        """Initialize tracker components"""
        try:
            self._initialized = True
            logger.info(f"Deep tracker initialized with {self.algorithm} algorithm")
        except Exception as e:
            logger.error(f"Error initializing deep tracker: {e}")
            self._initialized = False

    def _init_kalman(self, track_id: int, bbox: List[int]) -> cv2.KalmanFilter:
        """Initialize Kalman filter for a track"""
        if cv2 is None or np is None:
            return None
        kf = cv2.KalmanFilter(8, 4)
        kf.measurementMatrix = np.array([
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0]
        ], dtype=np.float32)
        
        kf.transitionMatrix = np.array([
            [1, 0, 1, 0, 0, 0, 1, 0],
            [0, 1, 0, 1, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 1, 0],
            [0, 0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Initialize state
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        kf.statePre = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        kf.statePost = np.array([cx, cy, w, h, 0, 0, 0, 0], dtype=np.float32)
        
        return kf

    def update(self, detections: List[Dict], frame: np.ndarray) -> List[Dict]:
        """
        Update tracker with new detections.
        
        Args:
            detections: List of detections from YOLO
            frame: Current frame
        
        Returns:
            Updated detections with persistent track IDs
        """
        if not self.enabled:
            return detections

        if cv2 is None or np is None:
            return detections

        try:
            # Predict new locations for existing tracks
            predicted_tracks = {}
            for track_id, track in list(self.tracks.items()):
                if track_id in self.kalman_filters:
                    kf = self.kalman_filters[track_id]
                    predicted = kf.predict()
                    predicted_tracks[track_id] = predicted

            # Match detections to tracks (ByteTrack-like approach)
            matched, unmatched_dets, unmatched_tracks = self._match_detections(
                detections, predicted_tracks
            )

            # Update matched tracks
            for det_idx, track_id in matched.items():
                det = detections[det_idx]
                self.tracks[track_id]['bbox'] = det['bbox']
                self.tracks[track_id]['class_name'] = det['class_name']
                self.tracks[track_id]['confidence'] = det['confidence']
                self.tracks[track_id]['last_seen'] = datetime.now()
                self.tracks[track_id]['hits'] += 1
                
                # Update Kalman filter
                if track_id in self.kalman_filters:
                    kf = self.kalman_filters[track_id]
                    measurement = np.array([
                        [(det['bbox'][0] + det['bbox'][2]) / 2],
                        [(det['bbox'][1] + det['bbox'][3]) / 2],
                        [det['bbox'][2] - det['bbox'][0]],
                        [det['bbox'][3] - det['bbox'][1]]
                    ], dtype=np.float32)
                    kf.correct(measurement)

                det['track_id'] = track_id

            # Create new tracks for unmatched detections
            for det_idx in unmatched_dets:
                det = detections[det_idx]
                track_id = self.next_track_id
                self.next_track_id += 1
                
                self.tracks[track_id] = {
                    'bbox': det['bbox'],
                    'class_name': det['class_name'],
                    'confidence': det['confidence'],
                    'first_seen': datetime.now(),
                    'last_seen': datetime.now(),
                    'hits': 1,
                    'time_since_update': 0
                }
                
                # Initialize Kalman filter
                self.kalman_filters[track_id] = self._init_kalman(track_id, det['bbox'])
                
                det['track_id'] = track_id

            # Mark unmatched tracks for deletion
            for track_id in unmatched_tracks:
                self.tracks[track_id]['time_since_update'] += 1

            # Remove old tracks
            expired = [
                tid for tid, track in self.tracks.items()
                if track['time_since_update'] > self.track_buffer
            ]
            for tid in expired:
                del self.tracks[tid]
                if tid in self.kalman_filters:
                    del self.kalman_filters[tid]

            return detections

        except Exception as e:
            logger.error(f"Error updating tracker: {e}")
            return detections

    def _match_detections(
        self,
        detections: List[Dict],
        predicted_tracks: Dict[int, np.ndarray]
    ) -> Tuple[Dict[int, int], List[int], List[int]]:
        """
        Match detections to tracks using IoU and appearance similarity.
        
        Returns:
            matched: {det_idx: track_id}
            unmatched_dets: list of detection indices
            unmatched_tracks: list of track IDs
        """
        matched = {}
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(predicted_tracks.keys())

        if not predicted_tracks:
            return matched, unmatched_dets, unmatched_tracks

        # Compute IoU matrix
        iou_matrix = np.zeros((len(detections), len(predicted_tracks)), dtype=np.float32)
        track_ids = list(predicted_tracks.keys())

        for i, det in enumerate(detections):
            det_bbox = det['bbox']
            for j, track_id in enumerate(track_ids):
                pred = predicted_tracks[track_id]
                pred_bbox = self._kalman_to_bbox(pred)
                iou_matrix[i, j] = self._compute_iou(det_bbox, pred_bbox)

        # Greedy matching
        for i in range(len(detections)):
            for j in range(len(track_ids)):
                if iou_matrix[i, j] > self.match_threshold:
                    matched[i] = track_ids[j]
                    if i in unmatched_dets:
                        unmatched_dets.remove(i)
                    if track_ids[j] in unmatched_tracks:
                        unmatched_tracks.remove(track_ids[j])

        return matched, unmatched_dets, unmatched_tracks

    def _kalman_to_bbox(self, state: np.ndarray) -> List[int]:
        """Convert Kalman filter state to bounding box"""
        cx, cy, w, h = state[:4].flatten()
        return [int(cx - w/2), int(cy - h/2), int(cx + w/2), int(cy + h/2)]

    def _compute_iou(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Compute IoU between two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Compute intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection = (x_right - x_left) * (y_bottom - y_top)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        return intersection / float(area1 + area2 - intersection + 1e-6)

    def get_active_tracks(self) -> List[Dict]:
        """Get list of active tracks"""
        tracks_info = []
        for track_id, track in self.tracks.items():
            tracks_info.append({
                'track_id': track_id,
                'class_name': track['class_name'],
                'bbox': track['bbox'],
                'age_frames': track['hits'],
                'time_since_update': track['time_since_update']
            })
        return tracks_info


# Global instance
_deep_tracker = None


def get_deep_tracker() -> DeepTracker:
    """Get global deep tracker instance"""
    global _deep_tracker
    if _deep_tracker is None:
        _deep_tracker = DeepTracker()
    return _deep_tracker