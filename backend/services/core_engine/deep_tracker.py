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
        # Fallback association radius for the centre-distance stage, expressed
        # as a multiple of the detection's own diagonal so it scales with how
        # near or far the object is. 1.5 tolerates the ~1s inter-frame gap of
        # CPU inference without merging genuinely distinct neighbours.
        self.max_center_distance = tracker_config.get('max_center_distance', 1.5)

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
        
        # Constant-velocity model. State is [cx, cy, w, h, vx, vy, vw, vh], so
        # each position term must advance by its OWN velocity:
        #     cx' = cx + vx      w' = w + vw
        # The previous matrix coupled position to size (cx' = cx + w + vw),
        # which made a stationary box "jump" every predict() — IoU then never
        # exceeded match_threshold and every detection spawned a new track id.
        kf.transitionMatrix = np.array([
            [1, 0, 0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1]
        ], dtype=np.float32)

        # OpenCV leaves errorCovPost zeroed, which makes the filter behave as if
        # the initial state were perfectly known and suppresses correction.
        kf.processNoiseCov = np.eye(8, dtype=np.float32) * 1e-2
        kf.measurementNoiseCov = np.eye(4, dtype=np.float32) * 1e-1
        kf.errorCovPost = np.eye(8, dtype=np.float32)
        
        # Initialize state
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        
        # NOTE: OpenCV expects the state to be a column vector of shape (8, 1).
        # A flat (8,) array makes predict() fail inside gemm with
        # "Assertion failed a_size.width == len", which silently kills tracking.
        kf.statePre = np.array(
            [[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32
        )
        kf.statePost = np.array(
            [[cx], [cy], [w], [h], [0], [0], [0], [0]], dtype=np.float32
        )
        
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

        # Greedy matching, best-IoU-first.
        # The pairs are sorted by descending IoU and each detection/track is
        # consumed at most once. The previous row-major scan let a single track
        # be assigned to several detections (last write wins) and could claim a
        # weak pair before a stronger one was considered.
        pairs = [
            (iou_matrix[i, j], i, j)
            for i in range(len(detections))
            for j in range(len(track_ids))
            if iou_matrix[i, j] > self.match_threshold
        ]
        pairs.sort(reverse=True)

        used_tracks = set()
        for _iou, i, j in pairs:
            track_id = track_ids[j]
            if i in matched or track_id in used_tracks:
                continue
            matched[i] = track_id
            used_tracks.add(track_id)
            if i in unmatched_dets:
                unmatched_dets.remove(i)
            if track_id in unmatched_tracks:
                unmatched_tracks.remove(track_id)

        # ── Second association stage: centre distance ──
        #
        # IoU alone only associates boxes that still physically overlap. That
        # holds when frames arrive back-to-back, but the pipeline analyses
        # roughly one frame per second on CPU while cameras run at 15-30 fps,
        # and in one second a walking person moves clear of their previous box.
        # IoU is then 0 for every pair, every detection looks new, and identity
        # churns: measured 89 distinct IDs across 10 processed frames of a
        # ~12-person scene, versus 15 when frames were consecutive.
        #
        # Churned IDs break everything keyed on identity - dwell time never
        # accumulates (loitering silently stops firing) while entry events
        # re-fire for the same person on every frame.
        #
        # So unmatched pairs get a second chance on centre distance, scaled by
        # object size (a box twice as large may move twice as far) and gated on
        # class so a person is never absorbed into a car's track.
        if unmatched_dets and unmatched_tracks:
            distance_pairs = []
            for i in list(unmatched_dets):
                det = detections[i]
                dx1, dy1, dx2, dy2 = det['bbox']
                det_cx, det_cy = (dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0
                det_diag = max(1.0, ((dx2 - dx1) ** 2 + (dy2 - dy1) ** 2) ** 0.5)

                for track_id in list(unmatched_tracks):
                    track = self.tracks.get(track_id)
                    if track is None:
                        continue
                    # Never merge across object classes.
                    if track.get('class_name') != det.get('class_name'):
                        continue

                    px1, py1, px2, py2 = self._kalman_to_bbox(predicted_tracks[track_id])
                    pred_cx, pred_cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                    distance = ((det_cx - pred_cx) ** 2 + (det_cy - pred_cy) ** 2) ** 0.5

                    # Reject implausible size changes - a genuine match keeps
                    # roughly the same scale between observations.
                    pred_diag = max(1.0, ((px2 - px1) ** 2 + (py2 - py1) ** 2) ** 0.5)
                    ratio = det_diag / pred_diag
                    if ratio < 0.5 or ratio > 2.0:
                        continue

                    normalised = distance / det_diag
                    if normalised <= self.max_center_distance:
                        distance_pairs.append((normalised, i, track_id))

            # Closest pairs win, each detection and track used at most once.
            distance_pairs.sort()
            for _dist, i, track_id in distance_pairs:
                if i in matched or track_id in used_tracks:
                    continue
                matched[i] = track_id
                used_tracks.add(track_id)
                if i in unmatched_dets:
                    unmatched_dets.remove(i)
                if track_id in unmatched_tracks:
                    unmatched_tracks.remove(track_id)

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