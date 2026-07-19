"""
Cross-Camera Person Tracking Service
Tracks persons across multiple cameras using appearance features and trajectory prediction
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
import threading

try:
    import numpy as np
except ImportError:
    np = None

try:
    from scipy.spatial.distance import cosine
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from backend.config.config import get_config
from backend.database.db import get_db

logger = logging.getLogger(__name__)


class CrossCameraTracker:
    """
    Cross-camera tracking engine that:
    - Maintains global person identities across all cameras
    - Tracks movement paths through camera network
    - Provides trajectory prediction across cameras
    - Allows targeted tracking by users
    """

    def __init__(self):
        self.db = get_db()
        self.lock = threading.Lock()

        # Global person tracks across all cameras
        self.global_tracks: Dict[str, Dict] = {}  # track_id -> full track data

        # Targeted persons for special tracking
        self.targeted_persons: Dict[str, Dict] = {}  # person_id -> target info

        # Camera transition graph (which cameras connect to which)
        self.camera_graph: Dict[int, List[int]] = {}

        # Appearance features cache
        self.appearance_features: Dict[str, np.ndarray] = {}

    # ==================== Targeted Tracking ====================

    def set_target(self, person_id: str, camera_id: int, reason: str = "") -> str:
        """
        Start tracking a person across all cameras.

        Args:
            person_id: Person identifier
            camera_id: Camera where person was spotted
            reason: Reason for tracking (optional)

        Returns:
            Global track ID
        """
        with self.lock:
            global_track_id = f"track_{datetime.now().strftime('%Y%m%d%H%M%S')}_{person_id}"

            self.targeted_persons[person_id] = {
                'global_track_id': global_track_id,
                'started_at': datetime.now().isoformat(),
                'started_camera': camera_id,
                'reason': reason,
                'path': [],
                'last_seen': datetime.now().isoformat(),
                'last_camera': camera_id,
                'status': 'active'
            }

            logger.info(f"Started targeted tracking for person {person_id}")
            return global_track_id

    def stop_target(self, person_id: str) -> None:
        """Stop targeted tracking of a person"""
        with self.lock:
            if person_id in self.targeted_persons:
                self.targeted_persons[person_id]['status'] = 'stopped'
                self.targeted_persons[person_id]['stopped_at'] = datetime.now().isoformat()
                logger.info(f"Stopped targeted tracking for person {person_id}")

    def is_targeted(self, person_id: str) -> bool:
        """Check if person is currently being tracked"""
        with self.lock:
            return (person_id in self.targeted_persons and 
                    self.targeted_persons[person_id].get('status') == 'active')

    def add_track_point(self, person_id: str, camera_id: int, bbox: List[int], 
                        features: Optional[np.ndarray] = None) -> None:
        """
        Add a position update for a tracked person.

        Updates the path and last seen info.
        """
        with self.lock:
            if person_id not in self.targeted_persons:
                return

            track = self.targeted_persons[person_id]
            if track.get('status') != 'active':
                return

            # Add to path
            track['path'].append({
                'camera_id': camera_id,
                'bbox': bbox,
                'timestamp': datetime.now().isoformat()
            })

            # Keep last 100 points
            if len(track['path']) > 100:
                track['path'] = track['path'][-100:]

            track['last_seen'] = datetime.now().isoformat()
            track['last_camera'] = camera_id

            # Update appearance features
            if features is not None and np is not None:
                self.appearance_features[person_id] = features

    # ==================== Cross-Camera Matching ====================

    def find_cross_camera_match(self, features: np.ndarray, 
                                 current_camera: int) -> Optional[str]:
        """
        Find matching person across cameras based on appearance features.

        Uses cosine similarity to match features.
        """
        if not SCIPY_AVAILABLE or features is None or np is None:
            return None

        best_match = None
        best_score = 1.0  # Lower is better for cosine similarity

        for person_id, cached_features in self.appearance_features.items():
            # Skip if person is already tracked in current camera
            if person_id in self.targeted_persons:
                if self.targeted_persons[person_id].get('last_camera') == current_camera:
                    continue

            # Check if person was recently seen in adjacent cameras
            if self.targeted_persons.get(person_id, {}).get('last_camera') in self._get_adjacent_cameras(current_camera):
                similarity = cosine(features.flatten(), cached_features.flatten())
                if similarity < best_score and similarity < 0.5:  # Threshold for match
                    best_score = similarity
                    best_match = person_id

        return best_match

    def _get_adjacent_cameras(self, camera_id: int) -> List[int]:
        """Get cameras that are spatially adjacent (for transition prediction)"""
        return self.camera_graph.get(camera_id, [])

    def set_camera_graph(self, graph: Dict[int, List[int]]) -> None:
        """Set camera adjacency graph for smarter matching"""
        self.camera_graph = graph

    # ==================== Path Prediction ====================

    def predict_next_camera(self, person_id: str) -> Optional[int]:
        """
        Predict which camera a person will appear in next.

        Based on movement patterns and camera adjacency.
        """
        with self.lock:
            if person_id not in self.targeted_persons:
                return None

            track = self.targeted_persons[person_id]
            path = track.get('path', [])

            if len(path) < 2:
                return None

            # Get last transition
            if len(path) >= 2:
                last_camera = path[-1]['camera_id']
                adjacent = self._get_adjacent_cameras(last_camera)

                # Simple prediction: most likely adjacent camera
                # Could be enhanced with real transition patterns
                if adjacent:
                    return adjacent[0]

            return None

    # ==================== Tracking Path ====================

    def get_tracking_path(self, person_id: str) -> List[Dict]:
        """Get the full movement path of a tracked person"""
        with self.lock:
            if person_id not in self.targeted_persons:
                return []
            return self.targeted_persons[person_id].get('path', [])

    def get_targeted_persons(self) -> Dict[str, Dict]:
        """Get all currently targeted persons"""
        with self.lock:
            return {
                pid: {
                    **data,
                    'total_points': len(data.get('path', []))
                }
                for pid, data in self.targeted_persons.items()
                if data.get('status') == 'active'
            }

    def get_tracking_summary(self) -> Dict:
        """Get summary of all tracking activity"""
        with self.lock:
            active = sum(1 for t in self.targeted_persons.values() if t.get('status') == 'active')
            stopped = sum(1 for t in self.targeted_persons.values() if t.get('status') == 'stopped')

            return {
                'total_tracked': len(self.targeted_persons),
                'active_tracks': active,
                'stopped_tracks': stopped,
                'total_path_points': sum(len(t.get('path', [])) for t in self.targeted_persons.values())
            }


# Global instance
_cross_camera_tracker = None


def get_cross_camera_tracker() -> CrossCameraTracker:
    """Get global cross-camera tracker instance"""
    global _cross_camera_tracker
    if _cross_camera_tracker is None:
        _cross_camera_tracker = CrossCameraTracker()
    return _cross_camera_tracker