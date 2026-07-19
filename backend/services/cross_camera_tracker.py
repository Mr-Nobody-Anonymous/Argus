"""
Cross-Camera Person Tracking Service
Tracks persons across multiple cameras using appearance features and trajectory prediction
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, TYPE_CHECKING
from collections import deque
from datetime import datetime, timedelta
import threading

if TYPE_CHECKING:
    import numpy as np

try:
    import numpy as np
    import cv2
except ImportError:
    np = None
    cv2 = None

try:
    from scipy.spatial.distance import cosine
    from scipy.spatial.distance import cdist
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from sklearn.cluster import DBSCAN
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

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
    - Uses ReID features for better matching
    - Integrates with Kafka for event streaming
    - Supports video file testing
    """

    def __init__(self):
        self.db = get_db()
        self.lock = threading.Lock()
        self.config = get_config()

        # Global person tracks across all cameras
        self.global_tracks: Dict[str, Dict] = {}  # track_id -> full track data

        # Targeted persons for special tracking
        self.targeted_persons: Dict[str, Dict] = {}  # person_id -> target info

        # Camera transition graph (which cameras connect to which)
        self.camera_graph: Dict[int, List[int]] = {}

        # Appearance features cache
        self.appearance_features: Dict[str, np.ndarray] = {}

        # Track history for trajectory analysis
        self.track_history: Dict[str, deque] = {}

        # Statistics
        self.stats = {
            "total_matches": 0,
            "successful_matches": 0,
            "failed_matches": 0,
            "active_tracks": 0,
        }

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

            # Initialize track history
            self.track_history[person_id] = deque(maxlen=1000)

            self.targeted_persons[person_id] = {
                'global_track_id': global_track_id,
                'started_at': datetime.now().isoformat(),
                'started_camera': camera_id,
                'reason': reason,
                'path': [],
                'last_seen': datetime.now().isoformat(),
                'last_camera': camera_id,
                'status': 'active',
                'features_history': [],
                'transition_history': []
            }

            self.stats['active_tracks'] = sum(1 for t in self.targeted_persons.values() if t.get('status') == 'active')

            logger.info(f"Started targeted tracking for person {person_id}")
            return global_track_id

    def stop_target(self, person_id: str) -> None:
        """Stop targeted tracking of a person"""
        with self.lock:
            if person_id in self.targeted_persons:
                self.targeted_persons[person_id]['status'] = 'stopped'
                self.targeted_persons[person_id]['stopped_at'] = datetime.now().isoformat()
                self.stats['active_tracks'] = sum(1 for t in self.targeted_persons.values() if t.get('status') == 'active')
                logger.info(f"Stopped targeted tracking for person {person_id}")

    def is_targeted(self, person_id: str) -> bool:
        """Check if person is currently being tracked"""
        with self.lock:
            return (person_id in self.targeted_persons and 
                    self.targeted_persons[person_id].get('status') == 'active')

    def add_track_point(self, person_id: str, camera_id: int, bbox: List[int], 
                        features: Optional[np.ndarray] = None, confidence: float = 1.0) -> None:
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
                'timestamp': datetime.now().isoformat(),
                'confidence': confidence
            })

            # Keep last 100 points
            if len(track['path']) > 100:
                track['path'] = track['path'][-100:]

            # Track history for trajectory analysis
            self.track_history[person_id].append({
                'camera_id': camera_id,
                'bbox': bbox,
                'timestamp': datetime.now().isoformat(),
            })

            track['last_seen'] = datetime.now().isoformat()
            track['last_camera'] = camera_id

            # Update appearance features with temporal weighting
            if features is not None and np is not None:
                self.appearance_features[person_id] = features.copy()
                track['features_history'].append({
                    'features': features.copy(),
                    'timestamp': datetime.now().isoformat()
                })
                # Keep last 50 feature samples
                if len(track['features_history']) > 50:
                    track['features_history'] = track['features_history'][-50:]

    def get_target_info(self, person_id: str) -> Optional[Dict]:
        """Get detailed info for a targeted person"""
        with self.lock:
            return self.targeted_persons.get(person_id)

    # ==================== Cross-Camera Matching ====================

    def find_cross_camera_match(self, features: np.ndarray, 
                                 current_camera: int,
                                 confidence: float = 0.9) -> Optional[Tuple[str, float]]:
        """
        Find matching person across cameras based on appearance features.

        Uses cosine similarity to match features with enhanced scoring.

        Returns:
            Tuple of (person_id, similarity_score) or None
        """
        if not SCIPY_AVAILABLE or features is None or np is None:
            return None

        self.stats['total_matches'] += 1
        best_match = None
        best_score = 1.0  # Lower is better for cosine similarity

        # Get adjacent cameras for context
        adjacent_cameras = self._get_adjacent_cameras(current_camera)

        for person_id, cached_features in self.appearance_features.items():
            # Skip if person is actively tracked in current camera
            if person_id in self.targeted_persons:
                track = self.targeted_persons[person_id]
                if track.get('status') == 'active' and track.get('last_camera') == current_camera:
                    # But still consider if recently appeared in adjacent camera
                    if track.get('last_camera') not in adjacent_cameras:
                        continue

            similarity = cosine(features.flatten(), cached_features.flatten())

            # Boost confidence for adjacent camera matches
            if person_id in self.targeted_persons:
                last_camera = self.targeted_persons[person_id].get('last_camera')
                if last_camera in adjacent_cameras:
                    similarity *= 0.8  # Boost adjacent camera matches

            if similarity < best_score and similarity < 0.5:  # Threshold for match
                best_score = similarity
                best_match = person_id

        if best_match:
            self.stats['successful_matches'] += 1
            return (best_match, float(best_score))
        else:
            self.stats['failed_matches'] += 1
            return None

    def batch_match_features(self, features_list: List[np.ndarray], 
                             camera_ids: List[int]) -> List[Optional[Tuple[str, float]]]:
        """
        Efficiently match multiple features against cached person features.
        
        Uses scipy's optimized distance calculation for better performance.
        """
        if not SCIPY_AVAILABLE or not features_list or not self.appearance_features:
            return [None] * len(features_list)

        if np is None:
            return [None] * len(features_list)

        # Extract cached features as matrix
        person_ids = list(self.appearance_features.keys())
        cached_features_matrix = np.array([self.appearance_features[pid].flatten() for pid in person_ids])
        input_features_matrix = np.array([f.flatten() for f in features_list])

        # Compute all pairwise distances efficiently
        distances = cdist(input_features_matrix, cached_features_matrix, metric='cosine')

        results = []
        for i, camera_id in enumerate(camera_ids):
            row = distances[i]
            min_idx = np.argmin(row)
            min_score = float(row[min_idx])

            if min_score < 0.5:
                results.append((person_ids[min_idx], min_score))
            else:
                results.append(None)

        return results

    def _get_adjacent_cameras(self, camera_id: int) -> List[int]:
        """Get cameras that are spatially adjacent (for transition prediction)"""
        return self.camera_graph.get(camera_id, [])

    def set_camera_graph(self, graph: Dict[int, List[int]]) -> None:
        """Set camera adjacency graph for smarter matching"""
        self.camera_graph = graph
        logger.info(f"Camera graph set with {len(graph)} nodes")

    # ==================== Path Prediction ====================

    def predict_next_camera(self, person_id: str) -> Optional[int]:
        """
        Predict which camera a person will appear in next.

        Based on movement patterns and camera adjacency with ML enhancement.
        """
        with self.lock:
            if person_id not in self.targeted_persons:
                return None

            track = self.targeted_persons[person_id]
            path = track.get('path', [])

            if len(path) < 2:
                return None

            # Analyze transition patterns
            transitions = track.get('transition_history', [])
            if not transitions:
                transitions = self._extract_transitions(path)

            last_camera = path[-1]['camera_id']
            adjacent = self._get_adjacent_cameras(last_camera)

            # Use frequency-based prediction
            if transitions:
                transition_counts = {}
                for t in transitions:
                    if t['from'] == last_camera:
                        to_camera = t['to']
                        transition_counts[to_camera] = transition_counts.get(to_camera, 0) + 1

                if transition_counts:
                    # Return most frequent next camera
                    return max(transition_counts.keys(), key=lambda k: transition_counts[k])

            # Fallback: Simple adjacency-based prediction
            if adjacent:
                return adjacent[0]

            return None

    def _extract_transitions(self, path: List[Dict]) -> List[Dict]:
        """Extract camera transitions from path history"""
        transitions = []
        for i in range(1, len(path)):
            if path[i]['camera_id'] != path[i-1]['camera_id']:
                transitions.append({
                    'from': path[i-1]['camera_id'],
                    'to': path[i]['camera_id'],
                    'timestamp': path[i]['timestamp']
                })
        return transitions

    def get_trajectory_prediction(self, person_id: str, horizon_seconds: int = 30) -> Optional[Dict]:
        """
        Predict future trajectory for a person.
        
        Returns predicted positions and likely cameras.
        """
        with self.lock:
            if person_id not in self.targeted_persons:
                return None

            track = self.targeted_persons[person_id]
            path = track.get('path', [])

            if len(path) < 3:
                return None

            # Simple linear prediction based on recent movements
            recent_positions = path[-5:] if len(path) >= 5 else path

            # Calculate average movement vector
            total_dx, total_dy = 0, 0
            for i in range(1, len(recent_positions)):
                prev = recent_positions[i-1]
                curr = recent_positions[i]
                bbox_prev = prev.get('bbox', [0, 0, 0, 0])
                bbox_curr = curr.get('bbox', [0, 0, 0, 0])
                total_dx += (bbox_curr[0] + bbox_curr[2]/2) - (bbox_prev[0] + bbox_prev[2]/2)
                total_dy += (bbox_curr[1] + bbox_curr[3]/2) - (bbox_prev[1] + bbox_prev[3]/2)

            avg_dx = total_dx / max(1, len(recent_positions) - 1)
            avg_dy = total_dy / max(1, len(recent_positions) - 1)

            # Predict next position
            last_bbox = path[-1].get('bbox', [0, 0, 100, 100])
            center_x = last_bbox[0] + last_bbox[2] / 2
            center_y = last_bbox[1] + last_bbox[3] / 2

            predicted_positions = []
            for i in range(1, horizon_seconds + 1):
                predicted_positions.append({
                    'x': center_x + avg_dx * i,
                    'y': center_y + avg_dy * i,
                    'timestamp_offset': i,
                    'camera_id': path[-1]['camera_id']
                })

            next_camera = self.predict_next_camera(person_id)

            return {
                'person_id': person_id,
                'predicted_positions': predicted_positions[:10],  # Top 10 predictions
                'predicted_next_camera': next_camera,
                'prediction_horizon_seconds': horizon_seconds,
                'confidence': 0.7 if len(path) > 5 else 0.4
            }

    def cluster_trajectories(self, threshold: float = 0.5) -> List[List[str]]:
        """
        Cluster persons with similar movement trajectories using DBSCAN.
        
        Returns clusters of person IDs.
        """
        if not SKLEARN_AVAILABLE or not self.track_history:
            return []

        # Extract trajectory features for each person
        trajectory_features = []
        person_ids = []

        for person_id, history in self.track_history.items():
            if len(history) >= 5:
                # Create feature vector: average position, movement patterns
                positions = [(h['bbox'][0] + h['bbox'][2]/2, h['bbox'][1] + h['bbox'][3]/2) for h in history]
                
                avg_x = sum(p[0] for p in positions) / len(positions)
                avg_y = sum(p[1] for p in positions) / len(positions)
                
                trajectory_features.append([avg_x, avg_y])
                person_ids.append(person_id)

        if len(trajectory_features) < 2:
            return []

        # Apply DBSCAN clustering
        features_array = np.array(trajectory_features)
        clustering = DBSCAN(eps=threshold, min_samples=2).fit(features_array)

        # Group persons by cluster
        clusters = {}
        for i, label in enumerate(clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(person_ids[i])

        return [clusters[label] for label in clusters if label >= 0]

    # ==================== Video Testing ====================

    def process_video_file(self, video_path: str, camera_ids: List[int], duration_seconds: int = 60) -> Dict:
        """
        Process a video file for cross-camera tracking testing.
        
        Simulates processing the video through multiple camera views
        for testing cross-camera tracking capabilities.
        """
        if cv2 is None:
            return {"status": "error", "message": "OpenCV not available"}
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return {"status": "error", "message": "Could not open video file"}
            
            frame_count = 0
            processed_frames = 0
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Process frames from the video
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Simulate cross-camera tracking every 10 frames
                if frame_count % 10 == 0 and camera_ids:
                    # Rotate through camera IDs for testing
                    current_camera_idx = (frame_count // 10) % len(camera_ids)
                    camera_id = camera_ids[current_camera_idx]
                    
                    # Simulate detection with random person ID for testing
                    person_id = f"test_person_{camera_id}_{frame_count}"
                    
                    # Add track point with mock features
                    if np is not None:
                        mock_features = np.random.rand(512).astype(np.float32)
                        bbox = [100, 100, 200, 400]  # Mock bounding box
                        
                        # Set as target if not already tracked
                        if person_id not in self.targeted_persons:
                            self.set_target(person_id, camera_id, "video_test")
                        
                        self.add_track_point(person_id, camera_id, bbox, mock_features)
                    
                    processed_frames += 1
                
                # Stop after specified duration
                if frame_count >= int(fps * duration_seconds):
                    break
            
            cap.release()
            
            return {
                "status": "success",
                "frames_processed": processed_frames,
                "total_frames": frame_count,
                "targets_created": len(self.targeted_persons),
                "message": f"Processed {processed_frames} test frames from video"
            }
            
        except Exception as e:
            logger.error(f"Error processing video file: {e}")
            return {"status": "error", "message": str(e)}

    def get_clusters(self) -> List[Dict]:
        """
        Get trajectory clusters with person details.
        Useful for video testing analysis.
        """
        clusters = self.cluster_trajectories()
        result = []
        
        for i, cluster in enumerate(clusters):
            cluster_info = {
                "cluster_id": i,
                "person_count": len(cluster),
                "persons": []
            }
            
            for person_id in cluster:
                track = self.targeted_persons.get(person_id, {})
                cluster_info["persons"].append({
                    "person_id": person_id,
                    "path_points": len(track.get('path', [])),
                    "cameras": list(set(p.get('camera_id') for p in track.get('path', [])))
                })
            
            result.append(cluster_info)
        
        return result

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
                'total_path_points': sum(len(t.get('path', [])) for t in self.targeted_persons.values()),
                'matching_stats': self.stats,
                'camera_graph_nodes': len(self.camera_graph)
            }

    def get_tracker_statistics(self) -> Dict:
        """Get enhanced statistics for the learning dashboard"""
        with self.lock:
            active_persons = self.get_targeted_persons()
            
            # Calculate trajectory patterns
            cameras_with_tracks = set()
            for track in active_persons.values():
                for point in track.get('path', []):
                    cameras_with_tracks.add(point.get('camera_id'))

            return {
                'total_tracked': len(self.targeted_persons),
                'active_tracks': self.stats['active_tracks'],
                'total_matches': self.stats['total_matches'],
                'successful_matches': self.stats['successful_matches'],
                'match_success_rate': round(
                    self.stats['successful_matches'] / max(1, self.stats['total_matches']) * 100, 2
                ),
                'cameras_in_graph': len(self.camera_graph),
                'cameras_with_active_tracks': len(cameras_with_tracks),
                'avg_path_length': round(
                    sum(len(t.get('path', [])) for t in active_persons.values()) / 
                    max(1, len(active_persons)), 2
                ) if active_persons else 0
            }

    def clear_old_tracks(self, max_age_hours: int = 24) -> int:
        """Clear tracks older than specified hours"""
        with self.lock:
            cutoff = datetime.now() - timedelta(hours=max_age_hours)
            to_remove = []

            for person_id, track in self.targeted_persons.items():
                last_seen = datetime.fromisoformat(track.get('last_seen', '2000-01-01'))
                if last_seen < cutoff and track.get('status') == 'active':
                    to_remove.append(person_id)

            for person_id in to_remove:
                self.stop_target(person_id)

            return len(to_remove)


# Global instance
_cross_camera_tracker = None


def get_cross_camera_tracker() -> CrossCameraTracker:
    """Get global cross-camera tracker instance"""
    global _cross_camera_tracker
    if _cross_camera_tracker is None:
        _cross_camera_tracker = CrossCameraTracker()
    return _cross_camera_tracker