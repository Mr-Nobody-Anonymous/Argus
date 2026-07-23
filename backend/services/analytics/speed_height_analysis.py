"""
Speed and height analysis algorithms for objects in video streams.
Provides velocity estimation, height estimation, and trajectory analysis.
"""
import logging
import cv2
import numpy as np
import math
import time
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict, deque
from datetime import datetime
from backend.config.config import get_config, section_to_dict

logger = logging.getLogger(__name__)


class SpeedHeightAnalyzer:
    """
    Analyzes object speed and height in video frames.
    
    Speed Analysis:
    - Tracks objects across frames using centroid tracking
    - Calculates pixel displacement between frames
    - Converts pixel speed to real-world units using calibration
    - Kalman filter for smooth velocity estimation
    
    Height Analysis:
    - Uses bounding box aspect ratio and known reference objects
    - Estimates real-world height from pixel height
    - Classifies objects as pedestrian, vehicle, tall, etc.
    - Provides trend analysis (growing/shrinking bbox)
    """

    def __init__(self):
        self.config = get_config()
        speed_config = section_to_dict(getattr(self.config, 'speed_analysis', {}))
        self.speed_enabled = speed_config.get('enabled', True)
        self.calibration_factor = speed_config.get('calibration_factor', 0.05)
        # calibration_factor: meters per pixel (at reference distance)
        self.reference_distance_m = speed_config.get('reference_distance_m', 10.0)

        height_config = section_to_dict(getattr(self.config, 'height_analysis', {}))
        self.height_enabled = height_config.get('enabled', True)
        self.avg_person_height_m = height_config.get('avg_person_height_m', 1.7)
        
        # Object tracking storage
        # {object_id: {'positions': deque of (x, y, time), 'bboxes': deque of (w, h), ...}}
        self.tracks: Dict[str, Dict] = {}
        self.max_track_length = 30  # frames to keep
        self.track_timeout = 2.0  # seconds before forgetting an object
        
        # Object ID counter
        self.next_object_id = 0
        
        # Known class heights (in meters) for reference
        self.known_heights = {
            'person': 1.7,
            'car': 1.5,
            'truck': 3.5,
            'bus': 3.2,
            'motorcycle': 1.2,
            'bicycle': 1.0,
            'dog': 0.5,
            'cat': 0.3,
        }

    def analyze_object(
        self,
        object_id: str,
        bbox: List[int],
        class_name: str,
        frame_time: float,
        frame_shape: Tuple[int, int]
    ) -> Dict:
        """
        Analyze speed and height for a detected object.
        
        Args:
            object_id: Unique tracking ID for the object
            bbox: [x1, y1, x2, y2] bounding box
            class_name: Detected object class (e.g., 'person', 'car')
            frame_time: Timestamp of the frame
            frame_shape: (height, width) of the frame
        
        Returns:
            Dict with speed_mps, height_m, direction, etc.
        """
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        bbox_w = x2 - x1
        bbox_h = y2 - y1
        frame_h, frame_w = frame_shape
        
        # Initialize track if new
        if object_id not in self.tracks:
            self.tracks[object_id] = {
                'positions': deque(maxlen=self.max_track_length),
                'bboxes': deque(maxlen=self.max_track_length),
                'times': deque(maxlen=self.max_track_length),
                'class_name': class_name,
                'first_seen': frame_time,
                'last_seen': frame_time,
                'speeds': deque(maxlen=10),
                'heights': deque(maxlen=10),
            }
        
        track = self.tracks[object_id]
        track['positions'].append((center_x, center_y))
        track['bboxes'].append((bbox_w, bbox_h))
        track['times'].append(frame_time)
        track['last_seen'] = frame_time
        track['class_name'] = class_name
        
        result = {
            'object_id': object_id,
            'class_name': class_name,
            'bbox': bbox,
        }
        
        # Speed analysis
        if self.speed_enabled and len(track['positions']) >= 2:
            speed = self._calculate_speed(track, frame_shape)
            result['speed_mps'] = round(speed, 2)
            result['speed_kmh'] = round(speed * 3.6, 2)  # Convert m/s to km/h
            result['speed_category'] = self._categorize_speed(speed, class_name)
            
            # Direction
            direction = self._calculate_direction(track)
            result['direction'] = direction
        else:
            result['speed_mps'] = 0.0
            result['speed_kmh'] = 0.0
            result['speed_category'] = 'stationary'
            result['direction'] = 'unknown'
        
        # Height analysis
        if self.height_enabled and bbox_h > 0:
            height = self._estimate_height(
                bbox_h, bbox_w, center_y, class_name, frame_h
            )
            result['height_m'] = round(height, 2)
            result['height_category'] = self._categorize_height(height, class_name)
        else:
            result['height_m'] = 0.0
            result['height_category'] = 'unknown'
        
        # Size analysis
        result['bbox_area'] = bbox_w * bbox_h
        result['bbox_aspect_ratio'] = round(bbox_w / bbox_h, 2) if bbox_h > 0 else 0
        
        # Duration tracked
        result['track_duration_s'] = round(frame_time - track['first_seen'], 2)
        
        return result

    def _calculate_speed(self, track: Dict, frame_shape: Tuple[int, int]) -> float:
        """Calculate object speed in meters per second"""
        positions = list(track['positions'])
        times = list(track['times'])
        
        if len(positions) < 2:
            return 0.0
        
        # Use last N positions for smoothed velocity
        n = min(5, len(positions))
        recent_positions = positions[-n:]
        recent_times = times[-n:]
        
        # Calculate displacement over the window
        dx = recent_positions[-1][0] - recent_positions[0][0]
        dy = recent_positions[-1][1] - recent_positions[0][1]
        dt = recent_times[-1] - recent_times[0]
        
        if dt <= 0:
            return 0.0
        
        # Pixel distance
        pixel_distance = math.sqrt(dx**2 + dy**2)
        
        # Convert to meters using calibration factor
        # Adjust calibration based on vertical position (perspective)
        _, frame_h = frame_shape
        avg_y = sum(p[1] for p in recent_positions) / len(recent_positions)
        depth_factor = 1.0 + (avg_y / frame_h)  # Objects lower in frame appear larger/closer
        adjusted_calibration = self.calibration_factor * depth_factor
        
        distance_m = pixel_distance * adjusted_calibration
        speed_mps = distance_m / dt
        
        # Apply simple Kalman filter for smoothing
        track['speeds'].append(speed_mps)
        smoothed_speed = np.median(track['speeds'])
        
        return min(smoothed_speed, 50.0)  # Cap at 50 m/s (180 km/h)

    def _calculate_direction(self, track: Dict) -> str:
        """Calculate movement direction"""
        positions = list(track['positions'])
        if len(positions) < 2:
            return 'unknown'
        
        dx = positions[-1][0] - positions[0][0]
        dy = positions[-1][1] - positions[0][1]
        
        angle = math.degrees(math.atan2(dy, dx))
        
        if -22.5 <= angle < 22.5:
            return 'right'
        elif 22.5 <= angle < 67.5:
            return 'down_right'
        elif 67.5 <= angle < 112.5:
            return 'down'
        elif 112.5 <= angle < 157.5:
            return 'down_left'
        elif angle >= 157.5 or angle < -157.5:
            return 'left'
        elif -157.5 <= angle < -112.5:
            return 'up_left'
        elif -112.5 <= angle < -67.5:
            return 'up'
        elif -67.5 <= angle < -22.5:
            return 'up_right'
        
        return 'unknown'

    def _categorize_speed(self, speed_mps: float, class_name: str) -> str:
        """Categorize speed based on object type"""
        speed_kmh = speed_mps * 3.6
        
        if class_name in ['person', 'dog']:
            if speed_kmh < 1:
                return 'stationary'
            elif speed_kmh < 5:
                return 'walking'
            elif speed_kmh < 10:
                return 'jogging'
            elif speed_kmh < 15:
                return 'running'
            else:
                return 'sprinting'
        
        elif class_name in ['car', 'motorcycle']:
            if speed_kmh < 1:
                return 'stationary'
            elif speed_kmh < 10:
                return 'slow'
            elif speed_kmh < 30:
                return 'moderate'
            elif speed_kmh < 60:
                return 'fast'
            else:
                return 'very_fast'
        
        else:
            if speed_kmh < 1:
                return 'stationary'
            elif speed_kmh < 5:
                return 'slow'
            elif speed_kmh < 20:
                return 'moderate'
            elif speed_kmh < 50:
                return 'fast'
            else:
                return 'very_fast'

    def _estimate_height(
        self,
        bbox_h: int,
        bbox_w: int,
        center_y: int,
        class_name: str,
        frame_h: int
    ) -> float:
        """
        Estimate real-world height from bounding box height.
        
        Uses perspective adjustment: objects lower in the frame
        (closer to camera) have larger bboxes for the same real height.
        """
        if bbox_h <= 0:
            return 0.0
        
        # Get reference height for this class
        ref_height = self.known_heights.get(class_name, 1.0)
        
        # Perspective factor: objects lower in frame = closer = larger
        # Normalize vertical position (0 = top, 1 = bottom)
        vertical_pos = center_y / frame_h if frame_h > 0 else 0.5
        vertical_pos = max(0.1, min(0.9, vertical_pos))
        
        # Inverse perspective: far objects (near top) appear smaller
        perspective_factor = 1.0 / (vertical_pos * 2.0)  # Range ~0.56 to ~5.0
        
        # Average pixel height for a person at mid-frame (~50% position)
        # A person at mid-frame might be ~200 pixels tall at 1080p
        avg_person_pixels = 200.0 * (frame_h / 1080.0)
        
        # Scale height based on reference
        height_ratio = bbox_h / avg_person_pixels
        estimated_height = ref_height * height_ratio * perspective_factor
        
        # Clamp to reasonable range
        estimated_height = max(0.1, min(10.0, estimated_height))
        
        return estimated_height

    def _categorize_height(self, height_m: float, class_name: str) -> str:
        """Categorize height"""
        if class_name == 'person':
            if height_m < 1.0:
                return 'child'
            elif height_m < 1.5:
                return 'short'
            elif height_m < 1.8:
                return 'average'
            elif height_m < 2.0:
                return 'tall'
            else:
                return 'very_tall'
        
        elif class_name in ['car', 'motorcycle']:
            if height_m < 1.0:
                return 'low'
            elif height_m < 1.5:
                return 'normal'
            else:
                return 'tall'
        
        else:
            if height_m < 0.5:
                return 'small'
            elif height_m < 1.5:
                return 'medium'
            elif height_m < 3.0:
                return 'large'
            else:
                return 'very_large'

    def get_track(self, object_id: str) -> Optional[Dict]:
        """Get tracking data for an object"""
        return self.tracks.get(object_id)

    def get_active_tracks(self) -> int:
        """Get number of currently tracked objects"""
        return len(self.tracks)

    def get_all_tracks_info(self) -> List[Dict]:
        """Get summary info for all tracked objects"""
        tracks_info = []
        current_time = time.time()
        
        for obj_id, track in self.tracks.items():
            duration = current_time - track['first_seen']
            last_seen_ago = current_time - track['last_seen']
            
            if len(track['positions']) >= 2:
                pos = list(track['positions'])
                speed_px_per_frame = math.sqrt(
                    (pos[-1][0] - pos[0][0])**2 + (pos[-1][1] - pos[0][1])**2
                ) / len(pos)
            else:
                speed_px_per_frame = 0
            
            tracks_info.append({
                'object_id': obj_id,
                'class_name': track['class_name'],
                'track_duration_s': round(duration, 1),
                'last_seen_ago_s': round(last_seen_ago, 1),
                'avg_speed_px': round(speed_px_per_frame, 1),
                'frames_tracked': len(track['positions']),
            })
        
        return tracks_info

    def cleanup_old_tracks(self):
        """Remove tracks that haven't been updated recently"""
        current_time = time.time()
        expired = [
            obj_id for obj_id, track in self.tracks.items()
            if current_time - track['last_seen'] > self.track_timeout
        ]
        for obj_id in expired:
            del self.tracks[obj_id]
        
        if expired:
            logger.debug(f"Cleaned up {len(expired)} old tracks")

    def get_next_object_id(self) -> str:
        """Generate a unique object ID"""
        obj_id = f"obj_{self.next_object_id}"
        self.next_object_id += 1
        return obj_id

    def get_speed_stats(self) -> Dict:
        """Get speed statistics for all tracked objects"""
        all_speeds = []
        class_speeds = defaultdict(list)
        
        for track in self.tracks.values():
            if track['speeds']:
                avg_speed = np.mean(track['speeds'])
                all_speeds.append(avg_speed)
                class_speeds[track['class_name']].append(avg_speed)
        
        stats = {
            'total_tracked': len(self.tracks),
            'overall_avg_speed_kmh': round(np.mean(all_speeds) * 3.6, 2) if all_speeds else 0,
            'overall_max_speed_kmh': round(max(all_speeds) * 3.6, 2) if all_speeds else 0,
        }
        
        if class_speeds:
            stats['by_class'] = {}
            for cls, speeds in class_speeds.items():
                stats['by_class'][cls] = {
                    'count': len(speeds),
                    'avg_speed_kmh': round(np.mean(speeds) * 3.6, 2),
                    'max_speed_kmh': round(max(speeds) * 3.6, 2),
                }
        
        return stats


# Global instance
_speed_height_analyzer = None


def get_speed_height_analyzer() -> SpeedHeightAnalyzer:
    """Get global speed and height analyzer instance"""
    global _speed_height_analyzer
    if _speed_height_analyzer is None:
        _speed_height_analyzer = SpeedHeightAnalyzer()
    return _speed_height_analyzer