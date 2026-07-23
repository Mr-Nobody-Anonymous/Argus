"""
Pose estimation service using MediaPipe and mmpose approaches.
Detects human keypoints and gestures in video frames.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
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


class PoseEstimator:
    """
    Pose estimation using MediaPipe for real-time inference.
    
    Features:
    - 33-body keypoints detection
    - Pose classification (standing, sitting, lying)
    - Gesture recognition
    - Fall detection
    - Activity analysis
    """

    # MediaPipe pose keypoint connections
    POSE_CONNECTIONS = [
        (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (25, 27), (27, 29), (29, 31),
        (24, 26), (26, 28), (28, 30), (30, 32)
    ]

    # Keypoint names
    KEYPOINT_NAMES = [
        'nose', 'left_eye_inner', 'left_eye', 'left_eye_outer',
        'right_eye_inner', 'right_eye', 'right_eye_outer', 'left_ear',
        'right_ear', 'mouth_left', 'mouth_right', 'left_shoulder',
        'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist',
        'right_wrist', 'left_pinky', 'right_pinky', 'left_index',
        'right_index', 'left_thumb', 'right_thumb', 'left_hip',
        'right_hip', 'left_knee', 'right_knee', 'left_ankle',
        'right_ankle', 'left_heel', 'right_heel', 'left_foot',
        'right_foot'
    ]

    def __init__(self):
        self.config = get_config()
        pose_config = section_to_dict(getattr(self.config, 'pose', {}))
        self.enabled = pose_config.get('enabled', True)
        self.min_detection_confidence = pose_config.get('min_detection_confidence', 0.5)
        self.min_tracking_confidence = pose_config.get('min_tracking_confidence', 0.5)

        self._initialized = False
        self.pose_model = None

        if self.enabled and cv2 is not None and np is not None:
            self._initialize()
        elif self.enabled:
            logger.warning("Pose estimator dependencies unavailable, using fallback mode")
            self._initialized = False

    def _initialize(self):
        """Initialize pose estimation model using MediaPipe"""
        try:
            # Try to use MediaPipe
            if cv2 is None or np is None:
                raise ImportError("OpenCV or NumPy unavailable")
            import mediapipe as mp
            
            self.mp_pose = mp.solutions.pose
            self.pose_model = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                enable_segmentation=False,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence
            )
            
            self.mp_draw = mp.solutions.drawing_utils
            self._initialized = True
            logger.info("MediaPipe pose estimator initialized")

        except ImportError:
            logger.warning("MediaPipe not available, using fallback pose detection")
            self._initialized = False
            self.pose_model = None
        except Exception as e:
            logger.error(f"Error initializing pose estimator: {e}")
            self._initialized = False

    def estimate_pose(self, frame: np.ndarray, person_bbox: List[int] = None) -> Optional[Dict]:
        """
        Estimate human pose in a frame.
        
        Args:
            frame: Input video frame
            person_bbox: Optional bounding box to focus on
        
        Returns:
            Pose data with keypoints and classification
        """
        if not self.enabled or not self._initialized:
            return self._fallback_pose_detection(frame, person_bbox)

        if cv2 is None or np is None:
            return self._fallback_pose_detection(frame, person_bbox)

        try:
            # Crop to person if bbox provided
            if person_bbox:
                x1, y1, x2, y2 = [max(0, int(v)) for v in person_bbox]
                person_frame = frame[y1:y2, x1:x2].copy()
                if person_frame.size == 0:
                    return None
            else:
                person_frame = frame.copy()

            # MediaPipe expects RGB
            rgb_frame = cv2.cvtColor(person_frame, cv2.COLOR_BGR2RGB)
            
            results = self.pose_model.process(rgb_frame)

            if not results.pose_landmarks:
                return None

            # Extract keypoints
            h, w = person_frame.shape[:2]
            keypoints = []
            
            for idx, landmark in enumerate(results.pose_landmarks.landmark):
                keypoints.append({
                    'name': self.KEYPOINT_NAMES[idx] if idx < len(self.KEYPOINT_NAMES) else f'keypoint_{idx}',
                    'x': float(landmark.x) * w,
                    'y': float(landmark.y) * h,
                    'z': float(landmark.z) if hasattr(landmark, 'z') else 0.0,
                    'visibility': float(landmark.visibility) if hasattr(landmark, 'visibility') else 1.0
                })

            # Classify pose
            pose_class = self._classify_pose(keypoints, h, w)
            
            # Detect gestures/falls
            gesture = self._detect_gesture(keypoints)
            fall_detected = self._detect_fall(keypoints)

            return {
                'keypoints': keypoints,
                'pose_class': pose_class,
                'gesture': gesture,
                'fall_detected': fall_detected,
                'num_keypoints': len(keypoints)
            }

        except Exception as e:
            logger.error(f"Error in pose estimation: {e}")
            return None

    def _fallback_pose_detection(self, frame: np.ndarray, person_bbox: List[int]) -> Optional[Dict]:
        """Fallback pose detection using simple heuristics"""
        if not person_bbox:
            return None

        try:
            x1, y1, x2, y2 = person_bbox
            bbox_h = y2 - y1
            bbox_w = x2 - x1

            # Simple pose classification based on aspect ratio
            aspect_ratio = bbox_h / bbox_w if bbox_w > 0 else 0

            if aspect_ratio > 2.0:
                pose_class = 'standing'
            elif aspect_ratio > 0.8:
                pose_class = 'sitting'
            else:
                pose_class = 'lying'
                # Check if lying is suspicious (could be fall)
                if aspect_ratio < 0.5 and bbox_w > 50:
                    return {
                        'keypoints': [],
                        'pose_class': pose_class,
                        'gesture': 'fall_risk',
                        'fall_detected': True,
                        'num_keypoints': 0,
                        'bbox_aspect_ratio': aspect_ratio
                    }

            return {
                'keypoints': [],
                'pose_class': pose_class,
                'gesture': 'none',
                'fall_detected': False,
                'num_keypoints': 0
            }

        except Exception:
            return None

    def _classify_pose(self, keypoints: List[Dict], frame_h: int, frame_w: int) -> str:
        """Classify human pose based on keypoints"""
        try:
            # Get key points
            left_shoulder = next((kp for kp in keypoints if kp['name'] == 'left_shoulder'), None)
            right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)
            left_hip = next((kp for kp in keypoints if kp['name'] == 'left_hip'), None)
            right_hip = next((kp for kp in keypoints if kp['name'] == 'right_hip'), None)

            if not all([left_shoulder, right_shoulder]):
                return 'unknown'

            # Calculate vertical positions
            shoulder_y = (left_shoulder['y'] + right_shoulder['y']) / 2
            hip_y = (left_hip['y'] + right_hip['y']) / 2 if left_hip and right_hip else shoulder_y

            # If shoulders are significantly above hips, likely standing
            if shoulder_y < hip_y - 20:
                return 'standing'
            else:
                return 'sitting'

        except Exception:
            return 'unknown'

    def _detect_gesture(self, keypoints: List[Dict]) -> str:
        """Detect basic gestures from keypoints"""
        try:
            # Check for raised hands
            left_wrist = next((kp for kp in keypoints if kp['name'] == 'left_wrist'), None)
            right_wrist = next((kp for kp in keypoints if kp['name'] == 'right_wrist'), None)
            left_shoulder = next((kp for kp in keypoints if kp['name'] == 'left_shoulder'), None)
            right_shoulder = next((kp for kp in keypoints if kp['name'] == 'right_shoulder'), None)

            if left_wrist and left_shoulder and left_wrist['y'] < left_shoulder['y']:
                return 'hands_up'

            return 'none'

        except Exception:
            return 'none'

    def _detect_fall(self, keypoints: List[Dict]) -> bool:
        """Detect potential fall based on keypoints"""
        try:
            # Check if keypoints indicate horizontal body position
            y_values = [kp['y'] for kp in keypoints]
            x_values = [kp['x'] for kp in keypoints]

            if len(y_values) < 10:
                return False

            y_range = max(y_values) - min(y_values)
            x_range = max(x_values) - min(x_values)

            # If horizontal spread is larger than vertical, might be a fall
            return x_range > y_range * 2

        except Exception:
            return False

    def get_pose_statistics(self) -> Dict:
        """Get pose estimation statistics"""
        return {
            'enabled': self.enabled,
            'initialized': self._initialized,
            'model': 'mediapipe' if self.pose_model else 'fallback'
        }


# Global instance
_pose_estimator = None


def get_pose_estimator() -> PoseEstimator:
    """Get global pose estimator instance"""
    global _pose_estimator
    if _pose_estimator is None:
        _pose_estimator = PoseEstimator()
    return _pose_estimator