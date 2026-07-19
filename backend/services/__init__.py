"""
SentinelSight Backend Services

Integrated features from forked repositories:
- ultralytics: YOLOv8 object detection (inference_engine.py)
- opencv: Computer vision operations (image_enhancement.py, face_recognition.py)
- insightface: Face recognition (face_recognition.py)
- mediapipe: Pose estimation (pose_estimator.py)
- deep_sort/BoT_SORT/ByteTrack: Object tracking (deep_tracker.py)
- PaddleOCR/EasyOCR: License plate recognition (license_plate_recognition.py)
- anomalib: Anomaly detection (anomaly_detector.py)
- deep-person-reid/fast-reid: Person re-identification (person_reid.py)

Infrastructure:
- Qdrant: Vector database (docker-compose.yml)
- Kafka: Event streaming (docker-compose.yml)
"""

from .camera_manager import get_camera_manager
from .zone_manager import get_zone_manager
from .event_store import get_event_store
from .processing_coordinator import get_processing_coordinator
from .inference_engine import get_inference_engine
from .mqtt_publisher import get_mqtt_publisher
from .image_enhancement import get_image_enhancement
from .face_recognition import get_face_recognition
from .speed_height_analysis import get_speed_height_analyzer
from .license_plate_recognition import get_license_plate_recognition
from .anomaly_detector import get_anomaly_detector
from .pose_estimator import get_pose_estimator
from .deep_tracker import get_deep_tracker
from .person_reid import get_person_reid

__all__ = [
    'get_camera_manager',
    'get_zone_manager',
    'get_event_store',
    'get_processing_coordinator',
    'get_inference_engine',
    'get_mqtt_publisher',
    'get_image_enhancement',
    'get_face_recognition',
    'get_speed_height_analyzer',
    'get_license_plate_recognition',
    'get_anomaly_detector',
    'get_pose_estimator',
    'get_deep_tracker',
    'get_person_reid',
]