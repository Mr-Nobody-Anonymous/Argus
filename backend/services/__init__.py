"""
Argus Backend Services

Integrated features from forked repositories:
- ultralytics: YOLOv8 object detection (inference_engine.py)
- opencv: Computer vision operations (image_enhancement.py, face_recognition.py)
- insightface: Face recognition (face_recognition.py)
- mediapipe: Pose estimation (pose_estimator.py)
- deep_sort/BoT_SORT/ByteTrack: Object tracking (deep_tracker.py)
- PaddleOCR/EasyOCR: License plate recognition (license_plate_recognition.py)
- anomalib: Anomaly detection (anomaly_detector.py)
- deep-person-reid/fast-reid: Person re-identification (person_reid.py)

Advanced Intelligence:
- adaptive_learning: Self-learning behavior patterns, emotion baselines, anomaly detection
"""

from backend.services.management.camera_manager import get_camera_manager
from backend.services.management.zone_manager import get_zone_manager
from backend.services.management.event_store import get_event_store
from backend.services.management.telemetry_monitor import get_telemetry_monitor
from backend.services.management.user_attention_tracker import get_user_attention_tracker
from backend.services.core_engine.processing_coordinator import get_processing_coordinator
from backend.services.core_engine.inference_engine import get_inference_engine
from backend.services.management.mqtt_publisher import get_mqtt_publisher
from backend.services.vision.image_enhancement import get_image_enhancement
from backend.services.vision.face_recognition import get_face_recognition
from backend.services.analytics.speed_height_analysis import get_speed_height_analyzer
from backend.services.vision.license_plate_recognition import get_license_plate_recognition
from backend.services.analytics.anomaly_detector import get_anomaly_detector
from backend.services.vision.pose_estimator import get_pose_estimator
from backend.services.core_engine.deep_tracker import get_deep_tracker
from backend.services.analytics.person_reid import get_person_reid
from backend.services.analytics.adaptive_learning import get_adaptive_learning_engine
from backend.services.analytics.cross_camera_tracker import get_cross_camera_tracker
from backend.services.core_engine.evolutionary_engine import get_evolutionary_engine
from backend.services.core_engine.consortium_broker import get_consortium_broker
from backend.services.core_engine.logic_mutator import get_logic_mutator
from backend.services.management.state_recovery_manager import get_state_recovery_manager
from backend.services.core_engine.yolo_detection_agent import get_yolo_detection_agent
from backend.services.vision.face_recognition_agent import get_face_recognition_agent
from backend.services.vision.lpr_agent import get_lpr_agent

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
    'get_adaptive_learning_engine',
    'get_cross_camera_tracker',
    'get_telemetry_monitor',
    'get_user_attention_tracker',
    'get_evolutionary_engine',
    'get_consortium_broker',
    'get_logic_mutator',
    'get_state_recovery_manager',
    'get_yolo_detection_agent',
    'get_face_recognition_agent',
    'get_lpr_agent',
]
