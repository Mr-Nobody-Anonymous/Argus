"""Argus service exports, loaded lazily to keep core modules lightweight.

Importing ``backend.services.perception`` should not load optional ML, image,
or geometry packages. Consumers that request a legacy service getter still
receive the same exported function; only that service's module is imported.
"""

from importlib import import_module

_EXPORT_MODULES = {
    "get_camera_manager": "backend.services.management.camera_manager",
    "get_zone_manager": "backend.services.management.zone_manager",
    "get_event_store": "backend.services.management.event_store",
    "get_telemetry_monitor": "backend.services.management.telemetry_monitor",
    "get_user_attention_tracker": "backend.services.management.user_attention_tracker",
    "get_processing_coordinator": "backend.services.core_engine.processing_coordinator",
    "get_inference_engine": "backend.services.core_engine.inference_engine",
    "get_mqtt_publisher": "backend.services.management.mqtt_publisher",
    "get_image_enhancement": "backend.services.vision.image_enhancement",
    "get_face_recognition": "backend.services.vision.face_recognition",
    "get_speed_height_analyzer": "backend.services.analytics.speed_height_analysis",
    "get_license_plate_recognition": "backend.services.vision.license_plate_recognition",
    "get_anomaly_detector": "backend.services.analytics.anomaly_detector",
    "get_pose_estimator": "backend.services.vision.pose_estimator",
    "get_deep_tracker": "backend.services.core_engine.deep_tracker",
    "get_person_reid": "backend.services.analytics.person_reid",
    "get_adaptive_learning_engine": "backend.services.analytics.adaptive_learning",
    "get_cross_camera_tracker": "backend.services.analytics.cross_camera_tracker",
    "get_evolutionary_engine": "backend.services.core_engine.evolutionary_engine",
    "get_consortium_broker": "backend.services.core_engine.consortium_broker",
    "get_logic_mutator": "backend.services.core_engine.logic_mutator",
    "get_state_recovery_manager": "backend.services.management.state_recovery_manager",
    "get_yolo_detection_agent": "backend.services.core_engine.yolo_detection_agent",
    "get_face_recognition_agent": "backend.services.vision.face_recognition_agent",
    "get_lpr_agent": "backend.services.vision.lpr_agent",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    """Import a service only when its public getter is requested."""
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
