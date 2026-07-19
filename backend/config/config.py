"""
Configuration management for SentinelSight
"""
import yaml
from pathlib import Path
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)


class SystemConfig(BaseModel):
    fps_target: int = 15
    max_cameras: int = 4
    snapshot_retention_days: int = 30
    log_level: str = "INFO"
    snapshot_dir: str = "data/snapshots"


class InferenceConfig(BaseModel):
    model: str = "yolov8n.pt"
    confidence_threshold: float = 0.5
    device: str = "cpu"
    classes: List[int] = [0, 2]  # person, car


class EnhancementConfig(BaseModel):
    enabled: bool = True
    auto_enhance: bool = True
    low_light_enhancement: bool = True
    denoise: bool = True
    sharpen: bool = True
    night_vision: bool = False
    deblur: bool = True
    hdr: bool = False


class FaceRecognitionConfig(BaseModel):
    model_config = {'protected_namespaces': ()}
    enabled: bool = False
    confidence_threshold: float = 0.6
    model_path: str = "models/face_recognition"
    max_faces_per_frame: int = 10


class LicensePlateConfig(BaseModel):
    enabled: bool = True
    region: str = "us"
    min_confidence: float = 0.6


class AnomalyConfig(BaseModel):
    enabled: bool = True
    sensitivity: float = 0.7
    window_size: int = 100


class ReIDConfig(BaseModel):
    enabled: bool = False
    similarity_threshold: float = 0.7
    feature_dim: int = 2048


class TrackerConfig(BaseModel):
    enabled: bool = True
    algorithm: str = "bytetrack"
    track_buffer: int = 30
    match_threshold: float = 0.6


class PoseConfig(BaseModel):
    enabled: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class QdrantConfig(BaseModel):
    enabled: bool = True
    host: str = "localhost"
    port: int = 6333


class KafkaConfig(BaseModel):
    enabled: bool = False
    host: str = "localhost"
    port: int = 9092


class CrossCameraTrackerConfig(BaseModel):
    enabled: bool = True
    similarity_threshold: float = 0.5
    max_track_age_hours: int = 24
    max_path_points: int = 100
    enable_trajectory_prediction: bool = True


class SpeedAnalysisConfig(BaseModel):
    enabled: bool = True
    calibration_factor: float = 0.05
    reference_distance_m: float = 10.0


class HeightAnalysisConfig(BaseModel):
    enabled: bool = True
    avg_person_height_m: float = 1.7


class MQTTConfig(BaseModel):
    enabled: bool = True
    broker: str = "localhost"
    port: int = 1883
    topic_prefix: str = "sentinelsight"
    qos: int = 1


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///../data/sentinelsight.db"


class RuleConfig(BaseModel):
    enabled: bool = True
    priority: str = "medium"
    description: str = ""
    threshold_seconds: Optional[int] = None


class Config(BaseModel):
    system: SystemConfig = SystemConfig()
    inference: InferenceConfig = InferenceConfig()
    enhancement: EnhancementConfig = EnhancementConfig()
    face_recognition: FaceRecognitionConfig = FaceRecognitionConfig()
    license_plate: LicensePlateConfig = LicensePlateConfig()
    anomaly: AnomalyConfig = AnomalyConfig()
    reid: ReIDConfig = ReIDConfig()
    tracker: TrackerConfig = TrackerConfig()
    pose: PoseConfig = PoseConfig()
    cross_camera_tracker: CrossCameraTrackerConfig = CrossCameraTrackerConfig()
    speed_analysis: SpeedAnalysisConfig = SpeedAnalysisConfig()
    height_analysis: HeightAnalysisConfig = HeightAnalysisConfig()
    mqtt: MQTTConfig = MQTTConfig()
    database: DatabaseConfig = DatabaseConfig()
    qdrant: QdrantConfig = QdrantConfig()
    kafka: KafkaConfig = KafkaConfig()
    cameras: List[dict] = []
    rules: Dict[str, RuleConfig] = {}


def section_to_dict(section: Any) -> Dict[str, Any]:
    """Normalize a config section to a plain dictionary."""
    if section is None:
        return {}
    if isinstance(section, dict):
        return section
    if hasattr(section, "model_dump"):
        return section.model_dump()
    if hasattr(section, "dict"):
        return section.dict()
    return {}


def load_config(config_path: str = None) -> Config:
    """Load configuration from YAML file"""
    if config_path is None:
        # Try multiple possible locations
        possible_paths = [
            Path("config/config.yaml"),  # From project root
            Path("../config/config.yaml"),  # From backend dir
            Path(__file__).parent.parent.parent / "config" / "config.yaml",  # Absolute from this file
        ]
        config_file = None
        for p in possible_paths:
            if p.exists():
                config_file = p
                break
    else:
        config_file = Path(config_path)
    
    if config_file is None or not config_file.exists():
        logger.warning(f"Config file not found, using defaults")
        return Config()
    
    try:
        with open(config_file, 'r') as f:
            config_data = yaml.safe_load(f)
        
        config = Config(**config_data)
        logger.info(f"Configuration loaded from {config_file}")
        return config
    
    except Exception as e:
        logger.error(f"Error loading config: {e}")
        logger.warning("Using default configuration")
        return Config()


# Global config instance
_config = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config