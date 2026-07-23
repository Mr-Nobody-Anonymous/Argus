"""
Configuration management for Argus
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


class SynthesizerConfig(BaseModel):
    enabled: bool = True
    context_window: int = 50
    min_generation_interval_s: int = 60
    max_generated_rules: int = 20


class EvaluatorConfig(BaseModel):
    enabled: bool = True
    evaluation_window: int = 1000
    fitness_weights: dict = {
        "inference_speed": 0.3,
        "tracking_accuracy": 0.25,
        "false_positive_ratio": 0.2,
        "kafka_latency_ms": 0.1,
        "rule_precision": 0.15,
    }
    survival_threshold: float = 0.4


class MutationVectorBounds(BaseModel):
    yolo_conf_threshold: list = [0.1, 0.9]
    iou_threshold: list = [0.1, 0.9]
    tracking_history_buffer: list = [5, 100]
    frame_skipping_cadence: list = [1, 5]
    rules_engine_cooldown: list = [0.5, 10.0]


class MutationConfig(BaseModel):
    enabled: bool = True
    initial_mutation_rate: float = 0.1
    adaptive_rate: bool = True
    max_mutation_attempts: int = 3
    population_cap: int = 50
    prune_interval_s: int = 300
    crossover_rate: float = 0.3
    vector_bounds: MutationVectorBounds = MutationVectorBounds()


class EvolutionaryEngineConfig(BaseModel):
    enabled: bool = True
    synthesizer: SynthesizerConfig = SynthesizerConfig()
    evaluator: EvaluatorConfig = EvaluatorConfig()
    mutation: MutationConfig = MutationConfig()


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
    topic_prefix: str = "argus"
    qos: int = 1


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///../data/argus.db"


class RuleConfig(BaseModel):
    enabled: bool = True
    priority: str = "medium"
    description: str = ""
    threshold_seconds: Optional[int] = None


class UserAttentionConfig(BaseModel):
    enabled: bool = True
    active_view_multiplier: float = 1.5
    click_interaction_boost: float = 2.0
    expanded_view_multiplier: float = 2.0
    click_boost_duration_s: int = 60
    decay_rate_seconds: int = 30
    viewport_poll_interval_s: int = 5
    unattended_camera_multiplier: float = 0.3


class LogicMutationConfig(BaseModel):
    enabled: bool = True
    max_code_line_length: int = 15
    stability_threshold: float = 0.10
    max_variants: int = 20
    test_frame_count: int = 100
    allowed_builtin_overrides: List[str] = [
        "abs", "min", "max", "round", "sum", "len",
        "any", "all", "sorted", "filter", "map",
    ]
    allowed_imports: List[str] = ["math"]
    sandbox_timeout_ms: int = 50


class StateRecoveryConfig(BaseModel):
    enabled: bool = True
    heartbeat_timeout_ms: int = 2000
    max_allowed_consecutive_errors: int = 10
    ledger_history_limit: int = 5
    watchdog_poll_interval_ms: int = 500
    auto_rollback_enabled: bool = True
    log_recovery_events: bool = True


class TelemetryThresholds(BaseModel):
    cpu_warning: float = 0.80
    cpu_critical: float = 0.95
    gpu_warning: float = 0.75
    gpu_critical: float = 0.90
    ram_warning: float = 0.80
    ram_critical: float = 0.95
    vram_warning: float = 0.80
    vram_critical: float = 0.95


class StressMultiplierConfig(BaseModel):
    min: float = 0.3
    max: float = 1.0
    curve_exponent: float = 2.0


class TelemetryConfig(BaseModel):
    enabled: bool = True
    sampling_interval_ms: int = 500
    cache_ttl_seconds: float = 1.0
    thresholds: TelemetryThresholds = TelemetryThresholds()
    stress_multiplier: StressMultiplierConfig = StressMultiplierConfig()


class ConsortiumConfig(BaseModel):
    enabled: bool = True
    sync_interval_ms: int = 100
    resource_bidding_enabled: bool = True
    max_agents: int = 8
    bidding_strategy: str = "proportional"


class LocalEvolutionConfig(BaseModel):
    enabled: bool = True
    evaluation_window: int = 1000
    mutation_rate: float = 0.1


class YoloAgentGeneBounds(BaseModel):
    yolo_conf_threshold: list = [0.1, 0.9]
    iou_threshold: list = [0.1, 0.9]
    input_resolution_scale: list = [0.5, 1.0]
    tracker_matching_threshold: list = [0.4, 0.9]


class YoloAgentConfig(BaseModel):
    enabled: bool = True
    local_evolution: LocalEvolutionConfig = LocalEvolutionConfig()
    gene_vector_bounds: YoloAgentGeneBounds = YoloAgentGeneBounds()
    compute_cost_per_frame: float = 15.0


class FaceAgentGeneBounds(BaseModel):
    match_distance_threshold: list = [0.3, 0.9]
    min_face_size_px: list = [30, 100]
    track_timeout_seconds: list = [1.0, 10.0]
    frame_skip_cadence: list = [1, 10]


class FaceAgentConfig(BaseModel):
    enabled: bool = True
    local_evolution: LocalEvolutionConfig = LocalEvolutionConfig()
    gene_vector_bounds: FaceAgentGeneBounds = FaceAgentGeneBounds()
    compute_cost_per_frame: float = 8.5


class LprAgentGeneBounds(BaseModel):
    segmentation_threshold: list = [0.3, 0.8]
    min_plate_height_px: list = [15, 50]
    resolution_downscale: list = [0.5, 1.0]
    detection_confidence: list = [0.4, 0.9]
    ocr_beam_width: list = [1, 5]


class LprAgentConfig(BaseModel):
    enabled: bool = True
    local_evolution: LocalEvolutionConfig = LocalEvolutionConfig()
    gene_vector_bounds: LprAgentGeneBounds = LprAgentGeneBounds()
    compute_cost_per_frame: float = 12.0


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
    consortium: ConsortiumConfig = ConsortiumConfig()
    telemetry: TelemetryConfig = TelemetryConfig()
    user_attention: UserAttentionConfig = UserAttentionConfig()
    logic_mutation: LogicMutationConfig = LogicMutationConfig()
    state_recovery: StateRecoveryConfig = StateRecoveryConfig()
    yolo_agent: YoloAgentConfig = YoloAgentConfig()
    face_agent: FaceAgentConfig = FaceAgentConfig()
    lpr_agent: LprAgentConfig = LprAgentConfig()
    evolutionary_engine: EvolutionaryEngineConfig = EvolutionaryEngineConfig()
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
            Path(__file__).parent.parent / "config" / "config.yaml",  # Absolute from this file
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