"""Canonical perception layer: one representation every analyser writes into.

Pure data structures plus adapters - no model imports, so this package stays
importable without torch, weights or a GPU.
"""

from .observation import (
    Attribute,
    BBox,
    Entity,
    EntityKind,
    Observation,
    RelationKind,
    Relationship,
    Scene,
    Source,
    LOW_CONFIDENCE,
    SINGLE_FRAME_PREDICATES,
    BEHAVIOURAL_PREDICATES,
)
from .adapters import (
    attach_face,
    attach_plate,
    attach_pose,
    attach_text,
    entity_from_detection,
    infer_spatial_relationships,
    kind_for_class,
    scene_from_detections,
)

from .temporal import (
    Track,
    TrackPoint,
    TrackStore,
    analyse_track,
    detect_abandonment,
    detect_disappearance,
    detect_dwell,
    detect_pacing,
)
from .scene_graph import (
    Edge,
    SceneGraph,
    detect_approach,
    detect_following,
    summarise_track,
    update_from_scene,
)
from .capabilities import (
    Capability,
    CapabilityRegistry,
    Tier,
    build_default_registry,
    get_registry,
)
from .evidence import (
    EvidenceChain,
    EvidenceItem,
    chain_from_edge,
    chain_from_observation,
    chain_from_track,
    explain_track,
)
from .relationships import infer_all as infer_relationships
from .change import ChangeDetector, CameraBaseline, RunningStat
from .scene_classifier import classify_scene, density_label, measure_frame
from .ocr import (
    OcrEngine,
    attach_text_to_entities,
    extract_text,
    find_text_regions,
    get_engine,
)
from .descriptors import (
    DIM as DESCRIPTOR_DIM,
    MatchResult,
    POSSIBLE_MATCH,
    STRONG_MATCH,
    active_backend,
    average,
    describe,
    grey_world,
    histogram_descriptor,
    rank,
    register_backend,
    similarity,
)
from .memory import (
    PerceptionMemory,
    QdrantVectorStore,
    SqliteVectorStore,
    StoredAppearance,
    StoredObservation,
    VectorStore,
    get_memory,
    reset_memory,
)
from .search import (
    AppearanceMatch,
    CrossCameraMatch,
    find_across_cameras,
    find_similar_appearances,
    recall,
    summarise_period,
    transit_plausibility,
)
from .pipeline import FrameResult, PerceptionPipeline, get_pipeline

__all__ = [
    "Attribute", "BBox", "Entity", "EntityKind", "Observation",
    "RelationKind", "Relationship", "Scene", "Source", "LOW_CONFIDENCE",
    "SINGLE_FRAME_PREDICATES", "BEHAVIOURAL_PREDICATES",
    "attach_face", "attach_plate", "attach_pose", "attach_text",
    "entity_from_detection", "infer_spatial_relationships",
    "kind_for_class", "scene_from_detections",
    "Track", "TrackPoint", "TrackStore", "analyse_track",
    "detect_abandonment", "detect_disappearance", "detect_dwell", "detect_pacing",
    "Edge", "SceneGraph", "detect_approach", "detect_following",
    "summarise_track", "update_from_scene",
    "Capability", "CapabilityRegistry", "Tier", "build_default_registry",
    "get_registry",
    "FrameResult", "PerceptionPipeline", "get_pipeline",
    "EvidenceChain", "EvidenceItem", "chain_from_edge",
    "chain_from_observation", "chain_from_track", "explain_track",
    "infer_relationships",
    "ChangeDetector", "CameraBaseline", "RunningStat",
    "classify_scene", "density_label", "measure_frame",
    "OcrEngine", "attach_text_to_entities", "extract_text",
    "find_text_regions", "get_engine",
    "DESCRIPTOR_DIM", "MatchResult", "POSSIBLE_MATCH", "STRONG_MATCH",
    "active_backend", "average", "describe", "grey_world",
    "histogram_descriptor", "rank", "register_backend", "similarity",
    "PerceptionMemory", "QdrantVectorStore", "SqliteVectorStore",
    "StoredAppearance", "StoredObservation", "VectorStore", "get_memory",
    "reset_memory",
    "AppearanceMatch", "CrossCameraMatch", "find_across_cameras",
    "find_similar_appearances", "recall", "summarise_period",
    "transit_plausibility",
]
