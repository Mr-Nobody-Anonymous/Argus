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
from .pipeline import FrameResult, PerceptionPipeline, get_pipeline

__all__ = [
    "Attribute", "BBox", "Entity", "EntityKind", "Observation",
    "RelationKind", "Relationship", "Scene", "Source", "LOW_CONFIDENCE",
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
]
