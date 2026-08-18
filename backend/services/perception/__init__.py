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

__all__ = [
    "Attribute", "BBox", "Entity", "EntityKind", "Observation",
    "RelationKind", "Relationship", "Scene", "Source", "LOW_CONFIDENCE",
    "attach_face", "attach_plate", "attach_pose", "attach_text",
    "entity_from_detection", "infer_spatial_relationships",
    "kind_for_class", "scene_from_detections",
]
