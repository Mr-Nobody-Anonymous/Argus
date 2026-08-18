"""Canonical perception model - the single representation every model writes into.

Today each analyser invents its own dict shape: the detector emits
``{'class_name','confidence','bbox'}``, LPR emits ``{'plate_text','vehicle_bbox'}``,
faces emit encodings, pose emits keypoints. Nothing can be joined, so the system
cannot answer "what is this person carrying" - the facts live in different
places with no shared identity.

This module defines that shared structure:

    Scene
    +-- Camera / timestamp / environment
    +-- Entity[]        what is it, where, what does it look like
    |   +-- Attribute[] every property, each with its own confidence + source
    +-- Relationship[]  entity <-> entity  (carrying, near, inside, following)
    +-- Observation[]   a durable statement about the scene

Three rules make this defensible rather than decorative:

1. **Every claim carries a confidence and a source.** An attribute is never a
   bare value; it records which analyser produced it. When two models disagree
   you can see who said what instead of silently overwriting.

2. **Absence is explicit.** A missing attribute means "not observed", never
   "not present". Cameras have occlusion, motion blur and limited resolution;
   a model that cannot distinguish those two cases will invent facts.

3. **Nothing here imports a model.** This layer is pure data, so it stays
   importable on a machine with no GPU, no torch and no weights - which is what
   lets the tests run anywhere and the schema be validated in CI.

The structures are plain dataclasses with ``to_dict``/``from_dict`` so they can
cross a process boundary, land in a database, or be diffed in a test without
dragging the inference stack along.
"""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ── vocabulary ───────────────────────────────────────────────────────────────

class EntityKind(str, Enum):
    """Coarse bucket an entity falls into.

    Deliberately small. The fine-grained label lives in ``Entity.category``
    (a free string) so an open-vocabulary detector can report "traffic cone"
    without anyone editing an enum. The kind exists only to route processing:
    a PERSON gets pose, a VEHICLE gets plate reading.
    """

    PERSON = "person"
    VEHICLE = "vehicle"
    ANIMAL = "animal"
    OBJECT = "object"
    TEXT = "text"
    UNKNOWN = "unknown"


class Source(str, Enum):
    """Which analyser produced a claim. Provenance is not optional."""

    DETECTOR = "detector"              # fixed-class detector (YOLO)
    OPEN_VOCAB = "open_vocabulary"     # open-vocabulary detector
    TRACKER = "tracker"
    POSE = "pose"
    FACE = "face"
    LPR = "lpr"
    OCR = "ocr"
    SEGMENTER = "segmenter"
    APPEARANCE = "appearance"
    VLM = "vision_language_model"
    SCENE = "scene_classifier"
    RULE = "rule_engine"
    TEMPORAL = "temporal_engine"
    HUMAN = "human"                    # operator confirmation outranks all


# Confidence any consumer should treat as "do not act on this alone".
LOW_CONFIDENCE = 0.4

# Ranking used when two sources claim the same attribute. A human always wins;
# a specialised model beats a general one at its own speciality.
_SOURCE_RANK: Dict[str, int] = {
    Source.HUMAN.value: 100,
    Source.LPR.value: 60,
    Source.FACE.value: 60,
    Source.OCR.value: 55,
    Source.POSE.value: 50,
    Source.SEGMENTER.value: 50,
    Source.VLM.value: 45,
    Source.OPEN_VOCAB.value: 40,
    Source.DETECTOR.value: 40,
    Source.APPEARANCE.value: 35,
    Source.TRACKER.value: 30,
    Source.SCENE.value: 30,
    Source.TEMPORAL.value: 25,
    Source.RULE.value: 20,
}


def _now() -> float:
    return time.time()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# ── geometry ─────────────────────────────────────────────────────────────────

@dataclass
class BBox:
    """Axis-aligned box in pixels, ``x1,y1`` top-left.

    Stored as floats: trackers and homographies produce sub-pixel values, and
    rounding at every hop accumulates drift.
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        # Normalise inverted boxes rather than propagating a negative area.
        if self.x2 < self.x1:
            self.x1, self.x2 = self.x2, self.x1
        if self.y2 < self.y1:
            self.y1, self.y2 = self.y2, self.y1

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> Tuple[float, float]:
        """Where an upright object meets the ground - the right anchor for
        distance on a ground plane. Box centres float in mid-air and make
        near/far comparisons wrong for objects of differing height."""
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def iou(self, other: "BBox") -> float:
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        inter = iw * ih
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def contains(self, other: "BBox", min_fraction: float = 0.85) -> bool:
        """True when ``other`` is mostly inside this box.

        Used for containment relationships (a bag within a person's box). Pure
        IoU cannot express this: a small box fully inside a large one has low
        IoU yet is entirely contained.
        """
        ix1, iy1 = max(self.x1, other.x1), max(self.y1, other.y1)
        ix2, iy2 = min(self.x2, other.x2), min(self.y2, other.y2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        return other.area > 0 and (inter / other.area) >= min_fraction

    def distance_to(self, other: "BBox") -> float:
        ax, ay = self.bottom_center
        bx, by = other.bottom_center
        return math.hypot(ax - bx, ay - by)

    def to_list(self) -> List[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    @classmethod
    def from_any(cls, value: Any) -> Optional["BBox"]:
        """Accept the several bbox shapes already in the codebase.

        Detections use ``[x1,y1,x2,y2]`` lists, some payloads use dicts, and
        stored rows use JSON strings. Rejecting any one of them would force
        call sites to normalise first - which is exactly the fragmentation this
        module exists to end.
        """
        if value is None:
            return None
        if isinstance(value, BBox):
            return value
        if isinstance(value, dict):
            if all(k in value for k in ("x1", "y1", "x2", "y2")):
                return cls(float(value["x1"]), float(value["y1"]),
                           float(value["x2"]), float(value["y2"]))
            if all(k in value for k in ("x", "y", "w", "h")):
                x, y = float(value["x"]), float(value["y"])
                return cls(x, y, x + float(value["w"]), y + float(value["h"]))
            return None
        if isinstance(value, str):
            import json
            try:
                return cls.from_any(json.loads(value))
            except Exception:
                return None
        if isinstance(value, (list, tuple)) and len(value) >= 4:
            try:
                return cls(*(float(v) for v in value[:4]))
            except (TypeError, ValueError):
                return None
        return None


# ── attributes ───────────────────────────────────────────────────────────────

@dataclass
class Attribute:
    """One property of one entity, with the evidence behind it.

    ``value`` stays untyped on purpose: a colour is a string, a speed a float,
    an embedding a list. What must never be lost is *who* claimed it and *how
    sure* they were.
    """

    name: str
    value: Any
    confidence: float = 1.0
    source: str = Source.DETECTOR.value
    observed_at: float = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        if isinstance(self.source, Source):
            self.source = self.source.value

    @property
    def is_reliable(self) -> bool:
        return self.confidence >= LOW_CONFIDENCE

    def outranks(self, other: "Attribute") -> bool:
        """Should this claim replace ``other``?

        Source authority first, confidence only as a tie-break. A face model at
        0.55 is worth more than a generic detector at 0.9 when the question is
        identity - comparing raw confidence across models is meaningless
        because they are not calibrated against each other.
        """
        mine = _SOURCE_RANK.get(self.source, 0)
        theirs = _SOURCE_RANK.get(other.source, 0)
        if mine != theirs:
            return mine > theirs
        return self.confidence > other.confidence


# ── entities ─────────────────────────────────────────────────────────────────

@dataclass
class Entity:
    """Something visible in one frame, plus everything known about it."""

    kind: str = EntityKind.UNKNOWN.value
    category: str = "unknown"          # free-form fine label ("delivery van")
    entity_id: str = field(default_factory=lambda: _new_id("ent"))
    track_id: Optional[int] = None     # stable across frames when tracked
    bbox: Optional[BBox] = None
    mask_rle: Optional[str] = None     # segmentation, when a segmenter ran
    confidence: float = 0.0
    source: str = Source.DETECTOR.value
    attributes: Dict[str, Attribute] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.kind, EntityKind):
            self.kind = self.kind.value
        if isinstance(self.source, Source):
            self.source = self.source.value
        self.bbox = BBox.from_any(self.bbox)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    # -- attribute access -----------------------------------------------------

    def set_attribute(self, name: str, value: Any, confidence: float = 1.0,
                      source: str = Source.DETECTOR.value) -> Attribute:
        """Record a property, keeping whichever claim is better sourced.

        Returns the attribute now in force, which may be the existing one.
        """
        incoming = Attribute(name=name, value=value,
                             confidence=confidence, source=source)
        current = self.attributes.get(name)
        if current is None or incoming.outranks(current):
            self.attributes[name] = incoming
            return incoming
        return current

    def get(self, name: str, default: Any = None) -> Any:
        attr = self.attributes.get(name)
        return default if attr is None else attr.value

    def get_attribute(self, name: str) -> Optional[Attribute]:
        return self.attributes.get(name)

    def observed(self, name: str) -> bool:
        """Whether anything claimed this attribute at all.

        The distinction from ``get(name) is None`` is the point: unobserved is
        not the same as absent.
        """
        return name in self.attributes

    def reliable_attributes(self) -> Dict[str, Attribute]:
        return {k: v for k, v in self.attributes.items() if v.is_reliable}

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "kind": self.kind,
            "category": self.category,
            "track_id": self.track_id,
            "bbox": self.bbox.to_list() if self.bbox else None,
            "mask_rle": self.mask_rle,
            "confidence": self.confidence,
            "source": self.source,
            "attributes": {
                k: {"value": a.value, "confidence": a.confidence,
                    "source": a.source, "observed_at": a.observed_at}
                for k, a in self.attributes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Entity":
        ent = cls(
            kind=data.get("kind", EntityKind.UNKNOWN.value),
            category=data.get("category", "unknown"),
            entity_id=data.get("entity_id") or _new_id("ent"),
            track_id=data.get("track_id"),
            bbox=BBox.from_any(data.get("bbox")),
            mask_rle=data.get("mask_rle"),
            confidence=float(data.get("confidence", 0.0)),
            source=data.get("source", Source.DETECTOR.value),
        )
        for name, a in (data.get("attributes") or {}).items():
            if isinstance(a, dict) and "value" in a:
                ent.attributes[name] = Attribute(
                    name=name, value=a["value"],
                    confidence=float(a.get("confidence", 1.0)),
                    source=a.get("source", Source.DETECTOR.value),
                    observed_at=float(a.get("observed_at", _now())),
                )
            else:  # tolerate a bare value
                ent.attributes[name] = Attribute(name=name, value=a)
        return ent


# ── relationships ────────────────────────────────────────────────────────────

class RelationKind(str, Enum):
    NEAR = "near"
    CARRYING = "carrying"
    INSIDE = "inside"
    RIDING = "riding"
    FOLLOWING = "following"
    APPROACHING = "approaching"
    OCCLUDING = "occluding"
    INTERACTING = "interacting"


@dataclass
class Relationship:
    """A directed edge between two entities: subject -> predicate -> object."""

    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 0.5
    source: str = Source.RULE.value
    evidence: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.predicate, RelationKind):
            self.predicate = self.predicate.value
        if isinstance(self.source, Source):
            self.source = self.source.value
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ── scene ────────────────────────────────────────────────────────────────────

@dataclass
class Scene:
    """Everything perceived in one frame of one camera."""

    camera_id: int
    frame_id: Optional[int] = None
    timestamp: float = field(default_factory=_now)
    scene_id: str = field(default_factory=lambda: _new_id("scn"))
    entities: List[Entity] = field(default_factory=list)
    relationships: List[Relationship] = field(default_factory=list)
    environment: Dict[str, Attribute] = field(default_factory=dict)

    # -- construction ---------------------------------------------------------

    def add_entity(self, entity: Entity) -> Entity:
        self.entities.append(entity)
        return entity

    def add_relationship(self, rel: Relationship) -> Relationship:
        self.relationships.append(rel)
        return rel

    def set_environment(self, name: str, value: Any, confidence: float = 1.0,
                        source: str = Source.SCENE.value) -> None:
        self.environment[name] = Attribute(name=name, value=value,
                                           confidence=confidence, source=source)

    # -- lookup ---------------------------------------------------------------

    def by_id(self, entity_id: str) -> Optional[Entity]:
        for e in self.entities:
            if e.entity_id == entity_id:
                return e
        return None

    def by_track(self, track_id: int) -> Optional[Entity]:
        for e in self.entities:
            if e.track_id == track_id:
                return e
        return None

    def of_kind(self, kind: str) -> List[Entity]:
        k = kind.value if isinstance(kind, EntityKind) else kind
        return [e for e in self.entities if e.kind == k]

    def relationships_for(self, entity_id: str) -> List[Relationship]:
        return [r for r in self.relationships
                if r.subject_id == entity_id or r.object_id == entity_id]

    # -- readable output ------------------------------------------------------

    def describe(self) -> str:
        """Plain-language summary. Useful in logs, alert bodies and tests,
        and it is the natural text to embed for semantic search later."""
        if not self.entities:
            return "nothing detected"
        counts: Dict[str, int] = {}
        for e in self.entities:
            counts[e.category] = counts.get(e.category, 0) + 1
        parts = [f"{n} {name}{'s' if n > 1 else ''}"
                 for name, n in sorted(counts.items(), key=lambda kv: -kv[1])]
        text = ", ".join(parts)
        if self.relationships:
            rels = ", ".join(
                f"{r.subject_id[:7]} {r.predicate} {r.object_id[:7]}"
                for r in self.relationships[:3]
            )
            text += f" ({rels})"
        return text

    def to_dict(self) -> Dict[str, Any]:
        return {
            "scene_id": self.scene_id,
            "camera_id": self.camera_id,
            "frame_id": self.frame_id,
            "timestamp": self.timestamp,
            "entities": [e.to_dict() for e in self.entities],
            "relationships": [r.to_dict() for r in self.relationships],
            "environment": {
                k: {"value": a.value, "confidence": a.confidence,
                    "source": a.source}
                for k, a in self.environment.items()
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Scene":
        scene = cls(
            camera_id=int(data.get("camera_id", 0)),
            frame_id=data.get("frame_id"),
            timestamp=float(data.get("timestamp", _now())),
            scene_id=data.get("scene_id") or _new_id("scn"),
        )
        scene.entities = [Entity.from_dict(e) for e in data.get("entities") or []]
        for r in data.get("relationships") or []:
            scene.relationships.append(Relationship(
                subject_id=r["subject_id"], predicate=r["predicate"],
                object_id=r["object_id"],
                confidence=float(r.get("confidence", 0.5)),
                source=r.get("source", Source.RULE.value),
                evidence=r.get("evidence") or {},
            ))
        for k, a in (data.get("environment") or {}).items():
            if isinstance(a, dict) and "value" in a:
                scene.environment[k] = Attribute(
                    name=k, value=a["value"],
                    confidence=float(a.get("confidence", 1.0)),
                    source=a.get("source", Source.SCENE.value))
        return scene


# ── observations ─────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """A durable statement worth remembering, with its evidence attached.

    Storing conclusions without evidence is how a surveillance system becomes
    indefensible. ``summary`` is what a human reads; ``evidence`` is what lets
    them disagree with it.
    """

    kind: str
    summary: str
    camera_id: Optional[int] = None
    observation_id: str = field(default_factory=lambda: _new_id("obs"))
    timestamp: float = field(default_factory=_now)
    confidence: float = 0.5
    source: str = Source.RULE.value
    entity_ids: List[str] = field(default_factory=list)
    track_ids: List[int] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.source, Source):
            self.source = self.source.value
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    @property
    def is_actionable(self) -> bool:
        """Confidence alone is not enough - an unexplained claim should never
        drive an alert, however sure the model says it is."""
        return self.confidence >= LOW_CONFIDENCE and bool(self.evidence)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Observation":
        return cls(
            kind=data["kind"], summary=data["summary"],
            camera_id=data.get("camera_id"),
            observation_id=data.get("observation_id") or _new_id("obs"),
            timestamp=float(data.get("timestamp", _now())),
            confidence=float(data.get("confidence", 0.5)),
            source=data.get("source", Source.RULE.value),
            entity_ids=list(data.get("entity_ids") or []),
            track_ids=list(data.get("track_ids") or []),
            evidence=list(data.get("evidence") or []),
            metadata=dict(data.get("metadata") or {}),
        )
