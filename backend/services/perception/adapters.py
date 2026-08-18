"""Translate the existing analysers' output into the canonical Scene.

The perception model is only worth anything if the code already running feeds
it. These adapters are deliberately one-directional and defensive: they take
whatever shape a module happens to emit today and produce Entities, without
requiring those modules to change. That means the canonical layer can be
adopted incrementally instead of via one large risky rewrite.

Every adapter follows the same rules:

* **Never raise into the pipeline.** A malformed detection must not kill a
  frame. Bad items are skipped, not guessed at.
* **Never invent a value.** If a field is absent, no attribute is written -
  the model can then tell "not observed" from "observed as absent".
* **Always stamp a source.** Provenance is what makes disagreement debuggable.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Iterable, List, Optional

from .observation import (
    BBox,
    Entity,
    EntityKind,
    Relationship,
    RelationKind,
    Scene,
    Source,
)

logger = logging.getLogger(__name__)


# COCO classes the fixed detector emits, mapped to a routing bucket. Anything
# absent falls through to OBJECT - an unmapped class is still a real thing that
# was seen, and dropping it would be worse than labelling it coarsely.
_KIND_BY_CLASS: Dict[str, EntityKind] = {
    "person": EntityKind.PERSON,
    "car": EntityKind.VEHICLE,
    "truck": EntityKind.VEHICLE,
    "bus": EntityKind.VEHICLE,
    "motorcycle": EntityKind.VEHICLE,
    "bicycle": EntityKind.VEHICLE,
    "train": EntityKind.VEHICLE,
    "boat": EntityKind.VEHICLE,
    "airplane": EntityKind.VEHICLE,
    "bird": EntityKind.ANIMAL,
    "cat": EntityKind.ANIMAL,
    "dog": EntityKind.ANIMAL,
    "horse": EntityKind.ANIMAL,
    "sheep": EntityKind.ANIMAL,
    "cow": EntityKind.ANIMAL,
    "elephant": EntityKind.ANIMAL,
    "bear": EntityKind.ANIMAL,
    "zebra": EntityKind.ANIMAL,
    "giraffe": EntityKind.ANIMAL,
}

# Things a person can plausibly carry. Used to promote a NEAR relationship to
# CARRYING; a car is near a person constantly and is obviously not carried.
CARRYABLE = {
    "backpack", "handbag", "suitcase", "umbrella", "bottle", "cup",
    "laptop", "cell phone", "book", "sports ball", "frisbee", "skateboard",
    "tie", "knife", "baseball bat", "tennis racket", "bag", "box", "package",
}


def kind_for_class(class_name: str) -> EntityKind:
    return _KIND_BY_CLASS.get((class_name or "").lower().strip(), EntityKind.OBJECT)


def entity_from_detection(det: Dict[str, Any],
                          source: str = Source.DETECTOR.value) -> Optional[Entity]:
    """Build an Entity from one detector dict.

    Accepts the shape ``inference_engine`` emits today
    (``class_name``/``confidence``/``bbox``) plus the ``track_id`` the tracker
    adds and the ``class``/``label`` aliases other modules use.
    """
    if not isinstance(det, dict):
        return None

    bbox = BBox.from_any(det.get("bbox") or det.get("box") or det.get("xyxy"))
    if bbox is None or bbox.area <= 0:
        # A detection with no usable geometry cannot be reasoned about.
        return None

    category = (det.get("class_name") or det.get("class")
                or det.get("label") or "unknown")
    category = str(category).lower().strip() or "unknown"

    try:
        confidence = float(det.get("confidence", det.get("score", 0.0)))
    except (TypeError, ValueError):
        confidence = 0.0

    track_id = det.get("track_id", det.get("id"))
    try:
        track_id = int(track_id) if track_id is not None else None
    except (TypeError, ValueError):
        track_id = None

    entity = Entity(
        kind=kind_for_class(category).value,
        category=category,
        track_id=track_id,
        bbox=bbox,
        confidence=confidence,
        source=source,
    )
    if det.get("class_id") is not None:
        entity.set_attribute("class_id", det["class_id"], 1.0, source)
    return entity


def scene_from_detections(camera_id: int,
                          detections: Iterable[Dict[str, Any]],
                          frame_id: Optional[int] = None,
                          timestamp: Optional[float] = None,
                          source: str = Source.DETECTOR.value) -> Scene:
    """Build a Scene from the detector output for a single frame."""
    scene = Scene(camera_id=camera_id, frame_id=frame_id)
    if timestamp is not None:
        scene.timestamp = float(timestamp)

    for det in detections or []:
        try:
            entity = entity_from_detection(det, source=source)
        except Exception as exc:  # noqa: BLE001 - one bad box must not kill a frame
            logger.debug(f"Skipping malformed detection: {exc}")
            continue
        if entity is not None:
            scene.add_entity(entity)
    return scene


# ── enrichment from the specialised analysers ────────────────────────────────

def attach_plate(scene: Scene, plate_text: str, vehicle_bbox: Any,
                 confidence: float = 0.6) -> Optional[Entity]:
    """Attach a plate reading to whichever vehicle it belongs to.

    LPR reports a plate and a box. Rather than storing that as an unrelated
    row, find the vehicle it overlaps so the plate becomes an attribute of a
    tracked entity - which is what makes "where has this plate been" answerable.
    """
    box = BBox.from_any(vehicle_bbox)
    if box is None or not plate_text:
        return None

    # IoU is the wrong metric: a plate is fully inside the vehicle yet scores
    # near zero against it (a 1500 px plate in a 28000 px car gives IoU 0.05).
    # Rank by how much of the plate box falls within the vehicle, and prefer
    # the smallest qualifying vehicle so an overlapping larger box behind it
    # does not steal the reading.
    best, best_area = None, None
    for ent in scene.of_kind(EntityKind.VEHICLE.value):
        if ent.bbox is None:
            continue
        if ent.bbox.contains(box, min_fraction=0.5):
            if best is None or ent.bbox.area < best_area:
                best, best_area = ent, ent.bbox.area

    if best is None:
        return None
    best.set_attribute("plate", str(plate_text).upper().strip(),
                       confidence, Source.LPR.value)
    return best


def attach_face(scene: Scene, name: Optional[str], face_bbox: Any,
                confidence: float = 0.5) -> Optional[Entity]:
    """Attach a face identity to the person containing that face."""
    box = BBox.from_any(face_bbox)
    if box is None:
        return None

    for ent in scene.of_kind(EntityKind.PERSON.value):
        if ent.bbox is not None and ent.bbox.contains(box, min_fraction=0.6):
            if name:
                ent.set_attribute("identity", name, confidence, Source.FACE.value)
            ent.set_attribute("face_visible", True, 1.0, Source.FACE.value)
            return ent
    return None


def attach_pose(scene: Scene, track_id: int, posture: str,
                confidence: float = 0.5) -> Optional[Entity]:
    ent = scene.by_track(track_id)
    if ent is None:
        return None
    ent.set_attribute("posture", posture, confidence, Source.POSE.value)
    return ent


def attach_text(scene: Scene, text: str, text_bbox: Any,
                confidence: float = 0.5) -> Entity:
    """Record scene text (a sign, a label) as a first-class entity.

    General OCR is not license-plate reading: the text may belong to no other
    object, so it becomes its own TEXT entity rather than an attribute.
    """
    box = BBox.from_any(text_bbox)
    ent = Entity(kind=EntityKind.TEXT.value, category="text", bbox=box,
                 confidence=confidence, source=Source.OCR.value)
    ent.set_attribute("text", str(text), confidence, Source.OCR.value)
    scene.add_entity(ent)
    return ent


# ── relationship inference ───────────────────────────────────────────────────

def infer_spatial_relationships(scene: Scene,
                                near_factor: float = 1.5) -> List[Relationship]:
    """Derive geometric relationships between the entities in one frame.

    Only what a single frame can support: proximity and containment. Anything
    needing history (following, approaching, loitering) belongs to the temporal
    engine and is deliberately not guessed here.

    ``near_factor`` is expressed in units of the subject's own height, not
    pixels, because a fixed pixel threshold means different real distances for
    a person near the camera versus far away.
    """
    rels: List[Relationship] = []
    people = scene.of_kind(EntityKind.PERSON.value)
    vehicles = scene.of_kind(EntityKind.VEHICLE.value)
    objects = scene.of_kind(EntityKind.OBJECT.value)

    # person carrying object: the object sits mostly inside the person's box.
    for person in people:
        if person.bbox is None:
            continue
        for obj in objects:
            if obj.bbox is None or obj.category not in CARRYABLE:
                continue
            if person.bbox.contains(obj.bbox, min_fraction=0.6):
                rels.append(Relationship(
                    subject_id=person.entity_id,
                    predicate=RelationKind.CARRYING.value,
                    object_id=obj.entity_id,
                    confidence=0.6,
                    source=Source.RULE.value,
                    evidence={"containment": round(
                        person.bbox.iou(obj.bbox), 3), "object": obj.category},
                ))

    # person near vehicle, scaled by the person's height.
    for person in people:
        if person.bbox is None or person.bbox.height <= 0:
            continue
        threshold = person.bbox.height * near_factor
        for veh in vehicles:
            if veh.bbox is None:
                continue
            dist = person.bbox.distance_to(veh.bbox)
            if dist <= threshold:
                rels.append(Relationship(
                    subject_id=person.entity_id,
                    predicate=RelationKind.NEAR.value,
                    object_id=veh.entity_id,
                    # Closer is more certain, capped so geometry alone never
                    # produces a near-certain claim.
                    confidence=round(min(0.9, 1.0 - (dist / (threshold * 2))), 3),
                    source=Source.RULE.value,
                    evidence={"pixel_distance": round(dist, 1),
                              "threshold": round(threshold, 1)},
                ))

    for rel in rels:
        scene.add_relationship(rel)
    return rels
