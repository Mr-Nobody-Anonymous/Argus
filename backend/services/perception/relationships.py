"""Richer relationship inference over a single frame.

`adapters.infer_spatial_relationships` covers the two relations the original
model needed (near, carrying). This module widens the vocabulary to what a
reviewer actually asks about - who is holding what, who is inside which
vehicle, what is on top of what, who is wearing a helmet - while holding the
line that a single frame supports geometry and *category priors only*.

The distinction that governs every function here:

* **Geometric** claims describe the frame ("A is inside B's box"). They can be
  made confidently, because they are measurements.
* **Semantic** claims interpret that geometry ("A is holding B"). Two boxes
  overlapping is not evidence that a hand is gripping anything, so these are
  capped well below certainty and always carry the geometry that produced them
  in `evidence`.

Behavioural predicates - following, entering, queuing - are absent here by
design. They need motion history and belong to `scene_graph`/`temporal`. A
single frame cannot see them, so this module does not pretend to.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

from .observation import (BBox, Entity, EntityKind, RelationKind, Relationship,
                          Scene, Source)

# Ceiling for any claim resting on geometry alone. Boxes overlapping is
# suggestive, never conclusive - the gap between 0.75 and 1.0 is exactly the
# room a real segmenter or pose model needs to improve on this.
GEOMETRY_CEILING = 0.75

# Ceiling for pure measurements ("A is above B"), which are as certain as the
# boxes themselves.
MEASUREMENT_CEILING = 0.95

# Objects small enough and of the right type to be held in a hand.
HOLDABLE = frozenset({
    "cell phone", "phone", "bottle", "cup", "book", "remote", "knife",
    "scissors", "banana", "apple", "sandwich", "wine glass", "mouse",
    "toothbrush", "hair drier", "tool", "camera",
})

# Objects carried against the body rather than gripped.
CARRIABLE = frozenset({
    "backpack", "handbag", "suitcase", "umbrella", "sports ball", "laptop",
    "box", "package", "bag", "briefcase",
})

# Objects normally worn.
WEARABLE = frozenset({
    "tie", "helmet", "hat", "cap", "glasses", "mask", "vest", "hard hat",
})

# Things a person rides rather than merely stands near.
RIDEABLE = frozenset({
    "bicycle", "motorcycle", "motorbike", "skateboard", "horse", "scooter",
    "surfboard", "snowboard", "skis",
})

# Vehicles a person can be inside.
ENCLOSING = frozenset({
    "car", "truck", "bus", "van", "train", "boat", "aeroplane", "airplane",
})

# Distance, in subject heights, beyond which two entities are unrelated.
FAR_FACTOR = 6.0

# Gap, in subject heights, under which two boxes count as adjacent.
ADJACENT_FACTOR = 0.25


def _containment(inner: BBox, outer: BBox) -> float:
    """Fraction of `inner`'s area that lies inside `outer`."""
    if inner is None or outer is None or inner.area <= 0:
        return 0.0
    ix1, iy1 = max(inner.x1, outer.x1), max(inner.y1, outer.y1)
    ix2, iy2 = min(inner.x2, outer.x2), min(inner.y2, outer.y2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    return ((ix2 - ix1) * (iy2 - iy1)) / inner.area


def _upper_body(box: BBox) -> BBox:
    """Top 45% of a person's box - roughly head, shoulders and chest."""
    return BBox(box.x1, box.y1, box.x2, box.y1 + box.height * 0.45)


def _hand_zone(box: BBox) -> BBox:
    """Middle band of a person's box, where held objects appear.

    Crude but honest: without pose there are no hand keypoints, so this is the
    region where a held object *would* be, and every claim built on it is
    capped accordingly.
    """
    return BBox(box.x1 - box.width * 0.25, box.y1 + box.height * 0.30,
                box.x2 + box.width * 0.25, box.y1 + box.height * 0.80)


def _rel(subject: Entity, predicate: str, obj: Entity, confidence: float,
         evidence: Dict, timestamp: float) -> Relationship:
    return Relationship(
        subject_id=subject.entity_id, predicate=predicate,
        object_id=obj.entity_id, confidence=round(confidence, 3),
        source=Source.RULE.value, evidence=evidence,
        first_observed=timestamp, last_observed=timestamp,
    )


def infer_containment(scene: Scene) -> List[Relationship]:
    """inside / outside / riding: one entity within another's extent."""
    out: List[Relationship] = []
    people = [e for e in scene.of_kind(EntityKind.PERSON.value) if e.bbox]
    vehicles = [e for e in scene.of_kind(EntityKind.VEHICLE.value) if e.bbox]
    ts = scene.timestamp

    for person in people:
        for vehicle in vehicles:
            fraction = _containment(person.bbox, vehicle.bbox)
            category = (vehicle.category or "").lower()

            if category in RIDEABLE and fraction > 0.25:
                # A cyclist's box overlaps the bicycle without enclosing it,
                # so a lower threshold is right here - and the person sits
                # above the vehicle's centre, which distinguishes riding from
                # merely standing in front of it.
                above = person.bbox.center[1] < vehicle.bbox.center[1]
                out.append(_rel(
                    person, RelationKind.RIDING.value, vehicle,
                    min(GEOMETRY_CEILING, 0.4 + fraction * 0.5 + (0.1 if above else 0.0)),
                    {"containment": round(fraction, 3),
                     "vehicle_type": category,
                     "subject_above_object": above,
                     "basis": "box overlap and vertical ordering, not pose"},
                    ts))
            elif category in ENCLOSING and fraction > 0.6:
                out.append(_rel(
                    person, RelationKind.INSIDE.value, vehicle,
                    min(GEOMETRY_CEILING, 0.35 + fraction * 0.45),
                    {"containment": round(fraction, 3),
                     "vehicle_type": category,
                     "basis": "person's box lies inside the vehicle's; "
                              "occlusion by the vehicle body is not verified"},
                    ts))
    return out


def infer_carried_objects(scene: Scene) -> List[Relationship]:
    """carrying / holding / wearing: object-to-person association by region."""
    out: List[Relationship] = []
    people = [e for e in scene.of_kind(EntityKind.PERSON.value) if e.bbox]
    objects = [e for e in scene.of_kind(EntityKind.OBJECT.value) if e.bbox]
    ts = scene.timestamp

    for person in people:
        upper = _upper_body(person.bbox)
        hands = _hand_zone(person.bbox)
        for obj in objects:
            category = (obj.category or "").lower()
            body_fraction = _containment(obj.bbox, person.bbox)
            if body_fraction < 0.3:
                continue

            if category in WEARABLE:
                head_fraction = _containment(obj.bbox, upper)
                if head_fraction > 0.5:
                    out.append(_rel(
                        person, RelationKind.WEARING.value, obj,
                        min(GEOMETRY_CEILING, 0.4 + head_fraction * 0.4),
                        {"upper_body_containment": round(head_fraction, 3),
                         "object": category,
                         "basis": "object lies in the upper-body region"},
                        ts))
            elif category in HOLDABLE:
                hand_fraction = _containment(obj.bbox, hands)
                if hand_fraction > 0.5:
                    out.append(_rel(
                        person, RelationKind.HOLDING.value, obj,
                        # Lower ceiling still: without hand keypoints this is
                        # the weakest of the three, and it should look weakest.
                        min(0.65, 0.3 + hand_fraction * 0.4),
                        {"hand_zone_containment": round(hand_fraction, 3),
                         "object": category,
                         "basis": "object lies where hands usually are; "
                                  "no pose keypoints available"},
                        ts))
            elif category in CARRIABLE:
                out.append(_rel(
                    person, RelationKind.CARRYING.value, obj,
                    min(GEOMETRY_CEILING, 0.35 + body_fraction * 0.45),
                    {"body_containment": round(body_fraction, 3),
                     "object": category,
                     "basis": "object overlaps the person's box"},
                    ts))
    return out


def infer_proximity(scene: Scene, near_factor: float = 1.5) -> List[Relationship]:
    """near / far_from / adjacent_to, scaled by the subject's own height."""
    out: List[Relationship] = []
    entities = [e for e in scene.entities
                if e.bbox is not None and e.bbox.height > 0
                and e.kind != EntityKind.TEXT.value]
    ts = scene.timestamp

    for i, subject in enumerate(entities):
        scale = subject.bbox.height
        for obj in entities[i + 1:]:
            distance = subject.bbox.distance_to(obj.bbox)
            near_threshold = scale * near_factor
            far_threshold = scale * FAR_FACTOR

            if distance <= scale * ADJACENT_FACTOR:
                out.append(_rel(
                    subject, RelationKind.ADJACENT_TO.value, obj,
                    min(MEASUREMENT_CEILING, 0.7 + (1.0 - distance / max(1.0, scale * ADJACENT_FACTOR)) * 0.2),
                    {"pixel_distance": round(distance, 1),
                     "threshold": round(scale * ADJACENT_FACTOR, 1),
                     "basis": "boxes almost touching"},
                    ts))
            elif distance <= near_threshold:
                out.append(_rel(
                    subject, RelationKind.NEAR.value, obj,
                    min(0.9, 1.0 - (distance / (near_threshold * 2))),
                    {"pixel_distance": round(distance, 1),
                     "threshold": round(near_threshold, 1),
                     "scale_reference": "subject height"},
                    ts))
            elif distance > far_threshold:
                # Explicit non-relationship. Recording it lets a search rule
                # out an association instead of inferring one from silence.
                out.append(_rel(
                    subject, RelationKind.FAR_FROM.value, obj,
                    min(MEASUREMENT_CEILING, 0.6 + min(0.3, distance / (far_threshold * 4))),
                    {"pixel_distance": round(distance, 1),
                     "threshold": round(far_threshold, 1)},
                    ts))
    return out


def infer_vertical(scene: Scene) -> List[Relationship]:
    """above / below for horizontally overlapping entities.

    Purely a measurement of image coordinates, and labelled as such: without
    camera calibration, "above in the image" is not the same as "above in the
    world", and the evidence says so.
    """
    out: List[Relationship] = []
    entities = [e for e in scene.entities
                if e.bbox is not None and e.kind != EntityKind.TEXT.value]
    ts = scene.timestamp

    for i, a in enumerate(entities):
        for b in entities[i + 1:]:
            overlap_x = (min(a.bbox.x2, b.bbox.x2) - max(a.bbox.x1, b.bbox.x1))
            if overlap_x <= 0:
                continue
            narrower = min(a.bbox.width, b.bbox.width) or 1.0
            if overlap_x / narrower < 0.5:
                continue
            gap = a.bbox.y1 - b.bbox.y2 if a.bbox.y1 > b.bbox.y2 else b.bbox.y1 - a.bbox.y2
            if gap < 0:
                continue   # vertically overlapping, so neither is above
            upper, lower = (b, a) if a.bbox.y1 > b.bbox.y2 else (a, b)
            out.append(_rel(
                upper, RelationKind.ABOVE.value, lower,
                MEASUREMENT_CEILING,
                {"vertical_gap_px": round(gap, 1),
                 "horizontal_overlap": round(overlap_x / narrower, 3),
                 "basis": "image coordinates; no ground-plane calibration, so "
                          "this is not a claim about real-world height"},
                ts))
    return out


def infer_occlusion(scene: Scene, min_iou: float = 0.15) -> List[Relationship]:
    """occluding: overlapping boxes where one is demonstrably in front.

    The nearer object is the one whose base sits lower in the image, which
    holds for a camera looking down at a ground plane and fails for anything
    airborne. The assumption is recorded in the evidence rather than hidden.
    """
    out: List[Relationship] = []
    entities = [e for e in scene.entities
                if e.bbox is not None and e.kind != EntityKind.TEXT.value]
    ts = scene.timestamp

    for i, a in enumerate(entities):
        for b in entities[i + 1:]:
            iou = a.bbox.iou(b.bbox)
            if iou < min_iou:
                continue
            front, behind = (a, b) if a.bbox.y2 > b.bbox.y2 else (b, a)
            out.append(_rel(
                front, RelationKind.OCCLUDING.value, behind,
                min(GEOMETRY_CEILING, 0.4 + iou),
                {"iou": round(iou, 3),
                 "basis": "lower box base assumed nearer the camera; "
                          "assumes a ground plane"},
                ts))
    return out


def infer_all(scene: Scene, near_factor: float = 1.5,
              include_negative: bool = False) -> List[Relationship]:
    """Every single-frame relationship, attached to the scene.

    `include_negative` controls `far_from`, which is correct but voluminous -
    O(n^2) on a busy frame. It is off by default and turned on by search
    indexing, where ruling an association out is worth the volume.
    """
    rels: List[Relationship] = []
    for inferred in (infer_containment(scene), infer_carried_objects(scene),
                     infer_proximity(scene, near_factor), infer_vertical(scene),
                     infer_occlusion(scene)):
        rels.extend(inferred)

    if not include_negative:
        rels = [r for r in rels if r.predicate != RelationKind.FAR_FROM.value]

    for rel in rels:
        scene.add_relationship(rel)
    return rels
