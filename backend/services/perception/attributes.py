"""Cheap visual attributes extracted from a crop.

Dominant clothing colour is what makes "the man in the red jacket" searchable,
and it costs almost nothing next to a detection pass. This module deliberately
covers only what is reliable on CPU at low resolution: colour, geometry, and a
coarse aspect-ratio posture hint.

Two decisions keep the output honest:

* **Colour is reported with a confidence derived from how dominant it actually
  is.** A crop that is 70% navy is a confident claim; one split evenly between
  five colours is not, and saying "blue, 0.2" is far more useful than picking a
  winner and pretending.
* **Nothing is emitted when the crop is too small to mean anything.** A 6x12
  pixel person is a handful of pixels; a colour drawn from it is noise. Below
  the minimum the attribute is simply not set, which the model reads as
  "not observed" rather than "no colour".

``numpy`` and ``cv2`` are imported lazily inside the functions, so importing
this module never drags OpenCV in - the perception package stays dependency
free, and the CI guard that enforces that keeps passing.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from .observation import Attribute, BBox, Entity, Source

logger = logging.getLogger(__name__)

# Below this a crop carries no reliable colour information.
MIN_CROP_PIXELS = 24 * 24

# Coarse named colours in HSV. Hue ranges are inclusive-exclusive; red wraps
# around 180 in OpenCV's 0-179 hue scale and so is listed twice.
_HUE_BANDS: List[Tuple[int, int, str]] = [
    (0, 8, "red"), (8, 22, "orange"), (22, 33, "yellow"),
    (33, 78, "green"), (78, 100, "cyan"), (100, 131, "blue"),
    (131, 160, "purple"), (160, 180, "red"),
]


def _colour_name(h: int, s: int, v: int) -> str:
    """Map one HSV pixel to a coarse colour name.

    Saturation and value are checked first: a dark pixel is "black" whatever
    its hue, and treating a near-black pixel as "blue" because of sensor noise
    is a classic source of nonsense attributes.
    """
    if v < 45:
        return "black"
    if s < 40:
        return "white" if v > 190 else "grey"
    for lo, hi, name in _HUE_BANDS:
        if lo <= h < hi:
            return name
    return "unknown"


def dominant_colour(crop) -> Optional[Tuple[str, float]]:
    """Most common coarse colour in a BGR crop, with its share as confidence.

    Returns None when the crop is too small to be meaningful.
    """
    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover - environment without OpenCV
        return None

    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    h, w = crop.shape[:2]
    if h * w < MIN_CROP_PIXELS:
        return None

    # Downsample before counting: colour statistics do not need full
    # resolution, and this keeps the cost near zero on a 2-core box.
    small = cv2.resize(crop, (24, 24), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

    counts: Dict[str, int] = {}
    for row in hsv.reshape(-1, 3):
        name = _colour_name(int(row[0]), int(row[1]), int(row[2]))
        if name != "unknown":
            counts[name] = counts.get(name, 0) + 1

    if not counts:
        return None
    total = sum(counts.values())
    name, count = max(counts.items(), key=lambda kv: kv[1])
    return name, round(count / total, 3)


def posture_hint(bbox: BBox) -> Optional[str]:
    """Very coarse posture from the box's aspect ratio.

    This is a hint, not pose estimation: a wide box may be a person lying down
    or two people merged into one detection. It is emitted at low confidence
    and is always outranked by the real pose model, which is exactly what the
    source-authority rule is for.
    """
    if bbox is None or bbox.height <= 0:
        return None
    ratio = bbox.width / bbox.height
    if ratio > 1.4:
        return "horizontal"
    if ratio < 0.55:
        return "upright"
    return None


def enrich_entity(entity: Entity, frame) -> Entity:
    """Attach cheap visual attributes to one entity, in place.

    Never raises: an enrichment failure must degrade the entity's richness, not
    drop the detection.
    """
    if entity.bbox is None or frame is None:
        return entity

    try:
        h, w = frame.shape[:2]
        x1 = max(0, int(entity.bbox.x1))
        y1 = max(0, int(entity.bbox.y1))
        x2 = min(w, int(entity.bbox.x2))
        y2 = min(h, int(entity.bbox.y2))
        if x2 <= x1 or y2 <= y1:
            return entity

        entity.set_attribute("width_px", round(entity.bbox.width, 1), 1.0,
                             Source.DETECTOR.value)
        entity.set_attribute("height_px", round(entity.bbox.height, 1), 1.0,
                             Source.DETECTOR.value)
        if entity.bbox.height > 0:
            entity.set_attribute(
                "aspect_ratio", round(entity.bbox.width / entity.bbox.height, 3),
                1.0, Source.DETECTOR.value)

        crop = frame[y1:y2, x1:x2]
        result = dominant_colour(crop)
        if result is not None:
            name, share = result
            entity.set_attribute("dominant_colour", name, share,
                                 Source.APPEARANCE.value)

        hint = posture_hint(entity.bbox)
        if hint is not None:
            # Low confidence on purpose: the pose model must win when present.
            entity.set_attribute("posture", hint, 0.3, Source.APPEARANCE.value)

    except Exception as exc:  # noqa: BLE001 - enrichment is best-effort
        logger.debug(f"Attribute enrichment failed: {exc}")
    return entity


def enrich_scene(scene, frame) -> int:
    """Enrich every entity in a scene. Returns how many were touched."""
    count = 0
    for entity in getattr(scene, "entities", []):
        enrich_entity(entity, frame)
        count += 1
    return count
