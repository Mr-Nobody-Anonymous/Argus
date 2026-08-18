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


def _colour_counts(crop, grid: int = 24) -> Dict[str, int]:
    """Count coarse colour names over a downsampled crop, vectorised.

    A per-pixel Python loop over even a 24x24 crop costs ~1.5 ms, which at a
    dozen entities per frame dominated the whole perception budget. The same
    classification expressed as numpy masks is ~50x cheaper and, being the
    single place the rules live, keeps ``dominant_colour`` and
    ``colour_palette`` from ever disagreeing.
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return {}
    if crop is None or getattr(crop, "size", 0) == 0:
        return {}
    h, w = crop.shape[:2]
    if h * w < MIN_CROP_PIXELS:
        return {}

    small = cv2.resize(crop, (grid, grid), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

    # Same precedence as the scalar rule: darkness, then desaturation, then
    # hue. Checking hue first is how near-black pixels become "blue".
    counts: Dict[str, int] = {}
    dark = val < 45
    grey_ish = (~dark) & (sat < 40)
    coloured = (~dark) & (~grey_ish)

    n_dark = int(dark.sum())
    if n_dark:
        counts["black"] = n_dark
    white = int((grey_ish & (val > 190)).sum())
    if white:
        counts["white"] = white
    grey = int(grey_ish.sum()) - white
    if grey:
        counts["grey"] = grey

    for lo, hi, name in _HUE_BANDS:
        n = int((coloured & (hue >= lo) & (hue < hi)).sum())
        if n:
            counts[name] = counts.get(name, 0) + n
    return counts


def dominant_colour(crop) -> Optional[Tuple[str, float]]:
    """Most common coarse colour in a BGR crop, with its share as confidence.

    Returns None when the crop is too small to be meaningful.
    """
    try:
        import cv2
        import numpy as np
    except Exception:  # pragma: no cover - environment without OpenCV
        return None

    counts = _colour_counts(crop)
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


def colour_palette(crop, top_n: int = 3) -> List[Tuple[str, float]]:
    """The top few colours with their shares, not just the winner.

    A person in a red jacket and dark trousers is two colours, and reporting
    only "red, 0.45" throws away half of what makes them findable. The shares
    sum to at most 1.0 over the whole crop.
    """
    counts = _colour_counts(crop, grid=32)
    if not counts:
        return []
    total = float(sum(counts.values())) or 1.0
    ranked = sorted(counts.items(), key=lambda kv: -kv[1])
    return [(name, round(n / total, 3)) for name, n in ranked[:top_n]]


def crop_brightness(crop) -> Optional[float]:
    """Mean intensity of a crop, 0-255.

    Useful on its own ("this object is in shadow") and as a caveat on every
    other appearance attribute: colour read from a very dark crop deserves
    less trust, and recording the brightness lets a consumer apply that
    judgement instead of guessing.
    """
    try:
        import cv2
    except Exception:
        return None
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    try:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return round(float(grey.mean()), 1)
    except Exception:
        return None


def sharpness(crop) -> Optional[float]:
    """Laplacian variance: high for crisp crops, low for blurred ones.

    A motion-blurred crop is a moving object, and it is also a crop whose
    colour and shape attributes are less trustworthy - both facts worth
    recording.
    """
    try:
        import cv2
    except Exception:
        return None
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    try:
        grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return round(float(cv2.Laplacian(grey, cv2.CV_64F).var()), 1)
    except Exception:
        return None


def size_class(bbox: BBox, frame_w: int, frame_h: int) -> Optional[str]:
    """How much of the frame the object occupies.

    Frame-relative rather than absolute, because absolute pixel size means
    nothing without knowing the camera's field of view. "Large" here means
    "close to this camera", which is the operationally useful reading.
    """
    if bbox is None or frame_w <= 0 or frame_h <= 0:
        return None
    share = bbox.area / float(frame_w * frame_h)
    if share < 0.005:
        return "tiny"
    if share < 0.03:
        return "small"
    if share < 0.15:
        return "medium"
    if share < 0.4:
        return "large"
    return "dominant"


def frame_position(bbox: BBox, frame_w: int, frame_h: int) -> Optional[str]:
    """Coarse nine-cell position of the object within the frame.

    Cheap spatial language ("top-left", "centre") that makes observations
    readable without a calibrated ground plane, and gives edge-of-frame
    reasoning something to work with.
    """
    if bbox is None or frame_w <= 0 or frame_h <= 0:
        return None
    cx, cy = bbox.center
    col = "left" if cx < frame_w / 3 else ("right" if cx > 2 * frame_w / 3
                                           else "centre")
    row = "top" if cy < frame_h / 3 else ("bottom" if cy > 2 * frame_h / 3
                                          else "middle")
    return row if col == "centre" and row == "middle" else f"{row}-{col}"


def touches_edge(bbox: BBox, frame_w: int, frame_h: int,
                 margin: int = 4) -> bool:
    """Whether the box is clipped by the frame edge.

    A clipped box has an unreliable size and aspect ratio, so posture hints
    and size classes derived from it should be discounted - and an object at
    the edge is one that is entering or leaving, which matters on its own.
    """
    if bbox is None:
        return False
    return (bbox.x1 <= margin or bbox.y1 <= margin
            or bbox.x2 >= frame_w - margin or bbox.y2 >= frame_h - margin)


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

        clipped = touches_edge(entity.bbox, w, h)
        entity.set_attribute("clipped_by_frame_edge", clipped, 1.0,
                             Source.DETECTOR.value)

        position = frame_position(entity.bbox, w, h)
        if position is not None:
            entity.set_attribute("frame_position", position, 1.0,
                                 Source.DETECTOR.value)

        size = size_class(entity.bbox, w, h)
        if size is not None:
            # A clipped box under-reports its true extent, so the size class
            # derived from it is a lower bound rather than a measurement.
            entity.set_attribute("size_class", size, 0.6 if clipped else 1.0,
                                 Source.DETECTOR.value)

        crop = frame[y1:y2, x1:x2]
        result = dominant_colour(crop)
        if result is not None:
            name, share = result
            entity.set_attribute("dominant_colour", name, share,
                                 Source.APPEARANCE.value)

        palette = colour_palette(crop)
        if palette:
            entity.set_attribute("colour_palette", palette,
                                 palette[0][1], Source.APPEARANCE.value)

        brightness = crop_brightness(crop)
        if brightness is not None:
            entity.set_attribute("brightness", brightness, 1.0,
                                 Source.APPEARANCE.value)
            if brightness < 40:
                # Recording *why* other appearance attributes are weak is more
                # useful than silently emitting them at full confidence.
                entity.set_attribute("appearance_reliable", False, 0.8,
                                     Source.APPEARANCE.value)

        focus = sharpness(crop)
        if focus is not None:
            entity.set_attribute("sharpness", focus, 1.0,
                                 Source.APPEARANCE.value)
            if focus < 25:
                entity.set_attribute("motion_blurred", True, 0.6,
                                     Source.APPEARANCE.value)

        hint = posture_hint(entity.bbox)
        if hint is not None and not clipped:
            # Low confidence on purpose: the pose model must win when present.
            # Skipped entirely when clipped, because a half-visible person has
            # a meaningless aspect ratio.
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
