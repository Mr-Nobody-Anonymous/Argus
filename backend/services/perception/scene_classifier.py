"""Lightweight scene understanding: the context everything else is read against.

A detection means different things in different places. Ten people in a
shopping street is normal; ten people in a restricted plant room at 03:00 is
not. Without scene context every threshold has to be hand-tuned per camera.

This module reports **measurements first, interpretations second** - and keeps
them in separate attributes so a consumer can use the numbers and ignore the
labels:

    measured      mean_brightness, contrast, edge_density, colour_balance
    interpreted   lighting (daylight/dim/dark), visibility, density

What it deliberately does **not** do is claim to recognise place categories
("parking_lot", "warehouse"). That needs a trained classifier such as
Places365; asserting it from pixel statistics would be invention. The
`scene_type` attribute is therefore only set when a real classifier backend is
present, and the capability registry reports it as unavailable otherwise.

That restraint is the point: it is better to report four honest measurements
than one confident guess.

OpenCV and numpy are imported lazily so the perception package stays
importable without them.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from .observation import Scene, Source

logger = logging.getLogger(__name__)

# Mean intensity (0-255) boundaries between lighting regimes. Wide bands,
# because these are coarse labels and precision here would be false.
DARK_BELOW = 45
DIM_BELOW = 95
BRIGHT_ABOVE = 190

# Laplacian variance below this suggests fog, heavy blur or a defocused lens.
LOW_DETAIL_VARIANCE = 40.0


def measure_frame(frame) -> Dict[str, float]:
    """Cheap global statistics for one frame. Pure measurement, no labels."""
    try:
        import cv2
        import numpy as np
    except Exception:
        return {}
    if frame is None or getattr(frame, "size", 0) == 0:
        return {}

    try:
        # Work on a small copy: global statistics do not need full resolution
        # and this keeps the whole function well under a millisecond.
        small = cv2.resize(frame, (96, 96), interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        mean = float(grey.mean())
        contrast = float(grey.std())
        detail = float(cv2.Laplacian(grey, cv2.CV_64F).var())

        edges = cv2.Canny(grey, 60, 160)
        edge_density = float((edges > 0).sum()) / float(edges.size)

        b, g, r = (float(small[:, :, i].mean()) for i in range(3))
        total = max(1.0, b + g + r)

        return {
            "mean_brightness": round(mean, 1),
            "contrast": round(contrast, 1),
            "detail_variance": round(detail, 1),
            "edge_density": round(edge_density, 4),
            "warmth": round((r - b) / total, 3),   # >0 warm, <0 cool
        }
    except Exception as exc:  # noqa: BLE001 - measurement is best-effort
        logger.debug(f"Frame measurement failed: {exc}")
        return {}


def interpret(measurements: Dict[str, float]) -> Dict[str, tuple]:
    """Turn measurements into coarse labels, each with a confidence.

    Every label here is an interpretation of the numbers above, so callers
    record them at lower confidence than the measurements themselves.
    """
    out: Dict[str, tuple] = {}
    if not measurements:
        return out

    mean = measurements.get("mean_brightness")
    if mean is not None:
        if mean < DARK_BELOW:
            out["lighting"] = ("dark", 0.75)
        elif mean < DIM_BELOW:
            out["lighting"] = ("dim", 0.65)
        elif mean > BRIGHT_ABOVE:
            out["lighting"] = ("bright", 0.7)
        else:
            out["lighting"] = ("daylight", 0.6)

    detail = measurements.get("detail_variance")
    contrast = measurements.get("contrast")
    if detail is not None and contrast is not None:
        if detail < LOW_DETAIL_VARIANCE and contrast < 30:
            # Could be fog, rain, a dirty lens or a defocused camera. Naming
            # one of those would be a guess, so report the observable fact.
            out["visibility"] = ("degraded", 0.55)
        else:
            out["visibility"] = ("clear", 0.6)
    return out


def density_label(entity_count: int) -> tuple:
    """Crowd density from the entity count alone.

    Frame-relative, not absolute: a wide-angle view of a plaza and a corridor
    camera will disagree, which is exactly why the number is reported
    alongside the label.
    """
    if entity_count == 0:
        return ("empty", 0.8)
    if entity_count <= 3:
        return ("sparse", 0.7)
    if entity_count <= 10:
        return ("moderate", 0.65)
    if entity_count <= 25:
        return ("busy", 0.6)
    return ("crowded", 0.6)


def classify_scene(scene: Scene, frame=None) -> Dict[str, Any]:
    """Attach environment attributes to a Scene. Returns what was set."""
    applied: Dict[str, Any] = {}

    measurements = measure_frame(frame) if frame is not None else {}
    for name, value in measurements.items():
        # Measurements are recorded at full confidence: they are what the
        # sensor reported, not an opinion about it.
        scene.set_environment(name, value, 1.0, Source.SCENE.value)
        applied[name] = value

    for name, (label, confidence) in interpret(measurements).items():
        scene.set_environment(name, label, confidence, Source.SCENE.value)
        applied[name] = label

    label, confidence = density_label(len(scene.entities))
    scene.set_environment("density", label, confidence, Source.SCENE.value)
    scene.set_environment("entity_count", len(scene.entities), 1.0,
                          Source.SCENE.value)
    applied["density"] = label
    return applied
