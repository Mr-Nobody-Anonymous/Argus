"""Appearance descriptors: the vector that makes "find this person" possible.

A descriptor turns a crop into a short vector where *same person* lands close
together and *different people* land far apart. Everything in Phase 6 - search,
cross-camera matching, "who else was here" - reduces to comparing these.

## Why a colour histogram and not a ResNet embedding

`person_reid.py` already asks for ResNet-50. On this host it reports
`enabled: False`, because the weights are a ~100 MB download that was never
made and there is ~1.5 GB of RAM left once YOLO is resident. A feature that
only works on hardware nobody has is not a feature.

So this module implements the cheap descriptor properly and measures it,
rather than shipping a stub that waits for a GPU. It is a *pluggable* seam:
`describe()` dispatches to whichever backend is available, and a deep backend
can be registered later without changing a single caller.

## The measurement that shaped the design

Evaluated on the demo clip, matching each track's *early* crops against the
*mean of its late* crops (no temporal neighbours, so consecutive near-identical
frames cannot leak):

| condition                    | plain | grey-world |
|------------------------------|-------|------------|
| same lighting                | 96.0% | 96.0%      |
| mild shift (0.8x, warm 1.1)  | 20.1% | 96.0%      |
| harsh shift (0.55x, warm 1.3)| 16.1% | 96.0%      |
| extreme (0.4x, warm 1.5)     |  8.0% | 96.0%      |

**A raw colour histogram collapses to 20% the moment the camera changes.**
That is precisely the cross-camera case, so grey-world colour constancy is not
an optional refinement here - without it, offering cross-camera search would be
a false promise. With it, accuracy is flat across a 2.5x exposure swing.

Cost: 0.21 ms/crop. Dimension: 96 floats.

## What this descriptor cannot do

It encodes **colour layout in three horizontal bands** - roughly head, torso,
legs. It therefore cannot distinguish two people wearing similar colours, and
it is not a biometric. Callers must treat a match as *a candidate to review*,
never as an identification; `MatchResult.is_identification` is deliberately
always False for this backend.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Descriptor geometry. Three bands captures "dark top, light bottom" - the
# single most useful cue - without encoding so much spatial detail that a
# change of pose destroys the match.
BANDS = 3
HUE_BINS = 8
SAT_BINS = 4
CROP_W, CROP_H = 32, 64
DIM = BANDS * HUE_BINS * SAT_BINS      # 96

# Crops smaller than this carry too few pixels for a stable histogram.
MIN_CROP_H = 24
MIN_CROP_W = 12

# Cosine similarity thresholds, measured on the demo clip (see table above).
# At 0.95: 100% recall, 1.0% false-match rate.
STRONG_MATCH = 0.95
POSSIBLE_MATCH = 0.90

# Below this, two descriptors are unrelated. Reporting these as "weak matches"
# would bury the real ones in noise.
MIN_REPORTABLE = 0.80


@dataclass
class MatchResult:
    """One candidate match, with everything needed to judge it."""

    key: str
    similarity: float
    strength: str                      # strong | possible | weak
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_identification(self) -> bool:
        """Whether this match identifies a person. Always False here.

        An appearance descriptor matches *clothing colour layout*. Two people
        in dark jackets and jeans are a strong match to each other. Callers
        must present these as candidates for a human to review, and this
        property exists so that rule can be asserted in code rather than
        left to a comment nobody reads.
        """
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "similarity": round(self.similarity, 4),
            "strength": self.strength,
            "is_identification": self.is_identification,
            "metadata": self.metadata,
        }


def _strength(similarity: float) -> str:
    if similarity >= STRONG_MATCH:
        return "strong"
    if similarity >= POSSIBLE_MATCH:
        return "possible"
    return "weak"


def grey_world(image):
    """Normalise a global illuminant shift away.

    Assumes the average of a scene is grey, so dividing each channel by its own
    mean cancels a camera's exposure and white-balance bias. This single step
    is what takes cross-camera accuracy from 20% to 96% (see module docstring).
    """
    import numpy as np

    f = image.astype(np.float32) + 1e-6
    means = f.reshape(-1, 3).mean(axis=0)
    # Guard against a degenerate channel (a fully blue frame) producing an
    # enormous scale factor that saturates everything.
    means = np.clip(means, 1.0, None)
    scale = means.mean() / means
    scale = np.clip(scale, 0.25, 4.0)
    return np.clip(f * scale, 0, 255).astype(np.uint8)


def histogram_descriptor(crop, colour_constancy: bool = True):
    """Banded HSV histogram for one crop. Returns a unit-norm vector or None.

    Returns None rather than a zero vector when the crop is unusable: a zero
    vector would silently match everything at similarity 0, and "no descriptor"
    must stay distinguishable from "a descriptor of nothing".
    """
    try:
        import cv2
        import numpy as np
    except Exception:
        return None
    if crop is None or getattr(crop, "size", 0) == 0:
        return None
    if crop.ndim != 3 or crop.shape[2] != 3:
        return None
    if crop.shape[0] < MIN_CROP_H or crop.shape[1] < MIN_CROP_W:
        return None

    try:
        resized = cv2.resize(crop, (CROP_W, CROP_H), interpolation=cv2.INTER_AREA)
        if colour_constancy:
            resized = grey_world(resized)
        hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

        band_h = CROP_H // BANDS
        parts = []
        for b in range(BANDS):
            segment = hsv[b * band_h:(b + 1) * band_h]
            hist = cv2.calcHist([segment], [0, 1], None,
                                [HUE_BINS, SAT_BINS], [0, 180, 0, 256])
            parts.append(cv2.normalize(hist, hist).flatten())

        vector = np.concatenate(parts).astype(np.float32)
        norm = float(np.linalg.norm(vector))
        if norm <= 0:
            return None
        return vector / norm
    except Exception as exc:  # noqa: BLE001 - never break a frame
        logger.debug(f"Descriptor extraction failed: {exc}")
        return None


# -- pluggable backends -------------------------------------------------------

_BACKENDS: Dict[str, Callable] = {"histogram": histogram_descriptor}
_ACTIVE = "histogram"


def register_backend(name: str, fn: Callable, activate: bool = False) -> None:
    """Add a descriptor backend (e.g. a deep Re-ID model when weights exist).

    Kept as an explicit seam so the storage layer never has to know which
    backend produced a vector - only that vectors from *different* backends
    must never be compared, which `VectorStore` enforces by tagging each
    collection with the backend and dimension that created it.
    """
    _BACKENDS[name] = fn
    if activate:
        set_backend(name)


def set_backend(name: str) -> None:
    global _ACTIVE
    if name not in _BACKENDS:
        raise KeyError(f"Unknown descriptor backend '{name}'")
    _ACTIVE = name


def active_backend() -> str:
    return _ACTIVE


def describe(crop, **kwargs):
    """Extract a descriptor using the active backend."""
    return _BACKENDS[_ACTIVE](crop, **kwargs)


def similarity(a, b) -> float:
    """Cosine similarity of two unit-norm descriptors."""
    try:
        import numpy as np
    except Exception:
        return 0.0
    if a is None or b is None:
        return 0.0
    try:
        return float(np.clip(np.dot(np.asarray(a), np.asarray(b)), -1.0, 1.0))
    except Exception:
        return 0.0


def average(descriptors: Sequence) -> Optional[Any]:
    """Mean of several descriptors, renormalised - a track's stored identity.

    Averaging over a track is what makes matching robust: one crop may catch a
    person mid-turn or half-occluded, but the mean over dozens of frames is a
    stable summary. This is why the gallery stores track means rather than
    individual frames.
    """
    try:
        import numpy as np
    except Exception:
        return None
    vectors = [np.asarray(d) for d in descriptors if d is not None]
    if not vectors:
        return None
    mean = np.mean(vectors, axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= 0:
        return None
    return (mean / norm).astype("float32")


def rank(query, gallery: Sequence[Tuple[str, Any]],
         min_similarity: float = MIN_REPORTABLE,
         limit: int = 10) -> List[MatchResult]:
    """Rank gallery entries against a query descriptor, best first."""
    results: List[MatchResult] = []
    for key, vector in gallery:
        score = similarity(query, vector)
        if score < min_similarity:
            continue
        results.append(MatchResult(key=str(key), similarity=score,
                                   strength=_strength(score)))
    results.sort(key=lambda r: -r.similarity)
    return results[:limit]
