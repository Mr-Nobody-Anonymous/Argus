"""Change detection: what is different from normal?

Detectors answer "what is in this frame". They cannot answer "what is different
from how this place usually looks" - and that question catches things no class
list anticipates: a door that is now open, a crate that appeared overnight, a
corridor that is unusually busy, a parked car that left.

Two independent mechanisms, because they fail in different ways:

1. **Entity-level change** - compares the set of tracked entities against a
   rolling baseline. Reliable, semantic, and only sees what the detector knows.
2. **Pixel-level change** - compares a downsampled greyscale signature of the
   frame against a baseline. Sees *anything*, including objects with no class,
   but is fooled by lighting and camera shake.

They are kept separate rather than merged into one score, because a pixel
change with no entity change means something different (lighting, weather, a
novel object) from an entity change with no pixel change (a tracking artefact).
Collapsing them would destroy that signal.

**Baselines are time-of-day aware.** A car park at 03:00 and at 13:00 are
different normals, and comparing against a single global average would report
every morning as an anomaly. Buckets are hourly.

Occupancy statistics use a running mean and variance rather than stored
samples, so a camera can run for months in constant memory.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .observation import Observation, Scene, Source

# How many hourly buckets. 24 keeps a day-shaped profile per camera.
BUCKETS = 24

# A bucket needs this many samples before its statistics mean anything. Below
# it, reporting an anomaly is just reporting inexperience.
MIN_SAMPLES = 30

# Standard deviations from the bucket mean before occupancy is unusual.
OCCUPANCY_SIGMA = 3.0

# Pixel-signature grid. Deliberately tiny: 16x16 = 256 cells is enough to spot
# a region-sized change and costs microseconds to compare.
GRID = 16

# Fraction of cells that must differ before the frame counts as changed.
PIXEL_CHANGE_FRACTION = 0.18

# Per-cell intensity delta (0-255) counted as a difference. Below this is
# sensor noise and compression artefacts.
CELL_DELTA = 28


def _now() -> float:
    return time.time()


def _bucket(timestamp: float) -> int:
    return int(time.localtime(timestamp).tm_hour) % BUCKETS


@dataclass
class RunningStat:
    """Welford's online mean/variance - constant memory, no sample list."""

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def push(self, value: float) -> None:
        self.count += 1
        delta = value - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (value - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def stdev(self) -> float:
        return math.sqrt(self.variance)

    def zscore(self, value: float) -> float:
        sd = self.stdev
        if sd < 1e-6:
            # No spread observed yet: treat any difference from the mean as
            # notable but bounded, rather than dividing by ~zero.
            return 0.0 if abs(value - self.mean) < 1e-6 else 3.0
        return (value - self.mean) / sd

    def to_dict(self) -> Dict[str, Any]:
        return {"count": self.count, "mean": round(self.mean, 2),
                "stdev": round(self.stdev, 2)}


@dataclass
class CameraBaseline:
    """What normal looks like for one camera, per hour of day."""

    camera_id: int
    occupancy: Dict[int, RunningStat] = field(
        default_factory=lambda: {i: RunningStat() for i in range(BUCKETS)})
    category_counts: Dict[int, Dict[str, RunningStat]] = field(
        default_factory=lambda: {i: {} for i in range(BUCKETS)})
    signature: Optional[List[int]] = None      # last pixel signature
    signature_baseline: Optional[List[float]] = None  # smoothed reference
    updated_at: float = field(default_factory=_now)

    def sample_count(self, timestamp: Optional[float] = None) -> int:
        return self.occupancy[_bucket(timestamp or _now())].count

    def is_ready(self, timestamp: Optional[float] = None) -> bool:
        """Whether this bucket has seen enough to judge anything."""
        return self.sample_count(timestamp) >= MIN_SAMPLES


class ChangeDetector:
    """Maintains per-camera baselines and reports departures from them."""

    def __init__(self):
        self._baselines: Dict[int, CameraBaseline] = {}
        self._lock = threading.RLock()
        self._last_entities: Dict[int, Dict[int, str]] = {}   # camera -> {track: category}

    # -- baseline -------------------------------------------------------------

    def baseline(self, camera_id: int) -> CameraBaseline:
        with self._lock:
            b = self._baselines.get(camera_id)
            if b is None:
                b = CameraBaseline(camera_id=camera_id)
                self._baselines[camera_id] = b
            return b

    def observe(self, scene: Scene) -> None:
        """Fold a scene into the baseline. Always called, even when judging."""
        b = self.baseline(scene.camera_id)
        bucket = _bucket(scene.timestamp)
        with self._lock:
            b.occupancy[bucket].push(float(len(scene.entities)))
            counts: Dict[str, int] = {}
            for e in scene.entities:
                counts[e.category] = counts.get(e.category, 0) + 1
            slot = b.category_counts[bucket]
            for category, n in counts.items():
                slot.setdefault(category, RunningStat()).push(float(n))
            # Categories absent this frame still get a zero sample, otherwise
            # the mean for a rare object is computed only from frames where it
            # appeared and is therefore never surprising.
            for category, stat in slot.items():
                if category not in counts:
                    stat.push(0.0)
            b.updated_at = scene.timestamp

    # -- entity-level change --------------------------------------------------

    def entity_changes(self, scene: Scene) -> List[Observation]:
        """Objects that appeared or vanished relative to the previous frame."""
        current = {e.track_id: e.category for e in scene.entities
                   if e.track_id is not None}
        with self._lock:
            previous = self._last_entities.get(scene.camera_id, {})
            self._last_entities[scene.camera_id] = current

        if not previous:
            return []   # first frame has nothing to compare against

        out: List[Observation] = []
        appeared = set(current) - set(previous)
        vanished = set(previous) - set(current)

        for tid in sorted(appeared):
            out.append(Observation(
                kind="object_appeared",
                summary=f"{current[tid]} {tid} entered the scene",
                camera_id=scene.camera_id, timestamp=scene.timestamp,
                confidence=0.55, source=Source.TEMPORAL.value,
                track_ids=[tid],
                evidence=[f"not present in the previous frame",
                          f"category {current[tid]}"],
            ))
        for tid in sorted(vanished):
            out.append(Observation(
                kind="object_left_frame",
                summary=f"{previous[tid]} {tid} left the scene",
                camera_id=scene.camera_id, timestamp=scene.timestamp,
                confidence=0.5, source=Source.TEMPORAL.value,
                track_ids=[tid],
                evidence=[f"present in the previous frame, absent now",
                          f"category {previous[tid]}"],
            ))
        return out

    # -- occupancy anomaly ----------------------------------------------------

    def occupancy_anomaly(self, scene: Scene) -> Optional[Observation]:
        """Unusually crowded or unusually empty for this camera at this hour."""
        b = self.baseline(scene.camera_id)
        bucket = _bucket(scene.timestamp)
        with self._lock:
            stat = b.occupancy[bucket]
            if stat.count < MIN_SAMPLES:
                return None
            n = float(len(scene.entities))
            z = stat.zscore(n)

        if abs(z) < OCCUPANCY_SIGMA:
            return None
        direction = "higher" if z > 0 else "lower"
        hour = time.strftime("%H:00", time.localtime(scene.timestamp))
        return Observation(
            kind="occupancy_anomaly",
            summary=(f"occupancy {int(n)} is unusually {direction} for "
                     f"{hour} on camera {scene.camera_id}"),
            camera_id=scene.camera_id, timestamp=scene.timestamp,
            confidence=min(0.85, 0.4 + abs(z) / 20.0),
            source=Source.TEMPORAL.value,
            evidence=[
                f"{int(n)} entities observed",
                f"baseline mean {stat.mean:.1f} +/- {stat.stdev:.1f} for {hour}",
                f"{abs(z):.1f} standard deviations from normal",
                f"baseline built from {stat.count} samples",
            ],
            metadata={"zscore": round(z, 2), "hour_bucket": bucket},
        )

    # -- pixel-level change ---------------------------------------------------

    def signature(self, frame) -> Optional[List[int]]:
        """Downsampled greyscale signature of a frame.

        Imported lazily so this module stays importable without OpenCV.
        """
        try:
            import cv2
        except Exception:
            return None
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        try:
            grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            small = cv2.resize(grey, (GRID, GRID), interpolation=cv2.INTER_AREA)
            return [int(v) for v in small.flatten()]
        except Exception:
            return None

    def pixel_change(self, camera_id: int, frame,
                     timestamp: Optional[float] = None) -> Optional[Observation]:
        """Regions that differ from the smoothed baseline image.

        The baseline is an exponential moving average, so gradual changes
        (the sun moving, shadows lengthening) are absorbed while abrupt ones
        stand out. That is the difference between a useful signal and an alarm
        every time a cloud passes.
        """
        sig = self.signature(frame)
        if sig is None:
            return None
        b = self.baseline(camera_id)
        ts = timestamp if timestamp is not None else _now()

        with self._lock:
            base = b.signature_baseline
            if base is None:
                b.signature_baseline = [float(v) for v in sig]
                b.signature = sig
                return None
            changed = sum(1 for a, bb in zip(sig, base) if abs(a - bb) > CELL_DELTA)
            fraction = changed / float(len(sig))
            # Update the baseline AFTER comparing, so a change is measured
            # against the old normal rather than partly against itself.
            b.signature_baseline = [0.95 * old + 0.05 * new
                                    for old, new in zip(base, sig)]
            b.signature = sig

        if fraction < PIXEL_CHANGE_FRACTION:
            return None
        return Observation(
            kind="scene_change",
            summary=f"{fraction * 100:.0f}% of camera {camera_id}'s view changed",
            camera_id=camera_id, timestamp=ts,
            confidence=min(0.8, 0.35 + fraction),
            source=Source.TEMPORAL.value,
            evidence=[
                f"{changed} of {len(sig)} regions differ from baseline",
                f"per-region threshold {CELL_DELTA} intensity levels",
                "baseline is an exponential moving average, so gradual "
                "lighting change is absorbed",
            ],
            metadata={"changed_fraction": round(fraction, 3)},
        )

    # -- combined -------------------------------------------------------------

    def analyse(self, scene: Scene, frame=None) -> List[Observation]:
        """Every change signal for one frame, then update the baseline."""
        out: List[Observation] = []
        try:
            out.extend(self.entity_changes(scene))
            anomaly = self.occupancy_anomaly(scene)
            if anomaly is not None:
                out.append(anomaly)
            if frame is not None:
                change = self.pixel_change(scene.camera_id, frame, scene.timestamp)
                if change is not None:
                    out.append(change)
        finally:
            # The baseline must learn even from anomalous frames, or a
            # persistent new normal is reported forever.
            self.observe(scene)
        return out

    def report(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "cameras": {
                    str(cid): {
                        "ready": b.is_ready(),
                        "samples_this_hour": b.sample_count(),
                        "min_samples": MIN_SAMPLES,
                        "occupancy_by_hour": {
                            str(h): s.to_dict()
                            for h, s in b.occupancy.items() if s.count
                        },
                    }
                    for cid, b in self._baselines.items()
                }
            }

    def reset(self) -> None:
        with self._lock:
            self._baselines.clear()
            self._last_entities.clear()
