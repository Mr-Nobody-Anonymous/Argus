"""Temporal intelligence: turn per-frame entities into tracks that accumulate.

A `Scene` is one instant. Almost nothing interesting is visible in one instant:
loitering, circling, abandonment, following and "something disappeared" are all
statements about *change*. This module keeps the history that makes them
answerable.

The central object is `Track` - everything Argus has reliably observed about
one entity over time:

    Track 17
      +-- first_seen / last_seen / duration
      +-- trajectory      (bounded ring of positions)
      +-- velocity / distance travelled
      +-- attributes      merged by source authority, with history
      +-- zones visited   and current zone
      +-- cameras seen    (cross-camera identity when Re-ID links them)
      +-- observations    durable statements, each with evidence

Design constraints that matter:

* **Bounded memory.** A camera running for a week must not accumulate an
  unbounded trajectory. Every history is a `deque` with a hard cap, so a track
  costs a predictable amount of RAM no matter how long it lives.
* **Absence is a fact.** A track that stops being seen is not deleted
  immediately - it becomes `stale`, then `lost`. Deleting on the first missed
  frame would make "this object disappeared" unobservable, and occlusion would
  look like departure.
* **Inference is never stored as evidence.** Derived claims (dwell time,
  loitering) are `Observation`s carrying the measurements that produced them,
  so a reviewer can disagree with the conclusion without doubting the data.
* **Stdlib only.** Same rule as the rest of the perception layer.
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from .observation import (
    Attribute,
    BBox,
    Entity,
    LOW_CONFIDENCE,
    Observation,
    Scene,
    Source,
)

# History caps. Chosen so a track is a few tens of KB at worst: at 5 fps a
# 512-point trajectory is ~100 s of motion, which is longer than any window we
# reason over, and far cheaper than keeping every frame.
MAX_TRAJECTORY = 512
MAX_ATTRIBUTE_HISTORY = 32
MAX_OBSERVATIONS = 64

# A track not seen for this long is stale (probably occluded); after the second
# threshold it is lost (probably gone). Two thresholds exist because occlusion
# and departure are different facts and collapsing them loses information.
STALE_AFTER_S = 2.0
LOST_AFTER_S = 10.0


def _now() -> float:
    return time.time()


@dataclass
class TrackPoint:
    """One position in time. Deliberately tiny - there may be hundreds."""

    timestamp: float
    x: float
    y: float
    camera_id: int
    bbox: Optional[List[float]] = None


@dataclass
class Track:
    """Accumulated knowledge about one entity across frames and cameras."""

    track_id: int
    kind: str
    category: str
    first_seen: float = field(default_factory=_now)
    last_seen: float = field(default_factory=_now)
    frame_count: int = 0

    trajectory: Deque[TrackPoint] = field(
        default_factory=lambda: deque(maxlen=MAX_TRAJECTORY))
    attributes: Dict[str, Attribute] = field(default_factory=dict)
    attribute_history: Dict[str, Deque[Attribute]] = field(default_factory=dict)

    cameras_seen: List[int] = field(default_factory=list)
    zones_visited: List[str] = field(default_factory=list)
    current_zones: List[str] = field(default_factory=list)
    observations: Deque[Observation] = field(
        default_factory=lambda: deque(maxlen=MAX_OBSERVATIONS))

    # Set when the track is retired, so "when did it disappear" stays answerable.
    lost_at: Optional[float] = None

    # -- lifecycle ------------------------------------------------------------

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    # Latest timestamp the owning store has seen, in the SAME clock as the
    # frames. Ages are measured against this rather than the wall clock so
    # replaying archived footage does not instantly mark everything lost.
    reference_time: Optional[float] = None

    def _reference(self, now: Optional[float] = None) -> float:
        if now is not None:
            return now
        if self.reference_time is not None:
            return self.reference_time
        return _now()

    def age(self, now: Optional[float] = None) -> float:
        """Seconds since this track was last observed, in stream time."""
        return max(0.0, self._reference(now) - self.last_seen)

    def status(self, now: Optional[float] = None) -> str:
        a = self.age(now)
        if self.lost_at is not None or a >= LOST_AFTER_S:
            return "lost"
        if a >= STALE_AFTER_S:
            return "stale"
        return "active"

    # -- updates --------------------------------------------------------------

    def observe(self, entity: Entity, camera_id: int,
                timestamp: Optional[float] = None) -> None:
        """Fold one frame's view of this entity into the accumulated track."""
        ts = float(timestamp if timestamp is not None else _now())
        self.last_seen = ts
        self.frame_count += 1
        self.lost_at = None  # a re-sighting revives it

        if entity.bbox is not None:
            x, y = entity.bbox.bottom_center
            self.trajectory.append(TrackPoint(
                timestamp=ts, x=x, y=y, camera_id=camera_id,
                bbox=entity.bbox.to_list()))

        if camera_id not in self.cameras_seen:
            self.cameras_seen.append(camera_id)

        # A more specific category from a better source should win, but a
        # detector flip-flopping between "car" and "truck" must not thrash the
        # label, so only upgrade when the entity is at least as authoritative.
        if entity.category not in ("unknown", "", None):
            if self.category == "unknown":
                self.category = entity.category

        for name, attr in entity.attributes.items():
            self.merge_attribute(attr)

    def merge_attribute(self, incoming: Attribute) -> Attribute:
        """Keep the best-sourced claim, but never throw the history away.

        Two models disagreeing is information. The current value is what the
        system acts on; the history is what a reviewer inspects.
        """
        hist = self.attribute_history.setdefault(
            incoming.name, deque(maxlen=MAX_ATTRIBUTE_HISTORY))
        hist.append(incoming)

        current = self.attributes.get(incoming.name)
        if current is None or incoming.outranks(current):
            self.attributes[incoming.name] = incoming
            return incoming
        return current

    def set_attribute(self, name: str, value: Any, confidence: float = 1.0,
                      source: str = Source.TEMPORAL.value) -> Attribute:
        return self.merge_attribute(Attribute(name=name, value=value,
                                              confidence=confidence, source=source))

    def get(self, name: str, default: Any = None) -> Any:
        attr = self.attributes.get(name)
        return default if attr is None else attr.value

    def observed(self, name: str) -> bool:
        return name in self.attributes

    def enter_zone(self, zone: str) -> bool:
        """Record a zone entry. Returns True only on an actual transition."""
        if zone in self.current_zones:
            return False
        self.current_zones.append(zone)
        if zone not in self.zones_visited:
            self.zones_visited.append(zone)
        return True

    def exit_zone(self, zone: str) -> bool:
        if zone not in self.current_zones:
            return False
        self.current_zones.remove(zone)
        return True

    def add_observation(self, obs: Observation) -> Observation:
        self.observations.append(obs)
        return obs

    # -- derived motion -------------------------------------------------------

    def displacement(self) -> float:
        """Straight-line pixels from first to last known position."""
        if len(self.trajectory) < 2:
            return 0.0
        a, b = self.trajectory[0], self.trajectory[-1]
        return math.hypot(b.x - a.x, b.y - a.y)

    def path_length(self) -> float:
        """Total distance actually walked, which can be far larger than the
        displacement - that gap is what reveals circling and pacing."""
        total = 0.0
        pts = list(self.trajectory)
        for a, b in zip(pts, pts[1:]):
            total += math.hypot(b.x - a.x, b.y - a.y)
        return total

    def velocity(self, window_s: float = 1.0) -> Tuple[float, float]:
        """(vx, vy) in pixels/second over the recent window.

        Measured over a window rather than the last two frames: consecutive
        detections jitter by a few pixels, and differencing them produces wild
        instantaneous speeds.
        """
        pts = list(self.trajectory)
        if len(pts) < 2:
            return (0.0, 0.0)
        latest = pts[-1]
        cutoff = latest.timestamp - window_s
        window = [p for p in pts if p.timestamp >= cutoff] or pts[-2:]
        first = window[0]
        dt = latest.timestamp - first.timestamp
        if dt <= 0:
            return (0.0, 0.0)
        return ((latest.x - first.x) / dt, (latest.y - first.y) / dt)

    def speed(self, window_s: float = 1.0) -> float:
        vx, vy = self.velocity(window_s)
        return math.hypot(vx, vy)

    def is_stationary(self, tolerance_px: float = 25.0,
                      window_s: float = 3.0) -> bool:
        """True when the entity has barely moved recently.

        Uses spread over the window rather than speed, because a person pacing
        in place has a high instantaneous speed but is not going anywhere.
        """
        pts = list(self.trajectory)
        if len(pts) < 2:
            return False
        cutoff = pts[-1].timestamp - window_s
        window = [p for p in pts if p.timestamp >= cutoff]
        if len(window) < 2:
            return False
        xs = [p.x for p in window]
        ys = [p.y for p in window]
        return (max(xs) - min(xs)) <= tolerance_px and (max(ys) - min(ys)) <= tolerance_px

    def direction(self) -> Optional[str]:
        """Compass-style heading, or None when there is no meaningful motion."""
        vx, vy = self.velocity()
        if math.hypot(vx, vy) < 1.0:  # px/s - below this it is noise
            return None
        # Screen coordinates: +y is downward, so north is negative y.
        angle = math.degrees(math.atan2(-vy, vx)) % 360
        for lo, hi, name in (
            (337.5, 360, "east"), (0, 22.5, "east"),
            (22.5, 67.5, "northeast"), (67.5, 112.5, "north"),
            (112.5, 157.5, "northwest"), (157.5, 202.5, "west"),
            (202.5, 247.5, "southwest"), (247.5, 292.5, "south"),
            (292.5, 337.5, "southeast"),
        ):
            if lo <= angle < hi:
                return name
        return None

    # -- serialisation --------------------------------------------------------

    def summary(self) -> Dict[str, Any]:
        """The accumulated picture, in the form an operator or API would read."""
        return {
            "track_id": self.track_id,
            "kind": self.kind,
            "category": self.category,
            "status": self.status(),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "duration_s": round(self.duration, 2),
            "frame_count": self.frame_count,
            "cameras_seen": list(self.cameras_seen),
            "zones_visited": list(self.zones_visited),
            "current_zones": list(self.current_zones),
            "speed_px_s": round(self.speed(), 2),
            "direction": self.direction(),
            "displacement_px": round(self.displacement(), 1),
            "path_length_px": round(self.path_length(), 1),
            "stationary": self.is_stationary(),
            "attributes": {
                name: {"value": a.value, "confidence": a.confidence,
                       "source": a.source}
                for name, a in self.attributes.items()
            },
            "observations": [o.to_dict() for o in self.observations],
        }

    def describe(self) -> str:
        """One readable line - the natural text to embed for semantic search."""
        bits = [f"{self.category} {self.track_id}"]
        if self.observed("identity"):
            bits.append(f"identified as {self.get('identity')}")
        if self.observed("plate"):
            bits.append(f"plate {self.get('plate')}")
        if self.observed("posture"):
            bits.append(str(self.get("posture")))
        d = self.direction()
        if d:
            bits.append(f"moving {d}")
        elif self.is_stationary():
            bits.append("stationary")
        if self.current_zones:
            bits.append(f"in {', '.join(self.current_zones)}")
        if self.duration >= 1:
            bits.append(f"for {self.duration:.0f}s")
        return ", ".join(bits)


class TrackStore:
    """Thread-safe registry of tracks for one deployment.

    The processing loop runs per camera in its own thread, and the API reads
    tracks from request handlers, so every mutation is lock-guarded. An
    `RLock` specifically, because the retirement sweep calls back into methods
    that take the same lock.
    """

    def __init__(self, max_tracks: int = 2048):
        self._tracks: Dict[int, Track] = {}
        self._lock = threading.RLock()
        self._max_tracks = max_tracks

    def __len__(self) -> int:
        with self._lock:
            return len(self._tracks)

    def get(self, track_id: int) -> Optional[Track]:
        with self._lock:
            return self._tracks.get(track_id)

    def all(self, include_lost: bool = False) -> List[Track]:
        with self._lock:
            tracks = list(self._tracks.values())
        if include_lost:
            return tracks
        return [t for t in tracks if t.status() != "lost"]

    def update_from_scene(self, scene: Scene) -> List[Track]:
        """Fold a whole scene into the store. The main entry point."""
        touched: List[Track] = []
        with self._lock:
            for entity in scene.entities:
                if entity.track_id is None:
                    # Untracked entities are real but not followable; they
                    # belong to the scene, not to a track.
                    continue
                track = self._tracks.get(entity.track_id)
                if track is None:
                    track = Track(track_id=entity.track_id, kind=entity.kind,
                                  category=entity.category,
                                  first_seen=scene.timestamp,
                                  last_seen=scene.timestamp)
                    self._tracks[entity.track_id] = track
                track.observe(entity, scene.camera_id, scene.timestamp)
                touched.append(track)
            # Advance every track's clock to the newest frame time, so tracks
            # that were NOT seen in this frame still age in stream time.
            for existing in self._tracks.values():
                if (existing.reference_time is None
                        or scene.timestamp > existing.reference_time):
                    existing.reference_time = scene.timestamp
            self._evict_if_needed()
        return touched

    def retire_stale(self, now: Optional[float] = None) -> List[Track]:
        """Mark tracks lost once they exceed the lost threshold.

        Returns the ones that transitioned on this call, so a caller can raise
        exactly one "disappeared" observation rather than one per frame.
        """
        newly_lost: List[Track] = []
        with self._lock:
            for track in self._tracks.values():
                # Each track ages against the stream clock it was updated with.
                ts = now if now is not None else track._reference()
                if track.lost_at is None and track.age(ts) >= LOST_AFTER_S:
                    track.lost_at = ts
                    newly_lost.append(track)
        return newly_lost

    def forget(self, track_id: int) -> bool:
        with self._lock:
            return self._tracks.pop(track_id, None) is not None

    def clear(self) -> None:
        with self._lock:
            self._tracks.clear()

    def _evict_if_needed(self) -> None:
        """Drop the longest-lost tracks when over capacity.

        Called with the lock held. Bounding this matters: a busy junction
        generates thousands of ids a day, and an unbounded dict is a slow leak
        that only shows up in production.
        """
        if len(self._tracks) <= self._max_tracks:
            return
        ordered = sorted(self._tracks.values(), key=lambda t: t.last_seen)
        for track in ordered[: len(self._tracks) - self._max_tracks]:
            self._tracks.pop(track.track_id, None)


# ── temporal observations ────────────────────────────────────────────────────

def detect_dwell(track: Track, threshold_s: float = 30.0,
                 tolerance_px: float = 25.0) -> Optional[Observation]:
    """Loitering: present a long time without going anywhere.

    Duration alone is not loitering - someone waiting at a bus stop and someone
    walking a long corridor both have long durations. The distinguishing fact
    is that the position barely changed, so both are required and both are
    reported as evidence.
    """
    if track.duration < threshold_s:
        return None
    if not track.is_stationary(tolerance_px=tolerance_px, window_s=min(10.0, threshold_s)):
        return None

    return Observation(
        kind="dwell",
        summary=f"{track.category} {track.track_id} stationary for {track.duration:.0f}s",
        confidence=min(0.9, 0.5 + track.duration / (threshold_s * 10)),
        source=Source.TEMPORAL.value,
        track_ids=[track.track_id],
        evidence=[
            f"duration {track.duration:.0f}s (threshold {threshold_s:.0f}s)",
            f"displacement {track.displacement():.0f}px within {tolerance_px:.0f}px tolerance",
            f"observed in {track.frame_count} frames",
        ],
        metadata={"duration_s": round(track.duration, 1),
                  "displacement_px": round(track.displacement(), 1)},
    )


def detect_pacing(track: Track, min_ratio: float = 3.0,
                  min_path_px: float = 200.0) -> Optional[Observation]:
    """Circling or pacing: walked a long way, ended up nowhere.

    The path-to-displacement ratio catches this cleanly. A straight walk has a
    ratio near 1; someone circling a car park has a very high one.
    """
    path = track.path_length()
    if path < min_path_px:
        return None
    disp = track.displacement()
    if disp < 1.0:
        disp = 1.0
    ratio = path / disp
    if ratio < min_ratio:
        return None

    return Observation(
        kind="pacing",
        summary=(f"{track.category} {track.track_id} covered {path:.0f}px "
                 f"but moved only {track.displacement():.0f}px"),
        confidence=min(0.85, 0.4 + ratio / 20.0),
        source=Source.TEMPORAL.value,
        track_ids=[track.track_id],
        evidence=[
            f"path length {path:.0f}px",
            f"net displacement {track.displacement():.0f}px",
            f"ratio {ratio:.1f} (threshold {min_ratio:.1f})",
        ],
        metadata={"ratio": round(ratio, 2)},
    )


def detect_disappearance(track: Track) -> Optional[Observation]:
    """Something that was there is not any more.

    Only meaningful for a track that was established - a one-frame detection
    vanishing is a false positive, not a disappearance, and reporting it would
    bury the real ones.
    """
    if track.lost_at is None or track.frame_count < 5:
        return None
    return Observation(
        kind="disappeared",
        summary=f"{track.category} {track.track_id} no longer visible",
        confidence=0.6,
        source=Source.TEMPORAL.value,
        track_ids=[track.track_id],
        evidence=[
            f"last seen {track.lost_at - track.last_seen:.0f}s before being declared lost",
            f"observed in {track.frame_count} frames over {track.duration:.0f}s",
            f"last position ({track.trajectory[-1].x:.0f}, {track.trajectory[-1].y:.0f})"
            if track.trajectory else "no recorded position",
        ],
        metadata={"cameras_seen": list(track.cameras_seen)},
    )


def detect_abandonment(track: Track, owner_gone: bool,
                       min_stationary_s: float = 30.0) -> Optional[Observation]:
    """An object left behind: stationary, and whoever brought it has gone.

    Requires the owner's departure as an input rather than inferring it here,
    because that fact belongs to the relationship layer - this function must
    not silently guess at it.
    """
    if track.kind not in ("object",):
        return None
    if track.duration < min_stationary_s or not track.is_stationary():
        return None
    if not owner_gone:
        return None

    return Observation(
        kind="abandoned_object",
        summary=f"{track.category} {track.track_id} left unattended",
        confidence=0.65,
        source=Source.TEMPORAL.value,
        track_ids=[track.track_id],
        evidence=[
            f"stationary for {track.duration:.0f}s",
            f"displacement {track.displacement():.0f}px",
            "associated person no longer present",
        ],
    )


def analyse_track(track: Track, owner_gone: bool = False) -> List[Observation]:
    """Run every temporal check that applies and attach what fires.

    Deduplicated by kind: a track that has already been reported as dwelling
    must not emit a fresh observation on every frame, which would flood the
    event store and make the feed useless.
    """
    already = {o.kind for o in track.observations}
    found: List[Observation] = []

    for candidate in (
        detect_dwell(track),
        detect_pacing(track),
        detect_disappearance(track),
        detect_abandonment(track, owner_gone=owner_gone),
    ):
        if candidate is not None and candidate.kind not in already:
            track.add_observation(candidate)
            found.append(candidate)
    return found
