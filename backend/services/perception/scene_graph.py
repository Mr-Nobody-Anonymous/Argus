"""Scene graph: relationships that persist and evolve across frames.

`adapters.infer_spatial_relationships` answers "what is true in this instant".
That is not enough. *Following* requires two entities to move together over
time; *approaching* requires a distance that is shrinking; a bag being *carried*
is a claim that strengthens the longer it holds. All of those are statements
about a relationship's history, so the graph has to remember.

    SceneGraph
      +-- edges keyed by (subject, predicate, object)
      +-- each edge: first_seen, last_seen, observation count, distance history
      +-- confidence grows with persistence, decays when unsupported

Two rules keep this honest:

* **Persistence raises confidence; it never manufactures certainty.** A
  relationship seen for 200 frames is more credible than one seen once, but the
  ceiling stays below 1.0 because geometry can only support so much.
* **An edge that stops being observed decays and then expires**, rather than
  lingering as a permanent claim. "Was true once" must not read as "is true".
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple
from collections import deque

from .observation import (
    Entity,
    Observation,
    RelationKind,
    Relationship,
    Scene,
    Source,
)
from .temporal import Track, TrackStore

# How long an unobserved edge survives before it is dropped. Long enough to
# ride out brief occlusion, short enough that a stale claim does not persist.
EDGE_TTL_S = 5.0

# Confidence a single frame's geometry can justify, and the ceiling that
# repeated observation can build to.
BASE_CONFIDENCE = 0.45
MAX_CONFIDENCE = 0.92

MAX_DISTANCE_HISTORY = 64


def _now() -> float:
    return time.time()


@dataclass
class Edge:
    """One relationship, with the history that justifies believing it."""

    subject_id: int
    predicate: str
    object_id: int
    first_seen: float = field(default_factory=_now)
    last_seen: float = field(default_factory=_now)
    observation_count: int = 0
    distances: Deque[float] = field(
        default_factory=lambda: deque(maxlen=MAX_DISTANCE_HISTORY))
    evidence: Dict[str, Any] = field(default_factory=dict)
    # Newest frame time the graph has seen, in the same clock as the frames.
    # Staleness is measured against this, not the wall clock, so replaying
    # archived footage does not decay every edge to zero immediately.
    reference_time: Optional[float] = None

    def _reference(self, now: Optional[float] = None) -> float:
        if now is not None:
            return now
        if self.reference_time is not None:
            return self.reference_time
        return _now()

    @property
    def key(self) -> Tuple[int, str, int]:
        return (self.subject_id, self.predicate, self.object_id)

    @property
    def duration(self) -> float:
        return max(0.0, self.last_seen - self.first_seen)

    def age(self, now: Optional[float] = None) -> float:
        return max(0.0, self._reference(now) - self.last_seen)

    def is_expired(self, now: Optional[float] = None) -> bool:
        return self.age(now) > EDGE_TTL_S

    @property
    def confidence(self) -> float:
        """Grows with sustained observation, decays once unsupported.

        The growth is deliberately slow (log-scaled): the difference between
        seeing something twice and ten times is meaningful, between 200 and 400
        times it is not.
        """
        support = BASE_CONFIDENCE + 0.12 * math.log1p(self.observation_count)
        value = min(MAX_CONFIDENCE, support)
        # Linear decay across the TTL, so a claim fades rather than vanishing.
        staleness = self.age()
        if staleness > 0:
            value *= max(0.0, 1.0 - staleness / EDGE_TTL_S)
        return round(max(0.0, value), 3)

    def trend(self) -> Optional[str]:
        """Is the pair converging or separating?

        Compares the mean of the first and last third of the distance history:
        endpoint comparison is far too sensitive to a single noisy detection.
        """
        if len(self.distances) < 6:
            return None
        pts = list(self.distances)
        third = max(2, len(pts) // 3)
        head = sum(pts[:third]) / third
        tail = sum(pts[-third:]) / third
        if head <= 0:
            return None
        change = (tail - head) / head
        if change < -0.25:
            return "approaching"
        if change > 0.25:
            return "separating"
        return "steady"

    def to_relationship(self) -> Relationship:
        ev = dict(self.evidence)
        ev.update({
            "observations": self.observation_count,
            "duration_s": round(self.duration, 1),
        })
        trend = self.trend()
        if trend:
            ev["trend"] = trend
        if self.distances:
            ev["mean_distance_px"] = round(
                sum(self.distances) / len(self.distances), 1)
        return Relationship(
            subject_id=str(self.subject_id),
            predicate=self.predicate,
            object_id=str(self.object_id),
            confidence=self.confidence,
            source=Source.TEMPORAL.value,
            evidence=ev,
        )


class SceneGraph:
    """Relationships between tracked entities, accumulated over time.

    Keyed by ``track_id`` rather than the per-frame ``entity_id``: an entity id
    is regenerated every frame, so an entity-keyed graph could never accumulate
    anything. This is the join that makes the graph temporal.
    """

    def __init__(self):
        self._edges: Dict[Tuple[int, str, int], Edge] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        with self._lock:
            return len(self._edges)

    def observe(self, subject_id: int, predicate: str, object_id: int,
                distance: Optional[float] = None,
                evidence: Optional[Dict[str, Any]] = None,
                timestamp: Optional[float] = None) -> Edge:
        """Record that a relationship holds right now."""
        if isinstance(predicate, RelationKind):
            predicate = predicate.value
        ts = timestamp if timestamp is not None else _now()
        key = (subject_id, predicate, object_id)
        with self._lock:
            edge = self._edges.get(key)
            if edge is None:
                edge = Edge(subject_id=subject_id, predicate=predicate,
                            object_id=object_id, first_seen=ts, last_seen=ts)
                self._edges[key] = edge
            edge.last_seen = ts
            edge.observation_count += 1
            # Advance the clock on every edge, so edges that were not observed
            # this frame still decay in stream time.
            for other in self._edges.values():
                if other.reference_time is None or ts > other.reference_time:
                    other.reference_time = ts
            if distance is not None:
                edge.distances.append(float(distance))
            if evidence:
                edge.evidence.update(evidence)
            return edge

    def get(self, subject_id: int, predicate: str, object_id: int) -> Optional[Edge]:
        if isinstance(predicate, RelationKind):
            predicate = predicate.value
        with self._lock:
            return self._edges.get((subject_id, predicate, object_id))

    def edges_for(self, track_id: int, include_expired: bool = False) -> List[Edge]:
        with self._lock:
            edges = [e for e in self._edges.values()
                     if e.subject_id == track_id or e.object_id == track_id]
        if include_expired:
            return edges
        return [e for e in edges if not e.is_expired()]

    def active_edges(self) -> List[Edge]:
        with self._lock:
            return [e for e in self._edges.values() if not e.is_expired()]

    def prune(self, now: Optional[float] = None) -> int:
        """Drop expired edges. Returns how many were removed."""
        with self._lock:
            dead = [k for k, e in self._edges.items() if e.is_expired(now)]
            for k in dead:
                self._edges.pop(k, None)
            return len(dead)

    def clear(self) -> None:
        with self._lock:
            self._edges.clear()

    def describe(self, track_id: int) -> List[str]:
        """Readable relationship lines for one track."""
        out = []
        for edge in sorted(self.edges_for(track_id),
                           key=lambda e: -e.confidence):
            other = edge.object_id if edge.subject_id == track_id else edge.subject_id
            arrow = "->" if edge.subject_id == track_id else "<-"
            out.append(f"{arrow} {edge.predicate} {other} "
                       f"({edge.confidence:.2f}, {edge.observation_count}x)")
        return out


# ── temporal relationship inference ──────────────────────────────────────────

def update_from_scene(graph: SceneGraph, scene: Scene,
                      tracks: TrackStore) -> List[Edge]:
    """Fold one frame's spatial relationships into the persistent graph.

    Only relationships between *tracked* entities can accumulate, so untracked
    pairs are skipped rather than being recorded under an id that will never
    recur.
    """
    by_entity: Dict[str, Optional[int]] = {
        e.entity_id: e.track_id for e in scene.entities}
    touched: List[Edge] = []

    for rel in scene.relationships:
        subj = by_entity.get(rel.subject_id)
        obj = by_entity.get(rel.object_id)
        if subj is None or obj is None:
            continue
        distance = rel.evidence.get("pixel_distance")
        edge = graph.observe(subj, rel.predicate, obj,
                             distance=distance, evidence=dict(rel.evidence),
                             timestamp=scene.timestamp)
        touched.append(edge)
    return touched


def detect_following(tracks: Iterable[Track], graph: SceneGraph,
                     min_duration_s: float = 5.0,
                     max_distance_px: float = 250.0,
                     min_speed_px_s: float = 5.0) -> List[Edge]:
    """Two entities moving together, one behind the other.

    Every condition here exists to suppress a specific false positive:

    * both must be *moving* - two people standing near each other are not
      following, they are queuing;
    * headings must agree - people passing in opposite directions are briefly
      close and clearly not following;
    * the distance must be sustained - a momentary pass is not a pursuit.
    """
    found: List[Edge] = []
    people = [t for t in tracks if t.kind == "person"]

    for a in people:
        va = a.velocity()
        speed_a = math.hypot(*va)
        if speed_a < min_speed_px_s:
            continue
        for b in people:
            if a.track_id == b.track_id:
                continue
            vb = b.velocity()
            speed_b = math.hypot(*vb)
            if speed_b < min_speed_px_s:
                continue

            if not a.trajectory or not b.trajectory:
                continue
            pa, pb = a.trajectory[-1], b.trajectory[-1]
            if pa.camera_id != pb.camera_id:
                continue
            distance = math.hypot(pa.x - pb.x, pa.y - pb.y)
            if distance > max_distance_px:
                continue

            # Headings must roughly agree (cosine similarity of velocities).
            dot = va[0] * vb[0] + va[1] * vb[1]
            cos = dot / (speed_a * speed_b) if speed_a and speed_b else 0.0
            if cos < 0.7:
                continue

            # `a` follows `b` only if `a` is behind `b` along `b`'s heading.
            behind = ((pb.x - pa.x) * vb[0] + (pb.y - pa.y) * vb[1]) > 0
            if not behind:
                continue

            edge = graph.observe(
                a.track_id, RelationKind.FOLLOWING.value, b.track_id,
                distance=distance,
                evidence={"heading_agreement": round(cos, 2),
                          "distance_px": round(distance, 1)})
            if edge.duration >= min_duration_s:
                found.append(edge)
    return found


def detect_approach(graph: SceneGraph) -> List[Edge]:
    """Edges whose distance history is consistently shrinking."""
    return [e for e in graph.active_edges()
            if e.trend() == "approaching" and e.observation_count >= 6]


def summarise_track(track: Track, graph: SceneGraph,
                    tracks: TrackStore) -> Dict[str, Any]:
    """Everything reliably known about one entity: the Phase 3 milestone.

    Deliberately separates what was *measured* from what was *inferred*, so a
    reader can always tell evidence from conclusion.
    """
    relationships = []
    for edge in graph.edges_for(track.track_id):
        other_id = (edge.object_id if edge.subject_id == track.track_id
                    else edge.subject_id)
        other = tracks.get(other_id)
        relationships.append({
            "predicate": edge.predicate,
            "direction": "outgoing" if edge.subject_id == track.track_id else "incoming",
            "other_track": other_id,
            "other_category": other.category if other else "unknown",
            "confidence": edge.confidence,
            "observations": edge.observation_count,
            "duration_s": round(edge.duration, 1),
            "trend": edge.trend(),
        })

    summary = track.summary()
    summary["relationships"] = relationships
    summary["description"] = track.describe()
    # Inference is kept apart from measurement, on purpose.
    summary["inferred"] = [
        {"kind": o.kind, "summary": o.summary, "confidence": o.confidence,
         "evidence": o.evidence}
        for o in track.observations
    ]
    return summary
