"""Retrieval: the two questions Phase 6 exists to answer.

The architecture doc set the acceptance bar precisely:

1. *"find this person across cameras"* returns ranked results by embedding
   similarity  -> `find_similar_appearances`, `find_across_cameras`
2. *"what happened near the loading bay yesterday"* answers from stored
   observations                                    -> `recall`

Everything here reads from `memory.py` and adds the judgement that raw rows
lack: how strong a match is, whether a physical journey between two cameras
was even possible in the time available, and - crucially - what the result
does **not** prove.

## The rule this module enforces

An appearance match is a **candidate for review, never an identification.**
The descriptor encodes colour layout in three bands (see `descriptors.py`);
two people in dark coats and jeans match strongly and are not the same person.
Every result therefore carries `is_identification: False` and a `caveat`, and
`CrossCameraMatch` refuses to promote a match to "confirmed" no matter how high
the cosine similarity climbs. Confidence measures descriptor agreement; it does
not measure truth.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .descriptors import (MIN_REPORTABLE, POSSIBLE_MATCH, STRONG_MATCH,
                          active_backend)

logger = logging.getLogger(__name__)

# Fastest plausible travel between two cameras, metres per second. A person
# reappearing on a camera 500 m away two seconds later is a descriptor
# coincidence, not a journey - and saying so is more useful than a high score.
MAX_WALK_SPEED_MS = 2.0
MAX_VEHICLE_SPEED_MS = 30.0

# Without a site map there is no distance between cameras, so the plausibility
# check falls back to this minimum transit time.
DEFAULT_MIN_TRANSIT_S = 1.0

_CAVEAT = ("Appearance matching compares clothing colour layout, not identity. "
           "Two people dressed alike will match strongly. Treat every result "
           "as a candidate for human review.")


@dataclass
class AppearanceMatch:
    """One ranked appearance candidate."""

    key: str
    camera_id: int
    track_id: int
    similarity: float
    strength: str
    first_seen: float
    last_seen: float
    frame_count: int
    category: str = "person"
    attributes: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_identification(self) -> bool:
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "similarity": round(self.similarity, 4),
            "strength": self.strength,
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "seen_at": time.strftime("%Y-%m-%dT%H:%M:%S",
                                     time.localtime(self.last_seen)),
            "frame_count": self.frame_count,
            "category": self.category,
            "attributes": self.attributes,
            "is_identification": self.is_identification,
        }


@dataclass
class CrossCameraMatch:
    """A candidate sighting of the same entity on a different camera."""

    match: AppearanceMatch
    from_camera: int
    to_camera: int
    time_gap_s: float
    plausible: bool
    plausibility_note: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            **self.match.to_dict(),
            "from_camera": self.from_camera,
            "to_camera": self.to_camera,
            "time_gap_s": round(self.time_gap_s, 1),
            "physically_plausible": self.plausible,
            "plausibility": self.plausibility_note,
        }


def _strength(score: float) -> str:
    if score >= STRONG_MATCH:
        return "strong"
    if score >= POSSIBLE_MATCH:
        return "possible"
    return "weak"


def _to_match(key: str, score: float, payload: Dict[str, Any]) -> AppearanceMatch:
    return AppearanceMatch(
        key=key, camera_id=int(payload.get("camera_id", -1)),
        track_id=int(payload.get("track_id", -1)), similarity=score,
        strength=_strength(score),
        first_seen=float(payload.get("first_seen", 0.0)),
        last_seen=float(payload.get("last_seen", 0.0)),
        frame_count=int(payload.get("frame_count", 0)),
        category=payload.get("category", "person"),
        attributes=payload.get("attributes", {}) or {})


# ── appearance search ────────────────────────────────────────────────────────

def find_similar_appearances(descriptor, memory=None, limit: int = 10,
                             min_similarity: float = MIN_REPORTABLE,
                             camera_id: Optional[int] = None,
                             exclude_key: Optional[str] = None,
                             since: Optional[float] = None
                             ) -> Dict[str, Any]:
    """Rank stored appearances against a descriptor."""
    from .memory import get_memory
    mem = memory or get_memory()

    where: Dict[str, Any] = {"backend": active_backend()}
    if camera_id is not None:
        where["camera_id"] = camera_id
    if exclude_key is not None:
        where["exclude_key"] = exclude_key
    if since is not None:
        where["since"] = since

    hits = mem.search_appearance_vectors(
        descriptor, limit=limit, min_similarity=min_similarity, where=where)
    matches = [_to_match(k, s, p) for k, s, p in hits]

    return {
        "matches": [m.to_dict() for m in matches],
        "count": len(matches),
        "descriptor_backend": active_backend(),
        "thresholds": {"strong": STRONG_MATCH, "possible": POSSIBLE_MATCH,
                       "reported_above": min_similarity},
        "caveat": _CAVEAT,
    }


def transit_plausibility(time_gap_s: float, distance_m: Optional[float] = None,
                         mode: str = "walk") -> Tuple[bool, str]:
    """Could an entity physically have travelled between the two cameras?

    A strong descriptor match with an impossible transit time is far more
    likely to be two similarly dressed people than one very fast one. Without
    a site map there are no inter-camera distances, so this reports honestly
    that it could only check the trivial case rather than implying a
    geographic check that never happened.
    """
    if time_gap_s < 0:
        return False, "the candidate sighting precedes the query sighting"
    if distance_m is None:
        if time_gap_s < DEFAULT_MIN_TRANSIT_S:
            return False, ("the two sightings overlap in time, so they cannot "
                           "be the same entity on different cameras")
        return True, ("no camera distances are configured, so only simultaneity "
                      "was checked - not travel time")
    speed = MAX_VEHICLE_SPEED_MS if mode == "vehicle" else MAX_WALK_SPEED_MS
    needed = distance_m / speed
    if time_gap_s < needed:
        return False, (f"{distance_m:.0f} m apart needs at least {needed:.0f} s "
                       f"at {speed:.0f} m/s, but only {time_gap_s:.0f} s elapsed")
    return True, (f"{time_gap_s:.0f} s is enough to cover {distance_m:.0f} m "
                  f"at {speed:.0f} m/s")


def find_across_cameras(camera_id: int, track_id: int, memory=None,
                        limit: int = 10,
                        min_similarity: float = POSSIBLE_MATCH,
                        distances: Optional[Dict[int, float]] = None
                        ) -> Dict[str, Any]:
    """"Where else has this entity been?" - the Phase 6 acceptance query.

    Results are annotated with whether the journey was physically possible, and
    implausible ones are kept but flagged rather than dropped: a reviewer
    deserves to see that two cameras saw matching clothing simultaneously,
    because that is evidence of *two people*, which is itself worth knowing.
    """
    from .memory import get_memory
    mem = memory or get_memory()

    key = f"{camera_id}:{track_id}"
    source = mem.get_appearance(key)
    if source is None:
        return {"error": f"no stored appearance for {key}", "matches": [],
                "count": 0, "caveat": _CAVEAT}

    hits = mem.search_appearance_vectors(
        source.descriptor, limit=limit * 3, min_similarity=min_similarity,
        where={"exclude_camera_id": camera_id, "backend": source.backend})

    out: List[CrossCameraMatch] = []
    for k, score, payload in hits:
        match = _to_match(k, score, payload)
        gap = match.first_seen - source.last_seen
        if gap < 0:
            gap = source.first_seen - match.last_seen
            gap = gap if gap > 0 else 0.0
        distance = (distances or {}).get(match.camera_id)
        plausible, note = transit_plausibility(gap, distance)
        out.append(CrossCameraMatch(
            match=match, from_camera=camera_id, to_camera=match.camera_id,
            time_gap_s=gap, plausible=plausible, plausibility_note=note))

    # Plausible first, then by similarity - an impossible journey should never
    # outrank a possible one just because the colours agreed slightly better.
    out.sort(key=lambda m: (not m.plausible, -m.match.similarity))
    out = out[:limit]

    return {
        "query": {"camera_id": camera_id, "track_id": track_id, "key": key,
                  "category": source.category,
                  "last_seen": source.last_seen,
                  "frames_observed": source.frame_count},
        "matches": [m.to_dict() for m in out],
        "count": len(out),
        "plausible_count": sum(1 for m in out if m.plausible),
        "descriptor_backend": source.backend,
        "caveat": _CAVEAT,
        "note": ("Ranked by appearance similarity. 'physically_plausible' is a "
                 "sanity check on travel time, not a confirmation of identity."),
    }


# ── observation recall ───────────────────────────────────────────────────────

def _parse_window(when: Optional[str], since: Optional[float],
                  until: Optional[float]) -> Tuple[Optional[float], Optional[float], str]:
    """Turn a coarse phrase like 'yesterday' into a concrete time window."""
    if since is not None or until is not None:
        return since, until, "explicit range"
    if not when:
        return None, None, "all time"

    now = time.time()
    phrase = when.strip().lower()
    day = 86400.0
    midnight = time.mktime(time.localtime(now)[:3] + (0, 0, 0, 0, 0, -1))

    if phrase == "today":
        return midnight, now, "today"
    if phrase == "yesterday":
        return midnight - day, midnight, "yesterday"
    if phrase in ("hour", "last_hour", "last hour"):
        return now - 3600, now, "the last hour"
    if phrase in ("week", "last_week", "last week"):
        return now - 7 * day, now, "the last 7 days"
    if phrase in ("24h", "day", "last_day", "last 24 hours"):
        return now - day, now, "the last 24 hours"
    return None, None, "all time (unrecognised period)"


def recall(text: Optional[str] = None, camera_id: Optional[int] = None,
           when: Optional[str] = None, kinds: Optional[Sequence[str]] = None,
           since: Optional[float] = None, until: Optional[float] = None,
           min_confidence: float = 0.0, limit: int = 100,
           memory=None) -> Dict[str, Any]:
    """"What happened near X, yesterday?" - the second acceptance query.

    Text and structured filters compose: free text narrows by what was said,
    the rest by camera, kind, time and confidence.
    """
    from .memory import get_memory
    mem = memory or get_memory()

    window_since, window_until, window_label = _parse_window(when, since, until)

    if text:
        rows = mem.search_observations(text, limit=limit * 4)
        # Apply structured filters in Python: the FTS index cannot express them
        # and a JOIN against a MATCH is not portable across SQLite builds.
        results = [
            o for o in rows
            if (camera_id is None or o.camera_id == camera_id)
            and (window_since is None or o.timestamp >= window_since)
            and (window_until is None or o.timestamp <= window_until)
            and (not kinds or o.kind in set(kinds))
            and o.confidence >= min_confidence
        ][:limit]
    else:
        results = mem.query_observations(
            camera_id=camera_id, kinds=kinds, since=window_since,
            until=window_until, min_confidence=min_confidence, limit=limit)

    by_kind: Dict[str, int] = {}
    by_camera: Dict[str, int] = {}
    for o in results:
        by_kind[o.kind] = by_kind.get(o.kind, 0) + 1
        by_camera[str(o.camera_id)] = by_camera.get(str(o.camera_id), 0) + 1

    grounded = [o for o in results if o.evidence]
    return {
        "query": {"text": text, "camera_id": camera_id, "period": window_label,
                  "kinds": list(kinds) if kinds else None,
                  "min_confidence": min_confidence},
        "window": {"since": window_since, "until": window_until},
        "count": len(results),
        "by_kind": by_kind,
        "by_camera": by_camera,
        "observations": [o.to_dict() for o in results],
        "grounded_count": len(grounded),
        "note": ("Every observation lists the evidence behind it. "
                 f"{len(grounded)} of {len(results)} carry evidence; any "
                 "without it are unsupported and must not be acted on."),
    }


def summarise_period(camera_id: Optional[int] = None, when: str = "today",
                     memory=None) -> Dict[str, Any]:
    """A plain-language digest of a period - the natural shift-handover answer."""
    result = recall(camera_id=camera_id, when=when, limit=1000, memory=memory)
    counts = result["by_kind"]
    if not counts:
        return {**result,
                "narrative": f"Nothing was recorded for {result['query']['period']}."}

    parts = [f"{n} x {kind.replace('_', ' ')}"
             for kind, n in sorted(counts.items(), key=lambda kv: -kv[1])]
    where = f" on camera {camera_id}" if camera_id is not None else ""
    narrative = (f"{result['count']} observations{where} "
                 f"for {result['query']['period']}: " + ", ".join(parts) + ".")
    return {**result, "narrative": narrative}
