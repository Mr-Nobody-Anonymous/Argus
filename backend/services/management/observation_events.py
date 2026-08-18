"""Promote perception observations into operator-visible events.

The perception pipeline computes dwell, pacing, abandonment, disappearance,
occupancy anomalies and scene changes, stores them in `perception_observations`,
and logs them. Until this module existed, that was *all* it did: none of those
findings ever reached the `events` table, so the Event Feed an operator actually
watches showed only intrusion and loitering. Argus was analysing far more than
it was reporting.

This bridge closes that gap, under three rules that keep it honest:

1. **Only actionable observations are promoted.** `Observation.is_actionable`
   already requires confidence >= LOW_CONFIDENCE, a non-empty item list, and
   grounding. An observation that fails it is not evidence; raising it to an
   operator alert would manufacture certainty the perception layer explicitly
   refused to claim.

2. **Promotion is not detection.** The event carries the observation's own
   evidence strings, confidence and source provenance in `metadata`, so an
   operator can see *why* it fired. Nothing is re-scored on the way through:
   an event's confidence is the observation's confidence.

3. **One event per observation, ever.** Observations are re-emitted while a
   condition persists (a person keeps dwelling). The perception layer already
   deduplicates per track by kind, but a restart or a re-tracked person can
   produce a second observation for the same real-world situation. A dedup key
   of (camera, kind, subject) with a per-kind cooldown stops the feed filling
   with the same standing person.

Priority is derived from the kind, not invented per event: an abandoned object
is high, a scene change is low. The mapping lives in `KIND_POLICY` so it is
inspectable and testable rather than buried in branches.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class KindPolicy:
    """How one observation kind becomes an event.

    `promote=False` means the kind is deliberately *not* an operator alert.
    Those observations remain queryable through the memory API - suppressing
    an alert must never mean discarding the record.
    """

    priority: str
    promote: bool = True
    cooldown_s: float = 60.0
    description: str = ""


# Every kind the perception layer can emit must appear here. A kind that is
# missing is a silent drop, so `promote_observations` fails loudly on unknown
# kinds in strict mode and a test enumerates the perception package to prove
# this table is complete.
KIND_POLICY: Dict[str, KindPolicy] = {
    "abandoned_object": KindPolicy(
        "high", True, 300.0, "Object left behind after its owner departed"
    ),
    "dwell": KindPolicy(
        "medium", True, 120.0, "Subject remained in place beyond the dwell threshold"
    ),
    "pacing": KindPolicy(
        "medium", True, 120.0, "Repetitive back-and-forth movement"
    ),
    "occupancy_anomaly": KindPolicy(
        "medium", True, 300.0, "Occupancy far from the learned baseline for this hour"
    ),
    "disappeared": KindPolicy(
        "low", True, 300.0, "Tracked subject vanished mid-scene rather than leaving"
    ),
    "scene_change": KindPolicy(
        "low", True, 600.0, "Persistent change in the static scene"
    ),
    # Deliberately not promoted: these are bookkeeping, not incidents. They
    # would drown the feed - every person walking past produces both.
    "object_appeared": KindPolicy("low", False, 0.0, "Bookkeeping: track opened"),
    "object_left_frame": KindPolicy("low", False, 0.0, "Bookkeeping: track closed"),
    "track_summary": KindPolicy("low", False, 0.0, "Bookkeeping: track statistics"),
}


class ObservationEventBridge:
    """Turns actionable observations into events, with dedup and provenance."""

    def __init__(self, event_store=None, max_keys: int = 4096):
        self._event_store = event_store
        self._recent: Dict[str, datetime] = {}
        self._lock = threading.RLock()
        self._max_keys = max_keys
        self.promoted = 0
        self.suppressed_not_actionable = 0
        self.suppressed_duplicate = 0
        self.suppressed_by_policy = 0

    # -- store is resolved lazily so importing this module never touches the DB
    def _store(self):
        if self._event_store is None:
            from backend.services.management.event_store import get_event_store

            self._event_store = get_event_store()
        return self._event_store

    @staticmethod
    def _subject(obs) -> str:
        """A stable identity for the thing the observation is about.

        Track-scoped observations dedup per track; scene-scoped ones (occupancy,
        scene change) carry no track id and dedup per camera+kind.
        """
        tids = getattr(obs, "track_ids", None) or ()
        if tids:
            return "track_" + "_".join(str(t) for t in sorted(tids))
        return "scene"

    def _is_duplicate(self, key: str, cooldown_s: float, now: datetime) -> bool:
        with self._lock:
            seen = self._recent.get(key)
            if seen is not None and (now - seen).total_seconds() < cooldown_s:
                return True
            self._recent[key] = now
            # Bound the dedup table. Without this it grows once per distinct
            # track for the life of the process - the same unbounded-state
            # defect already fixed in the tracker and descriptor caches.
            if len(self._recent) > self._max_keys:
                cutoff = sorted(self._recent.items(), key=lambda kv: kv[1])
                for k, _ in cutoff[: len(cutoff) // 4]:
                    self._recent.pop(k, None)
            return False

    def promote(
        self,
        camera_id: int,
        observations: Iterable[Any],
        frame=None,
        strict: bool = False,
    ) -> List[Dict]:
        """Promote each actionable observation to an event. Returns those created."""
        created: List[Dict] = []
        now = datetime.now()

        for obs in observations or ():
            kind = getattr(obs, "kind", None)
            policy = KIND_POLICY.get(kind)

            if policy is None:
                # A new observation kind with no declared policy. Loud, because
                # silently dropping it is how a feature goes missing.
                msg = f"No event policy declared for observation kind {kind!r}"
                if strict:
                    raise KeyError(msg)
                logger.error(msg)
                continue

            if not policy.promote:
                self.suppressed_by_policy += 1
                continue

            # The single most important check in this module.
            if not getattr(obs, "is_actionable", False):
                self.suppressed_not_actionable += 1
                continue

            key = f"{camera_id}_{kind}_{self._subject(obs)}"
            if self._is_duplicate(key, policy.cooldown_s, now):
                self.suppressed_duplicate += 1
                continue

            event = self._create(camera_id, obs, policy, frame)
            if event is not None:
                created.append(event)
                self.promoted += 1

        return created

    def _create(self, camera_id: int, obs, policy: KindPolicy, frame) -> Optional[Dict]:
        # An Observation records *which* tracks it concerns, not their geometry.
        # Resolve the box from the live track store so the event carries a
        # region an operator can look at; if the track has already been retired
        # the event is still created, just without a box. A missing bbox must
        # never suppress the alert itself.
        bbox, object_type = self._resolve_subject(obs)

        metadata: Dict[str, Any] = {
            "source": "perception",
            "observation_kind": obs.kind,
            "summary": obs.summary,
            # Evidence is the whole point: an operator must be able to see the
            # measurements behind the claim, not just its conclusion.
            "evidence": list(getattr(obs, "evidence", ()) or ()),
            "observation_source": getattr(obs, "source", None),
            "observation_id": getattr(obs, "observation_id", None),
            "track_ids": list(getattr(obs, "track_ids", ()) or ()),
            "policy_description": policy.description,
        }
        # Carry the observation's own metadata through, without letting it
        # overwrite the provenance keys above.
        for k, v in (getattr(obs, "metadata", None) or {}).items():
            metadata.setdefault(k, v)

        snapshot_path = None
        if frame is not None and bbox is not None:
            snapshot_path = self._save_snapshot(frame, bbox, camera_id, obs.kind)

        try:
            return self._store().create_event(
                camera_id=camera_id,
                rule_type=obs.kind,
                object_type=object_type,
                confidence=float(getattr(obs, "confidence", 0.0)),
                bbox=bbox,
                snapshot_path=snapshot_path,
                priority=policy.priority,
                metadata=metadata,
            )
        except Exception as exc:  # noqa: BLE001 - never break the frame loop
            logger.error(f"Failed to promote {obs.kind} observation: {exc}")
            return None

    @staticmethod
    def _resolve_subject(obs):
        """Best-effort bbox + class for the observation's primary track."""
        tids = list(getattr(obs, "track_ids", ()) or ())
        if not tids:
            return None, None
        try:
            from backend.services.perception import get_pipeline

            store = get_pipeline().tracks
            track = store.get(tids[0]) if hasattr(store, "get") else None
            if track is None:
                return None, None
            category = getattr(track, "category", None)
            # Geometry lives on the trajectory points, not the Track itself.
            # Walk backwards for the most recent point that carried a box.
            for point in reversed(getattr(track, "trajectory", ()) or ()):
                box = getattr(point, "bbox", None)
                if box:
                    return [int(v) for v in list(box)[:4]], category
            return None, category
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Could not resolve bbox for {getattr(obs,'kind',None)}: {exc}")
            return None, None

    def _save_snapshot(self, frame, bbox, camera_id: int, kind: str) -> Optional[str]:
        try:
            import cv2

            from backend.config.config import get_config, resolve_path

            out_dir = resolve_path(get_config().system.snapshot_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            annotated = frame.copy()
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 165, 255), 2)
            name = f"cam{camera_id}_{kind}_{datetime.now():%Y%m%d_%H%M%S}.jpg"
            path = out_dir / name
            cv2.imwrite(str(path), annotated)
            return str(path)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Snapshot for {kind} failed: {exc}")
            return None

    def stats(self) -> Dict[str, int]:
        return {
            "promoted": self.promoted,
            "suppressed_not_actionable": self.suppressed_not_actionable,
            "suppressed_duplicate": self.suppressed_duplicate,
            "suppressed_by_policy": self.suppressed_by_policy,
            "dedup_keys": len(self._recent),
        }


_bridge: Optional[ObservationEventBridge] = None
_bridge_lock = threading.Lock()


def get_observation_bridge() -> ObservationEventBridge:
    global _bridge
    if _bridge is None:
        with _bridge_lock:
            if _bridge is None:
                _bridge = ObservationEventBridge()
    return _bridge


def reset_observation_bridge() -> None:
    """Test hook: drop the singleton so dedup state does not leak between tests."""
    global _bridge
    with _bridge_lock:
        _bridge = None


def promotable_kinds() -> List[str]:
    return sorted(k for k, p in KIND_POLICY.items() if p.promote)
