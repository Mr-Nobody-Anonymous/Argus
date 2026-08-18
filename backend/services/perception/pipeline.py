"""The perception pipeline: one call that turns detections into understanding.

This is the bridge that stops the canonical model from being a parallel
universe. The processing loop currently does:

    YOLO -> detections -> rules engine -> events

which throws away everything between a box and an event. With this module it
becomes:

    YOLO -> Scene -> Track -> SceneGraph -> Observations -> events

and the old path keeps working unchanged, because `PerceptionPipeline.process`
takes exactly the detection list the coordinator already has and returns a
result the coordinator can ignore, log, or act on.

That property is deliberate. A migration that requires the pipeline to be
rewritten in one step will not happen; one that can be switched on for a single
camera and observed will.

Everything here is best-effort: a failure in perception must never stop a frame
from being processed by the existing rules engine.
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .adapters import infer_spatial_relationships, scene_from_detections
from .change import ChangeDetector
from .observation import Entity, Observation, Scene, Source
from .scene_graph import (
    SceneGraph,
    detect_approach,
    detect_following,
    summarise_track,
    update_from_scene,
)
from .temporal import Track, TrackStore, analyse_track

logger = logging.getLogger(__name__)


@dataclass
class FrameResult:
    """What one frame of perception produced."""

    scene: Scene
    tracks: List[Track] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def summary(self) -> Dict[str, Any]:
        return {
            "camera_id": self.scene.camera_id,
            "timestamp": self.scene.timestamp,
            "entities": len(self.scene.entities),
            "relationships": len(self.scene.relationships),
            "tracks": len(self.tracks),
            "observations": [o.to_dict() for o in self.observations],
            "description": self.scene.describe(),
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


# An object with no person within this many pixels is unattended. Generous,
# because a bag at a person's feet is often 100+ px from their centroid.
OWNER_PROXIMITY_PX = 150.0

# ...and it must stay unattended this long before "unowned" becomes
# "abandoned". Shorter than the 30 s stationary requirement in
# detect_abandonment, which remains the binding constraint.
OWNER_ABSENT_S = 10.0


class PerceptionPipeline:
    """Accumulates understanding across frames for one deployment.

    One instance serves every camera: tracks and relationships must be shared
    for cross-camera identity to be possible at all. All state is behind locks
    in `TrackStore` and `SceneGraph`, and the per-camera bookkeeping here has
    its own.
    """

    def __init__(self, enable_attributes: bool = True,
                 enable_scene_context: bool = True,
                 enable_change_detection: bool = True,
                 enable_rich_relationships: bool = True,
                 enable_text: bool = False,
                 text_interval_s: float = 2.0,
                 enable_memory: bool = True,
                 descriptor_interval_s: float = 1.0):
        self.tracks = TrackStore()
        self.graph = SceneGraph()
        self.change = ChangeDetector()
        self.enable_attributes = enable_attributes
        self.enable_scene_context = enable_scene_context
        self.enable_change_detection = enable_change_detection
        self.enable_rich_relationships = enable_rich_relationships
        # Text detection is off by default: at ~12 ms it is the most expensive
        # CPU stage here and most cameras never see readable writing. It is
        # opt-in per deployment rather than a tax on every frame.
        self.enable_text = enable_text
        self.text_interval_s = text_interval_s
        self._lock = threading.RLock()
        self._frame_counts: Dict[int, int] = {}
        self._last_retire = 0.0
        self._last_text: Dict[int, float] = {}
        self._stage_ms: Dict[str, float] = {}
        # Perception is worthless if it evaporates on restart, so observations
        # and appearances are written through to durable memory.
        self.enable_memory = enable_memory
        self.descriptor_interval_s = descriptor_interval_s
        self._last_descriptor: Dict[Tuple[int, int], float] = {}
        self._descriptors: Dict[Tuple[int, int], List[Any]] = {}
        self._persisted_observations = 0
        self._persisted_appearances = 0
        # Frame-clock timestamp at which each object track was first seen with
        # no person nearby. Feeds detect_abandonment's owner_gone argument.
        self._owner_absent_since: Dict[int, float] = {}

    # -- main entry point -----------------------------------------------------

    def process(self, camera_id: int, detections: List[Dict[str, Any]],
                frame=None, timestamp: Optional[float] = None) -> FrameResult:
        """Fold one frame's detections into the accumulated world model.

        Takes the detection dicts the coordinator already produces, so calling
        this requires no change to the detector, tracker or agents.
        """
        started = time.perf_counter()
        ts = timestamp if timestamp is not None else time.time()

        scene = scene_from_detections(camera_id, detections or [],
                                      timestamp=ts)

        # Cheap visual attributes, when a frame was supplied.
        if self.enable_attributes and frame is not None:
            with self._stage("attributes"):
                from .attributes import enrich_scene
                enrich_scene(scene, frame)

        # Appearance descriptors, rate-limited per track: the vector that
        # makes "find this person" possible later. Sampling once a second is
        # enough - consecutive frames of one person are near-identical, so
        # every frame would cost 25x more for no extra discrimination.
        if self.enable_memory and frame is not None:
            with self._stage("descriptors"):
                self._collect_descriptors(camera_id, scene, frame, ts)

        # Environment context: what the whole frame looks like. Runs before
        # the per-entity work so downstream stages can read the lighting and
        # visibility when deciding how far to trust appearance attributes.
        if self.enable_scene_context and frame is not None:
            with self._stage("scene_context"):
                from .scene_classifier import classify_scene
                classify_scene(scene, frame)

        # Text regions, rate-limited per camera. Signage does not change
        # between consecutive frames, so reading it every frame is waste.
        if self.enable_text and frame is not None and self._text_due(camera_id, ts):
            with self._stage("text"):
                from .ocr import attach_text_to_entities, extract_text
                extract_text(scene, frame)
                attach_text_to_entities(scene)

        # Single-frame geometry, then temporal accumulation.
        with self._stage("relationships"):
            if self.enable_rich_relationships:
                from .relationships import infer_all
                infer_all(scene)
            else:
                infer_spatial_relationships(scene)

        touched = self.tracks.update_from_scene(scene)

        try:
            update_from_scene(self.graph, scene, self.tracks)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Scene graph update failed: {exc}")

        observations: List[Observation] = []

        # Change detection: what is different from this camera's normal. Runs
        # against the baseline BEFORE the baseline absorbs this frame.
        if self.enable_change_detection:
            with self._stage("change"):
                observations.extend(self.change.analyse(scene, frame))

        # Following needs velocity from several frames, so it is only worth
        # evaluating once tracks have some history.
        try:
            active = [t for t in touched if len(t.trajectory) >= 4]
            if len(active) >= 2:
                detect_following(active, self.graph)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Following detection failed: {exc}")

        # Per-track temporal checks (dwell, pacing, abandonment).
        #
        # detect_abandonment() deliberately refuses to infer whether an object's
        # owner has left - that is a relationship fact, not a temporal one - and
        # takes it as an argument. Nothing ever supplied it, so the
        # abandoned_object check could never fire anywhere in the system. It is
        # computed here, where every track in the scene is visible at once.
        owner_gone_by_track = self._owner_gone(touched, ts)
        for track in touched:
            try:
                observations.extend(
                    analyse_track(
                        track,
                        owner_gone=owner_gone_by_track.get(track.track_id, False),
                    )
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Track analysis failed for {track.track_id}: {exc}")

        # Retirement runs on a timer, not per frame: a disappearance is a
        # deployment-wide fact and re-scanning every track each frame is waste.
        observations.extend(self._retire_if_due(ts))

        # Write through to durable memory. Only actionable observations are
        # stored: an unsupported claim is not worth the disk, and storing it
        # would let it resurface later looking like recorded fact.
        if self.enable_memory and observations:
            with self._stage("memory"):
                self._persist_observations(camera_id, observations)

        with self._lock:
            self._frame_counts[camera_id] = self._frame_counts.get(camera_id, 0) + 1

        return FrameResult(
            scene=scene, tracks=touched, observations=observations,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

    def _collect_descriptors(self, camera_id: int, scene, frame,
                             now: float) -> int:
        """Sample appearance descriptors for tracked entities."""
        from .descriptors import DIM, average, describe

        collected = 0
        for entity in scene.entities:
            if entity.track_id is None or entity.bbox is None:
                continue
            # Only people are worth re-identifying by clothing colour; a
            # descriptor of a car's colour bands is not discriminative.
            if entity.kind != "person":
                continue
            key = (camera_id, entity.track_id)
            with self._lock:
                last = self._last_descriptor.get(key, 0.0)
                if now - last < self.descriptor_interval_s:
                    continue
                self._last_descriptor[key] = now

            h, w = frame.shape[:2]
            x1, y1 = max(0, int(entity.bbox.x1)), max(0, int(entity.bbox.y1))
            x2, y2 = min(w, int(entity.bbox.x2)), min(h, int(entity.bbox.y2))
            if x2 <= x1 or y2 <= y1:
                continue
            vector = describe(frame[y1:y2, x1:x2])
            if vector is None:
                continue
            with self._lock:
                bucket = self._descriptors.setdefault(key, [])
                bucket.append(vector)
                # Cap the accumulation: the mean of 32 samples is already
                # stable, and an unbounded list is a slow memory leak on a
                # camera watching a doorway all day.
                if len(bucket) > 32:
                    del bucket[0]
            collected += 1
        return collected

    def _persist_observations(self, camera_id: int, observations) -> int:
        from .memory import get_memory
        memory = get_memory()
        written = 0
        for obs in observations:
            if not obs.is_actionable:
                continue
            try:
                memory.remember_observation(obs, camera_id=camera_id)
                written += 1
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Could not persist observation: {exc}")
        with self._lock:
            self._persisted_observations += written
        return written

    def _persist_track(self, track) -> bool:
        """Store a retired track's identity and mean appearance."""
        from .descriptors import active_backend, average
        from .memory import get_memory

        camera_id = track.cameras_seen[-1] if track.cameras_seen else 0
        key = (camera_id, track.track_id)
        with self._lock:
            vectors = self._descriptors.pop(key, [])
            self._last_descriptor.pop(key, None)

        memory = get_memory()
        try:
            attributes = {name: {"value": a.value, "confidence": a.confidence,
                                 "source": a.source}
                          for name, a in track.attributes.items()}
            memory.remember_track(
                camera_id=camera_id, track_id=track.track_id,
                category=track.category, first_seen=track.first_seen,
                last_seen=track.last_seen, frame_count=track.frame_count,
                summary=f"{track.category} {track.track_id}",
                attributes=attributes)

            mean = average(vectors)
            if mean is None:
                return False
            memory.remember_appearance(
                camera_id=camera_id, track_id=track.track_id, descriptor=mean,
                first_seen=track.first_seen, last_seen=track.last_seen,
                frame_count=track.frame_count, category=track.category,
                backend=active_backend(),
                attributes={"samples": len(vectors),
                            "colour": track.get("dominant_colour")
                            if hasattr(track, "get") else None})
            with self._lock:
                self._persisted_appearances += 1
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Could not persist track {track.track_id}: {exc}")
            return False

    @contextmanager
    def _stage(self, name: str):
        """Time one stage and swallow its failures.

        Every stage is optional by construction: perception enriches a frame,
        it must never be able to stop one being processed. The timing is what
        feeds measured costs back into the capability registry, replacing
        estimates with what this host actually does.
        """
        started = time.perf_counter()
        try:
            yield
        except Exception as exc:  # noqa: BLE001 - a stage failure is not fatal
            logger.debug(f"Perception stage '{name}' failed: {exc}")
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            with self._lock:
                previous = self._stage_ms.get(name)
                self._stage_ms[name] = (elapsed if previous is None
                                        else 0.9 * previous + 0.1 * elapsed)

    def _text_due(self, camera_id: int, now: float) -> bool:
        with self._lock:
            last = self._last_text.get(camera_id, 0.0)
            if now - last < self.text_interval_s:
                return False
            self._last_text[camera_id] = now
            return True

    def _owner_gone(self, touched, now: float) -> Dict[int, bool]:
        """Which stationary objects currently have no person near them.

        An object is "unowned" when no person track sits within
        OWNER_PROXIMITY_PX of it. That alone is not abandonment - a bag is
        unowned for a moment whenever its owner steps away - so the state must
        persist for OWNER_ABSENT_S before it counts. The elapsed time is
        measured on the frame clock, never wall-clock, so replayed footage
        behaves the same as live.
        """
        people = []
        objects = []
        for track in touched:
            pts = track.trajectory
            if not pts:
                continue
            last = pts[-1]
            if track.kind == "person":
                people.append((last.x, last.y))
            elif track.kind == "object":
                objects.append((track, last.x, last.y))

        result: Dict[int, bool] = {}
        for track, ox, oy in objects:
            near = any(
                ((ox - px) ** 2 + (oy - py) ** 2) ** 0.5 <= OWNER_PROXIMITY_PX
                for px, py in people
            )
            if near:
                # Owner present: reset the clock.
                self._owner_absent_since.pop(track.track_id, None)
                result[track.track_id] = False
                continue
            since = self._owner_absent_since.setdefault(track.track_id, now)
            result[track.track_id] = (now - since) >= OWNER_ABSENT_S

        # Bound the bookkeeping: drop entries for tracks no longer present.
        if len(self._owner_absent_since) > 512:
            live = {t.track_id for t, _, _ in objects}
            for tid in [k for k in self._owner_absent_since if k not in live]:
                del self._owner_absent_since[tid]
        return result

    def _retire_if_due(self, now: float, interval_s: float = 2.0) -> List[Observation]:
        with self._lock:
            if now - self._last_retire < interval_s:
                return []
            self._last_retire = now

        out: List[Observation] = []
        try:
            from .temporal import detect_disappearance
            for track in self.tracks.retire_stale(now):
                obs = detect_disappearance(track)
                if obs is not None:
                    track.add_observation(obs)
                    out.append(obs)
                # Retirement is the last moment this track exists in RAM. If
                # its appearance is not written now it is lost forever, and
                # cross-camera search would only ever see live tracks.
                if self.enable_memory:
                    self._persist_track(track)
            self.graph.prune(now)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Retirement sweep failed: {exc}")
        return out

    # -- queries --------------------------------------------------------------

    def describe_track(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Everything reliably known about one entity - the Phase 3 milestone."""
        track = self.tracks.get(track_id)
        if track is None:
            return None
        return summarise_track(track, self.graph, self.tracks)

    def active_tracks(self) -> List[Dict[str, Any]]:
        return [summarise_track(t, self.graph, self.tracks)
                for t in self.tracks.all()]

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            frames = dict(self._frame_counts)
        active = self.tracks.all()
        return {
            "frames_processed": frames,
            "tracks_total": len(self.tracks),
            "tracks_active": len(active),
            "relationships_active": len(self.graph.active_edges()),
            "observations": sum(len(t.observations) for t in active),
            "stage_ms": {k: round(v, 3) for k, v in sorted(self._stage_ms.items())},
            "change_baselines": self.change.report()["cameras"],
            "memory": {
                "enabled": self.enable_memory,
                "observations_written": self._persisted_observations,
                "appearances_written": self._persisted_appearances,
                "tracks_accumulating_descriptors": len(self._descriptors),
            },
            "stages_enabled": {
                "attributes": self.enable_attributes,
                "scene_context": self.enable_scene_context,
                "change_detection": self.enable_change_detection,
                "rich_relationships": self.enable_rich_relationships,
                "text": self.enable_text,
                "memory": self.enable_memory,
            },
        }

    def explain(self, track_id: int) -> Optional[Dict[str, Any]]:
        """Why Argus believes what it believes about one entity.

        Separates measurements from inferences rather than presenting a single
        confident narrative - the distinction an operator needs before acting.
        """
        track = self.tracks.get(track_id)
        if track is None:
            return None
        from .evidence import explain_track
        return explain_track(track, self.graph, self.tracks)

    def reset(self) -> None:
        self.tracks.clear()
        self.graph.clear()
        self.change.reset()
        with self._lock:
            self._frame_counts.clear()
            self._last_text.clear()
            self._stage_ms.clear()
            self._last_descriptor.clear()
            self._descriptors.clear()
            self._persisted_observations = 0
            self._persisted_appearances = 0


_PIPELINE: Optional[PerceptionPipeline] = None
_PIPELINE_LOCK = threading.Lock()


def get_pipeline() -> PerceptionPipeline:
    """Process-wide pipeline, created on first use."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = PerceptionPipeline()
        return _PIPELINE
