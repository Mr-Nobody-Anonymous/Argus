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
from typing import Any, Dict, List, Optional, Set

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
                 text_interval_s: float = 2.0):
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
        for track in touched:
            try:
                observations.extend(analyse_track(track))
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Track analysis failed for {track.track_id}: {exc}")

        # Retirement runs on a timer, not per frame: a disappearance is a
        # deployment-wide fact and re-scanning every track each frame is waste.
        observations.extend(self._retire_if_due(ts))

        with self._lock:
            self._frame_counts[camera_id] = self._frame_counts.get(camera_id, 0) + 1

        return FrameResult(
            scene=scene, tracks=touched, observations=observations,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )

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
            "stages_enabled": {
                "attributes": self.enable_attributes,
                "scene_context": self.enable_scene_context,
                "change_detection": self.enable_change_detection,
                "rich_relationships": self.enable_rich_relationships,
                "text": self.enable_text,
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


_PIPELINE: Optional[PerceptionPipeline] = None
_PIPELINE_LOCK = threading.Lock()


def get_pipeline() -> PerceptionPipeline:
    """Process-wide pipeline, created on first use."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = PerceptionPipeline()
        return _PIPELINE
