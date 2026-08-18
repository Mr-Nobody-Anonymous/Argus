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
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set

from .adapters import infer_spatial_relationships, scene_from_detections
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

    def __init__(self, enable_attributes: bool = True):
        self.tracks = TrackStore()
        self.graph = SceneGraph()
        self.enable_attributes = enable_attributes
        self._lock = threading.RLock()
        self._frame_counts: Dict[int, int] = {}
        self._last_retire = 0.0

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
            try:
                from .attributes import enrich_scene
                enrich_scene(scene, frame)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"Attribute enrichment skipped: {exc}")

        # Single-frame geometry, then temporal accumulation.
        try:
            infer_spatial_relationships(scene)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Relationship inference failed: {exc}")

        touched = self.tracks.update_from_scene(scene)

        try:
            update_from_scene(self.graph, scene, self.tracks)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Scene graph update failed: {exc}")

        observations: List[Observation] = []

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
        }

    def reset(self) -> None:
        self.tracks.clear()
        self.graph.clear()
        with self._lock:
            self._frame_counts.clear()


_PIPELINE: Optional[PerceptionPipeline] = None
_PIPELINE_LOCK = threading.Lock()


def get_pipeline() -> PerceptionPipeline:
    """Process-wide pipeline, created on first use."""
    global _PIPELINE
    with _PIPELINE_LOCK:
        if _PIPELINE is None:
            _PIPELINE = PerceptionPipeline()
        return _PIPELINE
