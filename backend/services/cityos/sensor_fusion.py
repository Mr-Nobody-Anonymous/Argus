"""
CityOS sensor fusion.

Merges observations from heterogeneous sensors - camera tracks, 4D LiDAR
clusters, radar returns - into ONE world model per intersection:

    camera track (normalised image) --calibration--> world (east, north)
    lidar cluster (sensor-local)    --pose---------> world (east, north)
                                        |
                          nearest-neighbour association
                                        |
                              fused world-model tracks

Design rules:

  - No single sensor is the source of truth. Position/speed estimates are
    confidence-weighted blends; each fused track records which sensors
    contributed.
  - Association is gated by physical distance plus per-sensor uncertainty,
    so a noisy LiDAR cluster cannot steal a camera track 20 m away.
  - Fused tracks expire quickly: identity remains ephemeral (privacy).
"""
import math
import threading
import time
from typing import Dict, List, Optional

# Association gate: base metres + per-source uncertainty allowance.
GATE_BASE_M = 4.0
SOURCE_UNCERTAINTY_M = {"camera": 1.5, "lidar": 2.0, "radar": 3.0}
TRACK_TTL_S = 6.0          # fused tracks expire fast (ephemeral ids)
MAX_TRACKS = 400


class Observation:
    """One measurement from one sensor at one instant."""

    __slots__ = ("source", "track_key", "east_m", "north_m", "speed_mps",
                 "heading_deg", "confidence", "timestamp")

    def __init__(self, source: str, east_m: float, north_m: float,
                 speed_mps: float = 0.0, heading_deg: Optional[float] = None,
                 confidence: float = 0.8, timestamp: Optional[float] = None,
                 track_key: Optional[str] = None):
        self.source = source
        self.track_key = track_key          # e.g. the camera tracker's id
        self.east_m = float(east_m)
        self.north_m = float(north_m)
        self.speed_mps = float(speed_mps or 0.0)
        self.heading_deg = heading_deg
        self.confidence = float(confidence)
        self.timestamp = float(timestamp if timestamp is not None
                               else time.time())


class FusedTrack:
    """A world-model object maintained across sensors."""

    __slots__ = ("id", "east_m", "north_m", "speed_mps", "heading_deg",
                 "sources", "last_seen", "created_at", "n_updates")

    def __init__(self, track_id: str, obs: Observation):
        self.id = track_id
        self.east_m = obs.east_m
        self.north_m = obs.north_m
        self.speed_mps = obs.speed_mps
        self.heading_deg = obs.heading_deg
        self.sources = {obs.source}
        self.created_at = obs.timestamp
        self.last_seen = obs.timestamp
        self.n_updates = 1

    def blend(self, obs: Observation):
        """Confidence-weighted position/speed update from a new observation."""
        w_new = obs.confidence
        w_old = max(self.n_updates, 1.0) * 0.8      # decay old weight
        total = w_new + w_old
        self.east_m = (self.east_m * w_old + obs.east_m * w_new) / total
        self.north_m = (self.north_m * w_old + obs.north_m * w_new) / total
        # Speeds blend only when both are plausible measurements.
        if obs.speed_mps > 0 or self.speed_mps > 0:
            self.speed_mps = (self.speed_mps * w_old +
                              obs.speed_mps * w_new) / total
        if obs.heading_deg is not None:
            if self.heading_deg is None:
                self.heading_deg = obs.heading_deg
            else:
                # Circular mean over the short arc between the two.
                d = ((obs.heading_deg - self.heading_deg + 180.0) % 360.0) \
                    - 180.0
                self.heading_deg = (self.heading_deg + d * w_new / total) \
                    % 360.0
        self.sources.add(obs.source)
        self.last_seen = max(self.last_seen, obs.timestamp)
        self.n_updates += 1

    def to_dict(self) -> Dict:
        return {
            "fused_id": self.id,
            "world": {"east_m": round(self.east_m, 2),
                      "north_m": round(self.north_m, 2)},
            "speed_mps": round(self.speed_mps, 2),
            "heading_deg": round(self.heading_deg, 1)
            if self.heading_deg is not None else None,
            "sources": sorted(self.sources),
            "age_s": round(time.time() - self.created_at, 1),
            "updates": self.n_updates,
        }


class SensorFusion:
    """Greedy nearest-neighbour multi-sensor track association."""

    def __init__(self):
        self._lock = threading.Lock()
        self.tracks: Dict[str, FusedTrack] = {}
        self._lidar_seq = 0
        self.stats = {
            "observations_processed": 0,
            "associations": 0,
            "tracks_created": 0,
        }

    def _gate_m(self, source_a: str, source_b: str) -> float:
        """Association gate for a candidate source pair."""
        worst = max(SOURCE_UNCERTAINTY_M.get(source_a, 2.5),
                    SOURCE_UNCERTAINTY_M.get(source_b, 2.5))
        return GATE_BASE_M + worst

    def update(self, observations: List[Observation]) -> List[FusedTrack]:
        """Fold one round of observations into the world model."""
        now = time.time()
        with self._lock:
            # Expire stale tracks BEFORE matching so a dead id can never be
            # resurrected by a new observation (ephemeral by design).
            expired = [tid for tid, tr in self.tracks.items()
                       if now - tr.last_seen > TRACK_TTL_S]
            for tid in expired:
                del self.tracks[tid]

            # Camera-keyed observations first: they anchor stable identities.
            ordered = sorted(observations,
                             key=lambda o: o.track_key is not None,
                             reverse=True)
            for obs in ordered:
                self.stats["observations_processed"] += 1
                best_id, best_dist = None, None
                for tid, tr in self.tracks.items():
                    dist = math.hypot(tr.east_m - obs.east_m,
                                      tr.north_m - obs.north_m)
                    sample_src = next(iter(tr.sources)) if tr.sources \
                        else obs.source
                    gate = self._gate_m(sample_src, obs.source)
                    if dist <= gate and (best_dist is None
                                         or dist < best_dist):
                        best_id, best_dist = tid, dist
                if best_id is not None:
                    # Blend moves the track toward the observation, so a
                    # second sensor seeing the same object also associates
                    # with it within this round.
                    self.tracks[best_id].blend(obs)
                    self.stats["associations"] += 1
                else:
                    key = obs.track_key or self._new_lidar_id()
                    self.tracks[key] = FusedTrack(key, obs)
                    self.stats["tracks_created"] += 1

            if len(self.tracks) > MAX_TRACKS:
                keep = sorted(self.tracks.values(),
                              key=lambda t: t.last_seen, reverse=True)
                self.tracks = {t.id: t for t in keep[:MAX_TRACKS]}
            return list(self.tracks.values())

    def _new_lidar_id(self) -> str:
        self._lidar_seq += 1
        return f"L{self._lidar_seq:06d}"

    def snapshot(self) -> List[Dict]:
        with self._lock:
            return [t.to_dict() for t in self.tracks.values()]

    def stats_dict(self) -> Dict:
        with self._lock:
            return {
                **self.stats,
                "active_tracks": len(self.tracks),
            }