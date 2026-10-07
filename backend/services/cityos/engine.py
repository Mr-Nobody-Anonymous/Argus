"""
CityOS Engine - per-intersection orchestration.

One CityOSEngine instance owns one intersection's full stack:

    Calibration + IntersectionMap (world coordinates, lanes)
        -> PerceptionEngine -> SafetyAnalytics -> TrafficFlowAnalyzer
        -> SignalOptimizer -> CorridorService

and exposes a `digital_twin()` snapshot: the machine-readable real-time model
of the intersection (objects with lane assignments and world coordinates,
signal + pedestrian state, queues, flow metrics, safety events, sensor
health) plus a bounded replay buffer for deterministic after-the-fact review.

Privacy governance: track ids are deliberately ephemeral. Completed trips,
corridor matches and replay snapshots expire on `id_retention_s`, and the
layer consumes only detection geometry - no imagery or biometrics.
"""
import logging
import threading
import time
from collections import deque
from typing import Dict, List, Optional

from backend.services.cityos.bike_analytics import BikeAnalytics
from backend.services.cityos.calibration import CameraCalibration
from backend.services.cityos.corridor import CorridorLink, CorridorService
from backend.services.cityos.incident_detection import IncidentDetector
from backend.services.cityos.intersection_map import IntersectionMap
from backend.services.cityos.ntcip import NTCIPControllerClient
from backend.services.cityos.perception_engine import PerceptionEngine
from backend.services.cityos.safety_analytics import SafetyAnalytics
from backend.services.cityos.sensor_abstraction import (
    LidarPoint, SensorFrame, SensorPose, SensorRegistry,
    cluster_point_cloud,
)
from backend.services.cityos.sensor_fusion import Observation, SensorFusion
from backend.services.cityos.signal_optimizer import SignalOptimizer
from backend.services.cityos.traffic_flow import TrafficFlowAnalyzer

logger = logging.getLogger(__name__)

# Edge-AI tick: heavy analysis runs at most this often, decoupled from the
# frame rate so a fast camera cannot burn CPU on pairwise TTC.
ANALYSIS_INTERVAL_S = 1.0
REPLAY_INTERVAL_S = 5.0
REPLAY_MAX = 360            # 30 minutes of 5-second snapshots
HEALTH_FPS_MIN = 0.3        # below this the sensor is degraded


class Intersection:
    """The full CityOS stack for one intersection."""

    def __init__(self, intersection_id: str, label: str = ""):
        self.id = intersection_id
        self.label = label or intersection_id.replace("_", " ").title()
        self.calibration = CameraCalibration()
        self.map = IntersectionMap()
        self.perception = PerceptionEngine(intersection_id)
        self.safety = SafetyAnalytics(intersection_id)
        self.flow = TrafficFlowAnalyzer(intersection_id)
        self.signal = SignalOptimizer(intersection_id)
        self.cameras: List[int] = []

        # Edge-node health counters.
        self.frames_ingested = 0
        self.last_ingest_at = 0.0
        self.ingest_latency_ms = 0.0
        self._frame_times: deque = deque(maxlen=120)
        self._last_analysis_at = 0.0
        self._last_replay_at = 0.0

        # Deterministic replay buffer: bounded twin snapshots.
        self.replay: deque = deque(maxlen=REPLAY_MAX)

        # ── Layers below perception ─────────────────────────────────
        # Physical sensors (4D LiDAR etc.) + multi-sensor world model.
        self.sensors = SensorRegistry()
        self.fusion = SensorFusion()
        self._lidar_observations: deque = deque(maxlen=64)   # (ts, obs)

        # ── Layers beside perception ────────────────────────────────
        self.bikes = BikeAnalytics(intersection_id)
        self.incidents = IncidentDetector(intersection_id)
        # NTCIP controller client: SIMULATION by default - no physical
        # device is ever actuated unless an operator explicitly opts in.
        self.controller = NTCIPControllerClient(intersection_id)

        # ── Privacy governance ──────────────────────────────────────
        self.retention_policy = {
            "completed_trips_s": 900.0,
            "corridor_matches_s": 900.0,
            "replay_snapshots_s": 1800.0,
        }
        self.governance_audit: deque = deque(maxlen=200)

    # ── Ingest ──────────────────────────────────────────────────────────

    def ingest(self, detections: List[Dict], analysis_results: List[Dict],
               frame_time: float):
        """Fold one processed frame into every layer of the stack."""
        t0 = time.perf_counter()
        self.perception.ingest(detections, analysis_results, frame_time)
        self.frames_ingested += 1
        now = time.time()
        self.last_ingest_at = now
        self._frame_times.append(now)

        if now - self._last_analysis_at >= ANALYSIS_INTERVAL_S:
            self._last_analysis_at = now
            users = self.perception.active_users()

            # Lane-level enrichment: world coords, lane id, stop-line distance.
            for u in users:
                east, north = self.calibration.to_world(
                    u["position"]["x"], u["position"]["y"])
                u["world"] = {"east_m": east, "north_m": north}
                u["true_heading"] = self.calibration.world_heading(
                    u.get("heading") or "-")
                self.map.classify(u, self.calibration)

            # Corridor continuity: new objects may complete a handoff.
            for u in users:
                match = self.corridor.register_entry(self.id, u) \
                    if hasattr(self, "corridor") else None
                if match:
                    logger.info(f"[cityos] corridor match {match['match_id']} "
                                f"({match['travel_time_s']}s)")

            self.flow.observe(users)
            self.flow.observe_queue(users)
            self.safety.process(self.perception, self.signal.approach_state)

            # Multi-sensor fusion: camera tracks (world coords) + recent
            # 4D-LiDAR cluster observations -> one world model.
            camera_obs = [
                Observation(
                    source="camera",
                    east_m=u["world"]["east_m"],
                    north_m=u["world"]["north_m"],
                    speed_mps=u.get("speed_mps", 0.0),
                    confidence=u.get("confidence", 0.8),
                    track_key=str(u["track_id"]),
                )
                for u in users
            ]
            lidar_obs = [
                Observation(
                    source="lidar",
                    east_m=o["east_m"], north_m=o["north_m"],
                    speed_mps=o.get("speed_mps", 0.0),
                    heading_deg=o.get("heading_deg"),
                    confidence=0.85,
                )
                for ts, obs in self._lidar_observations
                if now - ts < 2.0
                for o in obs
            ]
            self.fusion.update(camera_obs + lidar_obs)

            # Cyclist-specific analytics.
            self.bikes.process(users)

            demand = self.flow.demand_by_approach(users)
            for approach, weight in demand.items():
                if weight >= 1.0:
                    self.signal.record_detector_call(approach)
            self.signal.tick(demand)
            if self.signal.mode == "adaptive" and demand:
                rec = self.signal.recommend(demand)
                if rec.get("action") in ("extend", "terminate_early"):
                    self.signal.apply_recommendation(rec)

            # NTCIP controller reconciliation: poll the (simulated or real)
            # controller and compare against the commanded phase/state.
            poll = self.controller.poll(self.signal.phase, self.signal.state)
            if poll.get("ok"):
                self.signal.set_observed(poll.get("observed_phase"),
                                         poll.get("observed_state"))

            # Operational incidents from perception + queue state.
            self.incidents.process(users, self.flow.queue_status())

        if now - self._last_replay_at >= REPLAY_INTERVAL_S:
            self._last_replay_at = now
            self.replay.append(self._replay_snapshot())

        self.ingest_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    def attach_corridor(self, corridor: CorridorService):
        self.corridor = corridor

    # ── Sensor-layer ingest (below perception) ──────────────────────────

    def register_sensor(self, sensor_id: str, sensor_type: str,
                        pose: Optional[Dict] = None,
                        label: str = "") -> Dict:
        """Register a physical sensor (4D LiDAR, radar, ...) on this node."""
        sensor_pose = SensorPose.from_dict(pose) if pose else None
        sensor = self.sensors.register(sensor_id, sensor_type,
                                       pose=sensor_pose, label=label)
        return sensor.to_dict()

    def ingest_lidar(self, sensor_id: str, points: List[Dict],
                     timestamp: Optional[float] = None,
                     sequence: int = 0, expected_packets: int = 0,
                     received_packets: int = 0) -> Dict:
        """Ingest one 4D LiDAR frame: per-point x/y/z, radial velocity,
        intensity, return index.

        Points are sensor-local (forward/left/up metres); they are posed
        into world coordinates, clustered into candidate objects, and fed
        to the fusion layer alongside camera tracks. The camera detector is
        NOT assumed to be the source of truth.
        """
        timestamp = timestamp if timestamp is not None else time.time()
        sensor = self.sensors.get(sensor_id)
        if sensor is None:
            sensor = self.sensors.register(sensor_id, "lidar")
        frame = SensorFrame(
            sensor_id=sensor_id, sensor_type="lidar",
            timestamp=timestamp,
            points=[LidarPoint(
                p.get("x", 0.0), p.get("y", 0.0), p.get("z", 0.0),
                radial_velocity=p.get("radial_velocity", 0.0),
                intensity=p.get("intensity", 0.0),
                return_index=p.get("return_index", 0),
            ) for p in points],
            sequence=sequence,
            expected_packets=expected_packets,
            received_packets=received_packets,
        )
        self.sensors.record_frame(frame)
        observations = cluster_point_cloud(frame.points, sensor.pose)
        if observations:
            self._lidar_observations.append((time.time(), observations))
        return {
            "sensor_id": sensor_id,
            "points_received": len(points),
            "clusters": len(observations),
            "observations": observations,
        }

    def drain_trips(self):
        """Move completed trajectories into turning movements + corridors."""
        for trip in self.perception.recent_trips(limit=100):
            self.flow.record_trip(trip)
            if hasattr(self, "corridor"):
                self.corridor.register_exit(self.id, trip)

    def purge_expired_ids(self, retention_s: float) -> int:
        """Privacy governance: drop completed trips older than retention."""
        cutoff = time.time() - retention_s
        removed = 0
        with self.perception._lock:
            keep = deque(maxlen=self.perception.completed_trips.maxlen)
            for t in self.perception.completed_trips:
                if t.get("ended_at", 0) >= cutoff:
                    keep.append(t)
                else:
                    removed += 1
            self.perception.completed_trips.clear()
            self.perception.completed_trips.extend(keep)
        return removed

    # ── Sensor health ───────────────────────────────────────────────────

    def sensor_health(self) -> Dict:
        age = (time.time() - self.last_ingest_at) if self.last_ingest_at else None
        window = [t for t in self._frame_times if t > time.time() - 60]
        fps = round(len(window) / 60.0, 2)
        if age is None or age > 30 or fps < HEALTH_FPS_MIN / 4:
            status, note = "offline", "no frames ingested recently"
        elif fps < HEALTH_FPS_MIN or age > 10:
            status = "degraded"
            note = f"low throughput ({fps} fps) or stale ingest"
        else:
            status, note = "ok", "nominal"
        return {
            "status": status,
            "note": note,
            "fps_60s": fps,
            "last_ingest_age_s": round(age, 1) if age else None,
            "frames_total": self.frames_ingested,
            "ingest_latency_ms": self.ingest_latency_ms,
            "calibration_source": self.calibration.source,
            # Deep per-sensor diagnostics (4D LiDAR etc.).
            "sensor_registry": self.sensors.health_summary(),
            "fusion_stats": self.fusion.stats_dict(),
        }

    # ── Snapshots ───────────────────────────────────────────────────────

    def _compact_objects(self, users: List[Dict]) -> List[Dict]:
        return [{
            "track_id": u["track_id"],
            "category": u["category"],
            "position": u["position"],
            "speed_mps": u["speed_mps"],
            "heading": u["heading"],
            "lane_id": u.get("lane_id"),
            "approach": u.get("approach"),
        } for u in users]

    def _replay_snapshot(self) -> Dict:
        users = self.perception.active_users()
        return {
            "timestamp": round(time.time(), 3),
            "objects": self._compact_objects(users),
            "counts": dict(self.perception.counts_by_category()),
            "signal": {"phase": self.signal.phase, "state": self.signal.state},
            "queues": self.flow.queue_status(),
        }

    def export_replay(self) -> Dict:
        """Export the full replay buffer as a deterministic recording.

        A recording can be re-imported with CityOSEngine.reconstruct() to
        reproduce exactly what the twin believed at any recorded instant -
        e.g. for debugging a safety algorithm against August 24, 14:32:17.
        """
        return {
            "format_version": 1,
            "intersection_id": self.id,
            "label": self.label,
            "exported_at": round(time.time(), 3),
            "snapshot_interval_s": REPLAY_INTERVAL_S,
            "snapshots": list(self.replay),
        }

    def purge_expired_replay(self, retention_s: float) -> int:
        """Privacy governance: drop replay snapshots older than retention."""
        cutoff = time.time() - retention_s
        keep = deque(maxlen=self.replay.maxlen)
        removed = 0
        for snap in self.replay:
            if snap.get("timestamp", 0) >= cutoff:
                keep.append(snap)
            else:
                removed += 1
        self.replay.clear()
        self.replay.extend(keep)
        return removed

    def governance_report(self) -> Dict:
        """Formal privacy/data-governance posture for this intersection."""
        return {
            "intersection_id": self.id,
            "data_classes": {
                "imagery_or_biometrics": "never ingested (geometry-only)",
                "track_ids": "ephemeral - expire on retention policy",
                "completed_trips": f"{self.retention_policy['completed_trips_s']:.0f}s retention",
                "corridor_matches": f"{self.retention_policy['corridor_matches_s']:.0f}s retention",
                "replay_snapshots": f"{self.retention_policy['replay_snapshots_s']:.0f}s retention",
            },
            "retention_policy": dict(self.retention_policy),
            "audit_log_entries": len(self.governance_audit),
            "note": ("stable ids are deliberately ephemeral; exports carry "
                     "no imagery, embeddings or plate text"),
        }

    def audit_governance(self, action: str, **fields):
        """Append an entry to the privacy-governance audit log."""
        self.governance_audit.append({
            "ts": round(time.time(), 3),
            "action": action,
            **fields,
        })

    def replay_at(self, seconds_ago: float) -> Optional[Dict]:
        """Nearest recorded snapshot to `seconds_ago` in the past."""
        target = time.time() - max(seconds_ago, 0)
        best, best_diff = None, None
        for snap in self.replay:
            diff = abs(snap["timestamp"] - target)
            if best_diff is None or diff < best_diff:
                best, best_diff = snap, diff
        if best is None:
            return None
        out = dict(best)
        out["requested_seconds_ago"] = round(seconds_ago, 1)
        out["snapshot_age_error_s"] = round(best_diff, 1)
        return out

    # ── Digital twin ────────────────────────────────────────────────────

    def digital_twin(self) -> Dict:
        """Full machine-readable snapshot of the intersection."""
        users = self.perception.active_users()
        counts = self.perception.counts_by_category()
        demand = self.flow.demand_by_approach(users)

        # Lane-level enrichment for live consumers too.
        for u in users:
            east, north = self.calibration.to_world(
                u["position"]["x"], u["position"]["y"])
            u["world"] = {"east_m": east, "north_m": north}
            u["true_heading"] = self.calibration.world_heading(
                u.get("heading") or "-")
            self.map.classify(u, self.calibration)

        waiting_peds = sum(
            1 for u in users
            if u["category"] == "pedestrian"
            and not u.get("in_crosswalk")
            and abs(u.get("distance_to_stop_line_m") or 999) < 15.0
        )

        return {
            "intersection_id": self.id,
            "label": self.label,
            "cameras": list(self.cameras),
            "timestamp": round(time.time(), 3),
            "calibration": self.calibration.to_dict(),
            "map": self.map.to_dict(),
            "objects": users,
            "counts_by_category": counts,
            "lane_counts": self._lane_counts(users),
            "signal": self.signal.status(),
            "ped_signal": self.signal.ped_states(),
            "waiting_pedestrians": waiting_peds,
            "signal_recommendation": self.signal.recommend(demand),
            "demand_by_approach": {k: round(v, 2) for k, v in demand.items()},
            "queue_by_approach": self.flow.queue_status(),
            "flow": {
                "volume_series": self.flow.volume_series(minutes=15),
                "turning_matrix": self.flow.turning_matrix(),
                "speed_summary": self.flow.speed_summary(),
            },
            "safety": {
                **self.safety.stats(),
                "recent_events": self.safety.recent_events(limit=20),
            },
            # Multi-sensor world model + specialist layers.
            "fused_objects": self.fusion.snapshot(),
            "bikes": {
                **self.bikes.stats(),
                "volume_series": self.bikes.volume_series(minutes=15),
                "speed_summary": self.bikes.speed_summary(),
                "queue_by_approach": self.bikes.queue_status(users),
                "recent_events": self.bikes.recent_events(limit=10),
            },
            "incidents": {
                **self.incidents.stats(),
                "recent_incidents": self.incidents.recent_incidents(limit=10),
            },
            "controller": self.controller.status(),
            "signal_performance": self.signal.performance(),
            "perception_stats": self.perception.stats(),
            "sensor_health": self.sensor_health(),
            "edge_node": {
                "frames_ingested": self.frames_ingested,
                "ingest_latency_ms": self.ingest_latency_ms,
                "last_ingest_age_s": (
                    round(time.time() - self.last_ingest_at, 1)
                    if self.last_ingest_at else None
                ),
                "processing": "local (edge) - perception runs in-process",
            },
        }

    @staticmethod
    def _lane_counts(users: List[Dict]) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for u in users:
            lane = u.get("lane_id")
            if lane:
                counts[lane] = counts.get(lane, 0) + 1
        return counts

    def summary(self) -> Dict:
        """Lightweight status for lists of many intersections."""
        return {
            "intersection_id": self.id,
            "label": self.label,
            "cameras": list(self.cameras),
            "active_objects": len(self.perception.users),
            "counts_by_category": self.perception.counts_by_category(),
            "signal": {
                "phase": self.signal.phase,
                "state": self.signal.state,
                "mode": self.signal.mode,
            },
            "ped_signal": self.signal.ped_states(),
            "queues": self.flow.queue_status(),
            "sensor_status": self.sensor_health()["status"],
            "safety_counters": dict(self.safety.counters),
            "last_ingest_age_s": (
                round(time.time() - self.last_ingest_at, 1)
                if self.last_ingest_at else None
            ),
        }


class CityOSEngine:
    """Registry of intersections; the ingest entry point for the pipeline."""

    #: Privacy governance default: completed trips/corridor matches expire.
    ID_RETENTION_S = 900.0

    def __init__(self):
        self._lock = threading.Lock()
        self.intersections: Dict[str, Intersection] = {}
        self.camera_to_intersection: Dict[int, str] = {}
        self.corridor = CorridorService()
        self._last_purge = 0.0

    def get_intersection(self, intersection_id: str,
                         label: str = "") -> Intersection:
        with self._lock:
            inter = self.intersections.get(intersection_id)
            if inter is None:
                inter = Intersection(intersection_id, label=label)
                inter.attach_corridor(self.corridor)
                self.intersections[intersection_id] = inter
                logger.info(f"[cityos] created intersection '{intersection_id}'")
            return inter

    def bind_camera(self, camera_id: int, intersection_id: str):
        """Attach a camera to an intersection (default: its own)."""
        inter = self.get_intersection(intersection_id)
        with self._lock:
            self.camera_to_intersection[camera_id] = intersection_id
            if camera_id not in inter.cameras:
                inter.cameras.append(camera_id)

    def _intersection_for(self, camera_id: int) -> Intersection:
        with self._lock:
            iid = self.camera_to_intersection.get(camera_id)
        if iid is None:
            iid = f"camera_{camera_id}"
            self.bind_camera(camera_id, iid)
        return self.get_intersection(iid)

    def ingest(self, camera_id: int, detections: List[Dict],
               analysis_results: List[Dict], frame_time: float):
        """Pipeline hook: called once per processed frame per camera."""
        try:
            inter = self._intersection_for(camera_id)
            inter.ingest(detections, analysis_results, frame_time)
            # Cheap periodic maintenance instead of doing it every frame.
            if int(frame_time) % 10 == 0:
                inter.drain_trips()
                self._privacy_purge_if_due()
        except Exception as exc:  # noqa: BLE001 - never break the frame loop
            logger.debug(f"[cityos] ingest failed for camera {camera_id}: {exc}")

    def _privacy_purge_if_due(self):
        now = time.time()
        if now - self._last_purge < 300:
            return
        self._last_purge = now
        purged = 0
        for inter in self.list_intersections():
            purged += inter.purge_expired_ids(self.ID_RETENTION_S)
        if purged:
            logger.info(f"[cityos] privacy purge: expired {purged} "
                        f"completed-trip records (retention "
                        f"{self.ID_RETENTION_S:.0f}s)")

    def set_calibration(self, camera_id: int, cal_data: Dict) -> Dict:
        inter = self._intersection_for(camera_id)
        inter.calibration = CameraCalibration.from_dict(cal_data)
        return inter.calibration.to_dict()

    def set_map(self, camera_id: int, map_config: Dict) -> Dict:
        inter = self._intersection_for(camera_id)
        inter.map = IntersectionMap(map_config)
        return inter.map.to_dict()

    def add_corridor_link(self, link: CorridorLink):
        self.corridor.add_link(link)

    def twin(self, camera_id: Optional[int] = None) -> Dict:
        """Digital-twin snapshot for one camera's intersection, or all."""
        if camera_id is not None:
            with self._lock:
                iid = self.camera_to_intersection.get(camera_id)
            if iid is None:
                return {"error": f"no intersection bound to camera {camera_id}"}
            return self.get_intersection(iid).digital_twin()
        return {
            "intersections": [
                i.digital_twin() for i in self.list_intersections()
            ],
            "timestamp": round(time.time(), 3),
        }

    def list_intersections(self) -> List[Intersection]:
        with self._lock:
            return list(self.intersections.values())

    def summaries(self) -> List[Dict]:
        return [i.summary() for i in self.list_intersections()]

    def ingest_lidar(self, intersection_id: str, sensor_id: str,
                     points: List[Dict], timestamp: Optional[float] = None,
                     sequence: int = 0, expected_packets: int = 0,
                     received_packets: int = 0) -> Dict:
        """Sensor-layer entry point: one 4D LiDAR frame for an intersection."""
        inter = self.get_intersection(intersection_id)
        return inter.ingest_lidar(
            sensor_id, points, timestamp=timestamp, sequence=sequence,
            expected_packets=expected_packets,
            received_packets=received_packets,
        )

    def register_sensor(self, intersection_id: str, sensor_id: str,
                        sensor_type: str, pose: Optional[Dict] = None,
                        label: str = "") -> Dict:
        inter = self.get_intersection(intersection_id)
        return inter.register_sensor(sensor_id, sensor_type, pose=pose,
                                     label=label)

    def incidents(self, limit: int = 50) -> List[Dict]:
        """Merged incident feed across all intersections, newest first."""
        merged: List[Dict] = []
        for inter in self.list_intersections():
            merged.extend(inter.incidents.recent_incidents(limit=limit))
        merged.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return merged[:limit]

    def governance(self) -> Dict:
        """Fleet-wide privacy/data-governance posture."""
        inters = self.list_intersections()
        return {
            "mode": "geometry-only",
            "id_retention_s": self.ID_RETENTION_S,
            "retention_policies": {
                i.id: dict(i.retention_policy) for i in inters
            },
            "audit_log_entries": sum(len(i.governance_audit) for i in inters),
            "note": ("CityOS consumes detection geometry only - no face "
                     "embeddings, plate text or imagery enter this layer; "
                     "track ids are ephemeral and expire automatically"),
        }

    @staticmethod
    def reconstruct(recording: Dict) -> Dict:
        """Deterministic reconstruction from an exported replay recording.

        Returns a read-only view that answers 'what did the twin believe at
        time T?' without touching any live intersection state.
        """
        snapshots = sorted(recording.get("snapshots", []),
                           key=lambda s: s.get("timestamp", 0))
        if not snapshots:
            return {"error": "recording contains no snapshots"}
        return {
            "intersection_id": recording.get("intersection_id"),
            "label": recording.get("label"),
            "is_reconstruction": True,
            "snapshot_count": len(snapshots),
            "window": {
                "start": snapshots[0].get("timestamp"),
                "end": snapshots[-1].get("timestamp"),
            },
            "snapshots": snapshots,
        }

    def alerts(self, limit: int = 50, kind: Optional[str] = None) -> List[Dict]:
        """Merged alert feed across all intersections, newest first."""
        merged: List[Dict] = []
        for inter in self.list_intersections():
            merged.extend(inter.safety.recent_events(limit=limit, kind=kind))
        merged.sort(key=lambda e: e.get("timestamp", 0), reverse=True)
        return merged[:limit]

    def status(self) -> Dict:
        inters = self.summaries()
        return {
            "enabled": True,
            "intersections": inters,
            "total_intersections": len(inters),
            "corridors": self.corridor.summary(),
            "privacy": {
                "mode": "geometry-only",
                "id_retention_s": self.ID_RETENTION_S,
                "note": ("CityOS consumes detection geometry only - no face "
                         "embeddings, plate text or imagery enter this layer; "
                         "track ids are ephemeral and expire automatically"),
            },
        }


_cityos_engine: Optional[CityOSEngine] = None
_cityos_lock = threading.Lock()


def get_cityos_engine() -> CityOSEngine:
    """Global singleton."""
    global _cityos_engine
    if _cityos_engine is None:
        with _cityos_lock:
            if _cityos_engine is None:
                _cityos_engine = CityOSEngine()
    return _cityos_engine