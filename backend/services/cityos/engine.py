"""
CityOS Engine - per-intersection orchestration.

One CityOSEngine instance owns one intersection's full stack:

    PerceptionEngine -> SafetyAnalytics -> TrafficFlowAnalyzer -> SignalOptimizer

and exposes a `digital_twin()` snapshot: the machine-readable real-time model
of the intersection (objects, classes, positions, velocities, trajectories,
signal state, flow metrics, safety events) plus edge-node health.

The global registry maps camera_id -> intersection, so every Argus camera is
treated as one sensor covering an intersection approach. Two cameras can be
grouped into one intersection later via `bind_camera()`.
"""
import logging
import threading
import time
from typing import Dict, List, Optional

from backend.services.cityos.perception_engine import PerceptionEngine
from backend.services.cityos.safety_analytics import SafetyAnalytics
from backend.services.cityos.traffic_flow import TrafficFlowAnalyzer
from backend.services.cityos.signal_optimizer import SignalOptimizer

logger = logging.getLogger(__name__)

# Edge-AI tick: safety + signal processing runs at most this often, decoupled
# from the frame rate so a fast camera cannot burn CPU on pairwise TTC.
ANALYSIS_INTERVAL_S = 1.0


class Intersection:
    """The full CityOS stack for one intersection."""

    def __init__(self, intersection_id: str, label: str = ""):
        self.id = intersection_id
        self.label = label or intersection_id.replace("_", " ").title()
        self.perception = PerceptionEngine(intersection_id)
        self.safety = SafetyAnalytics(intersection_id)
        self.flow = TrafficFlowAnalyzer(intersection_id)
        self.signal = SignalOptimizer(intersection_id)
        self.cameras: List[int] = []
        # Edge-node health counters.
        self.frames_ingested = 0
        self.last_ingest_at = 0.0
        self.ingest_latency_ms = 0.0
        self._last_analysis_at = 0.0

    def ingest(self, detections: List[Dict], analysis_results: List[Dict],
               frame_time: float):
        """Fold one processed frame into every layer of the stack."""
        t0 = time.perf_counter()
        self.perception.ingest(detections, analysis_results, frame_time)
        self.frames_ingested += 1
        self.last_ingest_at = time.time()

        now = time.time()
        if now - self._last_analysis_at >= ANALYSIS_INTERVAL_S:
            self._last_analysis_at = now
            users = self.perception.active_users()
            self.flow.observe(users)
            self.safety.process(self.perception)
            demand = self.flow.demand_by_approach(users)
            status = self.signal.tick(demand)
            if self.signal.mode == "adaptive" and demand:
                rec = self.signal.recommend(demand)
                # Only auto-apply when the recommendation disagrees with the
                # current split; otherwise the log fills with no-ops.
                if rec.get("action") in ("extend", "terminate_early"):
                    self.signal.apply_recommendation(rec)

        self.ingest_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    def drain_trips(self):
        """Move completed trajectories into turning-movement statistics."""
        for trip in self.perception.recent_trips(limit=100):
            self.flow.record_trip(trip)

    def digital_twin(self) -> Dict:
        """Full machine-readable snapshot of the intersection."""
        users = self.perception.active_users()
        counts = self.perception.counts_by_category()
        demand = self.flow.demand_by_approach(users)
        return {
            "intersection_id": self.id,
            "label": self.label,
            "cameras": list(self.cameras),
            "timestamp": round(time.time(), 3),
            "objects": users,
            "counts_by_category": counts,
            "signal": self.signal.status(),
            "signal_recommendation": self.signal.recommend(demand),
            "demand_by_approach": {k: round(v, 2) for k, v in demand.items()},
            "flow": {
                "volume_series": self.flow.volume_series(minutes=15),
                "turning_matrix": self.flow.turning_matrix(),
                "speed_summary": self.flow.speed_summary(),
            },
            "safety": {
                **self.safety.stats(),
                "recent_events": self.safety.recent_events(limit=20),
            },
            "perception_stats": self.perception.stats(),
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
            "safety_counters": dict(self.safety.counters),
            "last_ingest_age_s": (
                round(time.time() - self.last_ingest_at, 1)
                if self.last_ingest_at else None
            ),
        }


class CityOSEngine:
    """Registry of intersections; the ingest entry point for the pipeline."""

    def __init__(self):
        self._lock = threading.Lock()
        self.intersections: Dict[str, Intersection] = {}
        self.camera_to_intersection: Dict[int, str] = {}

    def get_intersection(self, intersection_id: str,
                         label: str = "") -> Intersection:
        with self._lock:
            inter = self.intersections.get(intersection_id)
            if inter is None:
                inter = Intersection(intersection_id, label=label)
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
        except Exception as exc:  # noqa: BLE001 - never break the frame loop
            logger.debug(f"[cityos] ingest failed for camera {camera_id}: {exc}")

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
            "privacy": {
                "mode": "geometry-only",
                "note": ("CityOS consumes detection geometry only - no face "
                         "embeddings, plate text or imagery enter this layer"),
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