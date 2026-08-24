"""
CityOS Traffic Flow Analytics.

Continuous traffic-movement analysis over the perception model:

  - volume counts per road-user category, bucketed per minute
  - turning movements (entry approach -> exit approach) from completed trips
  - average / 85th-percentile speeds per category
  - occupancy and flow-rate estimates for signal optimisation

Buckets are bounded so memory stays flat on a permanently-running node.
"""
import threading
import time
from collections import deque, defaultdict
from typing import Dict, List

BUCKET_SECONDS = 60
MAX_BUCKETS = 120          # two hours of per-minute history
QUEUE_SPEED_MPS = 1.0      # below this a vehicle counts as queued
QUEUE_MAX_DIST_M = 100.0   # only count queues within this range of the line
QUEUE_HISTORY_S = 120      # growth-rate window


class TrafficFlowAnalyzer:
    """Per-intersection traffic-flow state."""

    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self._lock = threading.Lock()
        # minute-bucket -> {category: count}
        self.volume_buckets: Dict[int, Dict[str, int]] = {}
        self.turning_movements: deque = deque(maxlen=2000)   # completed trips
        self.speed_samples: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=500))
        self.peak_occupancy = 0
        # Queue analytics: approach -> deque[(ts, length_m)]
        self.queue_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=240))
        self.last_queue: Dict[str, Dict] = {}

    # ── Ingest ──────────────────────────────────────────────────────────

    def observe(self, users: List[Dict]):
        """Count currently-active users into the current minute bucket."""
        now = int(time.time() // BUCKET_SECONDS)
        with self._lock:
            bucket = self.volume_buckets.setdefault(now, defaultdict(int))
            for u in users:
                bucket[u["category"]] += 1
            self.peak_occupancy = max(self.peak_occupancy, len(users))
            if len(self.volume_buckets) > MAX_BUCKETS:
                cutoff = now - MAX_BUCKETS
                self.volume_buckets = {
                    k: v for k, v in self.volume_buckets.items() if k >= cutoff
                }
            for u in users:
                if u.get("speed_mps", 0) > 0.1:
                    self.speed_samples[u["category"]].append(u["speed_mps"])

    def record_trip(self, trip: Dict):
        """Fold a completed trajectory into turning-movement statistics."""
        with self._lock:
            self.turning_movements.append({
                "category": trip.get("category"),
                "entry": trip.get("entry"),
                "exit": trip.get("exit"),
                "ended_at": trip.get("ended_at"),
            })

    # ── Queries ─────────────────────────────────────────────────────────

    def volume_series(self, minutes: int = 30) -> List[Dict]:
        """Per-minute volume by category, oldest first."""
        now = int(time.time() // BUCKET_SECONDS)
        with self._lock:
            series = []
            for m in range(max(now - minutes + 1, 0), now + 1):
                counts = self.volume_buckets.get(m, {})
                series.append({
                    "minute": m * BUCKET_SECONDS,
                    "vehicle": counts.get("vehicle", 0),
                    "truck": counts.get("truck", 0),
                    "bus": counts.get("bus", 0),
                    "motorcycle": counts.get("motorcycle", 0),
                    "pedestrian": counts.get("pedestrian", 0),
                    "cyclist": counts.get("cyclist", 0),
                    "total": sum(counts.values()),
                })
            return series

    def turning_matrix(self, limit: int = 500) -> Dict[str, Dict[str, int]]:
        """Entry-approach -> exit-approach movement counts."""
        with self._lock:
            matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
            for t in list(self.turning_movements)[-limit:]:
                entry, exit_ = t.get("entry"), t.get("exit")
                if entry and exit_:
                    matrix[entry][exit_] += 1
            return {k: dict(v) for k, v in matrix.items()}

    def speed_summary(self) -> Dict[str, Dict]:
        with self._lock:
            out = {}
            for cat, samples in self.speed_samples.items():
                vals = sorted(samples)
                if not vals:
                    continue
                p85 = vals[min(int(len(vals) * 0.85), len(vals) - 1)]
                out[cat] = {
                    "avg_mps": round(sum(vals) / len(vals), 2),
                    "p85_mps": round(p85, 2),
                    "samples": len(vals),
                }
            return out

    # ── Queue detection & estimation ────────────────────────────────────

    def observe_queue(self, users: List[Dict]) -> Dict[str, Dict]:
        """Estimate queue depth/length/growth per approach.

        A vehicle is queued when it is nearly stopped and upstream of its
        stop line. Queue LENGTH is the distance from the stop line to the
        furthest queued vehicle - what an adaptive controller actually wants.
        """
        now = time.time()
        snapshot: Dict[str, Dict] = {}
        for u in users:
            if u.get("is_vru"):
                continue
            dist = u.get("distance_to_stop_line_m")
            if dist is None or not (0 < dist <= QUEUE_MAX_DIST_M):
                continue
            if u.get("speed_mps", 99) > QUEUE_SPEED_MPS:
                continue
            approach = u.get("approach") or "unknown"
            q = snapshot.setdefault(approach, {"count": 0, "length_m": 0.0})
            q["count"] += 1
            q["length_m"] = max(q["length_m"], dist)

        with self._lock:
            for approach, q in snapshot.items():
                hist = self.queue_history[approach]
                hist.append((now, q["length_m"]))
                # Growth rate vs the oldest sample inside the window.
                cutoff = now - QUEUE_HISTORY_S
                older = [l for t, l in hist if t >= cutoff]
                if len(older) >= 2 and hist[0][0] < now - 10:
                    dt = max(now - hist[0][0], 1.0)
                    q["growth_mpm"] = round(
                        (q["length_m"] - older[0]) / dt * 60.0, 2)
                else:
                    q["growth_mpm"] = 0.0
                q["stopped_count"] = q.pop("count")
            self.last_queue = {
                k: dict(v) for k, v in snapshot.items()
            }
        return {k: dict(v) for k, v in snapshot.items()}

    def queue_status(self) -> Dict[str, Dict]:
        with self._lock:
            return {k: dict(v) for k, v in self.last_queue.items()}

    def demand_by_approach(self, users: List[Dict]) -> Dict[str, float]:
        """Live vehicle demand per approach, used by the signal optimiser.

        Vehicles count fully; VRUs count at half weight since they also
        deserve green time but arrive more diffusely.
        """
        demand: Dict[str, float] = defaultdict(float)
        for u in users:
            approach = u.get("approach") or "unknown"
            weight = 0.5 if u.get("is_vru") else 1.0
            demand[approach] += weight
        return dict(demand)

    def stats(self) -> Dict:
        with self._lock:
            total_trips = len(self.turning_movements)
            return {
                "buckets": len(self.volume_buckets),
                "completed_trips_recorded": total_trips,
                "peak_occupancy": self.peak_occupancy,
                "speed_categories": sorted(self.speed_samples.keys()),
            }