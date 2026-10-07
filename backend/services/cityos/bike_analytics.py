"""
CityOS bicycle analytics.

Cyclist-specific intelligence beyond simple classification:

  - cyclist volumes (per-minute buckets) and speed statistics
  - bike-lane usage vs general-lane usage
  - wrong-way cycling (heading opposing the lane's legal heading)
  - cyclist queueing at stop lines
  - bike/pedestrian and bike/vehicle proximity conflicts

All geometry-only, deduplicated with cooldowns like SafetyAnalytics.
"""
import math
import threading
import time
from collections import deque, defaultdict
from typing import Dict, List, Optional

WRONGWAY_MIN_SPEED = 0.6     # m/s - slower cyclists have noisy headings
WRONGWAY_STREAK = 3
BIKE_VEHICLE_PROXIMITY_M = 3.5
BIKE_PED_PROXIMITY_M = 2.5
QUEUE_SPEED_MPS = 1.0
QUEUE_MAX_DIST_M = 60.0
COMPASS_IDX = {h: i for i, h in enumerate(
    ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])}


def _angular_diff(a: str, b: str) -> int:
    da = abs(COMPASS_IDX[a] - COMPASS_IDX[b]) % 8
    return min(da, 8 - da) * 45


class BikeAnalytics:
    """Per-intersection cyclist intelligence."""

    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self._lock = threading.Lock()
        self.volume_buckets: Dict[int, int] = {}       # minute -> cyclists
        self.speed_samples: deque = deque(maxlen=500)
        self._wrongway_streaks: Dict[str, int] = {}
        self._last_event_at: Dict[str, float] = {}
        self.events: deque = deque(maxlen=200)
        self.counters = {
            "wrong_way_cycling": 0,
            "bike_vehicle_conflict": 0,
            "bike_pedestrian_conflict": 0,
        }

    # ── Events ──────────────────────────────────────────────────────

    def _emit(self, kind: str, severity: str, message: str,
              actor: Dict, extra: Optional[Dict] = None) -> bool:
        now = time.time()
        key = f"{kind}:{actor.get('track_id')}"
        if now - self._last_event_at.get(key, 0) < 20.0:
            return False
        self._last_event_at[key] = now
        self.counters[kind] = self.counters.get(kind, 0) + 1
        self.events.append({
            "id": f"{self.intersection_id}-{int(now*1000)}-{kind}",
            "type": kind,
            "severity": severity,
            "message": message,
            "actors": [actor],
            "intersection_id": self.intersection_id,
            "timestamp": round(now, 3),
            **(extra or {}),
        })
        return True

    def recent_events(self, limit: int = 50,
                      kind: Optional[str] = None) -> List[Dict]:
        with self._lock:
            out = [e for e in self.events
                   if kind is None or e["type"] == kind][-limit:]
        out.reverse()
        return out

    # ── Frame processing ────────────────────────────────────────────

    def process(self, users: List[Dict]):
        """Analyse one perception snapshot for cyclist-specific patterns."""
        cyclists = [u for u in users if u["category"] == "cyclist"]
        vehicles = [u for u in users
                    if u["category"] in ("vehicle", "truck", "bus",
                                         "motorcycle")]
        peds = [u for u in users if u["category"] == "pedestrian"]
        now = time.time()

        with self._lock:
            bucket = int(now // 60)
            self.volume_buckets[bucket] = \
                self.volume_buckets.get(bucket, 0) + len(cyclists)
            if len(self.volume_buckets) > 120:
                cutoff = bucket - 120
                self.volume_buckets = {
                    k: v for k, v in self.volume_buckets.items()
                    if k >= cutoff
                }

        for c in cyclists:
            if c.get("speed_mps", 0) > 0.1:
                with self._lock:
                    self.speed_samples.append(c["speed_mps"])
            self._check_wrong_way(c)
            self._check_bike_conflicts(c, vehicles, peds)

    def _check_wrong_way(self, cyclist: Dict):
        if cyclist.get("speed_mps", 0) < WRONGWAY_MIN_SPEED:
            return
        heading = cyclist.get("heading")
        legal = cyclist.get("legal_heading")
        if not heading or heading == "-" or not legal:
            return
        key = str(cyclist["track_id"])
        if _angular_diff(heading, legal) > 135:
            streak = self._wrongway_streaks.get(key, 0) + 1
            self._wrongway_streaks[key] = streak
            if streak == WRONGWAY_STREAK:
                lane = cyclist.get("lane_id") or "general lane"
                self._emit(
                    "wrong_way_cycling", "high",
                    f"Wrong-way cycling: cyclist #{key} heading {heading} "
                    f"against the legal direction ({legal}) of {lane} at "
                    f"{cyclist.get('speed_kmh', '?')} km/h",
                    {"track_id": key, "category": "cyclist",
                     "position": cyclist["position"],
                     "heading": heading},
                    {"lane_id": cyclist.get("lane_id"),
                     "legal_heading": legal},
                )
        else:
            self._wrongway_streaks.pop(key, None)

    def _check_bike_conflicts(self, cyclist: Dict,
                              vehicles: List[Dict], peds: List[Dict]):
        cx, cy = cyclist["position"]["x"], cyclist["position"]["y"]
        scale = 30.0      # metres per normalised unit (approximate)

        for v in vehicles:
            d = math.hypot((cx - v["position"]["x"]) * scale,
                           (cy - v["position"]["y"]) * scale)
            if d < BIKE_VEHICLE_PROXIMITY_M:
                self._emit(
                    "bike_vehicle_conflict", "high",
                    f"Bike/vehicle conflict: cyclist "
                    f"#{cyclist['track_id']} vs {v['category']} "
                    f"#{v['track_id']} at {d:.1f} m",
                    {"track_id": cyclist["track_id"], "category": "cyclist",
                     "position": cyclist["position"]},
                    {"other_track_id": str(v["track_id"]),
                     "distance_m": round(d, 2)},
                )

        for p in peds:
            d = math.hypot((cx - p["position"]["x"]) * scale,
                           (cy - p["position"]["y"]) * scale)
            if d < BIKE_PED_PROXIMITY_M:
                self._emit(
                    "bike_pedestrian_conflict", "medium",
                    f"Bike/pedestrian conflict: cyclist "
                    f"#{cyclist['track_id']} vs pedestrian "
                    f"#{p['track_id']} at {d:.1f} m",
                    {"track_id": cyclist["track_id"], "category": "cyclist",
                     "position": cyclist["position"]},
                    {"other_track_id": str(p["track_id"]),
                     "distance_m": round(d, 2)},
                )

    # ── Queries ─────────────────────────────────────────────────────

    def volume_series(self, minutes: int = 30) -> List[Dict]:
        now = int(time.time() // 60)
        with self._lock:
            return [
                {"minute": m * 60,
                 "cyclists": self.volume_buckets.get(m, 0)}
                for m in range(max(now - minutes + 1, 0), now + 1)
            ]

    def speed_summary(self) -> Dict:
        with self._lock:
            vals = sorted(self.speed_samples)
            if not vals:
                return {}
            p85 = vals[min(int(len(vals) * 0.85), len(vals) - 1)]
            return {
                "avg_mps": round(sum(vals) / len(vals), 2),
                "p85_mps": round(p85, 2),
                "samples": len(vals),
            }

    def queue_status(self, users: List[Dict]) -> Dict[str, int]:
        """Stopped cyclists near their approach's stop line."""
        queues: Dict[str, int] = defaultdict(int)
        for u in users:
            if u["category"] != "cyclist":
                continue
            dist = u.get("distance_to_stop_line_m")
            if dist is None or not (0 < dist <= QUEUE_MAX_DIST_M):
                continue
            if u.get("speed_mps", 99) <= QUEUE_SPEED_MPS:
                queues[u.get("approach") or "unknown"] += 1
        return dict(queues)

    def stats(self) -> Dict:
        with self._lock:
            return {
                **self.counters,
                "events_buffered": len(self.events),
                "total_cyclists_bucketed":
                    sum(self.volume_buckets.values()),
            }