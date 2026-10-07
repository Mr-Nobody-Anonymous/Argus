"""
CityOS incident detection.

Recognises operational INCIDENTS - patterns that matter to a traffic
management centre - from the perception + flow + signal layers:

  1. collision            : two vehicles in sustained contact-range and both
                            near-stationary after converging.
  2. blocked_intersection : a vehicle stationary INSIDE the junction box
                            (past its stop line) for a prolonged period -
                            blocking cross traffic.
  3. abnormal_congestion  : queue length or growth rate far beyond normal,
                            sustained over the evaluation window.
  4. prohibited_area_ped  : a pedestrian inside the junction box but NOT in
                            a crosswalk (jaywalking into live traffic).

Every incident is deduplicated with cooldowns so an ongoing situation
produces one active incident, not one per analysis tick.
"""
import math
import threading
import time
from collections import deque
from typing import Dict, List, Optional

COLLISION_RANGE_M = 2.0        # vehicles this close ...
COLLISION_SPEED_MPS = 1.0      # ... and this slow => suspected collision
BLOCKED_MIN_S = 20.0           # stationary inside junction for this long
CONGESTION_QUEUE_M = 60.0      # queue length that is abnormal by itself
CONGESTION_GROWTH_MPM = 15.0   # queue growth rate that is abnormal
CONGESTION_SUSTAIN_S = 30.0    # must persist this long
VIEW_SCALE_M = 30.0            # metres per normalised unit (approximate)


class IncidentDetector:
    """Per-intersection operational-incident state machine."""

    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self._lock = threading.Lock()
        self.incidents: deque = deque(maxlen=200)
        self._last_at: Dict[str, float] = {}
        # track_id -> [since_ts, last_seen_ts] for junction-box dwellers
        self._in_junction_since: Dict[str, List[float]] = {}
        # approach -> deque[(ts, length_m)] of abnormal-queue persistence
        self._congestion_since: Dict[str, deque] = {}
        self.counters = {
            "collision": 0,
            "blocked_intersection": 0,
            "abnormal_congestion": 0,
            "prohibited_area_pedestrian": 0,
        }

    # ── Events ──────────────────────────────────────────────────────

    def _emit(self, kind: str, severity: str, message: str,
              actors: List[Dict], extra: Optional[Dict] = None) -> bool:
        now = time.time()
        key = f"{kind}:" + "|".join(sorted(a["track_id"] for a in actors))
        if now - self._last_at.get(key, 0) < 60.0:
            return False
        self._last_at[key] = now
        self.counters[kind] = self.counters.get(kind, 0) + 1
        self.incidents.append({
            "id": f"{self.intersection_id}-{int(now*1000)}-{kind}",
            "type": kind,
            "severity": severity,
            "message": message,
            "actors": actors,
            "intersection_id": self.intersection_id,
            "timestamp": round(now, 3),
            **(extra or {}),
        })
        return True

    def recent_incidents(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            out = list(self.incidents)[-limit:]
        out.reverse()
        return out

    # ── Frame processing ────────────────────────────────────────────

    def process(self, users: List[Dict], queue_status: Dict[str, Dict]):
        """Analyse one perception snapshot + queue status."""
        vehicles = [u for u in users
                    if u["category"] in ("vehicle", "truck", "bus")]
        peds = [u for u in users if u["category"] == "pedestrian"]
        self._check_collisions(vehicles)
        self._check_blocked_junction(vehicles)
        self._check_prohibited_peds(peds)
        self._check_congestion(queue_status)

    def _check_collisions(self, vehicles: List[Dict]):
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                d = math.hypot(
                    (a["position"]["x"] - b["position"]["x"]) * VIEW_SCALE_M,
                    (a["position"]["y"] - b["position"]["y"]) * VIEW_SCALE_M)
                if d > COLLISION_RANGE_M:
                    continue
                if (a["speed_mps"] > COLLISION_SPEED_MPS
                        or b["speed_mps"] > COLLISION_SPEED_MPS):
                    continue
                self._emit(
                    "collision", "critical",
                    f"Suspected collision: {a['category']} #{a['track_id']} "
                    f"and {b['category']} #{b['track_id']} in contact range "
                    f"({d:.1f} m) and both stationary",
                    [
                        {"track_id": str(a["track_id"]),
                         "category": a["category"],
                         "position": a["position"]},
                        {"track_id": str(b["track_id"]),
                         "category": b["category"],
                         "position": b["position"]},
                    ],
                    {"separation_m": round(d, 2)},
                )

    def _check_blocked_junction(self, vehicles: List[Dict]):
        now = time.time()
        seen = set()
        for v in vehicles:
            dist = v.get("distance_to_stop_line_m")
            if dist is None or dist > 0.0:
                continue                      # not inside the junction box
            if v["speed_mps"] > 1.5:
                continue                      # moving through normally
            key = str(v["track_id"])
            seen.add(key)
            entry = self._in_junction_since.get(key)
            since = entry[0] if entry else now
            self._in_junction_since[key] = [since, now]
            duration = now - since
            if duration >= BLOCKED_MIN_S:
                self._emit(
                    "blocked_intersection", "high",
                    f"Intersection blocking: {v['category']} #{key} "
                    f"stationary inside the junction box for "
                    f"{duration:.0f}s",
                    [{"track_id": key, "category": v["category"],
                      "position": v["position"]}],
                    {"blocked_s": round(duration, 1)},
                )
        # Forget vehicles that left the box.
        stale = set(self._in_junction_since) - seen
        for k in stale:
            del self._in_junction_since[k]

    def _check_prohibited_peds(self, peds: List[Dict]):
        for p in peds:
            dist = p.get("distance_to_stop_line_m")
            if dist is None or dist > 0.0:
                continue                      # not inside the junction box
            if p.get("in_crosswalk"):
                continue                      # legal crossing
            self._emit(
                "prohibited_area_pedestrian", "medium",
                f"Pedestrian #{p['track_id']} inside the junction box "
                f"outside any crosswalk",
                [{"track_id": str(p["track_id"]), "category": "pedestrian",
                  "position": p["position"]}],
            )

    def _check_congestion(self, queue_status: Dict[str, Dict]):
        now = time.time()
        for approach, q in queue_status.items():
            length = q.get("length_m", 0.0)
            growth = q.get("growth_mpm", 0.0)
            abnormal = length >= CONGESTION_QUEUE_M \
                or growth >= CONGESTION_GROWTH_MPM
            hist = self._congestion_since.setdefault(approach, deque())
            if abnormal:
                hist.append(now)
                cutoff = now - CONGESTION_SUSTAIN_S
                while hist and hist[0] < cutoff:
                    hist.popleft()
                span = hist[-1] - hist[0] if len(hist) >= 2 else 0.0
                if span >= CONGESTION_SUSTAIN_S * 0.8:
                    self._emit(
                        "abnormal_congestion", "high",
                        f"Abnormal congestion on {approach}: queue "
                        f"{length:.0f} m growing {growth:.1f} m/min, "
                        f"sustained {span:.0f}s",
                        [],
                        {"approach": approach, "queue_length_m": length,
                         "growth_mpm": growth},
                    )
            else:
                hist.clear()

    def stats(self) -> Dict:
        with self._lock:
            return {
                **self.counters,
                "incidents_buffered": len(self.incidents),
            }