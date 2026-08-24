"""
CityOS Safety Analytics.

Three Vision-Zero style analysers operating on the perception model:

  1. WrongWayDetector   - vehicles travelling against the dominant flow of
                          their approach, or against configured legal
                          headings.
  2. ConflictDetector   - near-miss / time-to-collision analysis between
                          pairs of road users whose trajectories converge.
  3. VRUGuard           - vehicle-vs-pedestrian/cyclist proximity and
                          yielding conflicts.

All three are geometry-only and deduplicate events with cooldowns so an
ongoing hazard produces one alert, not one per frame.
"""
import math
import threading
import time
from collections import deque
from typing import Dict, List, Optional

from backend.services.cityos.perception_engine import (
    PerceptionEngine, angle_of, VRU_CATEGORIES,
)

# Assumed real-world width of the camera view, used to convert normalised
# positions to metres for TTC maths. Per-camera calibration can override.
DEFAULT_VIEW_WIDTH_M = 30.0

TTC_ALERT_S = 2.5          # seconds-to-collision below which we alert
NEAR_MISS_DISTANCE_M = 3.0 # minimum separation that still counts as close
WRONGWAY_MIN_SPEED = 0.8   # m/s - slower objects have unreliable headings
WRONGWAY_STREAK = 4        # consecutive confirmations before an event
VRU_PROXIMITY_M = 4.0      # vehicle-VRU separation worth watching


def _angular_diff(a: float, b: float) -> float:
    """Smallest absolute difference between two angles in degrees."""
    return abs((a - b + 180.0) % 360.0 - 180.0)


class SafetyAnalytics:
    """Per-intersection safety state machine."""

    def __init__(self, intersection_id: str,
                 allowed_headings: Optional[Dict[str, List[str]]] = None,
                 view_width_m: float = DEFAULT_VIEW_WIDTH_M):
        self.intersection_id = intersection_id
        self.view_width_m = view_width_m
        # Optional operator-configured legal headings per approach,
        # e.g. {"north": ["S"], "south": ["N"]}. When absent the detector
        # learns the dominant heading per approach from observed traffic.
        self.allowed_headings = allowed_headings or {}
        self._lock = threading.Lock()
        self._wrongway_streaks: Dict[str, int] = {}
        self._dominant_heading: Dict[str, str] = {}     # approach -> heading
        self._heading_votes: Dict[str, Dict[str, int]] = {}
        self._last_event_at: Dict[str, float] = {}      # dedup key -> ts
        self.events: deque = deque(maxlen=300)
        self.counters = {
            "wrong_way": 0,
            "near_miss": 0,
            "vru_conflict": 0,
        }

    # ── Event plumbing ──────────────────────────────────────────────────

    def _emit(self, kind: str, severity: str, message: str,
              actors: List[Dict], extra: Optional[Dict] = None):
        now = time.time()
        actor_key = "|".join(sorted(a["track_id"] for a in actors))
        dedup_key = f"{kind}:{actor_key}"
        last = self._last_event_at.get(dedup_key, 0)
        if now - last < 20.0:      # one event per actor pair per 20 s
            return False
        self._last_event_at[dedup_key] = now
        self.counters[kind] = self.counters.get(kind, 0) + 1
        self.events.append({
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

    def recent_events(self, limit: int = 50,
                      kind: Optional[str] = None) -> List[Dict]:
        with self._lock:
            evts = [e for e in self.events
                    if kind is None or e["type"] == kind]
        evts.reverse()
        return evts[:limit]

    # ── Wrong-way detection ─────────────────────────────────────────────

    def _legal_headings_for(self, approach: str) -> Optional[List[float]]:
        """Configured legal headings (degrees) for an approach, if any."""
        labels = self.allowed_headings.get(approach)
        if not labels:
            return None
        angles = [angle_of(h) for h in labels]
        return [a for a in angles if a is not None]

    def _check_wrong_way(self, user_dict: Dict):
        if user_dict["category"] not in ("vehicle", "truck", "bus", "motorcycle"):
            return
        if user_dict["speed_mps"] < WRONGWAY_MIN_SPEED:
            return
        heading = user_dict["heading"]
        angle = angle_of(heading)
        if angle is None:
            return
        approach = user_dict.get("approach") or "unknown"

        legal = self._legal_headings_for(approach)
        if legal:
            violation = all(_angular_diff(angle, la) > 100.0 for la in legal)
        else:
            # Learn the dominant heading per approach from accumulated votes;
            # flag travel that opposes it by more than ~135 degrees.
            votes = self._heading_votes.setdefault(approach, {})
            votes[heading] = votes.get(heading, 0) + 1
            dominant = max(votes.items(), key=lambda kv: kv[1])[0]
            dom_angle = angle_of(dominant)
            violation = (
                dom_angle is not None
                and _angular_diff(angle, dom_angle) > 135.0
                and sum(votes.values()) >= 10
            )

        key = str(user_dict["track_id"])
        streak = self._wrongway_streaks.get(key, 0)
        if violation:
            streak += 1
            self._wrongway_streaks[key] = streak
            if streak == WRONGWAY_STREAK:
                self._emit(
                    "wrong_way", "critical",
                    f"Wrong-way {user_dict['category']} ({key}) heading "
                    f"{heading} on the {approach} approach at "
                    f"{user_dict['speed_kmh']} km/h",
                    [{
                        "track_id": key,
                        "category": user_dict["category"],
                        "position": user_dict["position"],
                        "heading": heading,
                        "speed_kmh": user_dict["speed_kmh"],
                    }],
                    {"approach": approach},
                )
        else:
            self._wrongway_streaks.pop(key, None)

    # ── Near-miss / TTC conflict detection ──────────────────────────────

    @staticmethod
    def _velocity_vector(user_dict: Dict) -> tuple:
        """(vx, vy) in normalised units per second from heading + speed."""
        angle = angle_of(user_dict["heading"])
        if angle is None or user_dict["speed_mps"] <= 0:
            return (0.0, 0.0)
        rad = math.radians(angle)
        # metres/sec -> normalised/sec using view width; y inverted (N = up).
        scale = user_dict["speed_mps"] / max(DEFAULT_VIEW_WIDTH_M, 1.0)
        return (math.sin(rad) * scale, -math.cos(rad) * scale)

    def _pair_ttc(self, a: Dict, b: Dict) -> Optional[float]:
        """Time-to-closest-approach between two users, plus min distance."""
        ax, ay = a["position"]["x"], a["position"]["y"]
        bx, by = b["position"]["x"], b["position"]["y"]
        avx, avy = self._velocity_vector(a)
        bvx, bvy = self._velocity_vector(b)
        rx, ry = ax - bx, ay - by
        vx, vy = avx - bvx, avy - bvy
        vv = vx * vx + vy * vy
        if vv < 1e-9:
            return None       # no relative motion -> no convergence
        t = -(rx * vx + ry * vy) / vv
        if t < 0 or t > 10.0:
            return None       # diverging, or too far out to matter
        cx, cy = rx + vx * t, ry + vy * t
        dist_norm = math.hypot(cx, cy)
        dist_m = dist_norm * self.view_width_m
        return t if dist_m < NEAR_MISS_DISTANCE_M else None

    def _check_conflicts(self, users: List[Dict]):
        vrus = [u for u in users if u["category"] in VRU_CATEGORIES]
        vehicles = [u for u in users if u["category"] not in VRU_CATEGORIES]

        # Vehicle-vs-vehicle near misses.
        for i in range(len(vehicles)):
            for j in range(i + 1, len(vehicles)):
                a, b = vehicles[i], vehicles[j]
                ttc = self._pair_ttc(a, b)
                if ttc is not None and ttc < TTC_ALERT_S:
                    self._emit(
                        "near_miss", "high",
                        f"Near miss: {a['category']} #{a['track_id']} and "
                        f"{b['category']} #{b['track_id']} converging, "
                        f"TTC {ttc:.1f}s",
                        [
                            {"track_id": a["track_id"],
                             "category": a["category"],
                             "position": a["position"],
                             "heading": a["heading"]},
                            {"track_id": b["track_id"],
                             "category": b["category"],
                             "position": b["position"],
                             "heading": b["heading"]},
                        ],
                        {"ttc_s": round(ttc, 2)},
                    )

        # Vehicle-vs-VRU conflicts (the Vision Zero core case).
        for v in vehicles:
            for p in vrus:
                dx = (v["position"]["x"] - p["position"]["x"]) * self.view_width_m
                dy = (v["position"]["y"] - p["position"]["y"]) * self.view_width_m
                dist_m = math.hypot(dx, dy)
                ttc = self._pair_ttc(v, p)
                if dist_m < VRU_PROXIMITY_M or (ttc is not None and ttc < TTC_ALERT_S):
                    severity = "critical" if dist_m < NEAR_MISS_DISTANCE_M else "high"
                    self._emit(
                        "vru_conflict", severity,
                        f"VRU conflict: {p['category']} #{p['track_id']} vs "
                        f"{v['category']} #{v['track_id']} at {dist_m:.1f} m",
                        [
                            {"track_id": v["track_id"],
                             "category": v["category"],
                             "position": v["position"],
                             "heading": v["heading"]},
                            {"track_id": p["track_id"],
                             "category": p["category"],
                             "position": p["position"],
                             "heading": p["heading"]},
                        ],
                        {"distance_m": round(dist_m, 2),
                         "ttc_s": round(ttc, 2) if ttc is not None else None},
                    )

    # ── Frame entry point ───────────────────────────────────────────────

    def process(self, perception: PerceptionEngine):
        """Run every analyser over the current perception snapshot."""
        users = perception.active_users()
        with self._lock:
            for u in users:
                self._check_wrong_way(u)
            self._check_conflicts(users)
            # Bound the dedup table.
            now = time.time()
            if len(self._last_event_at) > 500:
                cutoff = now - 120
                self._last_event_at = {
                    k: t for k, t in self._last_event_at.items() if t > cutoff
                }

    def stats(self) -> Dict:
        with self._lock:
            return {
                **self.counters,
                "events_buffered": len(self.events),
                "tracked_streaks": len(self._wrongway_streaks),
            }