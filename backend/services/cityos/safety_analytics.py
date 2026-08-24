"""
CityOS Safety Analytics.

Vision-Zero style analysers operating on the perception model:

  1. WrongWayDetector   - vehicles travelling against the dominant flow of
                          their approach, or against configured legal
                          headings.
  2. ConflictDetector   - near-miss / time-to-collision analysis between
                          pairs of road users whose trajectories converge.
  3. VRUGuard           - vehicle-vs-pedestrian/cyclist proximity and
                          yielding conflicts.
  4. RedLightGuard      - vehicles crossing the stop line while their
                          approach's signal is red.
  5. StoppedVehicleWatch - vehicles that should be moving (green signal)
                          but are not - potential breakdowns or obstructions.
  6. PetAnalyzer        - post-encroachment-time conflicts on an occupancy
                          grid: object B enters a cell object A vacated less
                          than PET_THRESHOLD_S ago.
  7. BrakingMonitor     - hard-braking events (rapid deceleration).

All analysers are geometry-only and deduplicate events with cooldowns so an
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

# Assumed real-world width of the camera view when no calibration exists.
DEFAULT_VIEW_WIDTH_M = 30.0

TTC_ALERT_S = 2.5           # seconds-to-collision below which we alert
NEAR_MISS_DISTANCE_M = 3.0  # minimum separation that still counts as close
WRONGWAY_MIN_SPEED = 0.8    # m/s - slower objects have unreliable headings
WRONGWAY_STREAK = 4         # consecutive confirmations before an event
VRU_PROXIMITY_M = 4.0       # vehicle-VRU separation worth watching
RED_LIGHT_MIN_SPEED = 1.5   # m/s - must be moving through, not creeping
STOPPED_SPEED_MPS = 0.3     # below this a vehicle counts as stopped
STOPPED_ON_GREEN_S = 45.0   # stopped this long on green => watch
PET_THRESHOLD_S = 5.0       # encroachment window for the PET analyser
GRID = 8                    # occupancy grid resolution (GRID x GRID)
BRAKE_DROP_MPS = 4.0        # speed loss that qualifies as hard braking
BRAKE_WINDOW_S = 1.5


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
        self._heading_votes: Dict[str, Dict[str, int]] = {}
        self._last_event_at: Dict[str, float] = {}      # dedup key -> ts
        self.events: deque = deque(maxlen=300)
        self.counters = {
            "wrong_way": 0,
            "near_miss": 0,
            "vru_conflict": 0,
            "red_light_running": 0,
            "stopped_vehicle": 0,
            "pet_conflict": 0,
            "hard_braking": 0,
        }
        # Red-light bookkeeping: track_id -> was past the stop line?
        self._was_past_line: Dict[str, bool] = {}
        # Stopped-vehicle bookkeeping: track_id -> (since_ts, last_seen_ts)
        self._stationary_since: Dict[str, List[float]] = {}
        # PET occupancy grid: (cx, cy) -> (vacate_ts, vacating_track_id)
        self._cell_vacated: Dict[tuple, tuple] = {}
        self._prev_cell: Dict[str, tuple] = {}
        # Braking history: track_id -> deque[(t, speed_mps)]
        self._speed_history: Dict[str, deque] = {}

    # ── Event plumbing ──────────────────────────────────────────────────

    def _emit(self, kind: str, severity: str, message: str,
              actors: List[Dict], extra: Optional[Dict] = None):
        now = time.time()
        actor_key = "|".join(sorted(a["track_id"] for a in actors))
        dedup_key = f"{kind}:{actor_key}"
        last = self._last_event_at.get(dedup_key, 0)
        if now - last < 20.0:      # one event per actor set per 20 s
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
                lane = user_dict.get("lane_id")
                where = f" in lane {lane}" if lane else ""
                self._emit(
                    "wrong_way", "critical",
                    f"Wrong-way {user_dict['category']} ({key}) heading "
                    f"{heading} on the {approach} approach{where} at "
                    f"{user_dict['speed_kmh']} km/h",
                    [{
                        "track_id": key,
                        "category": user_dict["category"],
                        "position": user_dict["position"],
                        "heading": heading,
                        "speed_kmh": user_dict["speed_kmh"],
                    }],
                    {"approach": approach, "lane_id": lane},
                )
        else:
            self._wrongway_streaks.pop(key, None)

    # ── Near-miss / TTC conflict detection ──────────────────────────────

    @staticmethod
    def _velocity_vector(user_dict: Dict, view_width_m: float) -> tuple:
        """(vx, vy) in normalised units per second from heading + speed."""
        angle = angle_of(user_dict["heading"])
        if angle is None or user_dict["speed_mps"] <= 0:
            return (0.0, 0.0)
        rad = math.radians(angle)
        scale = user_dict["speed_mps"] / max(view_width_m, 1.0)
        return (math.sin(rad) * scale, -math.cos(rad) * scale)

    def _pair_ttc(self, a: Dict, b: Dict) -> Optional[float]:
        """Time-to-closest-approach between two users, if they converge close."""
        ax, ay = a["position"]["x"], a["position"]["y"]
        bx, by = b["position"]["x"], b["position"]["y"]
        avx, avy = self._velocity_vector(a, self.view_width_m)
        bvx, bvy = self._velocity_vector(b, self.view_width_m)
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

    # ── Red-light / stop-line violations ────────────────────────────────

    def _check_red_light(self, user_dict: Dict, approach_signal):
        """Crossing the stop line while the approach shows red."""
        if user_dict["category"] not in ("vehicle", "truck", "bus", "motorcycle"):
            return
        if user_dict["speed_mps"] < RED_LIGHT_MIN_SPEED:
            return
        approach = user_dict.get("approach") or "unknown"
        dist = user_dict.get("distance_to_stop_line_m")
        if dist is None:
            return
        past = dist <= 0.0
        key = str(user_dict["track_id"])
        was_past = self._was_past_line.get(key, False)
        self._was_past_line[key] = past

        if past and not was_past:
            state = approach_signal(approach) if approach_signal else None
            if state == "red":
                lane = user_dict.get("lane_id")
                self._emit(
                    "red_light_running", "critical",
                    f"Red-light running: {user_dict['category']} #{key} "
                    f"crossed the {approach} stop line at "
                    f"{user_dict['speed_kmh']} km/h while the signal was RED"
                    + (f" (lane {lane})" if lane else ""),
                    [{
                        "track_id": key,
                        "category": user_dict["category"],
                        "position": user_dict["position"],
                        "heading": user_dict["heading"],
                        "speed_kmh": user_dict["speed_kmh"],
                    }],
                    {"approach": approach, "lane_id": lane,
                     "signal_state": state},
                )
        elif not past:
            # Left the approach region cleanly; forget stale flags.
            if key in self._was_past_line and len(self._was_past_line) > 500:
                self._was_past_line.clear()

    # ── Stopped / disabled vehicles ─────────────────────────────────────

    def _check_stopped(self, user_dict: Dict, approach_signal):
        """A vehicle not moving while its approach has green = obstruction."""
        if user_dict["category"] not in ("vehicle", "truck", "bus"):
            return
        key = str(user_dict["track_id"])
        now = time.time()
        if user_dict["speed_mps"] < STOPPED_SPEED_MPS:
            entry = self._stationary_since.get(key)
            since = entry[0] if entry else now
            self._stationary_since[key] = [since, now]
            duration = now - since
            approach = user_dict.get("approach") or "unknown"
            state = approach_signal(approach) if approach_signal else None
            dist = user_dict.get("distance_to_stop_line_m")
            upstream = dist is None or dist > 5.0
            # Disabled: motionless during green away from the stop line, or
            # motionless for a very long time anywhere.
            if ((state == "green" and duration >= STOPPED_ON_GREEN_S
                 and upstream) or duration >= 180.0):
                self._emit(
                    "stopped_vehicle", "high",
                    f"Stopped {user_dict['category']} #{key}: stationary "
                    f"{duration:.0f}s on the {approach} approach"
                    f"{' while signal GREEN' if state == 'green' else ''}"
                    f"{' - possible breakdown or obstruction' if upstream else ''}",
                    [{
                        "track_id": key,
                        "category": user_dict["category"],
                        "position": user_dict["position"],
                    }],
                    {"approach": approach, "stopped_s": round(duration, 1),
                     "signal_state": state},
                )
        else:
            self._stationary_since.pop(key, None)
        if len(self._stationary_since) > 500:
            cutoff = now - 600
            self._stationary_since = {
                k: v for k, v in self._stationary_since.items()
                if v[1] > cutoff
            }

    # ── Post-encroachment time (PET) ────────────────────────────────────

    def _check_pet(self, user_dict: Dict):
        """Object entering a cell another object recently vacated."""
        x = user_dict["position"]["x"]
        y = user_dict["position"]["y"]
        cell = (min(int(x * GRID), GRID - 1), min(int(y * GRID), GRID - 1))
        key = str(user_dict["track_id"])
        now = time.time()

        prev = self._prev_cell.get(key)
        if prev is not None and prev != cell:
            # Vacating the old cell right now.
            self._cell_vacated[prev] = (now, key)
        self._prev_cell[key] = cell

        vacated = self._cell_vacated.get(cell)
        if vacated is not None and vacated[1] != key:
            pet = now - vacated[0]
            if 0.0 < pet <= PET_THRESHOLD_S:
                self._emit(
                    "pet_conflict", "high",
                    f"PET conflict: {user_dict['category']} #{key} entered "
                    f"space vacated by #{vacated[1]} {pet:.1f}s ago "
                    f"(threshold {PET_THRESHOLD_S:.0f}s)",
                    [{"track_id": key,
                      "category": user_dict["category"],
                      "position": user_dict["position"]},
                     {"track_id": vacated[1], "category": "unknown"}],
                    {"pet_s": round(pet, 2)},
                )
        if len(self._cell_vacated) > GRID * GRID * 4:
            cutoff = now - PET_THRESHOLD_S * 4
            self._cell_vacated = {
                c: v for c, v in self._cell_vacated.items() if v[0] > cutoff
            }

    # ── Hard braking ────────────────────────────────────────────────────

    def _check_braking(self, user_dict: Dict):
        if user_dict["category"] not in ("vehicle", "truck", "bus", "motorcycle"):
            return
        key = str(user_dict["track_id"])
        now = time.time()
        hist = self._speed_history.setdefault(key, deque(maxlen=20))
        hist.append((now, user_dict["speed_mps"]))
        if len(hist) < 3:
            return
        latest_t, latest_v = hist[-1]
        for t, v in reversed(list(hist)[:-1]):
            dt = latest_t - t
            if dt <= 0 or dt > BRAKE_WINDOW_S:
                continue
            drop = v - latest_v
            if drop >= BRAKE_DROP_MPS and (drop / dt) >= 2.5:
                self._emit(
                    "hard_braking", "medium",
                    f"Hard braking: {user_dict['category']} #{key} lost "
                    f"{drop:.1f} m/s in {dt:.1f}s "
                    f"(deceleration {drop/dt:.1f} m/s²)",
                    [{"track_id": key,
                      "category": user_dict["category"],
                      "position": user_dict["position"]}],
                    {"drop_mps": round(drop, 1),
                     "decel_mps2": round(drop / dt, 1)},
                )
                break
        if len(self._speed_history) > 500:
            cutoff = now - 60
            self._speed_history = {
                k: h for k, h in self._speed_history.items()
                if h and h[-1][0] > cutoff
            }

    # ── Frame entry point ───────────────────────────────────────────────

    def process(self, perception: PerceptionEngine,
                approach_signal=None):
        """Run every analyser over the current perception snapshot.

        approach_signal: optional callable approach -> "green"/"yellow"/"red".
        Provided by the engine from the live signal model; without it the
        red-light and stopped-on-green analysers stay silent rather than
        guessing the signal state.
        """
        users = perception.active_users()
        with self._lock:
            for u in users:
                self._check_wrong_way(u)
                self._check_red_light(u, approach_signal)
                self._check_stopped(u, approach_signal)
                self._check_pet(u)
                self._check_braking(u)
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