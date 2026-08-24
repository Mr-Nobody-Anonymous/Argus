"""
CityOS Digital Perception Engine.

Transforms raw per-frame detections into a machine-readable model of the
intersection: every road user becomes an object with

  - a stable id (the tracker's persistent track id)
  - a class (car / truck / bus / motorcycle / bicycle / pedestrian / other)
  - a position (normalised [0..1] x/y of the camera view)
  - a velocity (m/s, from the speed analyser when calibrated, else from
    position deltas in normalised space)
  - a heading (compass direction of travel)
  - a trajectory (bounded history of recent positions)

The engine is deliberately geometry-only: it never receives image pixels,
face embeddings or plate text.
"""
import math
import threading
import time
from collections import deque
from typing import Dict, List, Optional, Tuple

# YOLO class -> CityOS road-user category
CLASS_MAP = {
    "car": "vehicle",
    "truck": "truck",
    "bus": "bus",
    "motorcycle": "motorcycle",
    "bicycle": "cyclist",
    "person": "pedestrian",
}

# Categories considered vulnerable road users (Vision Zero focus).
VRU_CATEGORIES = {"pedestrian", "cyclist"}

TRAJECTORY_MAX_POINTS = 60          # ~1-2 minutes of history at typical FPS
STALE_AFTER_SECONDS = 8.0           # drop objects not seen for this long

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def heading_from_delta(dx: float, dy: float) -> str:
    """Compass heading of a movement vector in normalised image space.

    Image y grows downward, so north (up on screen) is negative dy.
    """
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return "-"
    # atan2 with y inverted so up-screen = 0 deg = N, clockwise positive.
    angle = (math.degrees(math.atan2(dx, -dy)) + 360.0) % 360.0
    return COMPASS[int((angle + 22.5) // 45.0) % 8]


def angle_of(heading: str) -> Optional[float]:
    """Inverse of heading_from_delta: compass label -> degrees clockwise from N."""
    try:
        return COMPASS.index(heading) * 45.0
    except ValueError:
        return None


class RoadUser:
    """One tracked road user inside the intersection model."""

    __slots__ = (
        "track_id", "category", "raw_class", "confidence",
        "x", "y", "speed_mps", "heading", "trajectory",
        "first_seen", "last_seen", "approach", "exit_approach",
        "frames_seen", "peak_speed_mps",
    )

    def __init__(self, track_id, category, raw_class, confidence,
                 x, y, timestamp):
        self.track_id = track_id
        self.category = category
        self.raw_class = raw_class
        self.confidence = confidence
        self.x = x
        self.y = y
        self.speed_mps = 0.0
        self.heading = "-"
        self.trajectory = deque(maxlen=TRAJECTORY_MAX_POINTS)
        self.trajectory.append((round(x, 4), round(y, 4), round(timestamp, 3)))
        self.first_seen = timestamp
        self.last_seen = timestamp
        self.approach = None       # quadrant the object entered from
        self.exit_approach = None  # quadrant it left towards (set on removal)
        self.frames_seen = 1
        self.peak_speed_mps = 0.0

    def update(self, x, y, speed_mps, heading, confidence, timestamp):
        prev_x, prev_y = self.x, self.y
        self.x, self.y = x, y
        if speed_mps is not None:
            self.speed_mps = speed_mps
            self.peak_speed_mps = max(self.peak_speed_mps, speed_mps)
        if heading and heading != "-":
            self.heading = heading
        elif (abs(x - prev_x) > 0.002 or abs(y - prev_y) > 0.002):
            self.heading = heading_from_delta(x - prev_x, y - prev_y)
        self.confidence = max(self.confidence, confidence)
        self.last_seen = timestamp
        self.frames_seen += 1
        last_pt = self.trajectory[-1] if self.trajectory else None
        if last_pt is None or timestamp - last_pt[2] >= 0.2:
            self.trajectory.append((round(x, 4), round(y, 4), round(timestamp, 3)))

    @staticmethod
    def approach_of(x: float, y: float) -> str:
        """Which approach quadrant an object is in (screen-space)."""
        if x < 0.45:
            return "west" if y < 0.55 else "south-west"
        if x > 0.55:
            return "east" if y > 0.45 else "north-east"
        return "north" if y < 0.5 else "south"

    def to_dict(self) -> Dict:
        return {
            "track_id": self.track_id,
            "category": self.category,
            "class_name": self.raw_class,
            "confidence": round(self.confidence, 3),
            "position": {"x": round(self.x, 4), "y": round(self.y, 4)},
            "speed_mps": round(self.speed_mps, 2),
            "speed_kmh": round(self.speed_mps * 3.6, 1),
            "heading": self.heading,
            "is_vru": self.category in VRU_CATEGORIES,
            "approach": self.approach,
            "trajectory": list(self.trajectory),
            "age_s": round(time.time() - self.first_seen, 1),
        }


class PerceptionEngine:
    """Per-intersection digital perception state."""

    def __init__(self, intersection_id: str):
        self.intersection_id = intersection_id
        self._lock = threading.Lock()
        self.users: Dict[str, RoadUser] = {}
        self.completed_trips: deque = deque(maxlen=500)   # finished trajectories
        self.total_observed = 0
        self.last_update = 0.0

    # ── Ingest ──────────────────────────────────────────────────────────

    def ingest(self, detections: List[Dict], analysis_results: List[Dict],
               frame_time: float):
        """Fold one processed frame into the intersection model.

        detections:      [{track_id, class_name, confidence, bbox}, ...]
        analysis_results: [{track_id, speed_mps, direction, ...}, ...]
        """
        speeds_by_track = {}
        for r in analysis_results or []:
            tid = r.get("track_id")
            if tid is not None:
                speeds_by_track[str(tid)] = r

        now = time.time()
        seen_ids = set()
        with self._lock:
            for det in detections or []:
                track_id = det.get("track_id")
                if track_id is None:
                    continue
                key = str(track_id)
                bbox = det.get("bbox") or {}
                try:
                    if isinstance(bbox, dict):
                        x1, y1 = float(bbox.get("x1", 0)), float(bbox.get("y1", 0))
                        x2, y2 = float(bbox.get("x2", 0)), float(bbox.get("y2", 0))
                    else:
                        x1, y1, x2, y2 = (float(v) for v in bbox[:4])
                except (TypeError, ValueError):
                    continue
                cx = ((x1 + x2) / 2.0)
                cy = ((y1 + y2) / 2.0)
                # Guard against pixel coordinates that were never normalised.
                if cx > 1.5 or cy > 1.5:
                    frame_w = max(x2, 1.0)
                    frame_h = max(y2, 1.0)
                    cx /= frame_w
                    cy /= frame_h
                cx = min(max(cx, 0.0), 1.0)
                cy = min(max(cy, 0.0), 1.0)

                raw_class = str(det.get("class_name", "other")).lower()
                category = CLASS_MAP.get(raw_class, "other")
                confidence = float(det.get("confidence", 0.0))

                analysis = speeds_by_track.get(key, {})
                speed_mps = analysis.get("speed_mps")
                if speed_mps is None:
                    speed_kmh = analysis.get("speed_kmh")
                    speed_mps = (speed_kmh / 3.6) if speed_kmh is not None else None
                heading = analysis.get("direction")

                user = self.users.get(key)
                if user is None:
                    user = RoadUser(key, category, raw_class, confidence,
                                    cx, cy, frame_time)
                    user.approach = RoadUser.approach_of(cx, cy)
                    self.users[key] = user
                    self.total_observed += 1
                else:
                    if user.category != category and category != "other":
                        # Tracker re-classified; keep the newest confident class.
                        user.category = category
                        user.raw_class = raw_class
                    user.update(cx, cy, speed_mps, heading, confidence, frame_time)

                seen_ids.add(key)

            # Retire stale users, recording their exit approach. Judged on the
            # FRAME clock (frame_time), not wall-clock: tests and offline video
            # replay feed synthetic timestamps that may sit far from time.time().
            expired = [
                k for k, u in self.users.items()
                if frame_time - u.last_seen > STALE_AFTER_SECONDS
            ]
            for k in expired:
                user = self.users.pop(k)
                user.exit_approach = RoadUser.approach_of(user.x, user.y)
                self.completed_trips.append({
                    "track_id": user.track_id,
                    "category": user.category,
                    "entry": user.approach,
                    "exit": user.exit_approach,
                    "peak_speed_mps": round(user.peak_speed_mps, 2),
                    "duration_s": round(user.last_seen - user.first_seen, 1),
                    "ended_at": round(frame_time, 3),
                })
            self.last_update = now

    # ── Queries ─────────────────────────────────────────────────────────

    def active_users(self) -> List[Dict]:
        with self._lock:
            return [u.to_dict() for u in self.users.values()]

    def counts_by_category(self) -> Dict[str, int]:
        with self._lock:
            counts: Dict[str, int] = {}
            for u in self.users.values():
                counts[u.category] = counts.get(u.category, 0) + 1
            return counts

    def vru_count(self) -> int:
        with self._lock:
            return sum(1 for u in self.users.values() if u.category in VRU_CATEGORIES)

    def stats(self) -> Dict:
        with self._lock:
            return {
                "active_objects": len(self.users),
                "total_observed": self.total_observed,
                "completed_trips": len(self.completed_trips),
                "vru_active": sum(
                    1 for u in self.users.values() if u.category in VRU_CATEGORIES
                ),
                "last_update_age_s": (
                    round(time.time() - self.last_update, 1)
                    if self.last_update else None
                ),
            }

    def recent_trips(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            trips = list(self.completed_trips)[-limit:]
            trips.reverse()
            return trips
