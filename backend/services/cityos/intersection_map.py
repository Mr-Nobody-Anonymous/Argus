"""
CityOS intersection map - lane-level geometry.

Gives the perception model the vocabulary a traffic engineer uses:

  - approaches (north / east / south / west)
  - lanes with legal movements (through / left / right / bike)
  - stop lines
  - crosswalks

Every road user is enriched with: lane id, movement, distance to their
approach's stop line (metres, via calibration), and whether they are inside
a crosswalk. That turns "vehicle #184 at (0.42, 0.31)" into "vehicle #184 in
the southbound through lane, 12 m upstream of the stop line".

The default map is a symmetric four-approach junction sized from the
calibration. An operator can supply explicit geometry instead.
"""
from typing import Dict, List, Optional

APPROACHES = ("north", "east", "south", "west")
MOVEMENTS = ("through", "left", "right", "bike")

# Approach -> axis of its signal phase.
APPROACH_PHASE = {"north": "NS", "south": "NS", "east": "EW", "west": "EW"}

# Legal travel heading per approach movement (image-space compass).
# north approach = vehicles travelling southward, etc.
LEGAL_HEADINGS = {
    ("north", "through"): "S",
    ("north", "left"): "E",      # left turn onto the cross street
    ("north", "right"): "W",
    ("south", "through"): "N",
    ("south", "left"): "W",
    ("south", "right"): "E",
    ("east", "through"): "W",
    ("east", "left"): "S",
    ("east", "right"): "N",
    ("west", "through"): "E",
    ("west", "left"): "N",
    ("west", "right"): "S",
}

# Stop-line coordinate per approach (normalised). Vehicles on the north
# approach travel southward and cross y=0.32; symmetric for the others.
DEFAULT_STOP_LINES = {"north": 0.32, "south": 0.68, "west": 0.32, "east": 0.68}

# Lane band offsets across each approach road (fraction of road half-width).
LANE_OFFSETS = {"left": -0.28, "through": 0.0, "right": 0.28}


class Lane:
    __slots__ = ("id", "approach", "movement", "heading")

    def __init__(self, lane_id: str, approach: str, movement: str,
                 heading: str):
        self.id = lane_id
        self.approach = approach
        self.movement = movement
        self.heading = heading

    def to_dict(self) -> Dict:
        return {"id": self.id, "approach": self.approach,
                "movement": self.movement, "legal_heading": self.heading}


class IntersectionMap:
    """Lane-level geometry for one intersection."""

    def __init__(self, config: Optional[Dict] = None):
        config = config or {}
        self.stop_lines: Dict[str, float] = {
            **DEFAULT_STOP_LINES, **config.get("stop_lines", {})
        }
        self.lanes: Dict[str, Lane] = {}
        for approach in APPROACHES:
            for movement in MOVEMENTS:
                if movement == "bike" and not config.get("bike_lanes", True):
                    continue
                lane_id = f"{approach[0].upper()}_{movement}"
                self.lanes[lane_id] = Lane(
                    lane_id, approach, movement,
                    LEGAL_HEADINGS[(approach, movement)],
                )
        # Crosswalk bands: just outside each stop line.
        self.crosswalks: Dict[str, Dict] = config.get("crosswalks") or {
            a: {"offset_m": 2.5} for a in APPROACHES
        }

    # ── Classification ──────────────────────────────────────────────────

    def classify(self, user: Dict, cal) -> Dict:
        """Enrich a road-user dict with lane-level attributes (in place)."""
        x = user["position"]["x"]
        y = user["position"]["y"]
        approach = user.get("approach") or "unknown"

        best_lane, best_score = None, -1.0
        for lane in self.lanes.values():
            if lane.approach != approach:
                continue
            score = self._lane_score(lane, user)
            if score > best_score:
                best_lane, best_score = lane, score

        if best_lane is not None:
            user["lane_id"] = best_lane.id
            user["movement"] = best_lane.movement
            user["legal_heading"] = best_lane.heading
        else:
            user["lane_id"] = None
            user["movement"] = None
            user["legal_heading"] = None

        user["distance_to_stop_line_m"] = round(
            self.distance_to_stop_line(approach, x, y, cal), 1)

        user["in_crosswalk"] = self._in_crosswalk(approach, x, y, cal)
        return user

    def _lane_score(self, lane: Lane, user: Dict) -> float:
        """How well a user matches a lane: lateral band + heading agreement."""
        x, y = user["position"]["x"], user["position"]["y"]
        stop = self.stop_lines[lane.approach]
        if lane.approach in ("north", "south"):
            lateral = x
        else:
            lateral = y
        target = 0.5 + LANE_OFFSETS[lane.movement] * 0.18
        lateral_score = 1.0 - min(abs(lateral - target) / 0.25, 1.0)

        heading = user.get("heading")
        heading_score = 0.5
        if heading and heading != "-":
            if heading == lane.heading:
                heading_score = 1.0
            elif _angular_diff(heading, lane.heading) <= 45:
                heading_score = 0.6
            else:
                heading_score = 0.1
        return lateral_score * 0.6 + heading_score * 0.4

    def distance_to_stop_line(self, approach: str, x: float, y: float,
                              cal) -> float:
        """Metres upstream of the stop line; negative once past it."""
        stop = self.stop_lines.get(approach)
        if stop is None or approach == "unknown":
            return 0.0
        if approach == "north":
            d_norm = stop - y          # approaching from above (smaller y)
        elif approach == "south":
            d_norm = y - stop
        elif approach == "west":
            d_norm = stop - x
        else:                          # east
            d_norm = x - stop
        scale = (cal.view_height_m if approach in ("north", "south")
                 else cal.view_width_m)
        return d_norm * scale

    def _in_crosswalk(self, approach: str, x: float, y: float, cal) -> bool:
        offset = self.crosswalks.get(approach, {}).get("offset_m", 2.5)
        stop = self.stop_lines.get(approach)
        if stop is None:
            return False
        if approach == "north":
            d = (stop - y) * cal.view_height_m
        elif approach == "south":
            d = (y - stop) * cal.view_height_m
        elif approach == "west":
            d = (stop - x) * cal.view_width_m
        else:
            d = (x - stop) * cal.view_width_m
        return 0 <= d <= offset

    def past_stop_line(self, approach: str, x: float, y: float, cal) -> bool:
        return self.distance_to_stop_line(approach, x, y, cal) <= 0.0

    # ── Serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        return {
            "stop_lines": dict(self.stop_lines),
            "lanes": [l.to_dict() for l in self.lanes.values()],
            "crosswalks": {k: dict(v) for k, v in self.crosswalks.items()},
        }


COMPASS_IDX = {h: i for i, h in enumerate(
    ["N", "NE", "E", "SE", "S", "SW", "W", "NW"])}


def _angular_diff(a: str, b: str) -> int:
    """Smallest compass-step distance between two headings, in degrees."""
    da = abs(COMPASS_IDX[a] - COMPASS_IDX[b]) % 8
    return min(da, 8 - da) * 45
