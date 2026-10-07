"""
CityOS calibration - sensor coordinates -> world coordinates.

Bridges the gap between normalised image space and physical space:

    normalised (x, y) in [0..1]  --calibration-->  metres east/north of the
                                                   intersection centre

A calibration carries:
  - view_width_m / view_height_m : real-world size of the camera view
  - yaw_deg                      : rotation from IMAGE north (up-screen) to
                                   TRUE north, clockwise. 0 means the camera
                                   is aligned so up-screen is geographic north.
  - centre                       : where the intersection centre sits in the
                                   image (default 0.5, 0.5)

Headings are rotated too, so "E" in image space becomes the true compass
direction a traffic engineer would recognise. Without a calibration the
perception layer still works - distances just fall back to the assumed view
width, exactly as before.
"""
import math
from typing import Dict, Optional, Tuple

COMPASS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


class CameraCalibration:
    """Per-sensor extrinsic approximation for ground-plane projection."""

    def __init__(self, view_width_m: float = 30.0,
                 view_height_m: float = 22.5,
                 yaw_deg: float = 0.0,
                 centre: Tuple[float, float] = (0.5, 0.5),
                 source: str = "default",
                 geo_origin: Optional[Dict] = None,
                 mount_height_m: Optional[float] = None):
        self.view_width_m = float(view_width_m)
        self.view_height_m = float(view_height_m)
        self.yaw_deg = float(yaw_deg) % 360.0
        self.centre = (float(centre[0]), float(centre[1]))
        self.source = source
        # Geographic anchor of the intersection centre (WGS84). Optional:
        # without it world coordinates stay local ENU metres.
        self.geo_origin = dict(geo_origin) if geo_origin else None
        self.mount_height_m = (float(mount_height_m)
                               if mount_height_m is not None else None)

    # ── Geographic transform ────────────────────────────────────────────

    METRES_PER_DEG_LAT = 111320.0

    def to_geo(self, east_m: float,
               north_m: float) -> Optional[Tuple[float, float]]:
        """Local ENU metres -> (latitude, longitude) via the geo anchor."""
        if not self.geo_origin or "lat" not in self.geo_origin \
                or "lon" not in self.geo_origin:
            return None
        lat0 = float(self.geo_origin["lat"])
        lon0 = float(self.geo_origin["lon"])
        d_lat = north_m / self.METRES_PER_DEG_LAT
        metres_per_deg_lon = (self.METRES_PER_DEG_LAT
                              * math.cos(math.radians(lat0)))
        d_lon = east_m / max(metres_per_deg_lon, 1e-9)
        return (round(lat0 + d_lat, 7), round(lon0 + d_lon, 7))

    # ── Coordinate transforms ───────────────────────────────────────────

    def to_world(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """Normalised image coords -> metres (east, north) of the centre."""
        dx = (x_norm - self.centre[0]) * self.view_width_m
        dy = (self.centre[1] - y_norm) * self.view_height_m   # up-screen = +
        rad = math.radians(self.yaw_deg)
        # Rotate the image vector into world axes (clockwise yaw).
        east = dx * math.cos(rad) + dy * math.sin(rad)
        north = -dx * math.sin(rad) + dy * math.cos(rad)
        return (round(east, 2), round(north, 2))

    def to_image(self, east_m: float, north_m: float) -> Tuple[float, float]:
        """Inverse of to_world - useful for drawing map geometry."""
        rad = math.radians(-self.yaw_deg)
        dx = east_m * math.cos(rad) + north_m * math.sin(rad)
        dy = -east_m * math.sin(rad) + north_m * math.cos(rad)
        x = dx / self.view_width_m + self.centre[0]
        y = self.centre[1] - dy / self.view_height_m
        return (round(x, 4), round(y, 4))

    def world_heading(self, image_heading: str) -> str:
        """Rotate a compass label from image axes to true axes by yaw."""
        try:
            idx = COMPASS.index(image_heading)
        except ValueError:
            return image_heading
        shift = int(round(self.yaw_deg / 45.0)) % 8
        return COMPASS[(idx + shift) % 8]

    def _to_world_raw(self, x_norm: float, y_norm: float) -> Tuple[float, float]:
        """to_world without rounding - for exact distance maths."""
        dx = (x_norm - self.centre[0]) * self.view_width_m
        dy = (self.centre[1] - y_norm) * self.view_height_m
        rad = math.radians(self.yaw_deg)
        east = dx * math.cos(rad) + dy * math.sin(rad)
        north = -dx * math.sin(rad) + dy * math.cos(rad)
        return (east, north)

    def distance_m(self, ax, ay, bx, by) -> float:
        """Ground-plane distance in metres between two normalised points."""
        eax, nan_ = self._to_world_raw(ax, ay)
        ebx, nbn = self._to_world_raw(bx, by)
        return math.hypot(eax - ebx, nan_ - nbn)

    # ── Serialisation ───────────────────────────────────────────────────

    def to_dict(self) -> Dict:
        out = {
            "view_width_m": self.view_width_m,
            "view_height_m": self.view_height_m,
            "yaw_deg": self.yaw_deg,
            "centre": list(self.centre),
            "source": self.source,
        }
        if self.geo_origin:
            out["geo_origin"] = dict(self.geo_origin)
        if self.mount_height_m is not None:
            out["mount_height_m"] = self.mount_height_m
        return out

    @classmethod
    def from_dict(cls, data: Dict) -> "CameraCalibration":
        return cls(
            view_width_m=data.get("view_width_m", 30.0),
            view_height_m=data.get("view_height_m", 22.5),
            yaw_deg=data.get("yaw_deg", 0.0),
            centre=tuple(data.get("centre", (0.5, 0.5))),
            source=data.get("source", "operator"),
            geo_origin=data.get("geo_origin"),
            mount_height_m=data.get("mount_height_m"),
        )
