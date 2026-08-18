"""Per-camera ground-plane calibration, and the honesty rule that goes with it.

`speed_analysis.calibration_factor` in config.yaml is a single global constant
(0.05 m/px). It is applied to every camera regardless of lens, mounting height
or viewing angle, and nothing in the repository ever measured it. Multiplying
pixel displacement by a guess produces a number in m/s that *looks* like a
measurement, and a speed-violation event built on it would state "vehicle
travelled at 47 km/h" with no basis whatsoever.

A wrong speed is worse than no speed, because a wrong speed is actionable.

So this module makes calibration explicit and per-camera:

* A camera with a declared `meters_per_pixel` (or a reference object of known
  real width and its pixel width) is **calibrated**: speeds are real, and the
  speed-violation rule may fire.
* A camera without one is **uncalibrated**: `speed_mps` is still computed for
  relative comparison (is this thing faster than that thing?) but it is tagged
  `calibrated=False`, and the speed-violation rule refuses to fire. It reports
  its reason instead of a number.

This mirrors how the capability registry treats a missing model: unavailable
with a reason, never a silent fallback to a fabricated result.

Perspective note: a single scalar m/px is itself an approximation - objects far
from the camera cover fewer pixels per metre than near ones. That error is
bounded and documented here rather than hidden; `homography` support is the
correct fix and is declared as the upgrade path in the roadmap. For a fixed
overhead or shallow-angle camera the scalar is adequate within roughly +/-25%,
which is why the rule additionally requires a configurable margin over the
threshold before it fires.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)

# How far over the limit a speed must be before it is reported, to absorb the
# error inherent in a scalar metres-per-pixel approximation. 1.25 = 25% over.
DEFAULT_VIOLATION_MARGIN = 1.25


@dataclass(frozen=True)
class CameraCalibration:
    """What one camera knows about the size of the world it is looking at."""

    camera_id: int
    meters_per_pixel: Optional[float] = None
    source: str = "none"
    note: str = ""

    @property
    def is_calibrated(self) -> bool:
        return self.meters_per_pixel is not None and self.meters_per_pixel > 0

    def reason(self) -> str:
        if self.is_calibrated:
            return (
                f"calibrated at {self.meters_per_pixel:.4f} m/px from {self.source}"
            )
        return (
            f"camera {self.camera_id} has no ground-plane calibration, so pixel "
            f"motion cannot be converted to a real speed"
        )


class CalibrationRegistry:
    """Resolves calibration per camera from config, with an explicit unknown."""

    def __init__(self, config=None):
        self._config = config
        self._cache: Dict[int, CameraCalibration] = {}
        self._lock = threading.RLock()

    def _cfg(self):
        if self._config is None:
            from backend.config.config import get_config

            self._config = get_config()
        return self._config

    def get(self, camera_id: int) -> CameraCalibration:
        with self._lock:
            if camera_id in self._cache:
                return self._cache[camera_id]
            cal = self._resolve(camera_id)
            self._cache[camera_id] = cal
            return cal

    def _resolve(self, camera_id: int) -> CameraCalibration:
        try:
            from backend.config.config import section_to_dict

            cfg = self._cfg()
            section = section_to_dict(getattr(cfg, "camera_calibration", {})) or {}
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Calibration config unavailable: {exc}")
            section = {}

        entry = section.get(str(camera_id)) or section.get(camera_id)
        if isinstance(entry, dict):
            mpp = entry.get("meters_per_pixel")
            if mpp is None:
                # Derive from a reference object of known real size.
                real_m = entry.get("reference_width_m")
                px = entry.get("reference_width_px")
                if real_m and px:
                    try:
                        mpp = float(real_m) / float(px)
                    except (TypeError, ValueError, ZeroDivisionError):
                        mpp = None
            if mpp:
                try:
                    return CameraCalibration(
                        camera_id=camera_id,
                        meters_per_pixel=float(mpp),
                        source=entry.get("source", "config"),
                        note=entry.get("note", ""),
                    )
                except (TypeError, ValueError):
                    pass

        return CameraCalibration(camera_id=camera_id, meters_per_pixel=None)

    def reset(self) -> None:
        with self._lock:
            self._cache.clear()

    def report(self) -> Dict[str, object]:
        """What is calibrated and what is not - surfaced through the API."""
        with self._lock:
            cams = dict(self._cache)
        return {
            "calibrated": sorted(c for c, v in cams.items() if v.is_calibrated),
            "uncalibrated": sorted(c for c, v in cams.items() if not v.is_calibrated),
            "speed_violation_requires_calibration": True,
            "note": (
                "Speed violations are only reported for calibrated cameras. "
                "Uncalibrated cameras still measure relative motion, but no "
                "absolute speed is claimed."
            ),
        }


_registry: Optional[CalibrationRegistry] = None
_lock = threading.Lock()


def get_calibration_registry() -> CalibrationRegistry:
    global _registry
    if _registry is None:
        with _lock:
            if _registry is None:
                _registry = CalibrationRegistry()
    return _registry


def reset_calibration_registry() -> None:
    global _registry
    with _lock:
        _registry = None
