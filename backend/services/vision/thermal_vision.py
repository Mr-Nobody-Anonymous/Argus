"""
Thermal Vision and Infrared Analytics Service for Argus.

Provides:
- Thermal image ingestion (monochrome LWIR, 8-bit/16-bit radiometric frames)
- Thermal false-color palette generation (White-Hot, Black-Hot, Ironbow, Inferno, Jet, Rainbow, Bone)
- Radiometric temperature calibration and spot temperature measurements
- Thermal hotspot, elevated body temperature (fever), and fire/overheating detection
- Infrared thermal signature / object segmentation in zero-lux environments
- Cross-spectral (RGB + Thermal) image fusion
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

# Supported false-color palettes
PALETTE_WHITE_HOT = "white_hot"
PALETTE_BLACK_HOT = "black_hot"
PALETTE_IRONBOW = "ironbow"
PALETTE_INFERNO = "inferno"
PALETTE_JET = "jet"
PALETTE_RAINBOW = "rainbow"
PALETTE_BONE = "bone"
PALETTE_HOT = "hot"

SUPPORTED_PALETTES = [
    PALETTE_WHITE_HOT,
    PALETTE_BLACK_HOT,
    PALETTE_IRONBOW,
    PALETTE_INFERNO,
    PALETTE_JET,
    PALETTE_RAINBOW,
    PALETTE_BONE,
    PALETTE_HOT,
]


class ThermalVision:
    """
    Thermal Vision analyzer for surveillance feeds.
    Converts thermal radiance/grayscale feeds into calibrated temperature arrays,
    detects abnormal heat signatures, generates false-color palettes, and performs
    multispectral RGB-IR fusion.
    """

    def __init__(
        self,
        temp_min_c: float = -10.0,
        temp_max_c: float = 100.0,
        emissivity: float = 0.95,
        default_palette: str = PALETTE_IRONBOW,
    ):
        self.temp_min_c = float(temp_min_c)
        self.temp_max_c = float(temp_max_c)
        self.emissivity = max(0.1, min(1.0, float(emissivity)))
        self.default_palette = default_palette if default_palette in SUPPORTED_PALETTES else PALETTE_IRONBOW

    def update_calibration(
        self,
        temp_min_c: Optional[float] = None,
        temp_max_c: Optional[float] = None,
        emissivity: Optional[float] = None,
        default_palette: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Update radiometric sensor calibration settings."""
        if temp_min_c is not None:
            self.temp_min_c = float(temp_min_c)
        if temp_max_c is not None:
            self.temp_max_c = float(temp_max_c)
        if emissivity is not None:
            self.emissivity = max(0.1, min(1.0, float(emissivity)))
        if default_palette is not None and default_palette in SUPPORTED_PALETTES:
            self.default_palette = default_palette

        return self.get_calibration()

    def get_calibration(self) -> Dict[str, Any]:
        """Return current calibration and temperature scaling factors."""
        return {
            "temp_min_c": self.temp_min_c,
            "temp_max_c": self.temp_max_c,
            "emissivity": self.emissivity,
            "default_palette": self.default_palette,
            "supported_palettes": SUPPORTED_PALETTES,
        }

    def _ensure_gray(self, frame: np.ndarray) -> np.ndarray:
        """Ensure input frame is single-channel 8-bit or 16-bit grayscale."""
        if frame is None or frame.size == 0:
            raise ValueError("Empty or invalid frame provided to ThermalVision")

        if len(frame.shape) == 3:
            if cv2 is not None:
                return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            return np.mean(frame, axis=2).astype(np.uint8)
        return frame

    def raw_to_temperature(self, frame: np.ndarray) -> np.ndarray:
        """
        Convert pixel intensities into calibrated surface temperatures in degrees Celsius.
        Linear transfer model: T = temp_min + (intensity / max_val) * (temp_max - temp_min) / emissivity.
        """
        gray = self._ensure_gray(frame)
        max_val = 65535.0 if gray.dtype == np.uint16 else 255.0

        normalized = gray.astype(np.float32) / max_val
        temp_matrix = self.temp_min_c + normalized * (self.temp_max_c - self.temp_min_c)
        if self.emissivity < 1.0:
            temp_matrix = temp_matrix / self.emissivity

        return temp_matrix

    def get_roi_temperature(
        self,
        frame: np.ndarray,
        bbox: Optional[Dict[str, float] | List[float] | Tuple[float, float, float, float]] = None,
    ) -> Dict[str, float]:
        """
        Calculate temperature statistics for a region of interest (or full frame).
        Returns: {min_c, max_c, mean_c, std_c, min_f, max_f, mean_f}
        """
        temp_matrix = self.raw_to_temperature(frame)
        h, w = temp_matrix.shape[:2]

        if bbox is not None:
            if isinstance(bbox, dict):
                x1, y1 = int(bbox.get("x1", 0)), int(bbox.get("y1", 0))
                x2, y2 = int(bbox.get("x2", w)), int(bbox.get("y2", h))
            else:
                x1, y1, x2, y2 = [int(v) for v in bbox[:4]]

            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            roi = temp_matrix[y1:y2, x1:x2]
        else:
            roi = temp_matrix

        if roi.size == 0:
            return {"min_c": 0.0, "max_c": 0.0, "mean_c": 0.0, "std_c": 0.0}

        min_c = float(np.min(roi))
        max_c = float(np.max(roi))
        mean_c = float(np.mean(roi))
        std_c = float(np.std(roi))

        def _to_f(c):
            return round((c * 9.0 / 5.0) + 32.0, 2)

        return {
            "min_c": round(min_c, 2),
            "max_c": round(max_c, 2),
            "mean_c": round(mean_c, 2),
            "std_c": round(std_c, 2),
            "min_f": _to_f(min_c),
            "max_f": _to_f(max_c),
            "mean_f": _to_f(mean_c),
        }

    def colorize(self, frame: np.ndarray, palette: Optional[str] = None) -> np.ndarray:
        """
        Render a thermal grayscale image with the selected false-color surveillance palette.
        Returns 3-channel uint8 BGR image ready for display/streaming.
        """
        palette = palette or self.default_palette
        if palette not in SUPPORTED_PALETTES:
            palette = PALETTE_IRONBOW

        gray = self._ensure_gray(frame)
        if gray.dtype != np.uint8:
            gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U) if cv2 is not None else (
                (gray / np.max(gray)) * 255
            ).astype(np.uint8)

        if palette == PALETTE_WHITE_HOT:
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if cv2 is not None else np.stack([gray]*3, axis=-1)

        if palette == PALETTE_BLACK_HOT:
            inv = 255 - gray
            return cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR) if cv2 is not None else np.stack([inv]*3, axis=-1)

        if cv2 is None:
            return np.stack([gray]*3, axis=-1)

        colormap_mapping = {
            PALETTE_INFERNO: cv2.COLORMAP_INFERNO,
            PALETTE_JET: cv2.COLORMAP_JET,
            PALETTE_RAINBOW: cv2.COLORMAP_RAINBOW,
            PALETTE_BONE: cv2.COLORMAP_BONE,
            PALETTE_HOT: cv2.COLORMAP_HOT,
        }

        if palette in colormap_mapping:
            return cv2.applyColorMap(gray, colormap_mapping[palette])

        # Default: Custom Ironbow-style palette using TURBO / INFERNO blended contrast
        return cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO)

    def detect_hotspots(
        self,
        frame: np.ndarray,
        threshold_temp_c: float = 37.5,
        min_area_px: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Detect connected regions exceeding a given temperature threshold.
        Useful for Elevated Body Temperature (EBT / fever screening >= 37.5°C),
        industrial equipment overheating, or fire hotspot detection.
        """
        temp_matrix = self.raw_to_temperature(frame)
        h, w = temp_matrix.shape[:2]

        hot_mask = (temp_matrix >= threshold_temp_c).astype(np.uint8) * 255

        if cv2 is None:
            if np.any(hot_mask):
                max_t = float(np.max(temp_matrix))
                return [{
                    "bbox": [0, 0, w, h],
                    "peak_temp_c": round(max_t, 2),
                    "mean_temp_c": round(float(np.mean(temp_matrix[hot_mask > 0])), 2),
                    "area_px": int(np.sum(hot_mask > 0)),
                    "severity": "critical" if max_t >= 60.0 else "warning",
                }]
            return []

        contours, _ = cv2.findContours(hot_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        hotspots = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            roi_temp = temp_matrix[y:y+bh, x:x+bw]
            peak_temp = float(np.max(roi_temp))
            mean_temp = float(np.mean(roi_temp[hot_mask[y:y+bh, x:x+bw] > 0]))

            # Determine severity based on temperature
            if peak_temp >= 65.0:
                severity = "critical"      # Fire / extreme overheating
            elif peak_temp >= 38.5:
                severity = "high"          # Definite fever / abnormal heat
            elif peak_temp >= threshold_temp_c:
                severity = "warning"       # Threshold exceeded
            else:
                severity = "info"

            hotspots.append({
                "bbox": [x, y, x + bw, y + bh],
                "area_px": int(area),
                "peak_temp_c": round(peak_temp, 2),
                "mean_temp_c": round(mean_temp, 2),
                "severity": severity,
                "centroid": [round(x + bw / 2.0, 1), round(y + bh / 2.0, 1)],
            })

        hotspots.sort(key=lambda h: h["peak_temp_c"], reverse=True)
        return hotspots

    def detect_thermal_signatures(
        self,
        frame: np.ndarray,
        min_contrast_std: float = 1.5,
        min_area_px: int = 50,
    ) -> List[Dict[str, Any]]:
        """
        Segment distinct warm entities (people, operating vehicles) in pitch-black or low-contrast
        environments using adaptive Otsu thresholding on the thermal intensity matrix.
        """
        gray = self._ensure_gray(frame)
        if cv2 is None:
            return []

        # Smooth noise while preserving thermal step edges
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        temp_matrix = self.raw_to_temperature(gray)

        signatures = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area_px:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            roi_temp = temp_matrix[y:y+h, x:x+w]
            mean_t = float(np.mean(roi_temp))
            max_t = float(np.max(roi_temp))

            signatures.append({
                "bbox": [x, y, x + w, y + h],
                "area_px": int(area),
                "aspect_ratio": round(w / float(h), 2) if h > 0 else 1.0,
                "mean_temp_c": round(mean_t, 2),
                "peak_temp_c": round(max_t, 2),
            })

        return signatures

    def fuse_rgb_thermal(
        self,
        rgb_frame: np.ndarray,
        thermal_frame: np.ndarray,
        alpha: float = 0.5,
        palette: Optional[str] = None,
    ) -> np.ndarray:
        """
        Multispectral fusion: overlay a false-color thermal heatmap on an optical RGB image.
        fused = (1 - alpha) * rgb + alpha * thermal_colored
        """
        alpha = max(0.0, min(1.0, float(alpha)))
        thermal_colored = self.colorize(thermal_frame, palette)

        if cv2 is None:
            return rgb_frame

        # Resize thermal to match RGB if dimensions differ
        if rgb_frame.shape[:2] != thermal_colored.shape[:2]:
            h, w = rgb_frame.shape[:2]
            thermal_colored = cv2.resize(thermal_colored, (w, h), interpolation=cv2.INTER_LINEAR)

        fused = cv2.addWeighted(rgb_frame, 1.0 - alpha, thermal_colored, alpha, 0)
        return fused


_thermal_vision_instance: Optional[ThermalVision] = None


def get_thermal_vision() -> ThermalVision:
    """Singleton getter for ThermalVision service."""
    global _thermal_vision_instance
    if _thermal_vision_instance is None:
        _thermal_vision_instance = ThermalVision()
    return _thermal_vision_instance
