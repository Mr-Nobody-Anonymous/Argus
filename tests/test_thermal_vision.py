"""
Unit and integration tests for Thermal Vision in Argus.
Covers:
- Radiometric temperature conversion and calibration
- ROI temperature extraction
- False-color palette generation (white_hot, black_hot, ironbow, inferno, jet, etc.)
- Hotspot & anomaly detection (EBT fever screening, fire detection)
- Thermal signature segmentation
- Multispectral RGB + Thermal image fusion
- REST API endpoint verification
"""
import numpy as np
import pytest
import cv2

from backend.services.vision.thermal_vision import (
    ThermalVision, get_thermal_vision,
    PALETTE_WHITE_HOT, PALETTE_BLACK_HOT, PALETTE_IRONBOW, PALETTE_INFERNO,
    PALETTE_JET, PALETTE_RAINBOW, PALETTE_BONE, PALETTE_HOT,
    SUPPORTED_PALETTES,
)


@pytest.fixture
def sample_thermal_frame() -> np.ndarray:
    """Create a synthetic 100x100 8-bit thermal frame with cold background and warm target."""
    frame = np.full((100, 100), 40, dtype=np.uint8)  # ~15°C cold ambient
    # Add a warm person-like blob (~37°C body heat, pixel value ~110)
    cv2.circle(frame, (50, 50), 15, 110, -1)
    # Add a high-temperature hotspot (~80°C overheating hotspot, pixel value ~210)
    cv2.circle(frame, (20, 20), 8, 210, -1)
    return frame


@pytest.fixture
def sample_rgb_frame() -> np.ndarray:
    """Create a matching 100x100 3-channel optical RGB frame."""
    return np.full((100, 100, 3), 128, dtype=np.uint8)


class TestThermalCalibration:
    def test_default_calibration(self):
        tv = ThermalVision()
        cal = tv.get_calibration()
        assert cal["temp_min_c"] == -10.0
        assert cal["temp_max_c"] == 100.0
        assert cal["emissivity"] == 0.95
        assert cal["default_palette"] == PALETTE_IRONBOW
        assert len(cal["supported_palettes"]) == 8

    def test_update_calibration(self):
        tv = ThermalVision()
        updated = tv.update_calibration(temp_min_c=0.0, temp_max_c=150.0, emissivity=0.98, default_palette=PALETTE_INFERNO)
        assert updated["temp_min_c"] == 0.0
        assert updated["temp_max_c"] == 150.0
        assert updated["emissivity"] == 0.98
        assert updated["default_palette"] == PALETTE_INFERNO


class TestRadiometricTemperature:
    def test_raw_to_temperature_bounds(self):
        tv = ThermalVision(temp_min_c=0.0, temp_max_c=100.0, emissivity=1.0)
        black_frame = np.zeros((10, 10), dtype=np.uint8)
        white_frame = np.full((10, 10), 255, dtype=np.uint8)

        t_min = tv.raw_to_temperature(black_frame)
        t_max = tv.raw_to_temperature(white_frame)

        assert float(np.mean(t_min)) == pytest.approx(0.0, abs=0.1)
        assert float(np.mean(t_max)) == pytest.approx(100.0, abs=0.1)

    def test_roi_temperature_statistics(self, sample_thermal_frame):
        tv = ThermalVision(temp_min_c=-10.0, temp_max_c=100.0, emissivity=1.0)
        # Check full frame
        stats = tv.get_roi_temperature(sample_thermal_frame)
        assert stats["max_c"] > stats["min_c"]
        assert "min_f" in stats and "max_f" in stats and "mean_f" in stats

        # Check ROI around the hotspot at (20, 20)
        hotspot_roi = tv.get_roi_temperature(sample_thermal_frame, bbox=[10, 10, 30, 30])
        assert hotspot_roi["max_c"] > 70.0


class TestThermalColorization:
    def test_supported_palettes_generate_3_channels(self, sample_thermal_frame):
        tv = ThermalVision()
        for palette in SUPPORTED_PALETTES:
            colored = tv.colorize(sample_thermal_frame, palette=palette)
            assert colored.shape == (100, 100, 3)
            assert colored.dtype == np.uint8

    def test_white_hot_and_black_hot(self, sample_thermal_frame):
        tv = ThermalVision()
        white_hot = tv.colorize(sample_thermal_frame, palette=PALETTE_WHITE_HOT)
        black_hot = tv.colorize(sample_thermal_frame, palette=PALETTE_BLACK_HOT)

        # In white-hot, hotspot at (20, 20) should have high pixel intensity
        # In black-hot, hotspot at (20, 20) should be inverted (lower intensity)
        assert white_hot[20, 20, 0] > black_hot[20, 20, 0]


class TestHotspotDetection:
    def test_detect_hotspots_fever_and_fire(self, sample_thermal_frame):
        tv = ThermalVision(temp_min_c=-10.0, temp_max_c=100.0, emissivity=1.0)
        # Threshold at 35°C should catch both warm body (37°C) and hotspot (80°C)
        hotspots = tv.detect_hotspots(sample_thermal_frame, threshold_temp_c=35.0, min_area_px=10)
        assert len(hotspots) >= 2

        # Most intense hotspot should be critical (>65°C)
        assert hotspots[0]["severity"] == "critical"
        assert hotspots[0]["peak_temp_c"] > 75.0

    def test_detect_thermal_signatures(self, sample_thermal_frame):
        tv = ThermalVision(temp_min_c=-10.0, temp_max_c=100.0, emissivity=1.0)
        sigs = tv.detect_thermal_signatures(sample_thermal_frame, min_area_px=10)
        assert len(sigs) >= 1
        assert "bbox" in sigs[0]
        assert "mean_temp_c" in sigs[0]
        assert "peak_temp_c" in sigs[0]


class TestSpectralFusion:
    def test_fuse_rgb_thermal(self, sample_rgb_frame, sample_thermal_frame):
        tv = ThermalVision()
        fused = tv.fuse_rgb_thermal(sample_rgb_frame, sample_thermal_frame, alpha=0.4, palette=PALETTE_IRONBOW)
        assert fused.shape == (100, 100, 3)
        assert fused.dtype == np.uint8


class TestThermalApiEndpoints:
    @pytest.fixture
    def client(self):
        from starlette.testclient import TestClient
        from backend.api.main import app
        return TestClient(app)

    @pytest.fixture
    def auth_headers(self):
        from backend.api.auth import AuthUser, get_current_user
        from backend.api.main import app
        admin_user = AuthUser(id=1, username="admin", role="admin", is_superuser=True, is_staff=True)
        app.dependency_overrides[get_current_user] = lambda: admin_user
        yield {"Authorization": "Bearer mock-admin-token"}
        app.dependency_overrides.pop(get_current_user, None)

    def test_get_palettes_api(self, client, auth_headers):
        resp = client.get("/api/v1/thermal/palettes", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "palettes" in data
        assert "ironbow" in data["palettes"]
        assert "white_hot" in data["palettes"]

    def test_get_and_update_calibration_api(self, client, auth_headers):
        resp = client.get("/api/v1/thermal/calibration", headers=auth_headers)
        assert resp.status_code == 200
        cal = resp.json()
        assert "temp_min_c" in cal

        # Update calibration
        post_resp = client.post(
            "/api/v1/thermal/calibration",
            json={"temp_min_c": -5.0, "temp_max_c": 120.0, "default_palette": "inferno"},
            headers=auth_headers,
        )
        assert post_resp.status_code == 200
        updated = post_resp.json()["calibration"]
        assert updated["temp_min_c"] == -5.0
        assert updated["temp_max_c"] == 120.0

    def test_analyze_thermal_image_api(self, client, auth_headers, sample_thermal_frame):
        # Encode frame as png bytes
        ok, buf = cv2.imencode(".png", sample_thermal_frame)
        assert ok
        files = {"file": ("test_thermal.png", buf.tobytes(), "image/png")}

        resp = client.post(
            "/api/v1/thermal/analyze?threshold_temp_c=35.0",
            files=files,
            headers=auth_headers,
        )
        assert resp.status_code == 200
        res = resp.json()
        assert "overall_temperature" in res
        assert "hotspots" in res
        assert res["hotspots_count"] >= 1

