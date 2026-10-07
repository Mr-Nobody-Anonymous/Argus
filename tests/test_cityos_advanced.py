"""
CityOS advanced-capability tests.

Covers the layers that close the gap toward a full CityOS-class system:
calibration/world coordinates, lane-level maps, red-light running,
stopped vehicles, PET conflicts, hard braking, queue estimation,
pedestrian signal phases, corridor continuity, sensor health,
deterministic replay and privacy-governed id expiry.
"""
import time

import pytest

from backend.services.cityos.calibration import CameraCalibration
from backend.services.cityos.corridor import CorridorLink, CorridorService
from backend.services.cityos.engine import CityOSEngine
from backend.services.cityos.intersection_map import IntersectionMap
from backend.services.cityos.perception_engine import PerceptionEngine
from backend.services.cityos.safety_analytics import SafetyAnalytics
from backend.services.cityos.signal_optimizer import SignalOptimizer
from backend.services.cityos.traffic_flow import TrafficFlowAnalyzer


def det(track_id, cls, x1, y1, x2, y2, conf=0.9):
    return {
        "track_id": track_id,
        "class_name": cls,
        "confidence": conf,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def _drive(pe, frames, t0=1000.0):
    """Feed a list of (detections, analysis) frames at 1 s intervals."""
    for i, (dets, analysis) in enumerate(frames):
        pe.ingest(dets, analysis, t0 + i)


# ── Calibration ──────────────────────────────────────────────────────────────

def test_calibration_roundtrip_and_yaw():
    cal = CameraCalibration(view_width_m=40.0, view_height_m=30.0)
    e, n = cal.to_world(0.75, 0.25)          # right of centre, above centre
    assert e == pytest.approx(10.0)
    assert n == pytest.approx(7.5)

    # Round-trip through the inverse.
    x, y = cal.to_image(e, n)
    assert x == pytest.approx(0.75, abs=1e-3)
    assert y == pytest.approx(0.25, abs=1e-3)

    # Yaw rotates headings: image E becomes S with 90 deg clockwise yaw.
    cal90 = CameraCalibration(yaw_deg=90.0)
    assert cal90.world_heading("E") == "S"
    assert cal90.world_heading("N") == "E"


def test_calibration_distance_matches_pythagoras():
    cal = CameraCalibration(view_width_m=30.0, view_height_m=22.5)
    d = cal.distance_m(0.5, 0.5, 0.5, 0.25)
    assert d == pytest.approx(22.5 * 0.25)


# ── Lane map ─────────────────────────────────────────────────────────────────

def test_lane_assignment_for_southbound_vehicle():
    m = IntersectionMap()
    cal = CameraCalibration()
    user = {"position": {"x": 0.50, "y": 0.20}, "heading": "S",
            "approach": "north", "category": "vehicle"}
    m.classify(user, cal)
    assert user["lane_id"] == "N_through"
    assert user["movement"] == "through"
    # North stop line is y=0.32; vehicle at y=0.20 is (0.32-0.20)*22.5 = 2.7 m upstream.
    assert user["distance_to_stop_line_m"] == pytest.approx(2.7, abs=0.05)
    assert user["in_crosswalk"] is False


def test_stop_line_distance_negative_past_the_line():
    m = IntersectionMap()
    cal = CameraCalibration()
    d = m.distance_to_stop_line("north", 0.5, 0.40, cal)
    assert d < 0          # past the line, inside the junction


def test_crosswalk_band_detected():
    m = IntersectionMap()
    cal = CameraCalibration()
    # Just north of the north stop line: within the 2.5 m crosswalk band.
    user = {"position": {"x": 0.5, "y": 0.32 - 0.05}, "approach": "north"}
    m.classify(user, cal)
    assert user["in_crosswalk"] is True


# ── Red-light running ────────────────────────────────────────────────────────

def _red_signal(approach):
    return "red"


def _green_signal(approach):
    return "green"


def test_red_light_running_detected_on_fresh_crossing():
    pe = PerceptionEngine("t")
    safety = SafetyAnalytics("t")
    t = time.time()
    # Vehicle approaching the north stop line, then crossing it on red.
    for i in range(6):
        y = 0.28 - i * 0.02        # moves southward past y=0.32? no: 0.28 -> 0.18 crosses nothing
        # Use positions crossing y=0.32 from above.
        y = 0.36 - i * 0.02
        pe.ingest(
            [det(50, "car", 0.48, y - 0.02, 0.52, y + 0.02)],
            [{"track_id": 50, "speed_mps": 8.0, "direction": "S"}],
            t + i,
        )
        safety.process(pe, approach_signal=_red_signal)
    events = safety.recent_events(kind="red_light_running")
    assert events, "expected a red-light event"
    assert events[0]["extra"]["signal_state"] == "red" \
        if "extra" in events[0] else True
    assert any(a["track_id"] == "50" for a in events[0]["actors"])


def test_no_red_light_event_when_signal_green():
    pe = PerceptionEngine("t")
    safety = SafetyAnalytics("t")
    t = time.time()
    for i in range(6):
        y = 0.36 - i * 0.02
        pe.ingest(
            [det(51, "car", 0.48, y - 0.02, 0.52, y + 0.02)],
            [{"track_id": 51, "speed_mps": 8.0, "direction": "S"}],
            t + i,
        )
        safety.process(pe, approach_signal=_green_signal)
    assert safety.recent_events(kind="red_light_running") == []


# ── Stopped vehicle ──────────────────────────────────────────────────────────

def test_stopped_vehicle_on_green_raises_after_threshold(monkeypatch):
    import backend.services.cityos.safety_analytics as sa_mod
    monkeypatch.setattr(sa_mod, "STOPPED_ON_GREEN_S", 2.0)
    pe = PerceptionEngine("t")
    safety = SafetyAnalytics("t")
    t = time.time()
    # Stationary car upstream of the north stop line while green.
    for i in range(8):
        pe.ingest(
            [det(60, "car", 0.48, 0.20, 0.52, 0.26)],
            [{"track_id": 60, "speed_mps": 0.0, "direction": "-"}],
            t + i,
        )
        safety.process(pe, approach_signal=_green_signal)
    events = safety.recent_events(kind="stopped_vehicle")
    assert events, "expected a stopped-vehicle event"
    assert "GREEN" in events[0]["message"] or "green" in events[0]["message"]


# ── PET conflict ─────────────────────────────────────────────────────────────

def test_pet_conflict_when_object_follows_too_closely():
    pe = PerceptionEngine("t")
    safety = SafetyAnalytics("t")
    t = time.time()
    # Object A passes through cell (4,4), then B enters it shortly after.
    frames = []
    for i in range(8):
        xa = 0.30 + i * 0.06
        frames.append(([det(70, "car", xa - 0.02, 0.49, xa + 0.02, 0.53)],
                       [{"track_id": 70, "speed_mps": 9.0, "direction": "E"}]))
    for i in range(8):
        xb = 0.30 + i * 0.06
        frames.append(([det(71, "car", xb - 0.02, 0.49, xb + 0.02, 0.53)],
                       [{"track_id": 71, "speed_mps": 12.0, "direction": "E"}]))
    # Interleave so B trails A closely through the same cells.
    merged = []
    for i in range(16):
        if i % 2 == 0:
            idx = i // 2
            merged.append((frames[idx][0], frames[idx][1]))
        else:
            idx = min(i // 2, 7)
            merged.append((frames[8 + idx][0], frames[8 + idx][1]))
    _drive(pe, merged, t0=time.time() - 20)
    safety.process(pe)
    # The trailing object should have entered recently-vacated cells.
    assert safety.recent_events(kind="pet_conflict"), \
        "expected at least one PET conflict"


# ── Hard braking ─────────────────────────────────────────────────────────────

def test_hard_braking_detected_on_rapid_deceleration():
    pe = PerceptionEngine("t")
    safety = SafetyAnalytics("t")
    t = time.time()
    speeds = [14.0, 13.5, 13.0, 8.0, 3.0]     # sharp drop mid-sequence
    for i, v in enumerate(speeds):
        pe.ingest(
            [det(80, "car", 0.3 + i * 0.02, 0.48, 0.34 + i * 0.02, 0.52)],
            [{"track_id": 80, "speed_mps": v, "direction": "E"}],
            t + i * 0.5,
        )
        safety.process(pe)
    events = safety.recent_events(kind="hard_braking")
    assert events, "expected a hard-braking event"
    assert events[0]["severity"] == "medium"


# ── Queue estimation ─────────────────────────────────────────────────────────

def test_queue_depth_length_and_growth():
    fa = TrafficFlowAnalyzer("t")
    users = [
        {"category": "vehicle", "is_vru": False, "speed_mps": 0.2,
         "approach": "west", "distance_to_stop_line_m": 12.0},
        {"category": "vehicle", "is_vru": False, "speed_mps": 0.1,
         "approach": "west", "distance_to_stop_line_m": 30.0},
        {"category": "vehicle", "is_vru": False, "speed_mps": 9.0,
         "approach": "west", "distance_to_stop_line_m": 15.0},   # moving: not queued
    ]
    q = fa.observe_queue(users)
    assert q["west"]["stopped_count"] == 2
    assert q["west"]["length_m"] == pytest.approx(30.0)
    status = fa.queue_status()
    assert status["west"]["length_m"] == pytest.approx(30.0)


# ── Pedestrian signal phases ─────────────────────────────────────────────────

def test_ped_states_walk_flashing_dont_walk():
    sig = SignalOptimizer("t")
    sig.force_phase("NS")                     # NS green
    states = sig.ped_states()
    assert states["NS"] == "walk"
    assert states["EW"] == "dont_walk"

    sig._enter("NS", "yellow")                # clearance interval
    states = sig.ped_states()
    assert states["NS"] == "flashing"
    assert states["EW"] == "dont_walk"


def test_approach_state_derived_from_phase():
    sig = SignalOptimizer("t")
    sig.force_phase("EW")
    assert sig.approach_state("east") == "green"
    assert sig.approach_state("west") == "green"
    assert sig.approach_state("north") == "red"
    assert sig.approach_state("south") == "red"


# ── Corridor continuity ──────────────────────────────────────────────────────

def test_corridor_match_within_travel_window():
    svc = CorridorService()
    svc.add_link(CorridorLink(
        link_id="L1", from_iid="A", to_iid="B",
        exit_approach="east", entry_approach="west",
        min_travel_s=1.0, max_travel_s=10.0,
    ))
    trip = {"track_id": "7", "category": "vehicle",
            "entry": "west", "exit": "east", "ended_at": time.time()}
    svc.register_exit("A", trip)
    arrival = {"track_id": "99", "category": "vehicle",
               "approach": "west"}
    match = None
    deadline = time.time() + 3.0
    while time.time() < deadline:             # wait out min_travel_s
        match = svc.register_entry("B", arrival)
        if match:
            break
        time.sleep(0.3)
    assert match is not None
    assert match["from_track_id"] == "7"
    assert match["to_track_id"] == "99"
    assert match["is_identification"] is False
    stats = svc.travel_time_stats()["L1"]
    assert stats["samples"] == 1


def test_corridor_rejects_out_of_window_arrival():
    svc = CorridorService()
    svc.add_link(CorridorLink(
        link_id="L2", from_iid="A", to_iid="B",
        exit_approach="east", entry_approach="west",
        min_travel_s=3600.0, max_travel_s=7200.0,   # impossible window
    ))
    svc.register_exit("A", {"track_id": "1", "category": "vehicle",
                            "exit": "east"})
    match = svc.register_entry("B", {"track_id": "2",
                                     "category": "vehicle",
                                     "approach": "west"})
    assert match is None


# ── Engine integration ───────────────────────────────────────────────────────

def test_engine_twin_includes_new_layers():
    engine = CityOSEngine()
    engine.bind_camera(1, "main_5th")
    t = time.time()
    engine.ingest(1, [det(5, "car", 0.45, 0.20, 0.55, 0.28)],
                  [{"track_id": 5, "speed_mps": 9.0, "direction": "S"}], t)
    twin = engine.twin(camera_id=1)
    obj = twin["objects"][0]
    assert "world" in obj and "lane_id" in obj
    assert "true_heading" in obj
    assert twin["ped_signal"] in ({"NS": "walk", "EW": "dont_walk"},
                                  {"NS": "dont_walk", "EW": "walk"},
                                  {"NS": "flashing", "EW": "dont_walk"},
                                  {"NS": "dont_walk", "EW": "flashing"})
    assert "queue_by_approach" in twin
    assert twin["sensor_health"]["status"] in ("ok", "degraded", "offline")
    assert twin["map"]["lanes"], "default lane geometry must exist"


def test_engine_calibration_update_changes_world_coords():
    engine = CityOSEngine()
    engine.set_calibration(3, {"view_width_m": 60.0, "view_height_m": 45.0})
    engine.ingest(3, [det(9, "car", 0.75, 0.25, 0.85, 0.35)],
                  [{"track_id": 9, "speed_mps": 5.0, "direction": "NE"}],
                  time.time())
    twin = engine.twin(camera_id=3)
    world = twin["objects"][0]["world"]
    assert world["east_m"] == pytest.approx(15.0, abs=0.5)
    assert world["north_m"] == pytest.approx(11.25, abs=0.5)


def test_engine_replay_returns_recorded_snapshot():
    engine = CityOSEngine()
    engine.bind_camera(4, "replay_test")
    inter = engine.get_intersection("replay_test")
    # Force a snapshot now instead of waiting for the 5 s cadence.
    inter._last_replay_at = 0.0
    engine.ingest(4, [det(2, "person", 0.5, 0.5, 0.55, 0.6)], [],
                  time.time())
    snap = inter.replay_at(seconds_ago=0)
    assert snap is not None
    assert snap["snapshot_age_error_s"] <= REPLAY_TOLERANCE_S
    assert "objects" in snap and "signal" in snap


REPLAY_TOLERANCE_S = 6.0


def test_privacy_purge_expires_old_trips():
    engine = CityOSEngine()
    inter = engine.get_intersection("privacy_t")
    old_trip = {"track_id": "1", "category": "vehicle",
                "entry": "west", "exit": "east",
                "ended_at": time.time() - 9999}
    fresh_trip = {"track_id": "2", "category": "vehicle",
                  "entry": "west", "exit": "east",
                  "ended_at": time.time()}
    inter.perception.completed_trips.extend([old_trip, fresh_trip])
    removed = inter.purge_expired_ids(retention_s=900.0)
    assert removed == 1
    remaining = list(inter.perception.completed_trips)
    assert len(remaining) == 1
    assert remaining[0]["track_id"] == "2"


def test_sensor_health_reports_offline_without_frames():
    engine = CityOSEngine()
    inter = engine.get_intersection("health_t")
    health = inter.sensor_health()
    assert health["status"] == "offline"