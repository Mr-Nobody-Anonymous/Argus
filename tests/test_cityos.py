"""
CityOS intersection-intelligence tests.

Covers the geometry-only traffic layer end to end at unit level:
perception ingest (classification / position / trajectory), wrong-way
detection, near-miss TTC conflicts, VRU conflicts, flow analytics,
turning movements and the signal optimiser state machine.
"""
import time

import pytest

from backend.services.cityos.perception_engine import (
    PerceptionEngine, heading_from_delta, angle_of,
)
from backend.services.cityos.safety_analytics import SafetyAnalytics
from backend.services.cityos.traffic_flow import TrafficFlowAnalyzer
from backend.services.cityos.signal_optimizer import (
    SignalOptimizer, MAX_GREEN_S, YELLOW_S, ALL_RED_S,
)
from backend.services.cityos.engine import CityOSEngine


def det(track_id, cls, x1, y1, x2, y2, conf=0.9):
    return {
        "track_id": track_id,
        "class_name": cls,
        "confidence": conf,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


# ── Perception ───────────────────────────────────────────────────────────────

def test_heading_from_delta_compass():
    assert heading_from_delta(0, -1) == "N"
    assert heading_from_delta(1, 0) == "E"
    assert heading_from_delta(0, 1) == "S"
    assert heading_from_delta(-1, 0) == "W"


def test_angle_of_roundtrip():
    for h in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
        assert angle_of(h) is not None


def test_ingest_creates_tracked_user_with_trajectory():
    pe = PerceptionEngine("test")
    t = time.time()
    # A car moving right across the frame over several frames.
    for i in range(5):
        x = 0.1 + i * 0.05
        pe.ingest(
            [det(7, "car", x - 0.02, 0.48, x + 0.02, 0.52)],
            [{"track_id": 7, "speed_mps": 8.0, "direction": "E"}],
            t + i,
        )
    users = pe.active_users()
    assert len(users) == 1
    u = users[0]
    assert u["track_id"] == "7"
    assert u["category"] == "vehicle"
    assert u["heading"] == "E"
    assert u["speed_mps"] == pytest.approx(8.0)
    assert len(u["trajectory"]) >= 3
    stats = pe.stats()
    assert stats["active_objects"] == 1
    assert stats["total_observed"] == 1


def test_classifies_pedestrian_as_vru():
    pe = PerceptionEngine("test")
    pe.ingest([det(3, "person", 0.4, 0.4, 0.44, 0.5)], [], time.time())
    u = pe.active_users()[0]
    assert u["category"] == "pedestrian"
    assert u["is_vru"] is True


def test_stale_users_retire_into_completed_trips():
    pe = PerceptionEngine("test")
    t = time.time()
    pe.ingest([det(9, "car", 0.05, 0.45, 0.15, 0.55)], [], t)
    # Far-future frame retires the user.
    pe.ingest([], [], t + 60)
    trips = pe.recent_trips()
    assert len(trips) == 1
    assert trips[0]["track_id"] == "9"
    assert trips[0]["entry"] and trips[0]["exit"]


# ── Safety: wrong-way ────────────────────────────────────────────────────────

def _drive_wrong_way(safety, pe):
    """Feed a vehicle moving against the learned dominant flow."""
    t = time.time()
    # Frames 0..11 establish dominant eastbound flow on the west approach.
    for i in range(12):
        x = 0.1 + i * 0.03
        pe.ingest(
            [det(100, "car", x - 0.02, 0.30, x + 0.02, 0.34)],
            [{"track_id": 100, "speed_mps": 10.0, "direction": "E"}],
            t + i * 0.5,
        )
        safety.process(pe)
    # Now the offender travels westbound (against E) on the same approach.
    for i in range(6):
        x = 0.40 - i * 0.03
        pe.ingest(
            [det(200, "car", x - 0.02, 0.30, x + 0.02, 0.34)],
            [{"track_id": 200, "speed_mps": 10.0, "direction": "W"}],
            t + 20 + i * 0.5,
        )
        safety.process(pe)


def test_wrong_way_detected_against_dominant_flow():
    pe = PerceptionEngine("test")
    safety = SafetyAnalytics("test")
    _drive_wrong_way(safety, pe)
    events = safety.recent_events(kind="wrong_way")
    assert events, "expected a wrong-way event"
    assert events[0]["severity"] == "critical"
    assert any(a["track_id"] == "200" for a in events[0]["actors"])


def test_wrong_way_not_triggered_for_conforming_flow():
    pe = PerceptionEngine("test")
    safety = SafetyAnalytics("test")
    t = time.time()
    for i in range(12):
        x = 0.1 + i * 0.03
        pe.ingest(
            [det(300, "car", x - 0.02, 0.30, x + 0.02, 0.34)],
            [{"track_id": 300, "speed_mps": 10.0, "direction": "E"}],
            t + i * 0.5,
        )
        safety.process(pe)
    assert safety.recent_events(kind="wrong_way") == []


# ── Safety: near-miss / VRU ──────────────────────────────────────────────────

def test_near_miss_between_converging_vehicles():
    pe = PerceptionEngine("test")
    safety = SafetyAnalytics("test")
    t = time.time()
    # Two cars approaching the centre from opposite sides.
    frames = []
    for i in range(4):
        x_left = 0.25 + i * 0.08
        x_right = 0.75 - i * 0.08
        frames.append([
            det(1, "car", x_left - 0.02, 0.48, x_left + 0.02, 0.52),
            det(2, "car", x_right - 0.02, 0.48, x_right + 0.02, 0.52),
        ])
        pe.ingest(frames[-1], [
            {"track_id": 1, "speed_mps": 14.0, "direction": "E"},
            {"track_id": 2, "speed_mps": 14.0, "direction": "W"},
        ], t + i)
        safety.process(pe)
    events = safety.recent_events(kind="near_miss")
    assert events, "expected a near-miss event between converging vehicles"
    ids = {a["track_id"] for a in events[0]["actors"]}
    assert ids == {"1", "2"}


def test_vru_conflict_when_vehicle_close_to_pedestrian():
    pe = PerceptionEngine("test")
    safety = SafetyAnalytics("test")
    t = time.time()
    pe.ingest([
        det(10, "car", 0.50, 0.48, 0.58, 0.54),
        det(11, "person", 0.53, 0.50, 0.545, 0.56),
    ], [
        {"track_id": 10, "speed_mps": 8.0, "direction": "E"},
        {"track_id": 11, "speed_mps": 1.2, "direction": "S"},
    ], t)
    safety.process(pe)
    events = safety.recent_events(kind="vru_conflict")
    assert events, "expected a VRU conflict for co-located vehicle+pedestrian"
    cats = {a["category"] for a in events[0]["actors"]}
    assert "pedestrian" in cats


# ── Traffic flow ─────────────────────────────────────────────────────────────

def test_flow_volume_and_turning_matrix():
    fa = TrafficFlowAnalyzer("test")
    users = [
        {"category": "vehicle", "speed_mps": 8.0, "is_vru": False, "approach": "west"},
        {"category": "pedestrian", "speed_mps": 1.0, "is_vru": True, "approach": "north"},
    ]
    fa.observe(users)
    series = fa.volume_series(minutes=2)
    assert series[-1]["vehicle"] >= 1
    assert series[-1]["pedestrian"] >= 1
    fa.record_trip({"category": "vehicle", "entry": "west",
                    "exit": "north", "ended_at": time.time()})
    matrix = fa.turning_matrix()
    assert matrix["west"]["north"] == 1
    demand = fa.demand_by_approach(users)
    assert demand["west"] == pytest.approx(1.0)
    assert demand["north"] == pytest.approx(0.5)   # VRU half-weight


# ── Signal optimiser ─────────────────────────────────────────────────────────

def test_signal_phase_machine_progresses():
    sig = SignalOptimizer("test")
    s1 = sig.tick({})
    assert s1["state"] == "green"
    # Backdate the state timer past each threshold instead of sleeping -
    # deterministic and instant.
    sig._state_started -= MAX_GREEN_S + 0.1
    s2 = sig.tick({})          # green -> yellow
    assert s2["state"] == "yellow"
    sig._state_started -= YELLOW_S + 0.1
    s3 = sig.tick({})          # yellow -> all_red
    assert s3["state"] == "all_red"
    sig._state_started -= ALL_RED_S + 0.1
    s4 = sig.tick({})          # all_red -> next phase green
    assert s4["state"] == "green"
    assert s4["phase"] != s1["phase"]


def test_signal_recommendation_prefers_heavier_demand():
    sig = SignalOptimizer("test")
    rec = sig.recommend({"north": 8, "south": 4, "east": 1, "west": 1})
    assert rec["action"] in ("extend", "terminate_early")
    assert rec["ns_share"] > 0.5
    empty = sig.recommend({})
    assert empty["action"] == "hold"


def test_manual_mode_refuses_recommendations_but_forces_work():
    sig = SignalOptimizer("test")
    sig.set_mode("manual")
    cmd = sig.apply_recommendation({"action": "terminate_early"})
    assert cmd["applied"] is False
    status = sig.force_phase("EW")
    assert status["phase"] == "EW" and status["state"] == "green"
    with pytest.raises(ValueError):
        sig.force_phase("DIAGONAL")


# ── Engine orchestration ─────────────────────────────────────────────────────

def test_engine_binds_cameras_and_builds_twin():
    engine = CityOSEngine()
    engine.bind_camera(1, "main_and_5th")
    t = time.time()
    engine.ingest(1, [det(5, "car", 0.2, 0.4, 0.3, 0.5)],
                  [{"track_id": 5, "speed_mps": 9.0, "direction": "E"}], t)
    twin = engine.twin(camera_id=1)
    assert twin["intersection_id"] == "main_and_5th"
    assert twin["objects"], "twin should contain the ingested object"
    assert twin["signal"]["phase"] in ("NS", "EW")
    assert twin["edge_node"]["frames_ingested"] >= 1
    summaries = engine.summaries()
    assert summaries[0]["intersection_id"] == "main_and_5th"
    assert engine.status()["total_intersections"] == 1


def test_engine_unbound_camera_gets_own_intersection():
    engine = CityOSEngine()
    engine.ingest(42, [det(1, "person", 0.5, 0.5, 0.55, 0.6)], [], time.time())
    twin = engine.twin(camera_id=42)
    assert twin["intersection_id"] == "camera_42"