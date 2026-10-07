"""
CityOS sensor-layer tests.

Covers the layers closest to the hardware and the operational edges:

  - sensor abstraction: poses, 4D point-cloud clustering, per-sensor health
  - sensor fusion: multi-source world model with gated association
  - NTCIP controller client: polling, acks, validation, safe fallback
  - signal-performance analytics: cycles, splits, detector calls,
    commanded-vs-observed reconciliation
  - bicycle analytics: wrong-way cycling, conflicts, volumes
  - incident detection: collisions, junction blocking, prohibited-area
    pedestrians, abnormal congestion
  - engine integration: LiDAR ingest path, replay export/reconstruct,
    privacy-governance reporting
"""
import time

import pytest

from backend.services.cityos.bike_analytics import BikeAnalytics
from backend.services.cityos.engine import CityOSEngine
from backend.services.cityos.incident_detection import IncidentDetector
from backend.services.cityos.ntcip import CommandRejected, NTCIPControllerClient
from backend.services.cityos.sensor_abstraction import (
    LidarPoint, SensorPose, SensorRegistry, cluster_point_cloud,
)
from backend.services.cityos.sensor_fusion import Observation, SensorFusion
from backend.services.cityos.signal_optimizer import SignalOptimizer


# ── Sensor pose & clustering ─────────────────────────────────────────────────

def test_sensor_pose_yaw_transform():
    # Level sensor at origin facing north (yaw=0): forward = north.
    pose = SensorPose(east_m=0, north_m=0, up_m=5, yaw_deg=0, pitch_deg=0)
    e, n, u = pose.to_world(10, 0, 0)
    assert n == pytest.approx(10)
    assert e == pytest.approx(0)

    # Pitch tilts the forward vector upward but keeps its ground projection
    # (mount height 5 m + rise 10*sin(30) = 10 m total).
    pitched = SensorPose(pitch_deg=-30)
    _, n2, u2 = pitched.to_world(10, 0, 0)
    assert n2 == pytest.approx(10 * (3 ** 0.5) / 2)
    assert u2 == pytest.approx(10.0)

    # Yaw 90 deg clockwise: forward now points east.
    pose90 = SensorPose(yaw_deg=90, pitch_deg=0)
    e, n, _ = pose90.to_world(10, 0, 0)
    assert e == pytest.approx(10)
    assert n == pytest.approx(0)


def test_cluster_point_cloud_world_coords_and_speed():
    pose = SensorPose(east_m=0, north_m=-20, up_m=5, yaw_deg=0, pitch_deg=0)
    # A blob of returns ~10 m ahead of the sensor moving toward it.
    points = [
        LidarPoint(x=10 + dx * 0.1, y=dy * 0.1, z=1.0,
                   radial_velocity=-8.0, intensity=50)
        for dx in range(-3, 4) for dy in range(-3, 4)
    ]
    obs = cluster_point_cloud(points, pose)
    assert len(obs) == 1
    o = obs[0]
    assert o["n_points"] == 49
    assert o["east_m"] == pytest.approx(0, abs=0.5)
    assert o["north_m"] == pytest.approx(-10, abs=0.5)
    assert o["speed_mps"] == pytest.approx(8.0)   # |radial velocity|


def test_cluster_ignores_ground_and_noise():
    pose = SensorPose()
    points = [LidarPoint(x=5, y=5, z=0.05, radial_velocity=0.0)] * 30  # road
    points += [LidarPoint(x=8, y=8, z=1.0, radial_velocity=0.0)]       # lone pt
    assert cluster_point_cloud(points, pose) == []


# ── Sensor health ────────────────────────────────────────────────────────────

def test_sensor_health_flags_packet_loss_and_drift():
    reg = SensorRegistry()
    reg.register("lidar_1", "lidar")
    now = time.time()
    reg.record_frame(type("F", (), {
        "sensor_id": "lidar_1", "sensor_type": "lidar",
        "timestamp": now - 1.0, "points": [], "detections": [],
        "confidence": 1.0, "sequence": 1,
        "expected_packets": 100, "received_packets": 90,
    })())
    h = reg.get("lidar_1").health()
    assert h["packet_loss_pct"] == pytest.approx(10.0)
    assert any("packet loss" in p for p in h["problems"])
    assert h["status"] == "degraded"


def _frame(sensor_id, seq=1):
    now = time.time()
    return type("F", (), {
        "sensor_id": sensor_id, "sensor_type": "lidar",
        "timestamp": now, "points": [], "detections": [],
        "confidence": 1.0, "sequence": seq,
        "expected_packets": 0, "received_packets": 0,
    })()


def test_registry_coverage_reports_reduction():
    reg = SensorRegistry()
    reg.register("a", "lidar")
    reg.register("b", "camera")
    for i in range(20):                     # a is healthy (>= 0.3 fps)
        reg.record_frame(_frame("a", seq=i + 1))
    reg.get("b").connected = False          # one of two sensors down
    summary = reg.health_summary()
    assert summary["coverage_pct"] == pytest.approx(50.0)
    assert summary["status"] == "degraded"
    assert "coverage reduced" in summary["note"]


# ── Sensor fusion ────────────────────────────────────────────────────────────

def test_fusion_associates_camera_and_lidar_into_one_track():
    f = SensorFusion()
    t = time.time()
    cam = Observation(source="camera", east_m=5.0, north_m=5.0,
                      speed_mps=9.0, confidence=0.9, timestamp=t,
                      track_key="42")
    f.update([cam])
    lidar = Observation(source="lidar", east_m=5.6, north_m=5.2,
                        speed_mps=8.6, confidence=0.85, timestamp=t + 0.2)
    tracks = f.update([lidar])
    assert len(tracks) == 1
    tr = tracks[0]
    assert tr.id == "42"                    # camera anchors identity
    assert tr.sources == {"camera", "lidar"}
    # Blended position sits between the two measurements.
    assert 5.0 < tr.east_m < 5.6


def test_fusion_gates_out_far_observations():
    f = SensorFusion()
    t = time.time()
    f.update([Observation(source="camera", east_m=0, north_m=0,
                          timestamp=t, track_key="1")])
    tracks = f.update([Observation(source="lidar", east_m=40, north_m=40,
                                   timestamp=t)])
    assert len(tracks) == 2                 # no theft across a 55 m gap


def test_fusion_tracks_expire_quickly():
    f = SensorFusion()
    t = time.time() - 60                    # an hour-old observation
    f.update([Observation(source="camera", east_m=0, north_m=0,
                          timestamp=t, track_key="x")])
    # Next update happens much later -> stale track expired.
    tracks = f.update([Observation(source="camera", east_m=0, north_m=0,
                                   timestamp=time.time(), track_key="y")])
    assert {t.id for t in tracks} == {"y"}


# ── NTCIP controller client ──────────────────────────────────────────────────

def test_ntcip_simulation_poll_and_ack():
    c = NTCIPControllerClient("t", mode="simulation")
    r = c.poll("NS", "green")
    assert r["ok"] and r["observed_phase"] == "NS"
    ack = c.send_command("force_phase", phase="EW")
    assert ack["acked"] is True
    time.sleep(0.6)                          # simulated latency
    r = c.poll("NS", "green")
    assert r["observed_phase"] == "EW"       # controller applied the command
    st = c.status()
    assert st["mode"] == "simulation"
    assert "no physical device" in st["note"]


def test_ntcip_rejects_unsafe_cross_phase_green():
    c = NTCIPControllerClient("t")
    c.poll("NS", "green")
    with pytest.raises(CommandRejected):
        c.send_command("force_state", phase="EW", state="green")


def test_ntcip_safe_fallback_after_repeated_failures():
    c = NTCIPControllerClient("t")
    c.set_fault("poll_timeout")
    for _ in range(5):
        c.poll("NS", "green")
    assert c.status()["in_safe_fallback"] is True
    with pytest.raises(CommandRejected):
        c.send_command("hold")
    c.clear_fallback()
    assert c.status()["in_safe_fallback"] is False
    c.set_fault(None)
    assert c.poll("NS", "green")["ok"] is True


def test_ntcip_counts_disagreement_between_commanded_and_observed():
    c = NTCIPControllerClient("t")
    c.poll("NS", "green")
    c._sim_phase, c._sim_state = "EW", "green"     # controller drifts
    c._sim_apply_at = time.time() - 1
    c.poll("NS", "green")
    assert c.status()["commanded_vs_observed_disagreements"] >= 1


# ── Signal performance analytics ─────────────────────────────────────────────

def test_signal_performance_records_cycles_splits_and_calls():
    sig = SignalOptimizer("t")
    sig.force_phase("NS")
    sig._enter("NS", "yellow")
    sig._enter("NS", "all_red")
    sig._enter("EW", "green")
    perf = sig.performance()
    assert perf["intervals_recorded"] >= 3
    assert perf["avg_green_s"]["NS"] is not None or \
        perf["avg_yellow_s"]["NS"] is not None
    assert perf["splits_pct"] is None or \
        set(perf["splits_pct"]) == {"NS_pct", "EW_pct"}

    sig.record_detector_call("north")
    sig.record_detector_call("north")
    assert sig.performance()["detector_calls"]["north"] == 2


def test_signal_set_observed_counts_disagreements():
    sig = SignalOptimizer("t")
    sig.force_phase("NS")
    sig.set_observed("NS", "green")
    assert sig.performance()["observed_disagreements"] == 0
    sig.set_observed("EW", "green")            # controller disagrees
    assert sig.performance()["observed_disagreements"] == 1
    assert sig.performance()["observed_phase"] == "EW"


# ── Bicycle analytics ────────────────────────────────────────────────────────

def _cyclist(tid, x, y, speed, heading, legal=None):
    return {
        "track_id": tid, "category": "cyclist", "is_vru": True,
        "position": {"x": x, "y": y}, "speed_mps": speed,
        "speed_kmh": speed * 3.6, "heading": heading,
        "legal_heading": legal, "lane_id": "N_bike",
        "approach": "north", "distance_to_stop_line_m": 10.0,
    }


def test_wrong_way_cycling_detected_after_streak():
    b = BikeAnalytics("t")
    for _ in range(3):
        b.process([_cyclist(7, 0.5, 0.2, 3.0, "N", legal="S")])
    events = b.recent_events(kind="wrong_way_cycling")
    assert events and events[0]["actors"][0]["track_id"] == "7"


def test_bike_vehicle_conflict_on_proximity():
    b = BikeAnalytics("t")
    vehicle = {
        "track_id": 8, "category": "vehicle", "position": {"x": 0.51,
                                                           "y": 0.21},
        "speed_mps": 5.0, "heading": "S",
    }
    b.process([_cyclist(7, 0.5, 0.2, 2.0, "S", legal="S"), vehicle])
    assert b.recent_events(kind="bike_vehicle_conflict")


def test_bike_volume_bucketing_and_queue():
    b = BikeAnalytics("t")
    b.process([_cyclist(1, 0.5, 0.2, 4.0, "S", legal="S"),
               _cyclist(2, 0.5, 0.25, 0.2, "-", legal="S")])
    series = b.volume_series(minutes=5)
    assert series[-1]["cyclists"] == 2
    q = b.queue_status([_cyclist(2, 0.5, 0.25, 0.2, "-", legal="S")])
    assert q.get("north") == 1


# ── Incident detection ───────────────────────────────────────────────────────

def _vehicle(tid, x, y, speed, dist):
    return {
        "track_id": tid, "category": "vehicle", "is_vru": False,
        "position": {"x": x, "y": y}, "speed_mps": speed,
        "heading": "S", "distance_to_stop_line_m": dist,
        "in_crosswalk": False, "approach": "north",
    }


def test_collision_detected_for_contact_range_stationary_pair():
    inc = IncidentDetector("t")
    inc.process([
        _vehicle(1, 0.50, 0.50, 0.0, -2.0),
        _vehicle(2, 0.51, 0.51, 0.0, -2.0),
    ], {})
    kinds = [i["type"] for i in inc.recent_incidents()]
    assert "collision" in kinds


def test_blocked_intersection_after_sustained_dwell(monkeypatch):
    import backend.services.cityos.incident_detection as idet
    monkeypatch.setattr(idet, "BLOCKED_MIN_S", 0.0)
    inc = IncidentDetector("t")
    inc.process([_vehicle(3, 0.5, 0.5, 0.2, -3.0)], {})
    assert any(i["type"] == "blocked_intersection"
               for i in inc.recent_incidents())


def test_prohibited_area_pedestrian_flagged():
    inc = IncidentDetector("t")
    ped = {
        "track_id": 9, "category": "pedestrian", "is_vru": True,
        "position": {"x": 0.5, "y": 0.5}, "speed_mps": 1.2,
        "heading": "E", "distance_to_stop_line_m": -4.0,
        "in_crosswalk": False,
    }
    inc.process([ped], {})
    assert any(i["type"] == "prohibited_area_pedestrian"
               for i in inc.recent_incidents())


def test_abnormal_congestion_requires_sustained_abnormality(monkeypatch):
    import backend.services.cityos.incident_detection as idet
    monkeypatch.setattr(idet, "CONGESTION_SUSTAIN_S", 0.0)
    inc = IncidentDetector("t")
    queues = {"west": {"length_m": 80.0, "growth_mpm": 20.0}}
    inc.process([], queues)
    inc.process([], queues)      # two samples span the (zeroed) window
    assert any(i["type"] == "abnormal_congestion"
               for i in inc.recent_incidents())


# ── Engine integration ───────────────────────────────────────────────────────

def det(track_id, cls, x1, y1, x2, y2, conf=0.9):
    return {
        "track_id": track_id, "class_name": cls, "confidence": conf,
        "bbox": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
    }


def test_engine_lidar_ingest_creates_sensor_health_and_clusters():
    engine = CityOSEngine()
    inter = engine.get_intersection("lidar_t")
    inter.register_sensor("lidar_1", "lidar",
                          pose={"east_m": 0, "north_m": -20, "up_m": 5,
                                "yaw_deg": 0})
    points = [{"x": 10 + dx * 0.1, "y": dy * 0.1, "z": 1.0,
               "radial_velocity": -6.0, "intensity": 40}
              for dx in range(-3, 4) for dy in range(-3, 4)]
    result = inter.ingest_lidar("lidar_1", points, sequence=1)
    assert result["clusters"] == 1
    assert result["observations"][0]["speed_mps"] == pytest.approx(6.0)

    health = inter.sensor_health()["sensor_registry"]
    assert health["sensors"][0]["sensor_id"] == "lidar_1"
    assert health["sensors"][0]["frames_total"] == 1


def test_engine_twin_includes_new_layers():
    engine = CityOSEngine()
    engine.bind_camera(11, "full_t")
    t = time.time()
    engine.ingest(11, [det(5, "car", 0.45, 0.20, 0.55, 0.28)],
                  [{"track_id": 5, "speed_mps": 9.0, "direction": "S"}], t)
    twin = engine.twin(camera_id=11)
    assert "fused_objects" in twin
    assert "bikes" in twin and "incidents" in twin
    assert twin["controller"]["mode"] == "simulation"
    assert "signal_performance" in twin
    assert twin["signal_performance"]["intervals_recorded"] >= 0
    assert twin["sensor_health"]["fusion_stats"]["observations_processed"] > 0


def test_engine_camera_and_lidar_fuse_into_one_object():
    engine = CityOSEngine()
    inter = engine.get_intersection("fuse_t")
    inter.register_sensor("lidar_1", "lidar",
                          pose={"east_m": 0, "north_m": -20, "up_m": 5,
                                "yaw_deg": 0})
    # Camera sees a car near world (0, 5): normalised centre with default
    # calibration (30 m wide, 22.5 m tall) puts it ~5 m north of centre.
    inter._last_analysis_at = 0.0
    inter.ingest(
        [det(77, "car", 0.48, 0.28, 0.52, 0.34)],
        [{"track_id": 77, "speed_mps": 8.0, "direction": "S"}],
        time.time(),
    )
    # LiDAR sees a cluster within the association gate of the same spot.
    points = [{"x": 24 + dx * 0.1, "y": dy * 0.1, "z": 1.0,
               "radial_velocity": -7.5, "intensity": 40}
              for dx in range(-3, 4) for dy in range(-3, 4)]
    inter.ingest_lidar("lidar_1", points, sequence=1)
    inter._last_analysis_at = 0.0
    inter.ingest([det(77, "car", 0.48, 0.28, 0.52, 0.34)],
                 [{"track_id": 77, "speed_mps": 8.0, "direction": "S"}],
                 time.time())
    fused = inter.fusion.snapshot()
    matching = [f for f in fused if f["fused_id"] == "77"]
    assert matching, "camera track should exist in the world model"
    assert "lidar" in matching[0]["sources"], \
        "LiDAR observation should have been associated with the camera track"


def test_engine_replay_export_and_reconstruct_roundtrip():
    engine = CityOSEngine()
    engine.bind_camera(12, "replay_export_t")
    inter = engine.get_intersection("replay_export_t")
    inter._last_replay_at = 0.0
    engine.ingest(12, [det(3, "person", 0.5, 0.5, 0.55, 0.6)], [],
                  time.time())
    recording = inter.export_replay()
    assert recording["format_version"] == 1
    assert recording["snapshots"]

    view = CityOSEngine.reconstruct(recording)
    assert view["is_reconstruction"] is True
    assert view["snapshot_count"] == len(recording["snapshots"])
    assert view["window"]["start"] <= view["window"]["end"]

    empty = CityOSEngine.reconstruct({"snapshots": []})
    assert "error" in empty


def test_engine_governance_report_and_audit():
    engine = CityOSEngine()
    inter = engine.get_intersection("gov_t")
    inter.audit_governance("test_action", detail="x")
    report = inter.governance_report()
    assert report["data_classes"]["imagery_or_biometrics"].startswith("never")
    assert report["audit_log_entries"] == 1
    fleet = engine.governance()
    assert fleet["retention_policies"]["gov_t"]["completed_trips_s"] == 900.0


def test_calibration_geo_origin_transform():
    from backend.services.cityos.calibration import CameraCalibration
    cal = CameraCalibration(
        geo_origin={"lat": -1.286389, "lon": 36.817223})
    lat, lon = cal.to_geo(111.32, 0.0)
    assert lon == pytest.approx(36.818223, abs=1e-5)
    assert lat == pytest.approx(-1.286389, abs=1e-6)
    assert cal.to_dict()["geo_origin"]["lat"] == pytest.approx(-1.286389)
    # Without an anchor the transform declines to guess.
    assert CameraCalibration().to_geo(1, 1) is None