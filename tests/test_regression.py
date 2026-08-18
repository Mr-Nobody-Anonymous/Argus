"""
Regression tests for the Argus detection pipeline.

Every test here exists because a real bug shipped past the previous test suite.
The failure mode that motivated this file is *silent wrong output*: the pipeline
returned well-formed, plausible responses that were simply incorrect, while
/health reported green and no exception was ever raised.

Run:
    pytest tests/test_regression.py -v
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

cv2 = pytest.importorskip("cv2", reason="OpenCV required")

DEMO_CLIP = PROJECT_ROOT / "data" / "demo_clip.mp4"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def demo_frame() -> np.ndarray:
    """First frame of the demo clip - a real street scene with many people."""
    if not DEMO_CLIP.exists():
        pytest.skip(f"Fixture clip missing: {DEMO_CLIP}")
    cap = cv2.VideoCapture(str(DEMO_CLIP))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        pytest.skip("Could not decode demo clip")
    return frame


@pytest.fixture(scope="module")
def inference_engine():
    from backend.services.core_engine.inference_engine import get_inference_engine
    engine = get_inference_engine()
    if not engine.is_model_loaded():
        pytest.skip("YOLO weights unavailable")
    return engine


# ── Golden-frame detection ───────────────────────────────────────────────────

class TestGoldenFrame:
    """
    A known image must yield a known-good detection result.

    Guards against: model path regressions, preprocessing changes, and
    confidence-threshold drift silently zeroing the detector.
    """

    def test_detects_people_in_street_scene(self, inference_engine, demo_frame):
        detections = inference_engine.detect_objects(demo_frame)

        assert len(detections) >= 5, (
            f"Expected at least 5 detections in a crowded crosswalk scene, "
            f"got {len(detections)}. A near-zero count usually means the "
            f"confidence threshold ratcheted up or the model failed to load."
        )

        people = [d for d in detections if d.get("class_name") == "person"]
        assert len(people) >= 3, f"Expected multiple people, got {len(people)}"

    def test_detection_schema_is_stable(self, inference_engine, demo_frame):
        """Downstream consumers depend on these exact keys."""
        detections = inference_engine.detect_objects(demo_frame)
        assert detections, "No detections to validate schema against"

        for det in detections:
            assert {"bbox", "class_name", "confidence"} <= set(det)
            x1, y1, x2, y2 = det["bbox"]
            assert x2 > x1 and y2 > y1, f"Degenerate bbox: {det['bbox']}"
            assert 0.0 <= det["confidence"] <= 1.0

    def test_bboxes_are_within_frame(self, inference_engine, demo_frame):
        h, w = demo_frame.shape[:2]
        for det in inference_engine.detect_objects(demo_frame):
            x1, y1, x2, y2 = det["bbox"]
            assert -1 <= x1 < w and -1 <= y1 < h
            assert 0 < x2 <= w + 1 and 0 < y2 <= h + 1


# ── Tracker identity persistence ─────────────────────────────────────────────

class TestTrackerIdentity:
    """
    Regression guard for the Kalman filter defects.

    Two shipped bugs made the tracker mint a brand-new ID for every detection
    on every frame:
      1. statePre/statePost were (8,) instead of (8,1) -> predict() threw.
      2. The transition matrix coupled position to size, so a *stationary* box
         was predicted to jump (100,100) -> (150,180) and IoU never matched.

    Nothing raised. Detection still "worked". Only loitering (which needs dwell
    time) silently stopped firing.
    """

    def test_kalman_predicts_stationary_object_in_place(self):
        """A still object must not drift. This is the exact matrix bug."""
        from backend.services.core_engine.deep_tracker import get_deep_tracker

        tracker = get_deep_tracker()
        bbox = [100, 100, 150, 180]  # cx=125, cy=140, w=50, h=80
        kf = tracker._init_kalman(track_id=999, bbox=bbox)
        if kf is None:
            pytest.skip("Kalman filter unavailable")

        predicted = kf.predict()
        cx, cy, w, h = predicted[:4].flatten()

        assert abs(cx - 125) < 5, f"Stationary object drifted in x: {cx} != 125"
        assert abs(cy - 140) < 5, f"Stationary object drifted in y: {cy} != 140"
        assert abs(w - 50) < 5, f"Width mutated during predict: {w} != 50"
        assert abs(h - 80) < 5, f"Height mutated during predict: {h} != 80"

    def test_track_ids_persist_across_frames(self, demo_frame):
        """
        Feeding the same detections repeatedly must reuse IDs, not allocate new
        ones. Rising ID counts mean IoU matching is broken.
        """
        from backend.services.core_engine.deep_tracker import DeepTracker

        tracker = DeepTracker()
        if not tracker.enabled:
            pytest.skip("Deep tracker disabled")

        detections = [
            {"bbox": [100, 100, 150, 180], "class_name": "person", "confidence": 0.9, "class_id": 0},
            {"bbox": [300, 200, 360, 300], "class_name": "person", "confidence": 0.85, "class_id": 0},
        ]

        first = tracker.update([d.copy() for d in detections], demo_frame)
        first_ids = sorted(d["track_id"] for d in first)

        for _ in range(4):
            latest = tracker.update([d.copy() for d in detections], demo_frame)

        latest_ids = sorted(d["track_id"] for d in latest)
        assert latest_ids == first_ids, (
            f"Track IDs changed for stationary objects: {first_ids} -> {latest_ids}. "
            f"IoU matching is failing; loitering/dwell rules will never fire."
        )

    def test_one_track_cannot_claim_two_detections(self, demo_frame):
        """The greedy matcher must assign each track at most once."""
        from backend.services.core_engine.deep_tracker import DeepTracker

        tracker = DeepTracker()
        if not tracker.enabled:
            pytest.skip("Deep tracker disabled")

        detections = [
            {"bbox": [100, 100, 150, 180], "class_name": "person", "confidence": 0.9, "class_id": 0},
            {"bbox": [105, 105, 155, 185], "class_name": "person", "confidence": 0.88, "class_id": 0},
            {"bbox": [400, 300, 450, 380], "class_name": "person", "confidence": 0.8, "class_id": 0},
        ]
        tracker.update([d.copy() for d in detections], demo_frame)
        result = tracker.update([d.copy() for d in detections], demo_frame)

        ids = [d["track_id"] for d in result]
        assert len(ids) == len(set(ids)), f"Duplicate track IDs assigned: {ids}"


# ── Throttling must not fabricate emptiness ──────────────────────────────────

class TestThrottleSemantics:
    """
    Regression guard for the broker/agent starvation bug.

    The consortium broker budgeted 33 ms/frame (a 30 FPS GPU assumption) while
    CPU inference costs ~130 ms, pinning the detector at throttle 0.218. The
    agent then skipped 4 of every 5 frames and returned [] for them - which the
    analysis cache and WebSocket faithfully reported as "no objects present".
    """

    def test_skipped_frame_does_not_report_empty_scene(self, demo_frame):
        from backend.services.core_engine.yolo_detection_agent import YoloDetectionAgent

        agent = YoloDetectionAgent()
        if not agent.enabled:
            pytest.skip("YOLO agent disabled")

        real = agent.process_frame(demo_frame, camera_id=1)
        if not real:
            pytest.skip("No detections in fixture")

        # Force heavy throttling so the next calls are skipped.
        agent._current_throttle = 0.2
        skipped_results = [agent.process_frame(demo_frame, camera_id=1) for _ in range(3)]

        assert any(len(r) > 0 for r in skipped_results), (
            "Every throttled frame returned an empty list. A skipped frame means "
            "'not measured', not 'nothing there' - downstream consumers cannot "
            "distinguish the two and will report an empty scene."
        )

    def test_broker_does_not_starve_agent_on_cpu(self):
        """
        With a realistic CPU inference cost, the sole bidding agent must still
        receive a workable throttle rather than being pinned at the floor.
        """
        from backend.services.core_engine.consortium_broker import (
            ConsortiumBroker, AgentBid,
        )

        broker = ConsortiumBroker()
        if not broker.enabled:
            pytest.skip("Broker disabled")

        broker.submit_bid(AgentBid(
            agent_id="yolo_detection_agent",
            domain="detection",
            urgency=0.95,
            compute_cost=130.0,   # measured CPU cost, far above the 33 ms budget
            contextual_relevance=1.0,
            current_load=1.0,
        ))
        broker._last_sync_time = 0.0  # bypass the sync interval for the test

        allocations = broker.resolve_cycle()
        alloc = allocations.get("yolo_detection_agent")
        assert alloc is not None, "Broker returned no allocation for the only bidder"
        assert alloc.should_process, "Sole agent was told not to process at all"
        assert alloc.throttle_factor > 0.5, (
            f"Sole bidding agent throttled to {alloc.throttle_factor:.3f} purely "
            f"because CPU inference is slower than the nominal 33 ms budget."
        )

    def test_confidence_threshold_recovers_after_throttling(self):
        """
        The throttle branch used to raise the confidence threshold every cycle
        with no decay path, so the detector went permanently blind after any
        transient load.
        """
        from backend.services.core_engine.yolo_detection_agent import YoloDetectionAgent
        from backend.services.core_engine.consortium_broker import ResourceAllocation

        agent = YoloDetectionAgent()
        if not agent.enabled:
            pytest.skip("YOLO agent disabled")

        baseline = agent._gene_vector.yolo_conf_threshold

        throttled = ResourceAllocation(
            agent_id=agent.AGENT_ID, allocated_budget_ms=5.0,
            priority_boost=1.0, throttle_factor=0.2, should_process=True,
        )
        for _ in range(5):
            agent.apply_allocation(throttled)
        degraded = agent._gene_vector.yolo_conf_threshold
        assert degraded > baseline, "Throttling should raise the threshold"

        normal = ResourceAllocation(
            agent_id=agent.AGENT_ID, allocated_budget_ms=100.0,
            priority_boost=1.0, throttle_factor=1.0, should_process=True,
        )
        for _ in range(10):
            agent.apply_allocation(normal)

        assert agent._gene_vector.yolo_conf_threshold < degraded, (
            "Confidence threshold never decayed back after load subsided - "
            "the detector stays blind forever."
        )


# ── Stream ingestion ─────────────────────────────────────────────────────────

class TestStreamIngestion:
    """File sources must loop at EOF and be paced to their native frame rate."""

    def test_file_source_is_detected(self):
        from backend.services.management.stream_ingestion import StreamIngestion
        assert StreamIngestion is not None

        if not DEMO_CLIP.exists():
            pytest.skip("Demo clip missing")

        cap = cv2.VideoCapture(str(DEMO_CLIP))
        try:
            assert cap.isOpened(), "OpenCV cannot decode the demo clip"
            fps = cap.get(cv2.CAP_PROP_FPS)
            assert fps > 0, "Clip reports no frame rate; pacing would be impossible"
        finally:
            cap.release()

    def test_clip_rewinds_at_eof(self):
        """Seeking back to frame 0 must yield a decodable frame."""
        if not DEMO_CLIP.exists():
            pytest.skip("Demo clip missing")

        cap = cv2.VideoCapture(str(DEMO_CLIP))
        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(total - 1, 0))
            cap.read()
            cap.read()  # now past EOF
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
            assert ok and frame is not None, "Rewind after EOF failed"
        finally:
            cap.release()


# ── Configuration ────────────────────────────────────────────────────────────

class TestConfigContracts:
    """Config sections hold pydantic models, not dicts."""

    def test_section_to_dict_normalises_models(self):
        from backend.config.config import get_config, section_to_dict

        rules = get_config().rules
        assert rules, "No rules configured"
        for name, rule in rules.items():
            as_dict = section_to_dict(rule)
            assert isinstance(as_dict, dict), f"Rule {name} did not normalise"
            assert "enabled" in as_dict

    def test_env_interpolation(self, monkeypatch):
        from backend.config.config import _interpolate_env

        monkeypatch.setenv("ARGUS_TEST_SECRET", "s3cret")
        result = _interpolate_env({
            "a": "${ARGUS_TEST_SECRET}",
            "b": "${ARGUS_UNSET_VAR:-fallback}",
            "c": "literal",
        })
        assert result["a"] == "s3cret"
        assert result["b"] == "fallback"
        assert result["c"] == "literal"

    def test_unresolved_secrets_are_reported(self):
        from backend.config.config import find_unresolved_secrets

        unresolved = find_unresolved_secrets({"mqtt": {"password": "${DEFINITELY_UNSET_VAR}"}})
        assert unresolved, "Unresolved ${VAR} reference was not reported"

    def test_secrets_are_redacted(self):
        from backend.config.config import redact_secrets

        out = redact_secrets({"password": "hunter2", "api_key": "abc", "host": "localhost"})
        assert out["password"] == "***REDACTED***"
        assert out["api_key"] == "***REDACTED***"
        assert out["host"] == "localhost"


# ── Event store ──────────────────────────────────────────────────────────────

class TestEventStore:
    def test_query_events_returns_pair(self):
        """query_events returns (events, total) - not a bare list."""
        from backend.services.management.event_store import get_event_store

        result = get_event_store().query_events(limit=5)
        assert isinstance(result, tuple) and len(result) == 2
        events, total = result
        assert isinstance(events, list) and isinstance(total, int)

    def test_events_use_rule_type_column(self):
        from backend.services.management.event_store import get_event_store

        events, _ = get_event_store().query_events(limit=1)
        if not events:
            pytest.skip("No events stored")
        assert "rule_type" in events[0], (
            "Event schema changed; API consumers expect 'rule_type'."
        )


class TestPrimaryDetectorNeverStarved:
    """
    The swarm must never throttle the primary detector into replaying stale
    results. Everything downstream (tracking, zones, rules, events) derives
    from it, so starving it blinds the system rather than degrading it.

    Measured before the fix: throttle collapsed to the 0.10 floor and the swarm
    path reported 8.25 detections/frame vs 12.8 for the linear baseline - a 35%
    loss that looked like a throughput win.
    """

    def test_primary_agent_keeps_full_throttle_under_contention(self):
        from backend.services.core_engine.consortium_broker import ConsortiumBroker, AgentBid

        broker = ConsortiumBroker()
        if not broker.enabled:
            pytest.skip("Broker disabled")

        # Realistic CPU costs: all three agents bid, YOLO is the most expensive.
        for agent_id, cost in (
            ("yolo_agent", 130.0), ("face_agent", 8.5), ("lpr_agent", 12.0)
        ):
            broker.submit_bid(AgentBid(
                agent_id=agent_id, domain="d", urgency=0.9,
                compute_cost=cost, contextual_relevance=1.0, current_load=1.0,
            ))
        broker._last_sync_time = 0.0

        allocations = broker.resolve_cycle()
        primary = allocations.get("yolo_agent")
        assert primary is not None
        assert primary.throttle_factor >= 1.0, (
            f"Primary detector throttled to {primary.throttle_factor:.2f} under "
            f"contention - it will replay stale detections and the system goes blind."
        )

    def test_swarm_and_linear_agree_on_detection_count(self, demo_frame):
        """
        Both pipelines must see the same objects in the same frame. A faster
        pipeline that detects less is a regression, not an optimisation.
        """
        from backend.services.core_engine.inference_engine import get_inference_engine
        from backend.services.core_engine.yolo_detection_agent import YoloDetectionAgent

        engine_count = len(get_inference_engine().detect_objects(demo_frame))

        agent = YoloDetectionAgent()
        if not agent.enabled:
            pytest.skip("YOLO agent disabled")
        agent_count = len(agent.process_frame(demo_frame, camera_id=1))

        assert agent_count == engine_count, (
            f"Swarm path saw {agent_count} objects, linear path saw {engine_count}."
        )


class TestTrackIdentityAtRealFrameRates:
    """
    Identity must survive the gap between *processed* frames, not just between
    captured ones. CPU inference analyses ~1 frame/second while cameras run at
    15-30 fps, so association has to tolerate a full second of motion.

    Before the centre-distance fallback: 89 distinct IDs across 10 processed
    frames of a ~12-person scene. Every downstream feature keyed on identity
    (dwell time, entry events, cross-camera Re-ID) was corrupted by that churn.
    """

    def _count_ids(self, stride, processed_frames=10):
        import cv2
        from backend.services.core_engine.inference_engine import get_inference_engine
        from backend.services.core_engine.deep_tracker import get_deep_tracker

        engine = get_inference_engine()
        tracker = get_deep_tracker()
        tracker.tracks = {}
        tracker.kalman_filters = {}
        tracker.next_track_id = 1

        cap = cv2.VideoCapture(str(DEMO_CLIP))
        seen, processed, index = set(), 0, 0
        while processed < processed_frames:
            ok, frame = cap.read()
            if not ok:
                break
            if index % stride == 0:
                for det in tracker.update(engine.detect_objects(frame), frame):
                    if det.get("track_id") is not None:
                        seen.add(det["track_id"])
                processed += 1
            index += 1
        cap.release()
        return len(seen), processed

    def test_identity_survives_one_second_gaps(self):
        """A ~12-person clip must not mint dozens of IDs at the live rate."""
        n_ids, processed = self._count_ids(stride=15)
        assert processed >= 5, "Not enough frames decoded to judge"
        assert n_ids <= 30, (
            f"{n_ids} distinct track IDs across {processed} processed frames - "
            f"identity is churning, so dwell time and entry events are broken."
        )

    def test_subsampled_rate_close_to_consecutive(self):
        """Dropping to 1fps may cost some identity, but not an order of magnitude."""
        consecutive, _ = self._count_ids(stride=1)
        subsampled, _ = self._count_ids(stride=15)
        assert subsampled <= consecutive * 3, (
            f"Subsampled tracking produced {subsampled} IDs vs {consecutive} "
            f"consecutive - association is failing at realistic frame rates."
        )


class TestEventDeduplication:
    """
    A subject who stays put must produce one event, not one every dedup window.
    Measured before the fix: 209 events in 2 minutes from a single camera.
    """

    def test_ongoing_condition_produces_single_event(self):
        import datetime as _dt
        from backend.services.management.rules_engine import RulesEngine

        engine = RulesEngine()
        engine.dedup_window = _dt.timedelta(seconds=1)
        h = "cam1_zone1_intrusion_track_7"

        assert engine._is_duplicate_event(h) is False  # first sighting fires
        engine.recent_events[h] = _dt.datetime.now()

        # Continuous presence sampled across more than one window.
        for _ in range(6):
            time.sleep(0.4)
            assert engine._is_duplicate_event(h) is True, (
                "Ongoing condition re-fired - the dedup window is not sliding."
            )

    def test_event_rearms_after_absence(self):
        import datetime as _dt
        from backend.services.management.rules_engine import RulesEngine

        engine = RulesEngine()
        engine.dedup_window = _dt.timedelta(seconds=1)
        h = "cam1_zone1_intrusion_track_9"
        engine.recent_events[h] = _dt.datetime.now()

        time.sleep(1.3)  # subject leaves for longer than the window
        assert engine._is_duplicate_event(h) is False, (
            "Event never re-armed after the subject left - real re-entries "
            "would be silently dropped."
        )


class TestZoneAlertsPayloadShapes:
    """Zone checking used to assume `bbox` was always a list.

    The pipeline passes a list, but every serialised detection (WebSocket, REST)
    carries {"x1":..,"y1":..} and the class under "class" rather than
    "class_name". Feeding one of those back in raised KeyError: 0 and killed the
    zone check instead of degrading, and nothing covered the path.
    """

    @staticmethod
    def _manager():
        from backend.services.management.zone_alerts import ZoneAlerts
        za = ZoneAlerts()
        za.load_zones(1, [{
            "id": 1, "name": "vault", "type": "intrusion",
            "coordinates": "[[0,0],[100,0],[100,100],[0,100]]",
        }])
        return za

    def test_dict_shaped_bbox_does_not_raise(self):
        za = self._manager()
        det = [{"track_id": 8, "class": "person", "confidence": 0.9,
                "bbox": {"x1": 40, "y1": 40, "x2": 60, "y2": 60}}]
        za.check_zone_crossings(1, det)  # KeyError: 0 before the fix
        assert 8 in za.loitering_triggers, (
            "API-shaped bbox was not understood: the subject was not registered "
            "inside the zone."
        )

    def test_list_shaped_bbox_still_works(self):
        za = self._manager()
        det = [{"track_id": 7, "class_name": "person", "confidence": 0.9,
                "bbox": [40, 40, 60, 60]}]
        za.check_zone_crossings(1, det)
        assert 7 in za.loitering_triggers

    def test_malformed_bbox_degrades_instead_of_crashing(self):
        za = self._manager()
        for bad in (None, [1, 2], {}, "nonsense"):
            za.check_zone_crossings(
                1, [{"track_id": 1, "class_name": "person",
                     "confidence": 0.5, "bbox": bad}])

    def test_intrusion_actually_fires_after_dwell(self):
        za = self._manager()
        det = [{"track_id": 7, "class": "person", "confidence": 0.9,
                "bbox": {"x1": 40, "y1": 40, "x2": 60, "y2": 60}}]
        assert za.check_zone_crossings(1, det) == [], "fired before any dwell"
        za.loitering_triggers[7] = time.time() - 31
        events = za.check_zone_crossings(1, det)
        assert len(events) == 1, "loitering threshold never triggered an event"
        assert events[0].object_type == "person", (
            "class was read from the wrong key - events would be mislabelled"
        )

    def test_per_track_state_is_bounded(self):
        """A 24/7 feed mints new track ids forever; nothing evicted them."""
        za = self._manager()
        cap = za.MAX_TRACKED_IDS
        for tid in range(cap * 3):
            za._touch_track(tid)["last_center"] = (tid, tid)
            za.loitering_triggers[tid] = 1.0
        assert len(za.zone_triggers) <= cap, (
            f"zone_triggers grew to {len(za.zone_triggers)} - unbounded leak"
        )
        assert len(za.loitering_triggers) <= cap, (
            "loitering_triggers leaked while zone_triggers was capped"
        )

    def test_recently_seen_track_survives_eviction(self):
        za = self._manager()
        cap = za.MAX_TRACKED_IDS
        for tid in range(cap):
            za._touch_track(tid)["v"] = tid
        za._touch_track(0)["v"] = "refreshed"   # oldest, but seen again
        za._touch_track(10 ** 6)["v"] = "new"   # forces one eviction
        assert 0 in za.zone_triggers, "LRU refresh failed; an active track was dropped"
        assert 1 not in za.zone_triggers, "evicted the wrong entry"


class TestSnapshotDiskCeiling:
    """Time-based retention cannot bound disk usage inside its own window.

    At the measured event rate one camera writes roughly 4 GB of snapshots a
    day, so a 30-day policy only frees space after ~130 GB. The size cap is a
    second, independent bound that evicts oldest-first.
    """

    @staticmethod
    def _seed(tmp_dir, count=10, mb=1):
        import os
        paths = []
        for i in range(count):
            f = tmp_dir / f"cap_{i:02d}.jpg"
            f.write_bytes(b"x" * (mb * 1024 * 1024))
            age = time.time() - (count - i) * 86400   # oldest first
            os.utime(f, (age, age))
            paths.append(f)
        return paths

    @staticmethod
    def _dir():
        from backend.config.config import resolve_path, get_config
        d = resolve_path(get_config().system.snapshot_dir)
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.glob("cap_*.jpg"):
            stale.unlink()
        return d

    def _cleanup(self, d):
        for f in d.glob("cap_*.jpg"):
            f.unlink()

    def test_cap_evicts_oldest_until_under_budget(self):
        from backend.services.management.retention import enforce_snapshot_size_cap
        d = self._dir()
        try:
            self._seed(d, count=10, mb=1)
            removed = enforce_snapshot_size_cap(5)
            remaining = sorted(p.name for p in d.glob("cap_*.jpg"))
            total_mb = sum(p.stat().st_size for p in d.glob("cap_*.jpg")) / 1048576
            assert removed == 5, f"expected 5 evictions, got {removed}"
            assert total_mb <= 5, f"still {total_mb:.1f} MB over a 5 MB cap"
            assert remaining == [f"cap_{i:02d}.jpg" for i in range(5, 10)], (
                f"evicted the wrong files (must be oldest-first): {remaining}"
            )
        finally:
            self._cleanup(d)

    def test_cap_is_a_noop_when_under_budget(self):
        from backend.services.management.retention import enforce_snapshot_size_cap
        d = self._dir()
        try:
            self._seed(d, count=3, mb=1)
            assert enforce_snapshot_size_cap(100) == 0
            assert len(list(d.glob("cap_*.jpg"))) == 3, "deleted files while under cap"
        finally:
            self._cleanup(d)

    def test_cap_can_be_disabled(self):
        from backend.services.management.retention import enforce_snapshot_size_cap
        d = self._dir()
        try:
            self._seed(d, count=3, mb=1)
            assert enforce_snapshot_size_cap(0) == 0, "cap ran while disabled"
            assert len(list(d.glob("cap_*.jpg"))) == 3
        finally:
            self._cleanup(d)

    def test_retention_pass_reports_the_cap(self):
        from backend.services.management.retention import run_retention_once
        results = run_retention_once()
        assert "snapshots_over_cap" in results, (
            "run_retention_once() does not enforce the size cap"
        )


class TestDormantModulesStayHonest:
    """Six modules are imported by nothing and are documented as DORMANT.

    Two failure modes are worth catching: a dormant module quietly rotting until
    it no longer imports, and a module being wired up (or removed) without the
    documentation being updated.
    """

    DORMANT = [
        "backend.services.core_engine.yolo_tracker",
        "backend.services.core_engine.multistream_pipeline",
        "backend.services.core_engine.video_pipeline",
        "backend.services.core_engine.object_detection_tracker",
        "backend.services.core_engine.object_detection_tracker_refactored",
        "backend.services.management.model_optimizer",
    ]

    @pytest.mark.parametrize("module", DORMANT)
    def test_dormant_module_still_imports(self, module):
        import importlib
        importlib.import_module(module)

    @pytest.mark.parametrize("module", DORMANT)
    def test_dormant_module_is_labelled_in_docs(self, module):
        """If a module gets wired up, this fails until the docs are corrected."""
        import re
        name = module.rsplit(".", 1)[1] + ".py"
        doc = (PROJECT_ROOT / "FOLDER_STRUCTURE.md").read_text()
        row = next((l for l in doc.splitlines() if f"`{name}`" in l), None)
        assert row is not None, f"{name} is undocumented in FOLDER_STRUCTURE.md"

        src_root = PROJECT_ROOT / "backend"
        stem = name[:-3]
        importers = [
            f for f in src_root.rglob("*.py")
            if f.stem != stem
            and re.search(rf"(from|import)\s+\S*\b{stem}\b", f.read_text(errors="ignore"))
        ]
        if importers:
            assert "DORMANT" not in row, (
                f"{name} is now imported by "
                f"{[str(f.relative_to(PROJECT_ROOT)) for f in importers]} but is "
                f"still documented as DORMANT - update FOLDER_STRUCTURE.md."
            )
        else:
            assert "DORMANT" in row, (
                f"{name} is imported by nothing but is not labelled DORMANT in "
                f"FOLDER_STRUCTURE.md - readers will assume it is live code."
            )


class TestCameraLivenessIsNotPersisted:
    """Camera liveness is process state, not durable data.

    status/fps/last_frame_time were UPDATEd to SQLite about once per second per
    camera - 3.15 billion durable writes a year at 100 cameras - to store values
    that are meaningless once the process stops. It was also wrong: a clean
    shutdown reset them, but SIGKILL did not, so after a crash the API reported
    ('online', 14.73) for a camera that no longer existed.
    """

    def test_unknown_camera_reads_offline(self):
        from backend.services.management import camera_runtime as cr
        cr.clear_all()
        assert cr.get_status(4242) == {
            "status": "offline", "fps": 0.0, "last_frame_time": None
        }, "a camera nobody is ingesting must read back offline"

    def test_status_round_trips_in_memory(self):
        from backend.services.management import camera_runtime as cr
        cr.clear_all()
        cr.set_status(1, "online", 14.8)
        state = cr.get_status(1)
        assert state["status"] == "online"
        assert abs(state["fps"] - 14.8) < 0.001
        assert state["last_frame_time"] is not None
        cr.set_status(1, "offline")
        assert cr.get_status(1)["last_frame_time"] is None, (
            "an offline camera must not advertise a last-frame time"
        )
        cr.clear_all()

    def test_apply_to_overrides_whatever_the_row_says(self):
        """A stale row left by a crashed process must never win."""
        from backend.services.management import camera_runtime as cr
        cr.clear_all()
        stale_row = {"id": 2, "name": "cam", "status": "online", "fps": 14.7}
        assert cr.apply_to(stale_row)["status"] == "offline", (
            "a stale persisted 'online' survived a restart - this is the crash "
            "bug the in-memory store exists to prevent"
        )
        cr.set_status(2, "online", 9.0)
        assert cr.apply_to(stale_row)["fps"] == 9.0
        cr.clear_all()

    def test_liveness_columns_cannot_be_written_to_disk(self):
        """update_camera must refuse to persist runtime fields."""
        import inspect
        from backend.services.management import camera_manager
        src = inspect.getsource(camera_manager.CameraManager.update_camera)
        allowed = src.split("allowed_fields = ", 1)[1].split("]", 1)[0] + "]"
        for field in ("status", "fps", "last_frame_time"):
            assert f"'{field}'" not in allowed, (
                f"{field} is persistable again - a stale value could outlive "
                f"the process and be served as current. Found: {allowed}"
            )

    def test_update_status_performs_no_database_write(self):
        """The hot path must not touch the database at all."""
        from backend.services.management.camera_manager import CameraManager
        from backend.services.management import camera_runtime as cr

        class ExplodingDB:
            def execute(self, *a, **k):
                raise AssertionError(
                    "update_status wrote to the database - the per-second write "
                    "amplification has been reintroduced"
                )
            fetchone = fetchall = execute

        cr.clear_all()
        mgr = CameraManager.__new__(CameraManager)
        mgr.db = ExplodingDB()
        mgr.update_status(5, "online", 12.0)   # must not raise
        assert cr.get_status(5)["status"] == "online"
        cr.clear_all()

    def test_deleting_a_camera_drops_its_runtime_entry(self):
        """Otherwise state lingers and a recycled id inherits a dead camera's."""
        from backend.services.management.camera_manager import CameraManager
        from backend.services.management import camera_runtime as cr

        class NoopDB:
            def execute(self, *a, **k):
                return None

        cr.clear_all()
        mgr = CameraManager.__new__(CameraManager)
        mgr.db = NoopDB()
        mgr.update_status(11, "online", 20.0)
        assert 11 in cr.snapshot()
        mgr.delete_camera(11)
        assert 11 not in cr.snapshot(), "runtime state leaked after deletion"
        assert cr.get_status(11)["status"] == "offline"
        cr.clear_all()


class TestOneCommandLauncher:
    """The launcher is the first thing a new user touches.

    It must stay stdlib-only (it runs BEFORE dependencies exist), must not
    hard-code an OS, and must keep the documented commands available.
    """

    LAUNCHER = PROJECT_ROOT / "argus.py"

    def test_launcher_exists_and_compiles(self):
        import py_compile
        assert self.LAUNCHER.is_file(), "argus.py is missing"
        py_compile.compile(str(self.LAUNCHER), doraise=True)

    def test_launcher_imports_only_stdlib(self):
        """A third-party import here would crash before it could install it."""
        import ast
        import sys

        tree = ast.parse(self.LAUNCHER.read_text(encoding="utf-8"))
        roots = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for a in node.names:
                    roots.add(a.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.level == 0:
                    roots.add(node.module.split(".")[0])

        stdlib = set(getattr(sys, "stdlib_module_names", ()))
        offenders = sorted(
            r for r in roots
            if r and not r.startswith("_") and stdlib and r not in stdlib
        )
        assert not offenders, f"argus.py must be stdlib-only, found: {offenders}"

    def test_documented_subcommands_are_wired(self):
        src = self.LAUNCHER.read_text(encoding="utf-8")
        for cmd in ("start", "stop", "status", "doctor", "reset"):
            assert f'sub.add_parser("{cmd}"' in src, f"missing subcommand: {cmd}"

    def test_no_hardcoded_posix_only_paths(self):
        """Windows support breaks if the venv path is hard-coded to bin/."""
        src = self.LAUNCHER.read_text(encoding="utf-8")
        assert 'Scripts/python.exe' in src, "no Windows venv path"
        assert 'os.name == "nt"' in src, "no Windows detection"

    def test_windows_wrappers_use_crlf(self):
        """cmd.exe mishandles .bat files saved with bare LF endings."""
        for name in ("start.bat", "stop.bat"):
            p = PROJECT_ROOT / name
            assert p.is_file(), f"{name} is missing"
            data = p.read_bytes()
            lone_lf = data.replace(b"\r\n", b"").count(b"\n")
            assert lone_lf == 0, f"{name} has {lone_lf} bare LF line endings"

    def test_unix_wrappers_are_executable(self):
        import stat
        for name in ("start.command", "stop.command"):
            p = PROJECT_ROOT / name
            assert p.is_file(), f"{name} is missing"
            mode = p.stat().st_mode
            assert mode & stat.S_IXUSR, f"{name} is not executable"

    def test_launcher_state_is_gitignored(self):
        """A committed .env would leak the generated JWT secret."""
        ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
        for entry in (".venv/", ".argus/", ".env"):
            assert entry in ignored, f"{entry} must be gitignored"


class TestSinglePortDashboard:
    """The UI is served by FastAPI so one URL is all a user needs."""

    def _client(self):
        """Boot the real app with a built UI present, via TestClient."""
        import shutil
        from fastapi.testclient import TestClient

        dist = PROJECT_ROOT / "frontend" / "dist"
        created = False
        if not (dist / "index.html").is_file():
            dist.mkdir(parents=True, exist_ok=True)
            (dist / "index.html").write_text(
                "<!doctype html><title>argus</title>", encoding="utf-8"
            )
            created = True

        # main.py decides whether to mount the UI at import time.
        for mod in [m for m in list(sys.modules) if m.startswith("backend.api.main")]:
            del sys.modules[mod]
        from backend.api.main import app
        return TestClient(app), (dist / "index.html") if created else None

    def test_unknown_api_path_stays_json_404(self):
        """The SPA fallback must never swallow an unknown /api route.

        Exercised through real requests: a grep-based check silently passes
        when the guard is renamed or removed.
        """
        client, tmp_index = self._client()
        try:
            for path in ("/api/v1/nonexistent", "/api/bogus", "/openapi.json/x"):
                r = client.get(path)
                assert r.status_code == 404, f"{path} -> {r.status_code}, expected 404"
                assert "text/html" not in r.headers.get("content-type", ""), (
                    f"{path} returned the SPA shell instead of a JSON 404"
                )

            # ...while a browser route still gets the app shell.
            r = client.get("/events")
            assert r.status_code == 200, "SPA deep link must return index.html"
            assert "text/html" in r.headers.get("content-type", "")
        finally:
            if tmp_index is not None:
                tmp_index.unlink(missing_ok=True)

    def test_spa_fallback_handles_raised_404(self):
        """Starlette raises HTTPException(404); returning-only checks miss it."""
        src = (PROJECT_ROOT / "backend" / "api" / "main.py").read_text(encoding="utf-8")
        assert "StarletteHTTPException" in src, (
            "SPA fallback must catch Starlette's raised 404, not just a returned one"
        )

    def test_ui_mount_is_optional(self):
        """A source checkout with no build must still boot as an API."""
        src = (PROJECT_ROOT / "backend" / "api" / "main.py").read_text(encoding="utf-8")
        assert "_ui_is_built()" in src, "no guard for a missing frontend build"


class TestCanonicalPerceptionModel:
    """The perception layer must stay joinable, honest and dependency-free."""

    def test_importable_without_any_ml_dependency(self):
        """It must load on a box with no torch/cv2 so schema tests run anywhere.

        Implemented by blocking the imports outright rather than trusting that
        none crept in: a transitive import is easy to add and invisible in review.
        """
        import subprocess

        code = (
            "import sys\n"
            "BLOCKED={'torch','cv2','ultralytics','numpy','sklearn','scipy','onnxruntime','PIL'}\n"
            "class B:\n"
            "    def find_module(self,n,p=None):\n"
            "        return self if n.split('.')[0] in BLOCKED else None\n"
            "    def load_module(self,n):\n"
            "        raise ImportError('blocked '+n)\n"
            "sys.meta_path.insert(0,B())\n"
            "from backend.services.perception import Scene, scene_from_detections\n"
            "s=scene_from_detections(1,[{'class_name':'person','confidence':0.9,"
            "'bbox':[0,0,10,20]}])\n"
            "assert len(s.entities)==1\n"
            "print('ok')\n"
        )
        p = subprocess.run([sys.executable, "-c", code], cwd=str(PROJECT_ROOT),
                           capture_output=True, text=True, timeout=120)
        assert p.returncode == 0, f"perception layer needs an ML dep:\n{p.stderr[-800:]}"

    def test_real_detector_output_converts_losslessly(self):
        """Guards the adapter against the detector's actual dict shape."""
        from backend.services.perception import scene_from_detections

        dets = [
            {"class_id": 0, "class_name": "person", "confidence": 0.84,
             "bbox": [61, 1, 129, 147]},
            {"class_id": 2, "class_name": "car", "confidence": 0.71,
             "bbox": [300, 120, 520, 260]},
        ]
        scene = scene_from_detections(camera_id=2, detections=dets)
        assert len(scene.entities) == len(dets)
        assert len(scene.of_kind("person")) == 1
        assert len(scene.of_kind("vehicle")) == 1
        assert scene.entities[0].bbox.to_list() == [61.0, 1.0, 129.0, 147.0]

    def test_malformed_detection_is_skipped_not_fatal(self):
        """One bad box must never take down a frame."""
        from backend.services.perception import scene_from_detections

        dets = [
            {"class_name": "person", "confidence": 0.9, "bbox": [0, 0, 10, 20]},
            {"class_name": "ghost", "confidence": 0.5, "bbox": None},
            {"class_name": "broken", "confidence": 0.5, "bbox": [1, 2]},
            {"class_name": "zero_area", "confidence": 0.5, "bbox": [5, 5, 5, 5]},
            "not even a dict",
        ]
        scene = scene_from_detections(camera_id=1, detections=dets)
        assert len(scene.entities) == 1, "only the valid detection should survive"

    def test_better_sourced_claim_wins_regardless_of_confidence(self):
        """Raw confidence is not comparable across models.

        A specialised reader at 0.55 must beat a generic detector at 0.99, or
        every fusion decision becomes a coin toss between uncalibrated numbers.
        """
        from backend.services.perception import Entity, EntityKind, Source

        car = Entity(kind=EntityKind.VEHICLE.value, category="car",
                     bbox=[0, 0, 10, 10])
        car.set_attribute("plate", "GUESS99", 0.99, Source.DETECTOR.value)
        car.set_attribute("plate", "AA12BC", 0.55, Source.LPR.value)
        assert car.get("plate") == "AA12BC"

        # ...and a human overrides even the specialist.
        car.set_attribute("plate", "AA12BD", 0.30, Source.HUMAN.value)
        assert car.get("plate") == "AA12BD"

    def test_weaker_source_cannot_overwrite_stronger(self):
        from backend.services.perception import Entity, Source

        person = Entity(kind="person", category="person", bbox=[0, 0, 10, 10])
        person.set_attribute("identity", "Alice", 0.8, Source.FACE.value)
        person.set_attribute("identity", "Bob", 0.95, Source.APPEARANCE.value)
        assert person.get("identity") == "Alice", "appearance must not beat a face match"

    def test_unobserved_is_distinct_from_absent(self):
        """The single most dangerous confusion in a perception system."""
        from backend.services.perception import Entity

        person = Entity(kind="person", category="person", bbox=[0, 0, 10, 10])
        assert person.observed("carrying_bag") is False
        assert person.get("carrying_bag") is None

        person.set_attribute("carrying_bag", False, 0.9, "detector")
        assert person.observed("carrying_bag") is True
        assert person.get("carrying_bag") is False

    def test_confidence_is_clamped(self):
        from backend.services.perception import Attribute, Entity

        assert Attribute("x", 1, confidence=5.0).confidence == 1.0
        assert Attribute("x", 1, confidence=-2.0).confidence == 0.0
        assert Entity(confidence=99).confidence == 1.0

    def test_scene_round_trips_through_dict(self):
        """It has to survive a database or a queue without losing provenance."""
        from backend.services.perception import (
            Scene, Source, scene_from_detections, infer_spatial_relationships)

        scene = scene_from_detections(2, [
            {"class_name": "person", "confidence": 0.9,
             "bbox": [100, 100, 150, 300], "track_id": 7},
            {"class_name": "car", "confidence": 0.8, "bbox": [200, 180, 400, 320]},
        ])
        scene.entities[0].set_attribute("posture", "walking", 0.7, Source.POSE.value)
        scene.set_environment("lighting", "daylight", 0.8)
        infer_spatial_relationships(scene)

        restored = Scene.from_dict(scene.to_dict())
        assert restored.to_dict() == scene.to_dict()
        assert restored.by_track(7) is not None
        attr = restored.by_track(7).get_attribute("posture")
        assert attr.source == Source.POSE.value and attr.confidence == 0.7

    def test_plate_attaches_to_the_vehicle_not_a_loose_row(self):
        from backend.services.perception import attach_plate, scene_from_detections

        scene = scene_from_detections(1, [
            {"class_name": "car", "confidence": 0.8, "bbox": [200, 180, 400, 320],
             "track_id": 3},
            {"class_name": "person", "confidence": 0.9, "bbox": [10, 10, 40, 90]},
        ])
        hit = attach_plate(scene, "AA12BC", [280, 280, 340, 305], 0.7)
        assert hit is not None and hit.track_id == 3
        assert scene.by_track(3).get("plate") == "AA12BC"

    def test_plate_with_no_matching_vehicle_is_dropped(self):
        """Better to lose a reading than bind it to the wrong car."""
        from backend.services.perception import attach_plate, scene_from_detections

        scene = scene_from_detections(1, [
            {"class_name": "car", "confidence": 0.8, "bbox": [0, 0, 50, 50]},
        ])
        assert attach_plate(scene, "ZZ99ZZ", [900, 900, 950, 920]) is None

    def test_carrying_requires_containment_not_mere_overlap(self):
        from backend.services.perception import (
            infer_spatial_relationships, scene_from_detections)

        scene = scene_from_detections(1, [
            {"class_name": "person", "confidence": 0.9, "bbox": [100, 100, 200, 400]},
            {"class_name": "handbag", "confidence": 0.6, "bbox": [120, 200, 160, 260]},
            {"class_name": "handbag", "confidence": 0.6, "bbox": [600, 200, 640, 260]},
        ])
        rels = infer_spatial_relationships(scene)
        carrying = [r for r in rels if r.predicate == "carrying"]
        assert len(carrying) == 1, "only the contained bag is carried"

    def test_proximity_scales_with_subject_size(self):
        """A fixed pixel radius means different real distances at different depths."""
        from backend.services.perception import (
            infer_spatial_relationships, scene_from_detections)

        # Small (distant) person, vehicle 100 px away -> too far.
        far = scene_from_detections(1, [
            {"class_name": "person", "confidence": 0.9, "bbox": [0, 0, 10, 30]},
            {"class_name": "car", "confidence": 0.8, "bbox": [100, 0, 160, 40]},
        ])
        # Large (near) person, same 100 px -> within reach.
        near = scene_from_detections(1, [
            {"class_name": "person", "confidence": 0.9, "bbox": [0, 0, 60, 300]},
            {"class_name": "car", "confidence": 0.8, "bbox": [100, 0, 300, 300]},
        ])
        assert not [r for r in infer_spatial_relationships(far) if r.predicate == "near"]
        assert [r for r in infer_spatial_relationships(near) if r.predicate == "near"]

    def test_relationship_carries_its_evidence(self):
        """A claim with no evidence cannot be reviewed or disputed."""
        from backend.services.perception import (
            infer_spatial_relationships, scene_from_detections)

        scene = scene_from_detections(1, [
            {"class_name": "person", "confidence": 0.9, "bbox": [0, 0, 60, 300]},
            {"class_name": "car", "confidence": 0.8, "bbox": [100, 0, 300, 300]},
        ])
        for rel in infer_spatial_relationships(scene):
            assert rel.evidence, f"{rel.predicate} has no evidence"
            assert rel.confidence <= 0.9, "geometry alone must not be near-certain"

    def test_observation_needs_evidence_to_be_actionable(self):
        from backend.services.perception import Observation

        bare = Observation(kind="loitering", summary="person loitering",
                           confidence=0.95)
        assert not bare.is_actionable, "a confident claim with no evidence is not actionable"

        backed = Observation(kind="loitering", summary="person loitering",
                             confidence=0.6, evidence=["dwell 143s", "zone B"])
        assert backed.is_actionable

    def test_bbox_accepts_every_shape_already_in_the_codebase(self):
        from backend.services.perception import BBox

        expected = [1.0, 2.0, 3.0, 4.0]
        assert BBox.from_any([1, 2, 3, 4]).to_list() == expected
        assert BBox.from_any((1, 2, 3, 4)).to_list() == expected
        assert BBox.from_any({"x1": 1, "y1": 2, "x2": 3, "y2": 4}).to_list() == expected
        assert BBox.from_any({"x": 1, "y": 2, "w": 2, "h": 2}).to_list() == expected
        assert BBox.from_any("[1, 2, 3, 4]").to_list() == expected
        assert BBox.from_any(None) is None
        assert BBox.from_any("garbage") is None
        # Inverted corners are normalised, not left to produce negative area.
        assert BBox.from_any([3, 4, 1, 2]).area == 4.0
