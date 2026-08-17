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
