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

import json
import os
import sys
import time
import tomllib
from pathlib import Path

import yaml

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


def _snapshot_dir_config():
    from backend.config.config import get_config
    return get_config()


# Captured once so _cleanup can restore whatever the real configuration was.
_ORIGINAL_SNAPSHOT_DIR = None


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
        """An isolated snapshot directory for this test.

        enforce_snapshot_size_cap() evicts every *.jpg under the configured
        snapshot dir, not just this test's cap_*.jpg fixtures. Pointing the
        test at the shared data/snapshots meant any real snapshot written by
        the running pipeline - or by another test - was counted in the eviction
        total and silently changed the result. The test failed only when the
        directory happened to be non-empty, which is the worst kind of flake:
        green on CI, red on a developer machine that has actually run Argus.
        """
        import tempfile
        from pathlib import Path
        global _ORIGINAL_SNAPSHOT_DIR
        cfg = _snapshot_dir_config()
        if _ORIGINAL_SNAPSHOT_DIR is None:
            _ORIGINAL_SNAPSHOT_DIR = cfg.system.snapshot_dir
        d = Path(tempfile.mkdtemp(prefix="argus_snapcap_"))
        cfg.system.snapshot_dir = str(d)
        return d

    def _cleanup(self, d):
        import shutil
        if _ORIGINAL_SNAPSHOT_DIR is not None:
            _snapshot_dir_config().system.snapshot_dir = _ORIGINAL_SNAPSHOT_DIR
        shutil.rmtree(d, ignore_errors=True)

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

    @staticmethod
    def _purge_main():
        """Drop the UI-mounted app so later tests re-import the real one.

        Reloading main.py with a built UI present swaps `GET /` for a static
        mount. Leaving that module in sys.modules leaks a different route
        table into every subsequent test - which is why the README route
        count passed alone and failed in a full run.
        """
        for mod in [m for m in list(sys.modules) if m.startswith("backend.api.main")]:
            del sys.modules[mod]

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
            self._purge_main()

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


class TestTemporalIntelligence:
    """Tracks must accumulate honestly and bound their own memory."""

    def _walk(self, pipeline, track_id=1, frames=20, dx=10, t0=None, camera=1):
        import time as _t
        t0 = t0 if t0 is not None else _t.time()
        for i in range(frames):
            pipeline.process(camera, [{
                "class_name": "person", "confidence": 0.9,
                "bbox": [100 + i * dx, 100, 150 + i * dx, 300],
                "track_id": track_id,
            }], timestamp=t0 + i * 0.2)
        return t0

    def test_track_accumulates_across_frames(self):
        from backend.services.perception import PerceptionPipeline

        pipe = PerceptionPipeline(enable_attributes=False)
        self._walk(pipe, frames=20)
        track = pipe.tracks.get(1)
        assert track.frame_count == 20
        assert track.duration > 3.5
        assert track.direction() == "east"
        assert track.displacement() > 150

    def test_replaying_archived_footage_does_not_mark_tracks_lost(self):
        """Ages must be measured in stream time, not wall-clock time.

        Replay and forensic review both feed timestamps from the past. Aging
        against `time.time()` marks every track lost on arrival, which silently
        disables all temporal analysis on recorded video.
        """
        import time as _t
        from backend.services.perception import PerceptionPipeline

        pipe = PerceptionPipeline(enable_attributes=False)
        self._walk(pipe, frames=20, t0=_t.time() - 3600)  # an hour old

        track = pipe.tracks.get(1)
        assert track.status() == "active", "replayed footage must not age out"
        assert len(pipe.tracks.all()) == 1
        assert pipe.graph is not None

    def test_stationary_is_not_confused_with_absent_or_slow(self):
        from backend.services.perception import PerceptionPipeline

        pipe = PerceptionPipeline(enable_attributes=False)
        self._walk(pipe, track_id=2, frames=20, dx=0)
        assert pipe.tracks.get(2).is_stationary() is True

        pipe2 = PerceptionPipeline(enable_attributes=False)
        self._walk(pipe2, track_id=3, frames=20, dx=30)
        assert pipe2.tracks.get(3).is_stationary() is False

    def test_dwell_requires_both_duration_and_stillness(self):
        """A long walk is not loitering; standing still briefly is not either."""
        import time as _t
        from backend.services.perception import PerceptionPipeline, detect_dwell

        t0 = _t.time()
        walking = PerceptionPipeline(enable_attributes=False)
        self._walk(walking, track_id=4, frames=100, dx=15, t0=t0)
        assert detect_dwell(walking.tracks.get(4)) is None, "walking is not dwelling"

        still = PerceptionPipeline(enable_attributes=False)
        # 0.2 s spacing gives only ~20 s over 100 frames, below the 30 s
        # threshold - feed a longer span so the duration condition is actually met.
        for i in range(100):
            still.process(1, [{"class_name": "person", "confidence": 0.9,
                               "bbox": [200, 100, 250, 300], "track_id": 5}],
                          timestamp=t0 + i * 0.5)
        obs = detect_dwell(still.tracks.get(5))
        assert obs is not None and obs.evidence, "dwell must report its evidence"

    def test_pacing_uses_path_to_displacement_ratio(self):
        import time as _t
        from backend.services.perception import PerceptionPipeline, detect_pacing

        t0 = _t.time()
        pipe = PerceptionPipeline(enable_attributes=False)
        for i in range(80):
            x = 200 + (150 if (i // 10) % 2 else 0)
            pipe.process(1, [{"class_name": "person", "confidence": 0.9,
                              "bbox": [x, 100, x + 50, 300], "track_id": 6}],
                         timestamp=t0 + i * 0.25)
        obs = detect_pacing(pipe.tracks.get(6))
        assert obs is not None
        assert obs.metadata["ratio"] > 3.0

    def test_observation_fires_once_not_every_frame(self):
        """Re-reporting each frame would flood the feed and bury real events."""
        import time as _t
        from backend.services.perception import PerceptionPipeline

        t0 = _t.time()
        pipe = PerceptionPipeline(enable_attributes=False)
        fired = []
        for i in range(120):
            r = pipe.process(1, [{"class_name": "person", "confidence": 0.9,
                                  "bbox": [200, 100, 250, 300], "track_id": 7}],
                             timestamp=t0 + i * 0.5)
            fired += [o.kind for o in r.observations]
        assert fired.count("dwell") == 1, f"dwell fired {fired.count('dwell')} times"

    def test_disappearance_needs_an_established_track(self):
        """A one-frame blip vanishing is a false positive, not a disappearance."""
        from backend.services.perception import Track, detect_disappearance

        blip = Track(track_id=8, kind="person", category="person")
        blip.frame_count = 2
        blip.lost_at = blip.last_seen + 11
        assert detect_disappearance(blip) is None

    def test_trajectory_memory_is_bounded(self):
        """A camera running for a week must not grow an unbounded history."""
        import time as _t
        from backend.services.perception import PerceptionPipeline
        from backend.services.perception.temporal import MAX_TRAJECTORY

        pipe = PerceptionPipeline(enable_attributes=False)
        self._walk(pipe, track_id=9, frames=MAX_TRAJECTORY + 200, dx=1,
                   t0=_t.time())
        assert len(pipe.tracks.get(9).trajectory) == MAX_TRAJECTORY

    def test_track_store_evicts_when_over_capacity(self):
        from backend.services.perception import TrackStore
        from backend.services.perception.observation import Scene, Entity

        store = TrackStore(max_tracks=50)
        for i in range(200):
            scene = Scene(camera_id=1)
            scene.timestamp = 1000.0 + i
            scene.add_entity(Entity(kind="person", category="person",
                                    track_id=i, bbox=[0, 0, 10, 20]))
            store.update_from_scene(scene)
        assert len(store) <= 50

    def test_attribute_history_is_kept_when_sources_disagree(self):
        """Disagreement is information; overwriting it destroys the audit trail."""
        from backend.services.perception import Track, Source

        track = Track(track_id=10, kind="vehicle", category="car")
        track.set_attribute("plate", "AAA111", 0.9, Source.DETECTOR.value)
        track.set_attribute("plate", "BBB222", 0.5, Source.LPR.value)
        assert track.get("plate") == "BBB222", "specialist must win"
        assert len(track.attribute_history["plate"]) == 2, "history must be kept"


class TestSceneGraphOverTime:
    """Relationships must strengthen with evidence and decay without it."""

    def test_confidence_grows_with_sustained_observation(self):
        from backend.services.perception import SceneGraph

        graph = SceneGraph()
        once = graph.observe(1, "near", 2, distance=50.0, timestamp=1000.0)
        first = once.confidence
        for i in range(30):
            edge = graph.observe(1, "near", 2, distance=50.0,
                                 timestamp=1000.0 + i * 0.1)
        assert edge.confidence > first
        assert edge.confidence <= 0.92, "geometry alone must never reach certainty"

    def test_unsupported_edge_decays_then_expires(self):
        """'Was true once' must not read as 'is true'."""
        from backend.services.perception import SceneGraph
        from backend.services.perception.scene_graph import EDGE_TTL_S

        graph = SceneGraph()
        edge = graph.observe(1, "near", 2, distance=50.0, timestamp=1000.0)
        fresh = edge.confidence

        edge.reference_time = 1000.0 + EDGE_TTL_S / 2
        assert edge.confidence < fresh, "a stale claim must decay"
        edge.reference_time = 1000.0 + EDGE_TTL_S + 1
        assert edge.is_expired()
        assert graph.prune(now=1000.0 + EDGE_TTL_S + 1) == 1

    def test_approach_detected_from_shrinking_distance(self):
        from backend.services.perception import SceneGraph

        graph = SceneGraph()
        for i in range(12):
            graph.observe(1, "near", 2, distance=300.0 - i * 20,
                          timestamp=1000.0 + i * 0.2)
        assert graph.get(1, "near", 2).trend() == "approaching"

    def test_separating_is_distinguished_from_approaching(self):
        from backend.services.perception import SceneGraph

        graph = SceneGraph()
        for i in range(12):
            graph.observe(1, "near", 2, distance=50.0 + i * 20,
                          timestamp=1000.0 + i * 0.2)
        assert graph.get(1, "near", 2).trend() == "separating"

    def test_following_requires_movement_and_agreeing_headings(self):
        """Two people standing near each other are queuing, not following."""
        import time as _t
        from backend.services.perception import PerceptionPipeline, SceneGraph
        from backend.services.perception.scene_graph import detect_following

        t0 = _t.time()
        pipe = PerceptionPipeline(enable_attributes=False)
        for i in range(20):  # both stationary, side by side
            pipe.process(1, [
                {"class_name": "person", "confidence": .9,
                 "bbox": [100, 100, 150, 300], "track_id": 1},
                {"class_name": "person", "confidence": .9,
                 "bbox": [200, 100, 250, 300], "track_id": 2},
            ], timestamp=t0 + i * 0.2)
        graph = SceneGraph()
        assert not detect_following(pipe.tracks.all(), graph), \
            "stationary people must not be reported as following"

    def test_following_detected_when_moving_together(self):
        import time as _t
        from backend.services.perception import PerceptionPipeline, SceneGraph
        from backend.services.perception.scene_graph import detect_following

        t0 = _t.time()
        pipe = PerceptionPipeline(enable_attributes=False)
        for i in range(30):
            x = 100 + i * 12
            pipe.process(1, [
                {"class_name": "person", "confidence": .9,
                 "bbox": [x, 100, x + 50, 300], "track_id": 1},
                {"class_name": "person", "confidence": .9,
                 "bbox": [x - 80, 100, x - 30, 300], "track_id": 2},
            ], timestamp=t0 + i * 0.2)
        graph = SceneGraph()
        found = detect_following(pipe.tracks.all(), graph)
        # Repeat so the edge accumulates past the minimum duration.
        for _ in range(3):
            found = detect_following(pipe.tracks.all(), graph)
        assert graph.get(2, "following", 1) is not None, \
            "the trailing person should be following the leader"

    def test_graph_is_keyed_by_track_not_entity(self):
        """Entity ids are regenerated every frame; an entity-keyed graph could
        never accumulate anything."""
        import time as _t
        from backend.services.perception import PerceptionPipeline

        t0 = _t.time()
        pipe = PerceptionPipeline(enable_attributes=False)
        for i in range(15):
            pipe.process(1, [
                {"class_name": "person", "confidence": .9,
                 "bbox": [100, 100, 160, 300], "track_id": 1},
                {"class_name": "car", "confidence": .8,
                 "bbox": [200, 150, 400, 300], "track_id": 2},
            ], timestamp=t0 + i * 0.2)
        edges = pipe.graph.edges_for(1)
        assert edges, "no relationship accumulated"
        assert max(e.observation_count for e in edges) > 5, \
            "edges must accumulate across frames, not reset each frame"


class TestCapabilityRegistry:
    """The scheduler can only reason about capabilities it can describe."""

    def test_gpu_capabilities_unavailable_without_cuda(self):
        from backend.services.perception import get_registry
        from backend.services.perception.capabilities import Tier, _cuda_available

        reg = get_registry()
        if _cuda_available():
            pytest.skip("host has CUDA; the negative case cannot be checked")
        for cap in reg.by_tier(Tier.GPU):
            assert not cap.available
            assert "CUDA" in cap.unavailable_reason

    def test_availability_is_probed_not_assumed(self):
        """A package can be declared in requirements and still fail to load."""
        from backend.services.perception.capabilities import (
            Capability, CapabilityRegistry, Tier)

        reg = CapabilityRegistry()
        reg.register(Capability(name="fictional", tier=Tier.CPU, cost_ms=1.0,
                                module="a_module_that_does_not_exist"))
        reg.probe()
        cap = reg.get("fictional")
        assert cap.available is False
        assert "ModuleNotFoundError" in cap.unavailable_reason

    def test_plan_skips_capabilities_whose_output_is_already_known(self):
        from backend.services.perception import get_registry

        reg = get_registry()
        context = {"person", "vehicle", "detection", "track_id"}
        fresh = reg.plan(context, already_known=set(), budget_ms=500)
        assert fresh, "nothing planned at all"

        provided = set()
        for cap in fresh:
            provided |= cap.provides
        repeat = reg.plan(context, already_known=provided, budget_ms=500)
        assert len(repeat) < len(fresh), \
            "re-running work whose output is already known is waste"

    def test_plan_respects_the_time_budget(self):
        from backend.services.perception import get_registry

        reg = get_registry()
        chosen = reg.plan({"person", "vehicle", "detection", "track_id"},
                          budget_ms=10.0)
        assert sum(c.cost_ms for c in chosen) <= 10.0

    def test_plan_excludes_inapplicable_capabilities(self):
        """Running a plate reader on a frame with no vehicle is pure waste."""
        from backend.services.perception import get_registry

        reg = get_registry()
        chosen = reg.plan({"person", "detection", "track_id"}, budget_ms=500)
        assert "lpr" not in {c.name for c in chosen}

    def test_measured_cost_replaces_the_estimate(self):
        from backend.services.perception.capabilities import (
            Capability, CapabilityRegistry, Tier)

        reg = CapabilityRegistry()
        reg.register(Capability(name="thing", tier=Tier.CPU, cost_ms=100.0))
        reg.probe()
        reg.record_cost("thing", 20.0)
        cap = reg.get("thing")
        assert cap.measured is True and cap.cost_ms == 20.0
        reg.record_cost("thing", 30.0)   # smoothed, not replaced outright
        assert 20.0 < cap.cost_ms < 30.0


class TestPerceptionPipelineIntegration:
    """The pipeline must be safe to run inside the live processing loop."""

    def test_accepts_the_coordinator_detection_shape(self):
        from backend.services.perception import PerceptionPipeline

        pipe = PerceptionPipeline(enable_attributes=False)
        result = pipe.process(2, [
            {"class_id": 0, "class_name": "person", "confidence": 0.84,
             "bbox": [61, 1, 129, 147], "track_id": 3},
        ])
        assert len(result.scene.entities) == 1
        assert result.tracks and result.tracks[0].track_id == 3

    def test_never_raises_on_malformed_input(self):
        """Perception must never stop a frame reaching the rules engine."""
        from backend.services.perception import PerceptionPipeline

        pipe = PerceptionPipeline(enable_attributes=False)
        for bad in ([], None, [{}], [{"bbox": "nonsense"}], ["string"],
                    [{"class_name": "x", "bbox": [1, 2]}]):
            result = pipe.process(1, bad)
            assert result is not None

    def test_measurement_and_inference_stay_separated(self):
        """A reader must always be able to tell evidence from conclusion."""
        import time as _t
        from backend.services.perception import PerceptionPipeline

        t0 = _t.time()
        pipe = PerceptionPipeline(enable_attributes=False)
        for i in range(90):
            pipe.process(1, [{"class_name": "person", "confidence": 0.9,
                              "bbox": [200, 100, 250, 300], "track_id": 1}],
                         timestamp=t0 + i * 0.5)
        summary = pipe.describe_track(1)
        assert "attributes" in summary and "inferred" in summary
        for item in summary["inferred"]:
            assert item["evidence"], "an inference with no evidence is not defensible"

    def test_pipeline_can_be_disabled_by_env(self):
        """Mirrors ARGUS_NO_SWARM so the legacy path stays benchmarkable."""
        src = (PROJECT_ROOT / "backend" / "services" / "core_engine"
               / "processing_coordinator.py").read_text(encoding="utf-8")
        assert "ARGUS_NO_PERCEPTION" in src
        assert src.count("self.perception.process(") == 2, \
            "both the swarm and linear paths must feed perception"


# ─────────────────────────────────────────────────────────────────────────────
# Iteration 11: CPU perception expansion
#
# The rule every test here defends: Argus must never silently turn an inference
# into an observation. A measurement and a guess must stay distinguishable all
# the way to the API.
# ─────────────────────────────────────────────────────────────────────────────

class TestChangeDetection:
    """Departures from a per-camera, time-of-day baseline."""

    def _scene(self, camera_id=1, n=3, ts=None, kind="person"):
        from backend.services.perception import Entity, EntityKind, BBox, Scene
        scene = Scene(camera_id=camera_id, timestamp=ts or time.time())
        for i in range(n):
            scene.add_entity(Entity(
                kind=kind, category=kind, track_id=i,
                bbox=BBox(10.0 + i * 60, 10.0, 60.0 + i * 60, 150.0),
                confidence=0.9))
        return scene

    def test_no_anomaly_before_the_baseline_is_ready(self):
        """An immature baseline must stay silent rather than guess.

        Reporting an anomaly from five samples is reporting inexperience, and
        it is how anomaly detectors earn a reputation for crying wolf.
        """
        from backend.services.perception import ChangeDetector
        from backend.services.perception.change import MIN_SAMPLES
        cd = ChangeDetector()
        for _ in range(5):
            cd.observe(self._scene(n=3))
        assert cd.occupancy_anomaly(self._scene(n=500)) is None, \
            "must not judge before the baseline is mature"
        assert not cd.baseline(1).is_ready()
        assert MIN_SAMPLES > 5

    def test_occupancy_anomaly_after_baseline_matures(self):
        from backend.services.perception.change import ChangeDetector, MIN_SAMPLES
        cd = ChangeDetector()
        now = time.time()
        for i in range(MIN_SAMPLES + 5):
            cd.observe(self._scene(n=3 if i % 2 else 4, ts=now))
        obs = cd.occupancy_anomaly(self._scene(n=60, ts=now))
        assert obs is not None, "60 entities against a mean of ~3 is anomalous"
        assert obs.kind == "occupancy_anomaly"
        assert obs.evidence, "an anomaly must show the baseline it broke"
        assert any("baseline" in e for e in obs.evidence)
        assert any("standard deviation" in e for e in obs.evidence)

    def test_baselines_are_separated_by_hour_of_day(self):
        """03:00 and 13:00 are different normals for the same camera.

        Without this, every morning rush is an anomaly and the feature is
        useless on any camera with a daily rhythm.
        """
        from backend.services.perception.change import (BUCKETS, ChangeDetector,
                                                        MIN_SAMPLES, _bucket)
        cd = ChangeDetector()
        night = time.mktime(time.strptime("2026-01-01 03:30", "%Y-%m-%d %H:%M"))
        noon = time.mktime(time.strptime("2026-01-01 13:30", "%Y-%m-%d %H:%M"))
        assert _bucket(night) != _bucket(noon)

        for _ in range(MIN_SAMPLES + 5):
            cd.observe(self._scene(n=1, ts=night))
            cd.observe(self._scene(n=40, ts=noon))

        # 40 people at noon is normal; 40 people at 03:30 is not.
        assert cd.occupancy_anomaly(self._scene(n=40, ts=noon)) is None
        assert cd.occupancy_anomaly(self._scene(n=40, ts=night)) is not None

    def test_appearance_and_disappearance_are_reported(self):
        from backend.services.perception import ChangeDetector
        cd = ChangeDetector()
        cd.entity_changes(self._scene(n=2))
        obs = cd.entity_changes(self._scene(n=3))
        kinds = {o.kind for o in obs}
        assert "object_appeared" in kinds
        obs2 = cd.entity_changes(self._scene(n=1))
        assert "object_left_frame" in {o.kind for o in obs2}

    def test_first_frame_reports_nothing(self):
        """Nothing to compare against is not the same as nothing changed."""
        from backend.services.perception import ChangeDetector
        cd = ChangeDetector()
        assert cd.entity_changes(self._scene(n=5)) == []

    def test_baseline_learns_from_anomalous_frames_too(self):
        """A persistent new normal must stop being reported as an anomaly."""
        from backend.services.perception.change import ChangeDetector, MIN_SAMPLES
        cd = ChangeDetector()
        now = time.time()
        for _ in range(MIN_SAMPLES * 3):
            cd.analyse(self._scene(n=30, ts=now), None)
        assert cd.occupancy_anomaly(self._scene(n=30, ts=now)) is None, \
            "30 must be normal once it has been the norm for a long time"

    def test_running_stats_use_constant_memory(self):
        """Months of footage must not accumulate a sample list."""
        from backend.services.perception import RunningStat
        s = RunningStat()
        for v in range(5000):
            s.push(float(v % 10))
        assert s.count == 5000
        assert 4.0 < s.mean < 5.0
        assert not any(isinstance(v, (list, tuple, set, dict))
                       for v in vars(s).values()), \
            "RunningStat must not retain samples"


class TestSceneClassification:
    """Environment context, and the honesty of its limits."""

    def test_density_scales_with_entity_count(self):
        from backend.services.perception import density_label
        assert density_label(0)[0] == "empty"
        assert density_label(2)[0] == "sparse"
        assert density_label(8)[0] == "moderate"
        assert density_label(40)[0] == "crowded"

    def test_classify_attaches_density_without_a_frame(self):
        from backend.services.perception import Scene, classify_scene
        scene = Scene(camera_id=1)
        applied = classify_scene(scene, None)
        assert applied["density"] == "empty"
        assert "density" in scene.environment
        assert scene.environment["entity_count"].value == 0

    def test_measurements_outrank_interpretations(self):
        """The numbers are certain; the labels drawn from them are not."""
        from backend.services.perception.scene_classifier import interpret
        measured = {"mean_brightness": 20.0, "contrast": 10.0,
                    "detail_variance": 5.0}
        labels = interpret(measured)
        assert labels["lighting"][0] == "dark"
        for _, confidence in labels.values():
            assert confidence < 1.0, \
                "an interpretation must never claim certainty"

    def test_no_place_category_is_invented(self):
        """Pixel statistics cannot support 'this is a car park'."""
        from backend.services.perception.scene_classifier import interpret
        labels = interpret({"mean_brightness": 120.0, "contrast": 50.0,
                            "detail_variance": 300.0})
        assert "scene_type" not in labels
        assert "indoor_outdoor" not in labels

    def test_degraded_visibility_is_detected(self):
        from backend.services.perception.scene_classifier import interpret
        labels = interpret({"mean_brightness": 120.0, "contrast": 8.0,
                            "detail_variance": 3.0})
        assert labels["visibility"][0] == "degraded"


class TestPluggableOcr:
    """Text regions work everywhere; reading them is optional."""

    def test_engine_reports_unavailable_honestly(self):
        """No engine here, and the registry must say so with a reason."""
        from backend.services.perception import get_engine
        engine = get_engine()
        if not engine.available:
            assert engine.reason, "unavailability must come with a reason"

    def test_capability_registry_does_not_claim_ocr_it_cannot_do(self):
        """The bug this guards: pytesseract imports without the binary.

        An import-only probe marks OCR available and the pipeline then fails
        on the first real frame. Availability must be decided by running the
        backend, not importing it.
        """
        from backend.services.perception import build_default_registry, get_engine
        reg = build_default_registry()
        cap = reg.get("ocr")
        assert cap is not None
        assert cap.available == get_engine().available, \
            "the registry must agree with a real functional probe"
        if not cap.available:
            assert cap.unavailable_reason
            assert "text_regions" in cap.unavailable_reason, \
                "must point at the half that still works"

    def test_region_detection_needs_no_engine(self):
        """The useful half must work with no OCR engine installed at all."""
        import numpy as np
        import cv2
        from backend.services.perception import find_text_regions
        img = np.full((240, 640, 3), 245, dtype=np.uint8)
        cv2.putText(img, "LOADING BAY 7", (30, 130), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 0, 0), 5)
        regions = find_text_regions(img)
        assert regions, "MSER must find the painted text"
        assert any(r.width > 60 for r in regions)

    def test_unread_text_is_recorded_as_unread(self):
        """Seeing writing you cannot read is a real observation.

        It must be distinguishable both from reading it and from seeing none.
        """
        import numpy as np
        import cv2
        from backend.services.perception import (EntityKind, Scene, extract_text,
                                                 get_engine)
        img = np.full((240, 640, 3), 245, dtype=np.uint8)
        cv2.putText(img, "EXIT 12B", (40, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    2.2, (0, 0, 0), 6)
        scene = Scene(camera_id=1)
        created = extract_text(scene, img)
        assert created, "text regions must become entities"
        for entity in created:
            assert entity.kind == EntityKind.TEXT.value
            assert entity.observed("text_readable")
            if not get_engine().available:
                assert entity.get("text_readable") is False
                assert entity.get("ocr_unavailable"), \
                    "must record why the text could not be read"
                assert entity.get("text") is None, \
                    "no engine means no invented characters"

    def test_text_is_bound_to_the_object_it_sits_on(self):
        from backend.services.perception import (BBox, Entity, EntityKind, Scene,
                                                 attach_text_to_entities)
        scene = Scene(camera_id=1)
        van = Entity(kind=EntityKind.VEHICLE.value, category="truck",
                     bbox=BBox(0, 0, 400, 300), confidence=0.9)
        scene.add_entity(van)
        text = Entity(kind=EntityKind.TEXT.value, category="text",
                      bbox=BBox(100, 100, 200, 140), confidence=0.5)
        text.set_attribute("text", "ACME LOGISTICS", 0.7, "ocr")
        scene.add_entity(text)

        assert attach_text_to_entities(scene) == 1
        assert van.get("marking") == "ACME LOGISTICS"


class TestRelationshipVocabulary:
    """A wider vocabulary, with the burden of proof kept intact."""

    def _person(self, x=100.0):
        from backend.services.perception import BBox, Entity, EntityKind
        return Entity(kind=EntityKind.PERSON.value, category="person",
                      bbox=BBox(x, 100.0, x + 50, 250.0), confidence=0.9)

    def test_single_frame_never_asserts_behaviour(self):
        """The core guarantee: geometry cannot produce 'following'.

        Following, entering and queuing are claims about motion over time. A
        single frame has no motion, so inferring them there would be invention.
        """
        from backend.services.perception import (BEHAVIOURAL_PREDICATES, Scene,
                                                 infer_relationships)
        from backend.services.perception.relationships import infer_all
        scene = Scene(camera_id=1)
        for x in (100.0, 160.0, 220.0, 280.0):
            scene.add_entity(self._person(x))
        rels = infer_all(scene)
        assert rels, "four adjacent people must produce some relationships"
        for rel in rels:
            assert rel.predicate not in BEHAVIOURAL_PREDICATES, \
                f"single frame must not assert behavioural '{rel.predicate}'"

    def test_wearing_holding_and_carrying_are_distinguished(self):
        from backend.services.perception import (BBox, Entity, EntityKind,
                                                 RelationKind, Scene)
        from backend.services.perception.relationships import infer_all
        scene = Scene(camera_id=1)
        person = self._person(100.0)          # (100,100)-(150,250)
        scene.add_entity(person)
        # Helmet in the upper-body band.
        scene.add_entity(Entity(kind=EntityKind.OBJECT.value, category="helmet",
                                bbox=BBox(105, 105, 145, 150), confidence=0.8))
        # Phone in the hand zone (30-80% of height => y 145..220).
        scene.add_entity(Entity(kind=EntityKind.OBJECT.value,
                                category="cell phone",
                                bbox=BBox(115, 160, 140, 195), confidence=0.8))
        # Backpack overlapping the body generally.
        scene.add_entity(Entity(kind=EntityKind.OBJECT.value, category="backpack",
                                bbox=BBox(100, 130, 150, 220), confidence=0.8))

        predicates = {r.predicate for r in infer_all(scene)}
        assert RelationKind.WEARING.value in predicates
        assert RelationKind.HOLDING.value in predicates
        assert RelationKind.CARRYING.value in predicates

    def test_geometry_only_claims_stay_below_certainty(self):
        """Boxes overlapping is suggestive, never conclusive.

        The bound is hard-coded rather than imported from the module under
        test. Importing GEOMETRY_CEILING would make this tautological - raise
        the constant and the assertion rises with it, so the test could never
        catch the overclaim it exists to catch.
        """
        from backend.services.perception import BBox, Entity, EntityKind, Scene
        from backend.services.perception.relationships import (GEOMETRY_CEILING,
                                                               infer_all)
        ABSOLUTE_MAX = 0.8
        assert GEOMETRY_CEILING <= ABSOLUTE_MAX, \
            "the geometry ceiling itself must stay well below certainty"

        scene = Scene(camera_id=1)
        person = self._person(100.0)
        scene.add_entity(person)
        scene.add_entity(Entity(kind=EntityKind.OBJECT.value, category="backpack",
                                bbox=BBox(100, 100, 150, 250), confidence=0.9))
        semantic = {"carrying", "holding", "wearing", "riding", "inside",
                    "occluding"}
        checked = 0
        for rel in infer_all(scene):
            if rel.predicate in semantic:
                checked += 1
                assert rel.confidence <= ABSOLUTE_MAX, \
                    f"{rel.predicate} at {rel.confidence} overclaims"
        assert checked, "the fixture must produce semantic relationships"

    def test_riding_versus_inside(self):
        from backend.services.perception import (BBox, Entity, EntityKind,
                                                 RelationKind, Scene)
        from backend.services.perception.relationships import infer_containment

        cyclist = Scene(camera_id=1)
        cyclist.add_entity(Entity(kind=EntityKind.PERSON.value, category="person",
                                  bbox=BBox(100, 80, 150, 220), confidence=0.9))
        cyclist.add_entity(Entity(kind=EntityKind.VEHICLE.value,
                                  category="bicycle",
                                  bbox=BBox(90, 160, 170, 260), confidence=0.9))
        assert any(r.predicate == RelationKind.RIDING.value
                   for r in infer_containment(cyclist))

        driver = Scene(camera_id=1)
        driver.add_entity(Entity(kind=EntityKind.PERSON.value, category="person",
                                 bbox=BBox(120, 120, 160, 200), confidence=0.9))
        driver.add_entity(Entity(kind=EntityKind.VEHICLE.value, category="car",
                                 bbox=BBox(50, 80, 350, 280), confidence=0.9))
        assert any(r.predicate == RelationKind.INSIDE.value
                   for r in infer_containment(driver))

    def test_every_relationship_carries_its_basis(self):
        from backend.services.perception import BBox, Entity, EntityKind, Scene
        from backend.services.perception.relationships import infer_all
        scene = Scene(camera_id=1)
        scene.add_entity(self._person(100.0))
        scene.add_entity(self._person(170.0))
        scene.add_entity(Entity(kind=EntityKind.VEHICLE.value, category="car",
                                bbox=BBox(300, 120, 520, 260), confidence=0.9))
        rels = infer_all(scene)
        assert rels
        for rel in rels:
            assert rel.evidence, f"{rel.predicate} asserted with no evidence"
            assert rel.is_supported, f"{rel.predicate} is unsupported"

    def test_relationship_lifecycle_distinguishes_momentary_from_sustained(self):
        from backend.services.perception import Relationship
        now = time.time()
        rel = Relationship(subject_id="a", predicate="near", object_id="b",
                           evidence={"pixel_distance": 10.0},
                           first_observed=now, last_observed=now)
        assert rel.status == "momentary"
        for i in range(8):
            rel.reinforce(timestamp=now + i + 1)
        assert rel.status == "sustained"
        assert rel.duration >= 5.0
        assert rel.observation_count == 9

    def test_behavioural_claim_from_one_sighting_is_unsupported(self):
        """The schema itself refuses to call a single frame 'following'."""
        from backend.services.perception import Relationship
        rel = Relationship(subject_id="a", predicate="following", object_id="b",
                           confidence=0.9, evidence={"heading_cosine": 0.95})
        assert rel.is_behavioural
        assert not rel.is_supported, \
            "one frame cannot support a behavioural claim"
        rel.reinforce()
        assert rel.is_supported

    def test_relationship_lifecycle_survives_serialisation(self):
        """Restoring must not reset two minutes of evidence to a fresh sighting."""
        from backend.services.perception import BBox, Entity, EntityKind, Scene
        scene = Scene(camera_id=1)
        a = Entity(kind=EntityKind.PERSON.value, category="person",
                   bbox=BBox(0, 0, 50, 150), confidence=0.9)
        b = Entity(kind=EntityKind.PERSON.value, category="person",
                   bbox=BBox(60, 0, 110, 150), confidence=0.9)
        scene.add_entity(a)
        scene.add_entity(b)
        from backend.services.perception import Relationship
        rel = Relationship(subject_id=a.entity_id, predicate="near",
                           object_id=b.entity_id, evidence={"pixel_distance": 10})
        for _ in range(6):
            rel.reinforce()
        scene.add_relationship(rel)

        restored = Scene.from_dict(scene.to_dict())
        assert restored.to_dict() == scene.to_dict()
        assert restored.relationships[0].observation_count == 7
        assert restored.relationships[0].status == rel.status


class TestExpandedAttributes:
    """More cheap attributes, each with an honest confidence."""

    def _frame(self, colour=(40, 40, 200), size=(300, 400)):
        import numpy as np
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img[:, :] = colour
        return img

    def test_size_class_is_frame_relative(self):
        from backend.services.perception import BBox
        from backend.services.perception.attributes import size_class
        assert size_class(BBox(0, 0, 10, 10), 1920, 1080) == "tiny"
        assert size_class(BBox(0, 0, 1900, 1000), 1920, 1080) == "dominant"

    def test_frame_position_is_reported(self):
        from backend.services.perception import BBox
        from backend.services.perception.attributes import frame_position
        assert frame_position(BBox(0, 0, 40, 40), 900, 900) == "top-left"
        assert frame_position(BBox(400, 400, 500, 500), 900, 900) == "middle"

    def test_clipped_boxes_suppress_the_posture_hint(self):
        """A half-visible person has a meaningless aspect ratio.

        Emitting 'horizontal' for someone walking out of frame is exactly the
        kind of confident nonsense the source-authority rule exists to stop.
        """
        from backend.services.perception import BBox, Entity, EntityKind
        from backend.services.perception.attributes import enrich_entity
        frame = self._frame()
        clipped = Entity(kind=EntityKind.PERSON.value, category="person",
                         bbox=BBox(0, 250, 200, 300), confidence=0.9)
        enrich_entity(clipped, frame)
        assert clipped.get("clipped_by_frame_edge") is True
        assert not clipped.observed("posture"), \
            "a clipped box must not produce a posture claim"

    def test_dark_crops_are_flagged_as_unreliable(self):
        from backend.services.perception import BBox, Entity, EntityKind
        from backend.services.perception.attributes import enrich_entity
        dark = self._frame(colour=(5, 5, 5))
        entity = Entity(kind=EntityKind.PERSON.value, category="person",
                        bbox=BBox(100, 100, 160, 260), confidence=0.9)
        enrich_entity(entity, dark)
        assert entity.get("brightness") is not None
        assert entity.get("appearance_reliable") is False, \
            "colour read from a near-black crop must be marked unreliable"

    def test_palette_reports_more_than_the_winner(self):
        import numpy as np
        from backend.services.perception.attributes import colour_palette
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        crop[:50, :] = (200, 40, 40)     # blue-ish top (BGR)
        crop[50:, :] = (40, 40, 200)     # red-ish bottom
        palette = colour_palette(crop)
        assert len(palette) >= 2
        assert sum(share for _, share in palette) <= 1.0001
        names = {name for name, _ in palette}
        assert "red" in names and "blue" in names

    def test_tiny_crops_yield_no_colour_claim(self):
        import numpy as np
        from backend.services.perception.attributes import colour_palette
        assert colour_palette(np.zeros((4, 4, 3), dtype=np.uint8)) == []


class TestEvidenceApi:
    """Every conclusion must be able to show its work."""

    def _track_with_history(self):
        from backend.services.perception import BBox, Entity, EntityKind, Scene
        from backend.services.perception import TrackStore
        store = TrackStore()
        base = time.time()
        for i in range(12):
            scene = Scene(camera_id=1, timestamp=base + i * 0.5)
            scene.add_entity(Entity(
                kind=EntityKind.PERSON.value, category="person", track_id=7,
                bbox=BBox(100.0 + i, 100.0, 150.0 + i, 250.0), confidence=0.9))
            store.update_from_scene(scene)
        return store, store.get(7)

    def test_chain_separates_measurement_from_inference(self):
        from backend.services.perception import chain_from_track
        _, track = self._track_with_history()
        chain = chain_from_track(track)
        assert chain.items, "a chain with no items explains nothing"
        assert chain.supporting_measurements(), \
            "a grounded chain needs at least one measurement"
        for item in chain.supporting_measurements():
            assert item.is_measurement
        for item in chain.supporting_inferences():
            assert not item.is_measurement

    def test_ungrounded_chain_is_not_actionable(self):
        """A conclusion resting only on other inferences must not be actioned."""
        from backend.services.perception import EvidenceChain
        chain = EvidenceChain(assessment="person is loitering", confidence=0.95,
                              kind="dwell")
        chain.add("appears to be waiting", is_measurement=False)
        assert not chain.is_grounded
        assert not chain.is_actionable, \
            "high confidence must not rescue an ungrounded claim"
        assert chain.status in {"unsupported", "derived", "weak"}

    def test_grounded_high_confidence_chain_is_actionable(self):
        from backend.services.perception import EvidenceChain
        chain = EvidenceChain(assessment="stationary for 90s", confidence=0.8,
                              kind="dwell")
        chain.add("displacement 12 px over 90 s", is_measurement=True, value=12)
        chain.add("consistent with waiting", is_measurement=False)
        assert chain.is_grounded and chain.is_actionable
        assert chain.status == "supported"

    def test_explain_track_keeps_the_two_apart_and_warns(self):
        from backend.services.perception import explain_track
        _, track = self._track_with_history()
        out = explain_track(track)
        assert "measured" in out and "inferred" in out
        assert isinstance(out["measured"], list)
        assert out.get("note"), \
            "the explain payload must warn against presenting inference as fact"

    def test_pipeline_exposes_explain(self):
        from backend.services.perception import PerceptionPipeline
        pipeline = PerceptionPipeline()
        base = time.time()
        for i in range(10):
            pipeline.process(1, [{"class_name": "person", "confidence": 0.9,
                                  "bbox": [100 + i, 100, 150 + i, 250],
                                  "track_id": 3}], None, base + i * 0.4)
        explained = pipeline.explain(3)
        assert explained is not None
        assert "measured" in explained
        assert pipeline.explain(999) is None

    def test_evidence_text_is_human_readable(self):
        from backend.services.perception import EvidenceChain
        chain = EvidenceChain(assessment="vehicle stopped in a fire lane",
                              confidence=0.62, kind="zone_violation")
        chain.add("stationary 45 s", is_measurement=True, value=45)
        chain.add("inside zone 'fire_lane'", is_measurement=True)
        text = chain.explain()
        assert "fire lane" in text
        assert "45" in text


class TestPerceptionStagesStayHonest:
    """Cross-cutting guarantees over the whole expanded pipeline."""

    def test_capability_registry_reports_new_capabilities(self):
        from backend.services.perception import build_default_registry
        reg = build_default_registry()
        for name in ("text_regions", "scene_classification", "change_detection",
                     "rich_relationships"):
            cap = reg.get(name)
            assert cap is not None, f"{name} must be registered"
            assert cap.available, f"{name} is pure-CPU and must run here"
            assert cap.description

    def test_unavailable_capabilities_say_how_to_enable_them(self):
        """'Unavailable' with no reason is a dead end for an operator."""
        from backend.services.perception import build_default_registry
        reg = build_default_registry()
        for cap in reg.all():
            if not cap.available:
                assert cap.unavailable_reason, \
                    f"{cap.name} is unavailable with no explanation"

    def test_pipeline_never_raises_on_a_broken_frame(self):
        """Perception enriches a frame; it must never stop one."""
        import numpy as np
        from backend.services.perception import PerceptionPipeline
        pipeline = PerceptionPipeline(enable_text=True)
        rubbish = [
            None,
            np.zeros((0, 0, 3), dtype=np.uint8),
            np.zeros((10, 10), dtype=np.uint8),      # wrong channel count
            "not a frame",
        ]
        for frame in rubbish:
            result = pipeline.process(1, [{"class_name": "person",
                                           "confidence": 0.8,
                                           "bbox": [1, 1, 20, 60],
                                           "track_id": 1}], frame)
            assert result is not None

    def test_pipeline_reports_per_stage_cost(self):
        from backend.services.perception import PerceptionPipeline
        pipeline = PerceptionPipeline()
        for _ in range(3):
            pipeline.process(1, [{"class_name": "person", "confidence": 0.9,
                                  "bbox": [10, 10, 60, 150], "track_id": 1}])
        stats = pipeline.stats()
        assert stats["stage_ms"], "stage costs must be measured, not assumed"
        assert stats["stages_enabled"]["change_detection"] is True

    def test_text_stage_is_rate_limited(self):
        """12 ms per frame for signage that never changes is waste."""
        from backend.services.perception import PerceptionPipeline
        pipeline = PerceptionPipeline(enable_text=True, text_interval_s=5.0)
        now = time.time()
        assert pipeline._text_due(1, now) is True
        assert pipeline._text_due(1, now + 1.0) is False
        assert pipeline._text_due(1, now + 6.0) is True
        # Rate limiting is per camera, not global.
        assert pipeline._text_due(2, now + 6.0) is True


class TestPerceptionKeepsNoPersistentAppearanceData:
    """Appearance data is re-identifying; persisting it creates an obligation.

    Phase 2 shipped with perception entirely in memory, and this class held a
    tripwire that failed the moment it learned to write to disk. Phase 6 (memory)
    crossed that line deliberately - so the tripwire did its job and has been
    converted into the stricter guarantee it was always demanding: **every
    perception table that persists data must be covered by a retention policy.**

    The list is explicit rather than discovered, so adding a table without
    adding its expiry fails here.
    """

    #: Perception tables allowed to persist, each with the retention key that
    #: expires it. A new table must be added to BOTH or this test fails.
    PERSISTED_TABLES = {
        "perception_observations": "perception_observations_days",
        "perception_appearances": "perception_appearances_days",
        "perception_tracks": "perception_appearances_days",
    }

    def test_only_the_memory_module_persists_anything(self):
        """Storage stays in one auditable place.

        If descriptors could be written from five different modules, no
        reviewer could confirm they all expire. Confining writes to memory.py
        is what makes the retention guarantee checkable at all.
        """
        import re
        pkg = PROJECT_ROOT / "backend" / "services" / "perception"
        allowed = {"memory.py"}
        offenders = []
        for path in sorted(pkg.glob("*.py")):
            if path.name in allowed:
                continue
            src = path.read_text(encoding="utf-8")
            if re.search(r"INSERT\s+INTO|UPDATE\s+\w+\s+SET|sqlite3\.connect|"
                         r"CREATE\s+TABLE", src, re.IGNORECASE):
                offenders.append(path.name)
        assert not offenders, (
            f"{offenders} write to the database directly. Route persistence "
            "through backend/services/perception/memory.py so every table "
            "stays covered by a retention policy.")

    def test_every_persisted_table_has_a_retention_policy(self):
        from backend.services.management.retention import _DEFAULTS
        import inspect
        from backend.services.management import retention as retention_module

        source = inspect.getsource(retention_module.run_retention_once)
        for table, policy_key in self.PERSISTED_TABLES.items():
            assert policy_key in _DEFAULTS, \
                f"{table} has no retention default ({policy_key})"
            assert table in source, \
                (f"{table} is never purged by run_retention_once - "
                 "re-identifying data would accumulate forever")

    def test_every_perception_table_created_is_declared_here(self):
        """A new table must not slip in without an expiry."""
        import re
        src = (PROJECT_ROOT / "backend" / "services" / "perception"
               / "memory.py").read_text(encoding="utf-8")
        created = set(re.findall(
            r"CREATE TABLE IF NOT EXISTS\s+(\w+)", src))
        # FTS mirrors of an already-covered table carry no independent data.
        created = {t for t in created if not t.endswith("_fts")}
        undeclared = created - set(self.PERSISTED_TABLES)
        assert not undeclared, (
            f"{undeclared} are created but have no declared retention policy. "
            "Add them to PERSISTED_TABLES and to retention.py.")

    def test_track_and_attribute_history_are_bounded(self):
        """Memory-only is not a licence to grow without limit."""
        from backend.services.perception.temporal import (MAX_ATTRIBUTE_HISTORY,
                                                          MAX_TRAJECTORY,
                                                          TrackStore)
        assert MAX_TRAJECTORY <= 1024
        assert MAX_ATTRIBUTE_HISTORY <= 64
        store = TrackStore(max_tracks=8)
        from backend.services.perception import BBox, Entity, EntityKind, Scene
        base = time.time()
        for i in range(50):
            scene = Scene(camera_id=1, timestamp=base + i)
            scene.add_entity(Entity(kind=EntityKind.PERSON.value,
                                    category="person", track_id=i,
                                    bbox=BBox(0, 0, 10, 20), confidence=0.9))
            store.update_from_scene(scene)
        assert len(store) <= 8, "TrackStore must evict rather than grow"

    def test_change_baselines_store_statistics_not_imagery(self):
        """A baseline must never become an undeclared image store."""
        from backend.services.perception import ChangeDetector
        from backend.services.perception.change import GRID
        cd = ChangeDetector()
        baseline = cd.baseline(1)
        # The only pixel-derived state is a GRID x GRID intensity signature.
        assert GRID <= 32, "the signature must stay a thumbnail, not a frame"
        assert baseline.signature is None
        assert baseline.signature_baseline is None


# ─────────────────────────────────────────────────────────────────────────────
# Phase 6 — Memory: everything perceived, retrievable later
# ─────────────────────────────────────────────────────────────────────────────

class TestAppearanceDescriptors:
    """The vector that makes "find this person" possible."""

    def _person_crop(self, top=(200, 40, 40), bottom=(40, 40, 200), size=(96, 48)):
        import numpy as np
        img = np.zeros((size[0], size[1], 3), dtype=np.uint8)
        img[: size[0] // 2] = top
        img[size[0] // 2:] = bottom
        return img

    def _shift(self, img, gain, warm):
        import numpy as np
        out = img.astype(np.float32) * gain
        out[:, :, 2] *= warm
        out[:, :, 0] /= warm
        return np.clip(out, 0, 255).astype(np.uint8)

    def test_descriptor_has_the_declared_dimension_and_is_unit_norm(self):
        import numpy as np
        from backend.services.perception import DESCRIPTOR_DIM, describe
        d = describe(self._person_crop())
        assert d is not None
        assert len(d) == DESCRIPTOR_DIM
        assert abs(float(np.linalg.norm(d)) - 1.0) < 1e-5

    def test_unusable_crops_return_none_not_a_zero_vector(self):
        """A zero vector would match everything; None means "not observed"."""
        import numpy as np
        from backend.services.perception import describe
        assert describe(None) is None
        assert describe(np.zeros((0, 0, 3), dtype=np.uint8)) is None
        assert describe(np.zeros((8, 4, 3), dtype=np.uint8)) is None, \
            "a crop below the minimum size must not yield a descriptor"
        assert describe(np.zeros((40, 20), dtype=np.uint8)) is None

    def test_same_appearance_scores_higher_than_different(self):
        from backend.services.perception import describe, similarity
        a = describe(self._person_crop())
        b = describe(self._person_crop())
        c = describe(self._person_crop(top=(40, 200, 40), bottom=(200, 200, 40)))
        assert similarity(a, b) > similarity(a, c)
        assert similarity(a, b) > 0.95

    def test_colour_constancy_survives_a_lighting_change(self):
        """The measurement that decided the design.

        Without grey-world normalisation a colour histogram collapses from 96%
        to 20% rank-1 the moment the camera changes - which is precisely the
        cross-camera case this whole phase exists to serve. If this regresses,
        cross-camera search becomes a false promise, so it is pinned here.
        """
        from backend.services.perception import describe, similarity
        original = self._person_crop()
        harsh = self._shift(original, 0.55, 1.3)

        with_cc = similarity(describe(original, colour_constancy=True),
                             describe(harsh, colour_constancy=True))
        without_cc = similarity(describe(original, colour_constancy=False),
                                describe(harsh, colour_constancy=False))
        assert with_cc > 0.9, \
            f"colour constancy must survive a lighting shift (got {with_cc:.3f})"
        assert with_cc > without_cc, \
            "grey-world normalisation must beat the raw histogram under a shift"

    def test_average_is_renormalised(self):
        import numpy as np
        from backend.services.perception import average, describe
        crops = [self._person_crop(), self._person_crop(top=(190, 45, 45))]
        mean = average([describe(c) for c in crops])
        assert mean is not None
        assert abs(float(np.linalg.norm(mean)) - 1.0) < 1e-5
        assert average([]) is None
        assert average([None, None]) is None

    def test_a_match_is_never_an_identification(self):
        """Colour layout is not a biometric, and the type system says so."""
        from backend.services.perception import MatchResult
        m = MatchResult(key="1:1", similarity=0.999, strength="strong")
        assert m.is_identification is False
        assert m.to_dict()["is_identification"] is False


class TestPerceptionMemory:
    """Durable storage of observations, appearances and tracks."""

    def _memory(self, tmp_path):
        from backend.services.perception import PerceptionMemory
        return PerceptionMemory(db_path=str(tmp_path / "mem.db"))

    def _observation(self, kind="dwell", camera_id=1, ts=None, conf=0.8):
        from backend.services.perception import Observation
        return Observation(
            kind=kind, summary=f"person loitering near the loading bay",
            camera_id=camera_id, timestamp=ts or time.time(), confidence=conf,
            source="temporal_engine", track_ids=[7],
            evidence=["duration 143s (threshold 30s)", "displacement 12px"])

    def test_observations_survive_a_restart(self, tmp_path):
        """The whole point: perception that evaporates answers nothing later."""
        from backend.services.perception import PerceptionMemory
        path = str(tmp_path / "mem.db")
        first = PerceptionMemory(db_path=path)
        first.remember_observation(self._observation())
        # A completely fresh instance, as after a process restart.
        second = PerceptionMemory(db_path=path)
        rows = second.query_observations()
        assert len(rows) == 1
        assert rows[0].evidence, "evidence must survive persistence"
        assert rows[0].track_ids == [7]

    def test_query_filters_compose(self, tmp_path):
        mem = self._memory(tmp_path)
        now = time.time()
        mem.remember_observation(self._observation("dwell", 1, now - 10))
        mem.remember_observation(self._observation("pacing", 2, now - 20))
        mem.remember_observation(self._observation("dwell", 2, now - 8000))

        assert len(mem.query_observations(camera_id=2)) == 2
        assert len(mem.query_observations(kinds=["dwell"])) == 2
        assert len(mem.query_observations(camera_id=2, kinds=["dwell"])) == 1
        assert len(mem.query_observations(since=now - 100)) == 2

    def test_full_text_search_finds_by_words_and_evidence(self, tmp_path):
        mem = self._memory(tmp_path)
        mem.remember_observation(self._observation())
        assert len(mem.search_observations("loading")) == 1
        assert len(mem.search_observations("bay")) == 1
        assert len(mem.search_observations("helicopter")) == 0

    def test_malformed_search_text_does_not_raise(self, tmp_path):
        """Search text is user input; an FTS syntax error must not 500."""
        mem = self._memory(tmp_path)
        mem.remember_observation(self._observation())
        for bad in ('"unclosed', "AND OR", "*", "((", 'NEAR/"'):
            assert isinstance(mem.search_observations(bad), list)

    def test_appearance_round_trips_through_the_database(self, tmp_path):
        import numpy as np
        from backend.services.perception import similarity
        mem = self._memory(tmp_path)
        v = np.asarray(np.random.default_rng(3).random(96), dtype="float32")
        v /= np.linalg.norm(v)
        mem.remember_appearance(1, 5, v, time.time() - 10, time.time(), 20)

        stored = mem.get_appearance("1:5")
        assert stored is not None
        assert similarity(stored.descriptor, v) > 0.9999, \
            "the stored vector must be the vector that went in"

    def test_appearance_upsert_does_not_duplicate(self, tmp_path):
        import numpy as np
        mem = self._memory(tmp_path)
        v = np.ones(96, dtype="float32") / 9.79795897
        mem.remember_appearance(1, 5, v, 100.0, 200.0, 10)
        mem.remember_appearance(1, 5, v, 100.0, 300.0, 25)
        assert mem.count("appearances") == 1
        assert mem.get_appearance("1:5").frame_count == 25

    def test_vector_search_ranks_by_similarity(self, tmp_path):
        import numpy as np
        mem = self._memory(tmp_path)
        rng = np.random.default_rng(11)

        def unit(x):
            x = np.asarray(x, dtype="float32")
            return x / np.linalg.norm(x)

        target = unit(rng.random(96))
        mem.remember_appearance(1, 1, target, 1.0, 2.0, 5)
        mem.remember_appearance(1, 2, unit(target + 0.05 * rng.random(96)), 1.0, 2.0, 5)
        mem.remember_appearance(1, 3, unit(rng.random(96)), 1.0, 2.0, 5)

        hits = mem.search_appearance_vectors(target, limit=3)
        assert hits[0][0] == "1:1"
        assert hits[0][1] > hits[-1][1]

    def test_mismatched_descriptor_dimensions_do_not_silently_return_nothing(
            self, tmp_path):
        """A dimension clash means two backends, not "no matches"."""
        import numpy as np
        mem = self._memory(tmp_path)
        mem.remember_appearance(1, 1, np.ones(96, dtype="float32") / 9.798,
                                1.0, 2.0, 5)
        result = mem.search_appearance_vectors(np.ones(512, dtype="float32"))
        assert result == []      # refuses rather than comparing nonsense

    def test_report_warns_before_brute_force_gets_slow(self, tmp_path):
        from backend.services.perception import SqliteVectorStore
        mem = self._memory(tmp_path)
        report = mem.report()
        assert report["vector_backend"]["backend"] == "sqlite"
        assert "warning" not in report["vector_backend"], \
            "an empty store must not warn"
        assert SqliteVectorStore.SCALE_WARNING_AT <= 50_000, \
            "the warning must fire before a query exceeds ~200 ms"

    def test_qdrant_is_not_claimed_without_a_live_server(self, tmp_path):
        """config says qdrant.enabled=true; nothing is listening.

        Trusting that flag is the same defect as a capability registry that
        advertises an OCR engine it cannot run. The backend must be decided by
        connecting, not by reading config.
        """
        from backend.config.config import get_config
        from backend.services.perception import QdrantVectorStore
        mem = self._memory(tmp_path)
        assert getattr(get_config().qdrant, "enabled", False) is True, \
            "this test is meaningless if config no longer claims qdrant"
        assert mem.vectors().name == "sqlite", \
            "must fall back to SQLite when no Qdrant server answers"

        # The choice must come from a real connection attempt, not from
        # hard-coding SQLite: if try_connect() were removed, a deployment that
        # DOES run Qdrant would silently keep brute-forcing millions of rows.
        assert QdrantVectorStore.try_connect() is None, \
            "no server is running here, so the probe must return None"
        probed = {"called": False}
        original = QdrantVectorStore.try_connect

        @classmethod
        def spy(cls, *a, **k):
            probed["called"] = True
            return original(*a, **k)

        QdrantVectorStore.try_connect = spy
        try:
            mem._vectors = None
            assert mem.vectors().name == "sqlite"
            assert probed["called"], \
                "the vector backend must be chosen by probing for Qdrant"
        finally:
            QdrantVectorStore.try_connect = original


class TestMemorySearch:
    """The two questions Phase 6 was accepted against."""

    def _memory(self, tmp_path):
        from backend.services.perception import PerceptionMemory
        return PerceptionMemory(db_path=str(tmp_path / "search.db"))

    def _unit(self, rng):
        import numpy as np
        v = np.asarray(rng.random(96), dtype="float32")
        return v / np.linalg.norm(v)

    def test_find_this_person_across_cameras(self, tmp_path):
        """Acceptance criterion 1."""
        import numpy as np
        from backend.services.perception import find_across_cameras
        mem = self._memory(tmp_path)
        rng = np.random.default_rng(5)
        person = self._unit(rng)
        now = time.time()

        mem.remember_appearance(1, 10, person, now - 600, now - 500, 40)
        mem.remember_appearance(2, 20,
                                person + 0.01 * rng.random(96).astype("float32"),
                                now - 380, now - 300, 35)
        mem.remember_appearance(3, 30, self._unit(rng), now - 200, now - 100, 20)

        out = find_across_cameras(1, 10, memory=mem, limit=5)
        assert out["count"] >= 1
        top = out["matches"][0]
        assert top["camera_id"] == 2, "the matching appearance must rank first"
        assert top["is_identification"] is False
        assert out["caveat"]

    def test_impossible_journeys_are_flagged_and_demoted(self, tmp_path):
        """Two cameras seeing matching clothes at once means two people."""
        import numpy as np
        from backend.services.perception import find_across_cameras
        mem = self._memory(tmp_path)
        rng = np.random.default_rng(9)
        person = self._unit(rng)
        now = time.time()

        mem.remember_appearance(1, 10, person, now - 600, now - 500, 40)
        # Overlapping in time on another camera - cannot be the same entity.
        mem.remember_appearance(2, 20, person, now - 590, now - 510, 30)
        # Later, plausible.
        mem.remember_appearance(3, 30, person, now - 300, now - 200, 30)

        out = find_across_cameras(1, 10, memory=mem, limit=5)
        assert out["matches"][0]["physically_plausible"] is True, \
            "a plausible match must outrank an impossible one"
        implausible = [m for m in out["matches"]
                       if not m["physically_plausible"]]
        assert implausible, "the overlapping sighting must still be reported"
        assert implausible[0]["plausibility"], "with a reason"

    def test_transit_plausibility_uses_distance_when_known(self):
        from backend.services.perception import transit_plausibility
        ok, why = transit_plausibility(120.0, 5000.0, "walk")
        assert ok is False and "2500" in why
        ok, why = transit_plausibility(120.0, 100.0, "walk")
        assert ok is True
        ok, why = transit_plausibility(60.0, 1000.0, "vehicle")
        assert ok is True, "a vehicle covers 1 km in 60 s"

    def test_plausibility_admits_when_it_could_not_check(self):
        """Reporting an unperformed check as a pass would be a quiet lie."""
        from backend.services.perception import transit_plausibility
        ok, why = transit_plausibility(300.0, None)
        assert ok is True
        assert "no camera distances" in why

    def test_what_happened_yesterday(self, tmp_path):
        """Acceptance criterion 2."""
        from backend.services.perception import Observation, recall
        mem = self._memory(tmp_path)
        now = time.time()
        midnight = time.mktime(time.localtime(now)[:3] + (0, 0, 0, 0, 0, -1))

        for ts, summary in ((midnight - 3600, "van stopped at the loading bay"),
                            (midnight - 7200, "person near the loading bay"),
                            (now - 60, "person in the car park")):
            mem.remember_observation(Observation(
                kind="dwell", summary=summary, camera_id=1, timestamp=ts,
                confidence=0.7, evidence=["duration 60s"]))

        yesterday = recall(when="yesterday", memory=mem)
        assert yesterday["count"] == 2, "only yesterday's rows"
        today = recall(when="today", memory=mem)
        assert today["count"] == 1

        combined = recall(text="loading bay", when="yesterday", memory=mem)
        assert combined["count"] == 2
        assert combined["query"]["period"] == "yesterday"

    def test_recall_reports_how_many_results_are_grounded(self, tmp_path):
        from backend.services.perception import Observation, recall
        mem = self._memory(tmp_path)
        mem.remember_observation(Observation(
            kind="dwell", summary="grounded claim", camera_id=1,
            confidence=0.8, evidence=["measured 40s"]))
        mem.remember_observation(Observation(
            kind="guess", summary="ungrounded claim", camera_id=1,
            confidence=0.9, evidence=[]))
        out = recall(memory=mem)
        assert out["count"] == 2
        assert out["grounded_count"] == 1, \
            "an evidence-free row must not be counted as grounded"
        assert "unsupported" in out["note"]

    def test_summary_reads_as_a_sentence(self, tmp_path):
        from backend.services.perception import Observation, summarise_period
        mem = self._memory(tmp_path)
        for _ in range(3):
            mem.remember_observation(Observation(
                kind="occupancy_anomaly", summary="unusually busy",
                camera_id=1, confidence=0.7, evidence=["z=3.4"]))
        out = summarise_period(camera_id=1, when="today", memory=mem)
        assert "3" in out["narrative"]
        assert "occupancy anomaly" in out["narrative"]

    def test_empty_period_says_so(self, tmp_path):
        from backend.services.perception import summarise_period
        out = summarise_period(when="yesterday", memory=self._memory(tmp_path))
        assert "Nothing was recorded" in out["narrative"]


class TestMemoryRetentionIsEnforced:
    """Appearance data is re-identifying; it must expire on its own clock."""

    def test_perception_tables_are_in_the_retention_policy(self):
        """The obligation this phase inherited from the last one.

        Phase 2 shipped with no persistence and a test that fails the moment
        perception writes to disk. Phase 6 writes to disk, so the retention
        policy had to grow in the same change - this asserts it did.
        """
        from backend.services.management.retention import _DEFAULTS
        assert "perception_observations_days" in _DEFAULTS
        assert "perception_appearances_days" in _DEFAULTS
        assert (_DEFAULTS["perception_appearances_days"]
                < _DEFAULTS["perception_observations_days"]), \
            ("re-identifying descriptors must expire sooner than the text "
             "observations that cite them")

    def test_retention_actually_deletes_expired_rows(self, tmp_path):
        from backend.services.perception import Observation, PerceptionMemory
        import numpy as np
        mem = PerceptionMemory(db_path=str(tmp_path / "ret.db"))
        now = time.time()
        old, recent = now - 400 * 86400, now - 60

        mem.remember_observation(Observation(
            kind="dwell", summary="ancient", camera_id=1, timestamp=old,
            confidence=0.8, evidence=["x"]))
        mem.remember_observation(Observation(
            kind="dwell", summary="recent", camera_id=1, timestamp=recent,
            confidence=0.8, evidence=["x"]))
        v = np.ones(96, dtype="float32") / 9.79795897
        mem.remember_appearance(1, 1, v, old, old, 5)
        mem.remember_appearance(1, 2, v, recent, recent, 5)

        removed = mem.purge_expired(observation_days=60, appearance_days=7)
        assert removed["observations"] == 1
        assert removed["appearances"] == 1
        assert mem.count("observations") == 1
        assert mem.count("appearances") == 1

    def test_a_wrong_retention_column_raises_instead_of_logging(self, tmp_path):
        """A purge that silently deletes nothing is worse than none at all.

        Found in this codebase: `anomalies` and `license_plates` were purged on
        a `timestamp` column that neither table has, so plate reads accumulated
        forever behind a swallowed log line.
        """
        import sqlite3
        from backend.services.management.retention import (RetentionPolicyError,
                                                           _purge_table)
        conn = sqlite3.connect(str(tmp_path / "r.db"))
        conn.execute("CREATE TABLE license_plates (id INTEGER, detected_at TEXT)")
        with pytest.raises(RetentionPolicyError):
            _purge_table(conn, "license_plates", "timestamp", 30)
        conn.close()

    def test_real_retention_pass_covers_every_table_without_error(self):
        """The regression guard for the column-name defect."""
        from backend.services.management.retention import run_retention_once
        results = run_retention_once()
        for table in ("anomalies", "license_plates", "perception_observations",
                      "perception_appearances"):
            assert table in results, f"{table} is not covered by retention"


class TestMemoryPipelineIntegration:
    """Memory must be wired into the live pipeline, not sit beside it."""

    def _pipeline(self, tmp_path, **kw):
        from backend.services.perception import PerceptionMemory
        import backend.services.perception.memory as memory_module
        from backend.services.perception import PerceptionPipeline
        memory_module._MEMORY = PerceptionMemory(
            db_path=str(tmp_path / "pipe.db"))
        return PerceptionPipeline(**kw), memory_module._MEMORY

    def _frame(self):
        import numpy as np
        rng = np.random.default_rng(2)
        return rng.integers(0, 255, (240, 320, 3), dtype=np.uint8)

    def test_descriptors_are_rate_limited_per_track(self, tmp_path):
        """Consecutive frames of one person are near-identical; sampling
        every frame costs 25x more for no extra discrimination."""
        pipeline, _ = self._pipeline(tmp_path, descriptor_interval_s=1.0)
        frame = self._frame()
        base = time.time()
        for i in range(20):
            pipeline.process(1, [{"class_name": "person", "confidence": 0.9,
                                  "bbox": [50, 50, 90, 170], "track_id": 1}],
                             frame, base + i * 0.1)
        samples = len(pipeline._descriptors.get((1, 1), []))
        assert 1 <= samples <= 3, \
            f"2 seconds at 1 Hz should sample ~2 descriptors, got {samples}"

    def test_descriptor_accumulation_is_bounded(self, tmp_path):
        """A camera watching a doorway all day must not grow without limit."""
        pipeline, _ = self._pipeline(tmp_path, descriptor_interval_s=0.0)
        frame = self._frame()
        base = time.time()
        for i in range(120):
            pipeline.process(1, [{"class_name": "person", "confidence": 0.9,
                                  "bbox": [50, 50, 90, 170], "track_id": 1}],
                             frame, base + i)
        assert len(pipeline._descriptors[(1, 1)]) <= 32

    def test_actionable_observations_reach_the_database(self, tmp_path):
        pipeline, mem = self._pipeline(tmp_path)
        from backend.services.perception import Observation
        pipeline._persist_observations(1, [
            Observation(kind="dwell", summary="grounded", camera_id=1,
                        confidence=0.8, evidence=["measured"]),
            Observation(kind="guess", summary="ungrounded", camera_id=1,
                        confidence=0.95, evidence=[]),
        ])
        stored = mem.query_observations()
        assert len(stored) == 1, \
            "an unsupported claim must not be stored as recorded fact"
        assert stored[0].summary == "grounded"

    def test_retired_tracks_persist_their_appearance(self, tmp_path):
        """Retirement is the last moment the track exists in RAM."""
        pipeline, mem = self._pipeline(tmp_path, descriptor_interval_s=0.0)
        frame = self._frame()
        base = time.time()
        for i in range(6):
            pipeline.process(1, [{"class_name": "person", "confidence": 0.9,
                                  "bbox": [50, 50, 90, 170], "track_id": 1}],
                             frame, base + i * 0.3)
        # Jump far enough ahead that the track is declared lost.
        pipeline._retire_if_due(base + 1000, interval_s=0.0)
        assert mem.count("appearances") >= 1, \
            "a retired track's appearance must survive it"
        assert mem.count("tracks") >= 1

    def test_memory_can_be_disabled(self, tmp_path):
        pipeline, mem = self._pipeline(tmp_path, enable_memory=False)
        frame = self._frame()
        base = time.time()
        for i in range(5):
            pipeline.process(1, [{"class_name": "person", "confidence": 0.9,
                                  "bbox": [50, 50, 90, 170], "track_id": 1}],
                             frame, base + i)
        assert pipeline.stats()["memory"]["enabled"] is False
        assert mem.count("observations") == 0

    def test_pipeline_survives_an_unwritable_memory(self, tmp_path):
        """A storage failure must degrade recall, not stop the frame."""
        from backend.services.perception import PerceptionPipeline
        import backend.services.perception.memory as memory_module

        class Broken:
            def remember_observation(self, *a, **k):
                raise RuntimeError("disk full")
            def remember_appearance(self, *a, **k):
                raise RuntimeError("disk full")
            def remember_track(self, *a, **k):
                raise RuntimeError("disk full")

        original = memory_module._MEMORY
        try:
            memory_module._MEMORY = Broken()
            pipeline = PerceptionPipeline()
            result = pipeline.process(
                1, [{"class_name": "person", "confidence": 0.9,
                     "bbox": [10, 10, 50, 130], "track_id": 1}], self._frame())
            assert result is not None
        finally:
            memory_module._MEMORY = original


class TestDocumentedApiSurfaceMatchesReality:
    """Endpoint counts in the README must be introspected, never guessed."""

    def test_readme_route_counts_are_accurate(self):
        """The documented API surface must match the code.

        Counts the API only. `GET /` is deliberately excluded: it exists as a
        route when the frontend is unbuilt and is replaced by a StaticFiles
        mount once `frontend/dist` is present, so including it made this test
        depend on whether the UI happened to be built - it passed in CI and
        failed on a developer machine, or vice versa. The documented number
        must describe the API, not the build state of the dashboard.
        """
        from backend.api.main import app
        ops = set()
        for route in app.routes:
            methods = getattr(route, "methods", None)
            if not methods or route.path == "/":
                continue
            for m in methods - {"HEAD", "OPTIONS"}:
                ops.add((route.path, m))
        v1 = {o for o in ops if o[0].startswith("/api/v1")}

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert f"{len(ops)} registered routes" in readme, \
            f"README must state {len(ops)} registered routes"
        assert f"{len(v1)} under `/api/v1`" in readme, \
            f"README must state {len(v1)} operations under /api/v1"

    def test_readme_protected_count_is_accurate(self):
        import inspect
        from backend.api.main import app
        public = 0
        total = 0
        for route in app.routes:
            methods = getattr(route, "methods", None)
            if not methods or not route.path.startswith("/api/v1"):
                continue
            for _ in methods - {"HEAD", "OPTIONS"}:
                total += 1
                src = ""
                if hasattr(route, "endpoint"):
                    try:
                        src = inspect.getsource(route.endpoint)
                    except (OSError, TypeError):
                        src = ""
                guarded = bool(getattr(route, "dependencies", [])) or \
                    "require_role" in src or "get_current_user" in src
                if not guarded:
                    public += 1

        readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
        assert f"**{total - public} require a token and {public} are public**" \
            in readme, (f"README must state {total - public} protected and "
                        f"{public} public /api/v1 operations")


# ═══════════════════════════════════════════════════════════════════════════
# Analysis coverage: everything presented to Argus must actually be analysed,
# and anything it cannot analyse must say so rather than fail silently.
# ═══════════════════════════════════════════════════════════════════════════


class TestConfiguredRulesAreImplemented:
    """A rule declared `enabled: true` must have code behind it.

    config.yaml advertised speed_violation, fall_detection and
    abandoned_object as enabled while the engine implemented only intrusion
    and loitering. Nothing warned; the three simply never fired. Configuration
    that promises analysis nobody performs is indistinguishable from a working
    system right up until the incident review.
    """

    def test_every_enabled_rule_is_implemented(self):
        from backend.services.management.rules_engine import get_rules_engine

        status = get_rules_engine().rule_status()
        unimplemented = [
            name for name, v in status.items()
            if v["configured_enabled"] and not v["implemented"]
        ]
        assert not unimplemented, (
            f"config.yaml enables rules with no implementation: {unimplemented}"
        )

    def test_rule_status_reports_blockers_rather_than_claiming_success(self):
        from backend.services.management.rules_engine import get_rules_engine

        status = get_rules_engine().rule_status()
        for name, v in status.items():
            if v["blockers"]:
                assert not v["can_fire"], (
                    f"{name} reports blockers {v['blockers']} yet claims it can fire"
                )

    def test_rule_config_keys_survive_parsing(self):
        """Pydantic must not silently discard per-rule tuning keys.

        RuleConfig was a closed model, so `classes`, `move_tolerance_px` and
        friends were parsed, dropped, and replaced by hard-coded defaults with
        no warning anywhere.
        """
        from backend.config.config import get_config, section_to_dict

        rules = get_config().rules
        abandoned = section_to_dict(rules.get("abandoned_object"))
        assert abandoned.get("classes"), (
            "abandoned_object.classes was dropped by the config parser"
        )
        speed = section_to_dict(rules.get("speed_violation"))
        assert speed.get("violation_margin"), (
            "speed_violation.violation_margin was dropped by the config parser"
        )


class TestSpeedRequiresCalibration:
    """A speed in km/h must come from a measurement, not a global guess."""

    @staticmethod
    def _engine():
        from backend.services.management.rules_engine import RulesEngine

        engine = RulesEngine()
        engine._created = []
        engine.event_store = type(
            "FakeStore", (),
            {"create_event": lambda self, **kw: engine._created.append(kw)},
        )()
        engine.zone_manager.get_zones_by_camera = lambda cid: []
        return engine

    @staticmethod
    def _drive(engine, camera_id, frames=6):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(frames):
            engine.process_detections(
                camera_id, frame,
                [{"class_name": "car", "confidence": 0.9,
                  "bbox": [i * 60, 200, i * 60 + 50, 260], "track_id": 77}],
                frame_time=1000.0 + i * 0.2,
            )

    def test_uncalibrated_camera_reports_no_speed(self):
        engine = self._engine()
        self._drive(engine, 1)
        assert not [e for e in engine._created
                    if e["rule_type"] == "speed_violation"], (
            "reported a speed violation on a camera with no ground-plane "
            "calibration - the km/h figure would be fabricated"
        )

    def test_calibrated_camera_reports_speed_with_evidence(self):
        from backend.services.management.calibration import CameraCalibration

        engine = self._engine()
        engine.calibration._cache[2] = CameraCalibration(
            2, meters_per_pixel=0.05, source="test"
        )
        self._drive(engine, 2)
        events = [e for e in engine._created
                  if e["rule_type"] == "speed_violation"]
        assert events, "calibrated camera failed to report a clear violation"
        meta = events[0]["metadata"]
        assert meta["speed_kmh"] > meta["threshold_kmh"]
        assert meta["meters_per_pixel"] == 0.05
        assert any("m/px" in e for e in meta["evidence"]), (
            "speed event must cite the calibration it relied on"
        )


class TestFallDetectionRefusesHeuristics:
    """A medical alert must not come from a bounding-box aspect ratio."""

    @staticmethod
    def _engine():
        from backend.services.management.rules_engine import RulesEngine

        engine = RulesEngine()
        engine._created = []
        engine.event_store = type(
            "FakeStore", (),
            {"create_event": lambda self, **kw: engine._created.append(kw)},
        )()
        engine.zone_manager.get_zones_by_camera = lambda cid: []
        return engine

    def _fire(self, engine, keypoints):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        engine.process_detections(
            1, frame,
            [{"class_name": "person", "confidence": 0.9,
              "bbox": [10, 10, 50, 120], "track_id": 4}],
            pose_results=[{
                "fall_detected": True, "num_keypoints": keypoints,
                "pose_class": "lying", "track_id": 4,
                "detection": [10, 10, 50, 120],
            }],
            frame_time=1000.0,
        )
        return [e for e in engine._created if e["rule_type"] == "fall_detection"]

    def test_no_keypoints_means_no_alert(self):
        assert not self._fire(self._engine(), 0), (
            "raised a fall alert from the aspect-ratio fallback, which cannot "
            "distinguish a fall from crouching or lying down"
        )

    def test_real_keypoints_raise_the_alert(self):
        events = self._fire(self._engine(), 17)
        assert events, "suppressed a fall backed by real pose keypoints"
        assert events[0]["priority"] == "high"


class TestAbandonmentNeedsAnAbsentOwner:
    """detect_abandonment takes owner_gone as an input; something must supply it."""

    @staticmethod
    def _run(owner_leaves, track_id):
        from backend.services.perception.pipeline import PerceptionPipeline

        pipeline = PerceptionPipeline(enable_memory=False)
        fired = []
        t0 = 1000.0
        for i in range(160):
            ts = t0 + i * 0.5
            dets = [{"class_name": "backpack", "confidence": 0.9,
                     "bbox": [300, 300, 340, 350], "track_id": track_id}]
            if (not owner_leaves) or ts < t0 + 5:
                dets.append({"class_name": "person", "confidence": 0.9,
                             "bbox": [300, 200, 350, 340],
                             "track_id": track_id + 500})
            for obs in pipeline.process(1, dets, frame=None,
                                        timestamp=ts).observations:
                if obs.kind == "abandoned_object":
                    fired.append(obs)
        return fired

    def test_abandonment_fires_once_the_owner_has_gone(self):
        fired = self._run(owner_leaves=True, track_id=1)
        assert fired, (
            "abandoned_object never fired: owner_gone was never computed, so "
            "this check was dead code everywhere in the system"
        )
        assert fired[0].evidence, "abandonment claim carries no evidence"

    def test_abandonment_stays_silent_while_the_owner_is_present(self):
        assert not self._run(owner_leaves=False, track_id=2), (
            "declared an object abandoned while its owner stood beside it"
        )


class TestObservationsReachOperators:
    """Perception that never reaches the event feed is perception nobody sees."""

    @staticmethod
    def _bridge():
        from backend.services.management.observation_events import (
            ObservationEventBridge,
        )

        created = []
        store = type("FakeStore", (),
                     {"create_event": lambda self, **kw: created.append(kw) or kw})()
        return ObservationEventBridge(event_store=store), created

    @staticmethod
    def _obs(kind="dwell", evidence=("duration 45s",), confidence=0.8, tracks=(7,)):
        from backend.services.perception.observation import Observation, Source

        return Observation(
            kind=kind, summary="test observation", confidence=confidence,
            source=Source.TEMPORAL.value, track_ids=list(tracks),
            evidence=list(evidence),
        )

    def test_every_observation_kind_has_a_declared_policy(self):
        """A new observation kind must not be silently dropped."""
        import re

        from backend.services.management.observation_events import KIND_POLICY

        emitted = set()
        for path in (PROJECT_ROOT / "backend/services/perception").glob("*.py"):
            emitted |= set(
                re.findall(r'kind="([a-z_]+)"', path.read_text(encoding="utf-8"))
            )
        missing = sorted(emitted - set(KIND_POLICY))
        assert not missing, (
            f"observation kinds with no event policy (silently dropped): {missing}"
        )

    def test_actionable_observation_becomes_an_event_with_its_evidence(self):
        bridge, created = self._bridge()
        bridge.promote(1, [self._obs()], frame=None)
        assert len(created) == 1, "actionable observation never reached the feed"
        assert created[0]["rule_type"] == "dwell"
        assert created[0]["metadata"]["evidence"] == ["duration 45s"], (
            "event dropped the evidence behind the claim"
        )

    def test_ungrounded_observation_is_never_promoted(self):
        bridge, created = self._bridge()
        bridge.promote(1, [self._obs(evidence=(), confidence=0.99)], frame=None)
        assert not created, (
            "promoted an observation with no evidence to an operator alert, "
            "however confident it claimed to be"
        )

    def test_repeat_observations_do_not_flood_the_feed(self):
        bridge, created = self._bridge()
        obs = self._obs()
        for _ in range(20):
            bridge.promote(1, [obs], frame=None)
        assert len(created) == 1, (
            f"a persisting condition produced {len(created)} events"
        )

    def test_bookkeeping_kinds_stay_out_of_the_feed(self):
        bridge, created = self._bridge()
        bridge.promote(1, [self._obs(kind="object_appeared")], frame=None)
        assert not created, "track bookkeeping was promoted to an operator alert"

    def test_dedup_table_is_bounded(self):
        bridge, _ = self._bridge()
        for i in range(6000):
            bridge.promote(1, [self._obs(tracks=(i,))], frame=None)
        assert len(bridge._recent) <= bridge._max_keys, (
            "dedup table grows without bound - one entry per track forever"
        )


class TestEventLifecycle:
    """An alert nobody can prove was reviewed is not an audit trail."""

    @staticmethod
    def _store():
        from backend.services.management.event_store import get_event_store

        return get_event_store()

    @staticmethod
    def _camera_id():
        """A camera that definitely exists.

        Hardcoding camera_id=2 passed locally and failed on a fresh clone:
        events.camera_id is a foreign key, and a newly initialised database has
        no cameras at all. A test that depends on data it did not create is
        testing the developer's machine.
        """
        from backend.database.db import get_db

        db = get_db()
        rows = db.fetchall("SELECT id FROM cameras LIMIT 1")
        if rows:
            row = rows[0]
            return row["id"] if isinstance(row, dict) else row[0]
        cursor = db.execute(
            "INSERT INTO cameras (name, rtsp_url, status) VALUES (?, ?, ?)",
            ("lifecycle-test-camera", "data/demo_clip.mp4", "inactive"),
        )
        return cursor.lastrowid

    def _event(self):
        return self._store().create_event(
            camera_id=self._camera_id(), rule_type="dwell",
            confidence=0.5, priority="low",
        )

    def test_new_events_start_in_a_known_state(self):
        from backend.services.management.event_store import EventStore

        assert self._event()["status"] in EventStore.ALLOWED_TRANSITIONS

    def test_acknowledgement_records_who_and_when(self):
        store = self._store()
        event = store.update_event_status(
            self._event()["id"], "acknowledged", actor="tester"
        )
        assert event["acknowledged_by"] == "tester"
        assert event["acknowledged_at"], "acknowledged without recording when"

    def test_cannot_skip_acknowledgement(self):
        store = self._store()
        with pytest.raises(ValueError):
            store.update_event_status(self._event()["id"], "resolved")

    def test_terminal_states_cannot_be_reopened(self):
        store = self._store()
        eid = self._event()["id"]
        store.update_event_status(eid, "acknowledged", actor="t")
        store.update_event_status(eid, "resolved", actor="t")
        with pytest.raises(ValueError):
            store.update_event_status(eid, "open")

    def test_unknown_status_is_rejected(self):
        store = self._store()
        with pytest.raises(ValueError):
            store.update_event_status(self._event()["id"], "aknowledged")


class TestSpeedAnalysisAccumulatesHistory:
    """Speed needs a stable identity across frames or it is structurally zero."""

    def test_stable_object_id_yields_a_real_speed(self):
        from backend.services.analytics.speed_height_analysis import (
            SpeedHeightAnalyzer,
        )

        analyzer = SpeedHeightAnalyzer()
        result = {}
        for i in range(8):
            result = analyzer.analyze_object(
                "cam1_track_7", [i * 30, 200, i * 30 + 50, 300],
                "person", 1000.0 + i * 0.2, (480, 640),
            )
        assert result["speed_mps"] > 0, (
            "speed is 0.0 for a subject that moved 210px in 1.4s - the object "
            "id is not stable across frames, so no history accumulates"
        )
        assert len(analyzer.tracks) == 1, (
            f"one moving subject produced {len(analyzer.tracks)} tracks"
        )

    def test_coordinator_uses_the_tracker_id(self):
        """The coordinator must not mint a throwaway id per detection.

        Asserted behaviourally rather than by scanning the source: the source
        legitimately *mentions* get_next_object_id in the comment explaining
        why it must not be used, so a text check would fail on the fix's own
        documentation.
        """
        from backend.services.analytics.speed_height_analysis import (
            SpeedHeightAnalyzer,
        )
        from backend.services.core_engine.processing_coordinator import (
            ProcessingCoordinator,
        )

        coordinator = ProcessingCoordinator()
        coordinator.speed_height_analyzer = SpeedHeightAnalyzer()
        results = []
        for i in range(8):
            results = coordinator._run_speed_height_analysis(
                1,
                [{"class_name": "person", "confidence": 0.9,
                  "bbox": [i * 30, 200, i * 30 + 50, 300], "track_id": 7}],
                1000.0 + i * 0.2, (480, 640),
            )
        assert len(coordinator.speed_height_analyzer.tracks) == 1, (
            "one subject over 8 frames produced "
            f"{len(coordinator.speed_height_analyzer.tracks)} analyzer tracks - "
            "the coordinator is minting a fresh object id per detection"
        )
        assert results and results[0]["speed_mps"] > 0, (
            "coordinator reported 0.0 m/s for a subject that clearly moved"
        )

    def test_cleanup_uses_the_frame_clock(self):
        """Wall-clock cleanup evicts everything when replaying archived footage."""
        from backend.services.analytics.speed_height_analysis import (
            SpeedHeightAnalyzer,
        )

        analyzer = SpeedHeightAnalyzer()
        analyzer.analyze_object(
            "cam1_track_1", [10, 200, 60, 300], "person", 1000.0, (480, 640)
        )
        analyzer.cleanup_old_tracks(1000.0)
        assert len(analyzer.tracks) == 1, (
            "cleanup compared a frame timestamp against time.time() and "
            "evicted the track on the frame it was created"
        )


class TestTripwiresFire:
    """Line zones were accepted by the API and then silently never evaluated."""

    def test_line_zone_produces_a_crossing_event(self):
        from backend.services.management.rules_engine import RulesEngine

        engine = RulesEngine()
        created = []
        engine.event_store = type(
            "FakeStore", (),
            {"create_event": lambda self, **kw: created.append(kw)},
        )()
        zone = {"id": 9, "name": "Gate", "type": "line",
                "coordinates": [[320, 0], [320, 480]], "camera_id": 1}
        engine.zone_manager.get_zones_by_camera = lambda cid: [zone]
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in (250, 290, 310, 330, 370):
            engine.process_detections(
                1, frame,
                [{"class_name": "person", "confidence": 0.9,
                  "bbox": [x - 20, 200, x + 20, 320], "track_id": 5}],
                frame_time=1000.0 + x,
            )
        crossings = [e for e in created if e["rule_type"] == "line_crossing"]
        assert crossings, (
            "a subject walked through a tripwire and nothing fired: "
            "zone_manager.is_point_in_zone has no branch for type 'line'"
        )
        assert crossings[0]["metadata"]["zone_id"] == 9, (
            "tripwire event lost the zone id, so it cannot be attributed"
        )


class TestLearningEngineIsFed:
    """/stats/learning must report the learner, not numbers borrowed elsewhere."""

    def test_learning_stats_come_from_the_learning_engine(self):
        import inspect

        from backend.api import main

        src = inspect.getsource(main.get_learning_stats)
        assert "get_adaptive_learning_engine" in src, (
            "/stats/learning relabels cross-camera tracker counts as behaviour "
            "profiles; the learning engine is never consulted"
        )

    def test_coordinator_feeds_the_learner(self):
        import inspect

        from backend.services.core_engine.processing_coordinator import (
            ProcessingCoordinator,
        )

        src = inspect.getsource(ProcessingCoordinator._run_speed_height_analysis)
        assert "learn_behavior" in src, (
            "adaptive_learning is imported but never called, so it can only "
            "ever report having learned nothing"
        )


class TestDocumentedPathsExist:
    """A path named in a doc must exist, or be explicitly marked conditional.

    Documentation drifts silently: modules were reorganised into core_engine/,
    analytics/, vision/, management/ and perception/, and nine docs kept
    pointing at the pre-restructure locations. A reader following those paths
    finds nothing and cannot tell whether the file moved or the feature was
    never built.
    """

    # Tokens that legitimately do not resolve to a repository file.
    ALLOWED_ABSENT = {
        "/openapi.json",          # a URL, not a file
        "../config/config.yaml",  # a documented search path
        "yolov8m.pt",             # documented as not bundled
        "data/qdrant/",           # created only when the service runs
        "data/kafka/",
        "data/streams/",
    }

    def test_every_documented_path_resolves(self):
        import re

        docs = sorted(
            list(PROJECT_ROOT.glob("*.md")) + list(PROJECT_ROOT.glob("docs/*.md"))
        )
        assert docs, "no markdown files found to check"

        real_names = set()
        real_paths = set()
        for path in PROJECT_ROOT.rglob("*"):
            text = str(path)
            if any(skip in text for skip in
                   ("/node_modules/", "/.git/", "__pycache__", "/dist/")):
                continue
            if path.is_file():
                real_names.add(path.name)
                real_paths.add(str(path.relative_to(PROJECT_ROOT)))

        pattern = re.compile(
            r"`([A-Za-z0-9_./-]+\.(?:py|md|yaml|yml|json|jsx|js|pt|sh|bat|command))`"
        )
        missing = {}
        for doc in docs:
            for match in pattern.finditer(doc.read_text(encoding="utf-8")):
                token = match.group(1)
                if token in self.ALLOWED_ABSENT:
                    continue
                if "/" in token:
                    if token in real_paths or any(
                        p.endswith(token) for p in real_paths
                    ):
                        continue
                elif token in real_names:
                    continue
                missing.setdefault(doc.name, set()).add(token)

        assert not missing, (
            "documentation references paths that do not exist: "
            + "; ".join(f"{k}: {sorted(v)}" for k, v in sorted(missing.items()))
        )


class TestAlertDelivery:
    """Events must actually be *sent*, not just recorded.

    The original defect: MQTTPublisher.publish_event() was fully implemented
    and mqtt.enabled was true in config, but grep showed no caller anywhere in
    the codebase. Every event Argus ever produced was delivered nowhere, and
    nothing reported that. These tests pin the delivery path and, just as
    importantly, the honesty of its reporting.
    """

    @staticmethod
    def _event(**kw):
        from datetime import datetime
        base = dict(id=1, camera_id=2, rule_type="dwell", priority="high",
                    confidence=0.8, timestamp=datetime.now(),
                    metadata={"evidence": ["duration 45s"]})
        base.update(kw)
        return base

    def test_webhook_actually_delivers_over_http(self):
        """A real HTTP server must receive a real request with the payload."""
        import json as _json
        import threading
        from http.server import BaseHTTPRequestHandler, HTTPServer
        from backend.services.management.notifications import (
            AlertPolicy, NotificationService, WebhookTransport,
        )

        received = []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                received.append(_json.loads(self.rfile.read(length)))
                self.send_response(202)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        port = server.server_address[1]
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            service = NotificationService(
                transports=[WebhookTransport(url=f"http://127.0.0.1:{port}/h")],
                policy=AlertPolicy(min_priority="low"),
                synchronous=True,
            )
            results = service.notify(self._event())
            assert results and results[0].delivered, (
                f"webhook did not deliver: {[r.to_dict() for r in results or []]}"
            )
            assert len(received) == 1, "server received no request"
            assert received[0]["rule_type"] == "dwell"
            # Evidence is the reason an operator trusts the alert; it must survive.
            assert received[0]["metadata"]["evidence"] == ["duration 45s"]
            # datetimes must be JSON-safe or delivery silently fails in prod.
            assert isinstance(received[0]["timestamp"], str)
        finally:
            server.shutdown()

    def test_failed_delivery_is_reported_not_swallowed(self):
        """An unreachable endpoint must report delivered=False with a reason."""
        from backend.services.management.notifications import WebhookTransport

        # Port 9 (discard) refuses connections.
        result = WebhookTransport(url="http://127.0.0.1:9/x", timeout_s=1.0).send(
            self._event()
        )
        assert result.delivered is False
        assert result.reason, "a failure with no reason is undiagnosable"

    def test_unavailable_channel_is_not_reported_as_sent(self):
        """Availability is probed live, never assumed from the config flag.

        mqtt.enabled: true with no broker running must NOT count as delivered.
        A stub publisher is used rather than the ambient broker state, because
        skipping when the channel looks available makes the test unable to
        fail - which is how a mutation deleting the liveness probe survived.
        """
        from backend.services.management.notifications import MqttTransport

        class DisconnectedPublisher:
            """Config says enabled; the socket says otherwise."""

            class config:
                class mqtt:
                    enabled = True
                    broker = "localhost"
                    port = 1883

            def is_connected(self):
                return False

            def publish_event(self, event):
                raise AssertionError(
                    "published to a broker that is not connected"
                )

        transport = MqttTransport(publisher=DisconnectedPublisher())
        available, reason = transport.available()
        assert available is False, (
            "mqtt.enabled=true was treated as proof of a live broker"
        )
        assert reason, "unavailability must explain itself"
        result = transport.send(self._event())
        assert result.delivered is False, (
            "reported an alert as sent with no broker connected"
        )

    def test_priority_floor_filters(self):
        from backend.services.management.notifications import AlertPolicy

        policy = AlertPolicy(min_priority="high")
        assert policy.evaluate(self._event(priority="low"))[0] is False
        assert policy.evaluate(self._event(priority="high"))[0] is True
        assert policy.evaluate(self._event(priority="critical"))[0] is True

    def test_deny_list_beats_allow_list(self):
        from backend.services.management.notifications import AlertPolicy

        policy = AlertPolicy(min_priority="low", rules_allow=["dwell"],
                             rules_deny=["dwell"])
        assert policy.evaluate(self._event(rule_type="dwell"))[0] is False

    def test_quiet_hours_wrap_midnight(self):
        """22:00-06:00 must suppress at 03:00 and allow at noon."""
        from datetime import datetime
        from backend.services.management.notifications import AlertPolicy

        policy = AlertPolicy(min_priority="low", quiet_hours={"dwell": [22, 6]})
        assert policy.evaluate(
            self._event(), now=datetime(2026, 1, 1, 3, 0))[0] is False
        assert policy.evaluate(
            self._event(), now=datetime(2026, 1, 1, 23, 0))[0] is False
        assert policy.evaluate(
            self._event(), now=datetime(2026, 1, 1, 12, 0))[0] is True

    def test_rate_limit_caps_a_flapping_camera(self):
        """One noisy camera must not exhaust a pager."""
        from backend.services.management.notifications import (
            AlertPolicy, NotificationService,
        )

        service = NotificationService(
            transports=[],
            policy=AlertPolicy(min_priority="low", rate_limit_per_minute=3),
            synchronous=True,
        )
        for i in range(10):
            service.notify(self._event(id=i))
        assert service.counters["suppressed_rate_limit"] == 7, service.counters

    def test_dropped_alerts_are_counted(self):
        """Suppression must be visible in status(), never silent."""
        from backend.services.management.notifications import (
            AlertPolicy, NotificationService,
        )

        service = NotificationService(
            transports=[], policy=AlertPolicy(min_priority="critical"),
            synchronous=True,
        )
        service.notify(self._event(priority="low"))
        status = service.status()
        assert status["counters"]["suppressed_by_policy"] == 1
        assert "channels" in status

    def test_event_creation_survives_a_broken_transport(self):
        """Recording the event is the guarantee; delivery is best-effort."""
        from backend.services.management.notifications import (
            AlertPolicy, NotificationService, Transport,
        )

        class Exploding(Transport):
            name = "exploding"

            def available(self):
                return True, "ok"

            def send(self, event):
                raise RuntimeError("transport is on fire")

        service = NotificationService(
            transports=[Exploding()], policy=AlertPolicy(min_priority="low"),
            synchronous=True,
        )
        results = service.notify(self._event())
        assert results and results[0].delivered is False
        assert "fire" in results[0].reason

    def test_create_event_actually_dispatches(self):
        """Creating a real event must reach the notification service.

        Asserting on source text is not enough here: the `import` line alone
        contains the function name, so a scan still passes after the call
        itself is deleted. Mutation testing caught exactly that, so this
        drives a real event through the real code path instead.
        """
        from backend.services.management import notifications
        from backend.services.management.event_store import get_event_store

        seen = []

        class Recorder:
            def notify(self, event):
                seen.append(event)
                return []

        original = notifications.get_notification_service
        notifications.get_notification_service = lambda: Recorder()
        try:
            camera_id = TestEventLifecycle._camera_id()
            event = get_event_store().create_event(
                camera_id=camera_id, rule_type="dwell", priority="high",
                confidence=0.9, metadata={"evidence": ["dispatch probe"]},
            )
        finally:
            notifications.get_notification_service = original

        assert event, "event was not created"
        assert seen, (
            "create_event recorded the event but never dispatched it - the "
            "original defect: every event delivered nowhere"
        )
        assert seen[0]["rule_type"] == "dwell"


class TestEvidenceClips:
    """Pre-event footage must exist, be bounded, and never be faked."""

    @staticmethod
    def _frames(n=30, seed=0):
        import numpy as np
        rng = np.random.default_rng(seed)
        return [rng.integers(0, 255, (120, 160, 3), dtype=np.uint8)
                for _ in range(n)]

    def test_clip_is_a_real_playable_file(self):
        import tempfile
        from pathlib import Path
        from backend.services.management.evidence_clips import EvidenceClipService

        out = Path(tempfile.mkdtemp())
        service = EvidenceClipService(seconds=5, fps=10, output_dir=out)
        for i, frame in enumerate(self._frames()):
            service.record(1, frame, timestamp=1000.0 + i * 0.1)

        result = service.write_clip(1, "intrusion", event_id=7)
        if not result.written and "encoder" in result.reason:
            pytest.skip(f"no mp4 encoder in this OpenCV build: {result.reason}")
        assert result.written, result.reason
        path = Path(result.path)
        assert path.exists() and path.stat().st_size > 0
        # A path to an unreadable file is worse than no clip.
        import cv2
        capture = cv2.VideoCapture(str(path))
        try:
            assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) > 1
        finally:
            capture.release()

    def test_buffer_is_bounded_in_bytes_not_just_frames(self):
        """Frame size varies ~40x with content, so a frame cap is not a memory cap."""
        from backend.services.management.evidence_clips import FrameRingBuffer

        # Generous frame cap, tiny byte cap: bytes must bind.
        buffer = FrameRingBuffer(seconds=100, fps=10, quality=90, max_mb=1.0)
        for i, frame in enumerate(self._frames(400, seed=3)):
            buffer.append(frame, timestamp=1000.0 + i * 0.1)
        stats = buffer.stats()
        assert stats["mb"] <= stats["max_mb"], stats
        assert stats["frames"] < 1000, "frame cap alone was doing the bounding"

    def test_unclipped_rule_is_declined_with_a_reason(self):
        import tempfile
        from pathlib import Path
        from backend.services.management.evidence_clips import EvidenceClipService

        service = EvidenceClipService(output_dir=Path(tempfile.mkdtemp()))
        result = service.write_clip(1, "scene_change")
        assert result.written is False
        assert result.path is None, "declined clip must not return a path"
        assert "clip_rules" in result.reason

    def test_empty_buffer_never_returns_a_path(self):
        import tempfile
        from pathlib import Path
        from backend.services.management.evidence_clips import EvidenceClipService

        service = EvidenceClipService(output_dir=Path(tempfile.mkdtemp()))
        result = service.write_clip(99, "intrusion")
        assert result.written is False and result.path is None

    def test_clips_have_retention(self):
        """New files on disk with no expiry is a disk-exhaustion bug."""
        import os
        import tempfile
        import time as _time
        from pathlib import Path
        from unittest import mock
        import backend.services.management.retention as retention

        root = Path(tempfile.mkdtemp())
        clips = root / "clips"
        clips.mkdir()
        snapshots = root / "snapshots"
        snapshots.mkdir()
        for i in range(10):
            path = clips / f"c{i}.mp4"
            path.write_bytes(b"x" * (1024 * 1024))
            if i < 5:
                old = _time.time() - 20 * 86400
                os.utime(path, (old, old))

        with mock.patch.object(retention, "resolve_path", lambda _: snapshots):
            assert retention.purge_clips(14) == 5
            # Time expiry alone cannot bound disk: enforce the ceiling too.
            assert retention.enforce_clip_size_cap(3) == 2
        survivors = sorted(p.name for p in clips.glob("*.mp4"))
        assert survivors == ["c7.mp4", "c8.mp4", "c9.mp4"], survivors

    def test_clip_retention_runs_in_the_scheduled_pass(self):
        """A purge function nobody calls does not bound anything.

        purge_clips() being correct is irrelevant if run_retention_once()
        never invokes it - which mutation testing showed this suite missed.
        """
        import inspect
        import backend.services.management.retention as retention

        source = inspect.getsource(retention.run_retention_once)
        assert "purge_clips(" in source and "enforce_clip_size_cap(" in source, (
            "clips are written but never expired by the retention pass"
        )
        results = retention.run_retention_once()
        assert "clips" in results and "clips_over_cap" in results, (
            f"retention pass does not report clips: {sorted(results)}"
        )

    def test_coordinator_records_frames(self):
        """A buffer nothing writes to is the dormant-code defect again."""
        import inspect
        from backend.services.core_engine import processing_coordinator

        source = inspect.getsource(processing_coordinator)
        # Both the swarm and legacy frame paths must buffer. Checking mere
        # presence let a mutation delete one path and still pass.
        assert source.count("self.evidence.record(") >= 2, (
            "a frame path does not fill the ring buffer, so events on that "
            "path can never have a clip"
        )


class TestVideoWallComponent:
    """Guards on the live video tile.

    These are source-level checks because the component is browser code with
    no JS test runner in this repo. They are deliberately narrow: each one
    pins a specific bug that was observed live, not a coding style.
    """

    TILE = PROJECT_ROOT / "frontend" / "src" / "components" / "CameraTile.jsx"

    def _src(self):
        assert self.TILE.is_file(), f"missing {self.TILE}"
        return self.TILE.read_text(encoding="utf-8")

    def test_websocket_effect_does_not_depend_on_the_draw_callback(self):
        """The bug: the stream reconnected ~20 times a second.

        `draw` is rebuilt by useCallback on every detection update. Listing it
        in the transport effect's dependency array tore the WebSocket down and
        rebuilt it on each repaint, so the socket almost never lived long
        enough to deliver a frame - measured as 20+ 'WebSocket stream opened'
        lines per second in the backend log, a blank overlay, and 4.4 fps
        instead of 13.6.
        """
        src = self._src()
        assert "}, [cameraId]);" in src, (
            "the transport effect must depend on cameraId alone"
        )
        assert "}, [cameraId, draw]);" not in src, (
            "transport effect depends on `draw`, which is recreated on every "
            "detection update - this reconnects the WebSocket continuously"
        )
        # A ref must both exist and be kept current, otherwise the transport
        # calls a draw closure captured on first render and overlays freeze.
        assert "drawRef = useRef(" in src, (
            "the transport must reach the latest draw through a ref rather "
            "than by re-subscribing"
        )
        assert "drawRef.current = draw" in src, (
            "the draw ref is never updated, so the transport would call a "
            "stale closure from the first render"
        )

    def test_overlay_accounts_for_letterboxing(self):
        """The bug: boxes were scaled by the element rect, not the image.

        The frame is rendered with object-fit: contain, so a 640x480 frame in a
        412x238 box is letterboxed. Scaling detections by the element size
        stretched every box across the black bars and offset it from the
        subject it was supposed to mark.
        """
        src = self._src()
        assert "Math.min(rect.width / img.naturalWidth" in src, (
            "overlay must derive a single contain-scale from both axes"
        )
        for token in ("const offX =", "const offY ="):
            assert token in src, (
                f"overlay must compute the letterbox margin ({token!r} missing)"
            )
        # The margins must actually be applied to the drawn coordinates, not
        # merely computed: mapping helpers are the only consumers.
        assert "offX + v * scale" in src and "offY + v * scale" in src, (
            "letterbox offsets are computed but never applied to coordinates"
        )

    def test_liveness_is_measured_not_inferred_from_socket_state(self):
        """An open socket does not mean frames are arriving.

        A stalled camera holds the connection open indefinitely; without a
        frame clock the tile would keep showing an old picture labelled LIVE.
        """
        src = self._src()
        assert "STALE_AFTER_MS" in src and "OFFLINE_AFTER_MS" in src, (
            "tile must degrade LIVE -> STALE -> OFFLINE on its own frame clock"
        )
        assert "lastFrameAt" in src, "tile must track when the last frame arrived"

    def test_reconnect_backs_off(self):
        """A fixed retry from every tile is a request flood against a down backend."""
        src = self._src()
        assert "retryRef" in src and "Math.min" in src, (
            "reconnect must back off rather than hammer a fixed interval"
        )


class TestProtectedEvidenceIsFetchedWithAuth:
    """Snapshots and clips are role-protected; <img>/<video> cannot carry a token."""

    SRC = PROJECT_ROOT / "frontend" / "src"

    def test_snapshots_are_not_requested_from_the_unauthenticated_mount(self):
        """The bug: a broken image icon sat next to real evidence.

        /snapshots is only mounted when auth is disabled. The event dialog
        pointed <img src="/snapshots/{file}"> at it, which 404s on every
        authenticated deployment. The real route is
        GET /api/snapshots/{camera_id}/{filename}.
        """
        feed = (self.SRC / "pages" / "EventFeed.jsx").read_text(encoding="utf-8")
        assert 'src={`/snapshots/' not in feed, (
            "event dialog requests the unauthenticated /snapshots mount, "
            "which 404s whenever auth is enabled"
        )
        api = (self.SRC / "services" / "api.js").read_text(encoding="utf-8")
        assert "snapshotAPI" in api and "/snapshots/${cameraId}/" in api, (
            "no authenticated snapshot fetch exists"
        )

    def test_clip_endpoint_is_fetched_as_an_authenticated_blob(self):
        api = (self.SRC / "services" / "api.js").read_text(encoding="utf-8")
        assert "fetchClip" in api and "responseType: 'blob'" in api, (
            "clips must be fetched through the authenticated client, not a "
            "bare <video src> which cannot send a bearer token"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Deployment portability
#
# These assert the properties that let Argus run on a machine nobody has logged
# into: a platform-assigned port, a mounted data disk, and a dashboard that can
# be hosted apart from the API. Each one, when broken, produces a deploy that
# fails on the user's server rather than in CI.
# ─────────────────────────────────────────────────────────────────────────────
class TestDeploymentPortability:
    ROOT = Path(__file__).resolve().parent.parent

    def test_entrypoint_binds_the_platform_assigned_port(self):
        """Render/Railway/Fly/Cloud Run inject $PORT and health-check it.

        Hardcoding 8000 makes the platform's probe fail against a perfectly
        healthy app, which surfaces as a deploy timeout with no error in the
        logs - among the most confusing failures to diagnose remotely.
        """
        ep = (self.ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        assert 'PORT="${PORT:-8000}"' in ep, "must read $PORT with a local default"
        assert '--port "$PORT"' in ep, "uvicorn must bind the resolved $PORT"

    def test_entrypoint_execs_uvicorn_for_signal_delivery(self):
        """Without exec the shell stays PID 1 and never forwards SIGTERM, so
        every deploy waits out the platform's kill timeout."""
        ep = (self.ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        assert "exec python -m uvicorn" in ep

    def test_entrypoint_refuses_to_start_without_a_signing_key(self):
        """An ephemeral JWT secret signs everyone out on each restart. That is
        an intermittent bug users blame on the browser, so fail loudly."""
        ep = (self.ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        assert 'if [ -z "${ARGUS_JWT_SECRET:-}" ]' in ep
        assert "exit 1" in ep

    def test_mutable_state_is_redirectable_to_a_mounted_disk(self):
        """Container filesystems are ephemeral. Both the SQLite database and
        the snapshot directory must follow ARGUS_DATA_DIR or a redeploy wipes
        users, cameras and recorded evidence."""
        cfg = (self.ROOT / "config" / "config.yaml").read_text(encoding="utf-8")
        assert "${ARGUS_DATA_DIR:-data}/snapshots" in cfg
        assert "sqlite:///${ARGUS_DATA_DIR:-data}/argus.db" in cfg

    def test_data_dir_env_var_actually_relocates_state(self):
        """Interpolation is only useful if the loader applies it. Asserts the
        resolved values change, not merely that the placeholder is present."""
        import importlib
        import backend.config.config as cfgmod

        prev = os.environ.get("ARGUS_DATA_DIR")
        try:
            os.environ["ARGUS_DATA_DIR"] = "/mnt/argus-test"
            importlib.reload(cfgmod)
            cfg = cfgmod.get_config()
            assert cfg.system.snapshot_dir == "/mnt/argus-test/snapshots"
            # Four slashes: sqlite:/// plus an absolute /mnt path.
            assert cfg.database.url == "sqlite:////mnt/argus-test/argus.db"
        finally:
            if prev is None:
                os.environ.pop("ARGUS_DATA_DIR", None)
            else:
                os.environ["ARGUS_DATA_DIR"] = prev
            importlib.reload(cfgmod)
            cfgmod.get_config()

    def test_frontend_api_origin_is_build_time_configurable(self):
        """A static host (Vercel/Netlify) cannot run the Python API, so the
        dashboard must be buildable against a backend on another origin."""
        api = (self.ROOT / "frontend" / "src" / "services" / "api.js").read_text(
            encoding="utf-8"
        )
        assert "VITE_API_ORIGIN" in api
        assert "${API_ORIGIN}/api/v1" in api

    def test_websocket_url_follows_the_api_origin(self):
        """The regression this guards: deriving the socket host from
        window.location unconditionally. On a split deploy the browser then
        opens a socket back at the static host, which speaks no WebSocket, and
        the video wall is permanently dead with a confusing console error.
        """
        api = (self.ROOT / "frontend" / "src" / "services" / "api.js").read_text(
            encoding="utf-8"
        )
        idx = api.index("export const buildStreamUrl")
        # Slice to the end of the function, not a fixed character count: a
        # window that stops short can pass or fail on comment length alone.
        body = api[idx : api.index("\n};", idx)]
        assert "API_ORIGIN" in body, "stream URL must consider API_ORIGIN"
        assert "new URL(API_ORIGIN" in body, "must resolve the configured origin"
        # https pages must not open ws:// - the browser blocks it as mixed content.
        assert "'https:'" in body and "wss:" in body

    def test_snapshot_client_honours_the_api_origin(self):
        """snapshotAPI overrides baseURL (it lives under /api, not /api/v1). A
        hardcoded '/api' there breaks evidence on a split deploy even when
        every other call works - a subtle, partial failure."""
        api = (self.ROOT / "frontend" / "src" / "services" / "api.js").read_text(
            encoding="utf-8"
        )
        assert "baseURL: `${API_ORIGIN}/api`" in api
        assert "baseURL: '/api'" not in api

    def test_vercel_config_deploys_only_the_static_dashboard(self):
        """Vercel cannot host this backend: ~955 MB of torch/opencv against a
        500 MB function limit, no WebSocket server, and no persistent process
        for the retention thread. The config must not pretend otherwise."""
        vercel = json.loads((self.ROOT / "vercel.json").read_text(encoding="utf-8"))
        assert vercel["outputDirectory"] == "frontend/dist"
        assert "functions" not in vercel and "builds" not in vercel, (
            "no Python function may be declared - the backend cannot run here"
        )

    def test_single_worker_is_enforced(self):
        """Argus holds the coordinator, tracker state and a retention thread
        in-process and writes to SQLite. A second worker duplicates the thread
        and races on the database; events then vanish depending on which worker
        served the request, which reads as data loss rather than misconfig."""
        ep = (self.ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
        assert "--workers" not in ep.split("exec python -m uvicorn")[1], (
            "the serve command must not pass --workers"
        )

    def test_deploy_workflow_smoke_tests_the_image_before_publishing(self):
        """A published image that cannot boot fails on the user's server. The
        workflow must prove both the API and the dashboard respond."""
        wf = yaml.safe_load(
            (self.ROOT / ".github" / "workflows" / "deploy.yml").read_text(
                encoding="utf-8"
            )
        )
        steps = wf["jobs"]["build"]["steps"]
        smoke = [s for s in steps if "Smoke" in str(s.get("name", ""))]
        assert smoke, "deploy workflow must smoke-test the image"
        run = smoke[0]["run"]
        assert "/api/v1/health" in run, "must verify the API answers"
        assert "docker logs" in run, "must surface logs when the boot fails"

    def test_platform_configs_mount_a_disk_at_the_configured_data_dir(self):
        """A volume mounted anywhere other than ARGUS_DATA_DIR silently
        persists nothing: the app writes to the container filesystem while an
        empty disk sits alongside it."""
        render = yaml.safe_load((self.ROOT / "render.yaml").read_text(encoding="utf-8"))
        svc = render["services"][0]
        env = {e["key"]: e.get("value") for e in svc["envVars"]}
        assert svc["disk"]["mountPath"] == env["ARGUS_DATA_DIR"]

        fly = tomllib.loads((self.ROOT / "fly.toml").read_text(encoding="utf-8"))
        assert fly["mounts"][0]["destination"] == fly["env"]["ARGUS_DATA_DIR"]


class TestOneCommandCoversDockerToo:
    """`start` and `stop` must mean the same thing in every runtime.

    The launcher picks Docker automatically when the daemon is responding, so a
    Docker user and a native user run the identical command and must get the
    identical result: a working dashboard, and nothing left running afterwards.
    """

    ROOT = Path(__file__).resolve().parent.parent

    def test_docker_mode_starts_the_image_that_contains_the_dashboard(self):
        """The regression this guards: the launcher pointed Docker mode at
        docker-compose.yml, whose backend image copies only backend/ and
        config/ - no frontend/dist. `start` then printed
        "Dashboard http://localhost:8000" while that URL returned
        {"dashboard": "not built"}. The production compose serves both.
        """
        src = (self.ROOT / "argus.py").read_text(encoding="utf-8")
        assert 'COMPOSE_FILE = ROOT / "docker-compose.prod.yml"' in src, (
            "docker mode must use the production compose file, whose image "
            "contains the built dashboard"
        )

    def test_the_compose_image_actually_builds_the_dashboard(self):
        """Asserting the filename is not enough - the image it builds must
        genuinely contain the UI, or the fix above is cosmetic."""
        compose = yaml.safe_load(
            (self.ROOT / "docker-compose.prod.yml").read_text(encoding="utf-8")
        )
        dockerfile = compose["services"]["argus"]["build"]["dockerfile"]
        text = (self.ROOT / dockerfile).read_text(encoding="utf-8")
        assert "npm run build" in text
        assert "/ui/dist ./frontend/dist" in text, (
            "the runtime stage must copy the built UI into the image"
        )

    def test_stop_tears_down_both_compose_files(self):
        """`stop` must mean nothing is left running. A user who previously ran
        the development stack would otherwise keep containers holding the port,
        and the next `start` reports a healthy server it does not manage."""
        src = (self.ROOT / "argus.py").read_text(encoding="utf-8")
        stop_fn = src[src.index("def cmd_stop") : src.index("def cmd_status")]
        assert "DEV_COMPOSE_FILE" in stop_fn, (
            "stop must also bring down the development compose stack"
        )

    def test_docker_start_passes_the_required_secret_through(self):
        """docker-compose.prod.yml guards ARGUS_JWT_SECRET with `:?`, which
        aborts the whole command when unset. The launcher writes a real secret
        to .env, so it must pass that environment to compose or the one-command
        path dies on a variable the user was never asked to set."""
        src = (self.ROOT / "argus.py").read_text(encoding="utf-8")
        fn = src[src.index("def start_docker") : src.index("def start_native")]
        assert "load_env_file()" in fn
        assert "env=env" in fn, "compose must receive the loaded environment"

    def test_host_port_is_configurable_so_a_busy_port_is_survivable(self):
        """The launcher falls back to the next free port when 8000 is taken. A
        hardcoded host port in compose would bind 8000 anyway and fail."""
        compose_text = (self.ROOT / "docker-compose.prod.yml").read_text(
            encoding="utf-8"
        )
        assert "${ARGUS_HOST_PORT:-8000}:8000" in compose_text
        src = (self.ROOT / "argus.py").read_text(encoding="utf-8")
        assert 'env["ARGUS_HOST_PORT"]' in src

    def test_makefile_targets_delegate_to_the_real_launcher(self):
        """make is a convenience, not a second implementation. Windows users
        have no make, so the logic must live in argus.py and the targets must
        stay thin - two divergent code paths is how one of them rots."""
        mk = (self.ROOT / "Makefile").read_text(encoding="utf-8")
        for target in ("start:", "stop:"):
            body = mk[mk.index(target) + len(target) :].split("\n\n")[0]
            assert "argus.py" in body, f"`make {target[:-1]}` must call argus.py"

    def test_makefile_generates_a_valid_secret_for_docker(self):
        """`docker compose up` fails outright without ARGUS_JWT_SECRET. The
        .env target must produce a key that satisfies the >=32 char rule, or
        the one-command Docker path stops at an error message."""
        mk = (self.ROOT / "Makefile").read_text(encoding="utf-8")
        assert ".env:" in mk
        env_target = mk[mk.index(".env:") :].split("\n\n")[0]
        assert "token_urlsafe(48)" in env_target, (
            "must generate a key comfortably over the 32-character minimum"
        )
        assert "docker-start: .env" in mk, (
            "the docker target must depend on .env so the key exists first"
        )
