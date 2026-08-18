"""
Rules engine for event generation based on detections and zones
"""
import logging
import cv2
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from collections import defaultdict, deque
from backend.services.management.zone_manager import get_zone_manager
from backend.services.management.event_store import get_event_store
from backend.services.management.calibration import (
    DEFAULT_VIOLATION_MARGIN,
    get_calibration_registry,
)
from backend.services.core_engine.inference_engine import get_inference_engine
from backend.config.config import get_config, resolve_path, section_to_dict

logger = logging.getLogger(__name__)


class RulesEngine:
    def __init__(self):
        self.zone_manager = get_zone_manager()
        self.event_store = get_event_store()
        self.inference_engine = get_inference_engine()
        self.config = get_config()
        
        # Track objects in zones for loitering detection
        self.zone_occupancy: Dict[int, Dict] = defaultdict(dict)  # {zone_id: {object_key: first_seen_time}}
        
        # Event deduplication
        self.recent_events: Dict[str, datetime] = {}  # {event_hash: timestamp}
        self.dedup_window = timedelta(seconds=5)
        
        # Snapshot directory
        self.snapshot_dir = resolve_path(self.config.system.snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        # RLock: _check_loitering_rule() calls _is_duplicate_event() while
        # already holding the lock, which would deadlock a plain Lock.
        self.lock = threading.RLock()

        # Scene-scoped rule state.
        self.calibration = get_calibration_registry()
        self._speed_history: Dict[tuple, deque] = {}
        self._stationary: Dict[tuple, Dict] = {}
        self._uncalibrated_warned: set = set()
        self._fall_inactive_logged = False

        # Tripwire support (line zones), previously dormant.
        from backend.services.management.zone_alerts import get_zone_alerts
        self.zone_alerts = get_zone_alerts()
        self._tripwire_zone_sig: Dict[int, tuple] = {}

    def process_detections(self, camera_id: int, frame, detections: List[Dict],
                           pose_results: Optional[List[Dict]] = None,
                           frame_time: Optional[float] = None):
        """Process detections and generate events based on rules.

        Zone rules (intrusion, loitering) require zones. Scene rules
        (speed_violation, fall_detection, abandoned_object) do not: they are
        properties of an object anywhere in view. Returning early when a camera
        has no zones - as this method used to - meant those three rules could
        never fire on any camera without a drawn zone, which is every camera by
        default.
        """
        if not detections:
            return

        # ── Zone-scoped rules ──
        zones = self.zone_manager.get_zones_by_camera(camera_id)
        if zones:
            # Tripwires. zone_manager.is_point_in_zone() has no branch for
            # type 'line', so a line zone was silently never matched by the
            # loop below and no tripwire could ever fire. zone_alerts.py
            # already implements segment-intersection crossing properly but
            # nothing imported it; this is that wiring.
            self._check_tripwires(camera_id, zones, detections, frame)

            for detection in detections:
                bbox = detection['bbox']
                center = self.inference_engine.get_bbox_center(bbox)

                for zone in zones:
                    if self.zone_manager.is_point_in_zone(center, zone):
                        # Object is in zone
                        self._check_intrusion_rule(
                            camera_id, zone, detection, frame
                        )
                        self._check_loitering_rule(
                            camera_id, zone, detection, frame
                        )

        # ── Scene-scoped rules: no zone required ──
        for detection in detections:
            self._check_speed_violation_rule(camera_id, detection, frame, frame_time)
        self._check_fall_detection_rule(camera_id, detections, pose_results, frame)
        self._check_abandoned_object_rule(camera_id, detections, frame, frame_time)

        # Clean up old zone occupancy data
        self._cleanup_zone_occupancy()
        self._cleanup_recent_events()
        self._cleanup_stationary(frame_time)

    def _check_intrusion_rule(self, camera_id: int, zone: Dict, detection: Dict, frame):
        """Check intrusion rule: object enters restricted zone"""
        rule_config = section_to_dict(self.config.rules.get('intrusion'))
        if not rule_config.get('enabled', True):
            return

        # Intrusion is an *entry* event: it should fire once when a subject
        # crosses into the zone, not every dedup window for as long as they
        # stand there. Keying the hash on the track ID gives one event per
        # person per entry; the class-name key it replaced collapsed every
        # person in the zone into a single "person" bucket that then re-fired
        # every 5 seconds forever (observed: 174 intrusion events in 24h).
        track_id = detection.get('track_id')
        subject = f"track_{track_id}" if track_id is not None else detection['class_name']
        event_hash = f"{camera_id}_{zone['id']}_intrusion_{subject}"

        if self._is_duplicate_event(event_hash):
            return

        # Generate event
        snapshot_path = self._save_snapshot(
            frame, detection['bbox'], camera_id, 'intrusion'
        )

        priority = rule_config.get('priority', 'high')
        
        metadata = {
            'zone_id': zone['id'],
            'zone_name': zone['name'],
            'inference_time_ms': self.inference_engine.get_avg_inference_time()
        }

        self.event_store.create_event(
            camera_id=camera_id,
            rule_type='intrusion',
            object_type=detection['class_name'],
            confidence=detection['confidence'],
            bbox=detection['bbox'],
            snapshot_path=snapshot_path,
            priority=priority,
            metadata=metadata
        )

        # Mark as recent event
        with self.lock:
            self.recent_events[event_hash] = datetime.now()

        logger.info(f"Intrusion event: {detection['class_name']} in zone '{zone['name']}' (camera {camera_id})")

    def _check_loitering_rule(self, camera_id: int, zone: Dict, detection: Dict, frame):
        """Check loitering rule: object remains in zone > threshold seconds"""
        rule_config = section_to_dict(self.config.rules.get('loitering'))
        if not rule_config.get('enabled', True):
            return

        threshold_seconds = rule_config.get('threshold_seconds', 30)
        
        # Only track persons for loitering
        if detection['class_name'] != 'person':
            return

        # Identify the loitering subject.
        #
        # Prefer the tracker's persistent ID: it follows a person as they move,
        # which is exactly what dwell-time measurement requires. The previous
        # grid-cell key (center // 50) was a stand-in from before tracking
        # worked - it treated every 50px cell as a separate "object", so a busy
        # scene produced one loitering event per occupied cell per dedup window
        # (observed: 377 loitering events in 24h from a single camera), while a
        # person who simply walked across cells never accumulated dwell time.
        track_id = detection.get('track_id')
        if track_id is not None:
            object_key = f"track_{track_id}"
        else:
            center = self.inference_engine.get_bbox_center(detection['bbox'])
            object_key = f"grid_{center[0]//50}_{center[1]//50}"
        
        zone_id = zone['id']
        current_time = datetime.now()

        with self.lock:
            if object_key not in self.zone_occupancy[zone_id]:
                # First time seeing this object in this zone
                self.zone_occupancy[zone_id][object_key] = current_time
                return
            
            # Check how long object has been in zone
            first_seen = self.zone_occupancy[zone_id][object_key]
            duration = (current_time - first_seen).total_seconds()

            if duration >= threshold_seconds:
                # Loitering detected
                event_hash = f"{camera_id}_{zone_id}_loitering_{object_key}"
                
                if self._is_duplicate_event(event_hash):
                    return

                # Generate event
                snapshot_path = self._save_snapshot(
                    frame, detection['bbox'], camera_id, 'loitering'
                )

                priority = rule_config.get('priority', 'medium')
                
                metadata = {
                    'zone_id': zone['id'],
                    'zone_name': zone['name'],
                    'duration_seconds': int(duration),
                    'inference_time_ms': self.inference_engine.get_avg_inference_time()
                }

                self.event_store.create_event(
                    camera_id=camera_id,
                    rule_type='loitering',
                    object_type=detection['class_name'],
                    confidence=detection['confidence'],
                    bbox=detection['bbox'],
                    snapshot_path=snapshot_path,
                    priority=priority,
                    metadata=metadata
                )

                # Mark as recent event
                self.recent_events[event_hash] = current_time
                
                # Reset tracking for this object
                del self.zone_occupancy[zone_id][object_key]

                logger.info(f"Loitering event: person in zone '{zone['name']}' for {int(duration)}s (camera {camera_id})")

    # ──────────────────────────────────────────────────────────────────
    # Scene-scoped rules
    #
    # config.yaml declared speed_violation, fall_detection and
    # abandoned_object as `enabled: true` while the engine implemented
    # neither, so all three were silently never evaluated. Config that
    # advertises a capability the code does not have is the same defect
    # class as a registry advertising an OCR engine it cannot run.
    # ──────────────────────────────────────────────────────────────────

    def _check_speed_violation_rule(self, camera_id: int, detection: Dict, frame,
                                    frame_time: Optional[float]):
        """Vehicle exceeds the configured speed threshold.

        Refuses to fire on an uncalibrated camera. Pixel displacement times a
        guessed metres-per-pixel constant is not a speed measurement, and an
        event asserting "47 km/h" from a guess is a fabricated fact. See
        calibration.py for the full reasoning.
        """
        rule_config = section_to_dict(self.config.rules.get('speed_violation'))
        if not rule_config.get('enabled', True):
            return

        vehicle_classes = rule_config.get(
            'classes', ['car', 'truck', 'bus', 'motorcycle', 'bicycle']
        )
        class_name = detection.get('class_name')
        if class_name not in vehicle_classes:
            return

        calibration = self.calibration.get(camera_id)
        if not calibration.is_calibrated:
            # Report the refusal once per camera rather than per frame.
            if camera_id not in self._uncalibrated_warned:
                self._uncalibrated_warned.add(camera_id)
                logger.info(
                    f"Speed violation rule inactive on camera {camera_id}: "
                    f"{calibration.reason()}"
                )
            return

        track_id = detection.get('track_id')
        if track_id is None:
            return  # speed needs a track; a single box has no velocity

        analysis = self._speed_for(camera_id, track_id, detection, frame, frame_time)
        if not analysis:
            return

        speed_px_s = analysis.get('speed_px_s')
        if not speed_px_s:
            return

        speed_mps = speed_px_s * calibration.meters_per_pixel
        speed_kmh = speed_mps * 3.6
        threshold_kmh = float(rule_config.get('threshold_kmh', 30))
        margin = float(rule_config.get('violation_margin', DEFAULT_VIOLATION_MARGIN))

        # Require a margin over the limit to absorb scalar-calibration error.
        if speed_kmh < threshold_kmh * margin:
            return

        event_hash = f"{camera_id}_speed_violation_track_{track_id}"
        if self._is_duplicate_event(event_hash):
            return

        snapshot_path = self._save_snapshot(
            frame, detection['bbox'], camera_id, 'speed_violation'
        )
        self.event_store.create_event(
            camera_id=camera_id,
            rule_type='speed_violation',
            object_type=class_name,
            confidence=detection.get('confidence'),
            bbox=detection['bbox'],
            snapshot_path=snapshot_path,
            priority=rule_config.get('priority', 'high'),
            metadata={
                'speed_kmh': round(speed_kmh, 1),
                'threshold_kmh': threshold_kmh,
                'margin_applied': margin,
                'meters_per_pixel': calibration.meters_per_pixel,
                'calibration_source': calibration.source,
                'track_id': track_id,
                'evidence': [
                    f"{speed_px_s:.1f} px/s over {analysis.get('samples', 0)} samples",
                    f"{calibration.meters_per_pixel:.4f} m/px ({calibration.source})",
                    f"{speed_kmh:.1f} km/h vs {threshold_kmh:.0f} km/h limit "
                    f"(x{margin} margin)",
                ],
            },
        )
        with self.lock:
            self.recent_events[event_hash] = datetime.now()
        logger.info(
            f"Speed violation: {class_name} at {speed_kmh:.1f} km/h "
            f"(limit {threshold_kmh}) on camera {camera_id}"
        )

    def _speed_for(self, camera_id: int, track_id, detection: Dict, frame,
                   frame_time: Optional[float]) -> Optional[Dict]:
        """Pixel-space speed for a track, measured over its recent history."""
        now = frame_time if frame_time is not None else time.time()
        x1, y1, x2, y2 = detection['bbox'][:4]
        centre = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
        key = (camera_id, track_id)

        with self.lock:
            history = self._speed_history.setdefault(key, deque(maxlen=10))
            history.append((now, centre))
            if len(history) < 3:
                return None
            t0, p0 = history[0]
            t1, p1 = history[-1]

        dt = t1 - t0
        if dt <= 0:
            return None
        dist = ((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2) ** 0.5
        return {'speed_px_s': dist / dt, 'samples': len(history)}

    def _check_fall_detection_rule(self, camera_id: int, detections: List[Dict],
                                   pose_results: Optional[List[Dict]], frame):
        """Person fall detected via pose estimation.

        Only fires on real pose output. MediaPipe is absent on this host, so the
        estimator falls back to a bbox aspect-ratio heuristic; that heuristic
        cannot tell a fallen person from someone crouching, lying down
        deliberately, or a low camera angle. Firing a high-priority medical
        alert from it would be guessing. When pose is unavailable the rule
        reports itself inactive rather than degrading silently.
        """
        rule_config = section_to_dict(self.config.rules.get('fall_detection'))
        if not rule_config.get('enabled', True):
            return

        if not pose_results:
            if not self._fall_inactive_logged:
                self._fall_inactive_logged = True
                logger.info(
                    f"Fall detection rule inactive: no pose keypoints available "
                    f"(camera {camera_id}). Requires MediaPipe; the bbox "
                    f"aspect-ratio fallback is not accurate enough to raise a "
                    f"medical alert."
                )
            return

        require_keypoints = rule_config.get('require_keypoints', True)

        for pose in pose_results:
            if not pose.get('fall_detected'):
                continue
            # A fall claim without keypoints came from the aspect-ratio
            # fallback. Do not promote a heuristic to an alert.
            num_kp = pose.get('num_keypoints', 0)
            if require_keypoints and num_kp <= 0:
                continue

            track_id = pose.get('track_id', 'unknown')
            event_hash = f"{camera_id}_fall_detection_{track_id}"
            if self._is_duplicate_event(event_hash):
                continue

            bbox = pose.get('detection') or pose.get('bbox')
            snapshot_path = (
                self._save_snapshot(frame, bbox, camera_id, 'fall_detection')
                if bbox else None
            )
            self.event_store.create_event(
                camera_id=camera_id,
                rule_type='fall_detection',
                object_type='person',
                confidence=pose.get('confidence', 0.7),
                bbox=bbox,
                snapshot_path=snapshot_path,
                priority=rule_config.get('priority', 'high'),
                metadata={
                    'pose_class': pose.get('pose_class'),
                    'gesture': pose.get('gesture'),
                    'num_keypoints': num_kp,
                    'track_id': track_id,
                    'evidence': [
                        f"pose classified as {pose.get('pose_class')}",
                        f"{num_kp} keypoints used",
                    ],
                },
            )
            with self.lock:
                self.recent_events[event_hash] = datetime.now()
            logger.warning(
                f"Fall detected: track {track_id} on camera {camera_id}"
            )

    def _check_abandoned_object_rule(self, camera_id: int, detections: List[Dict],
                                     frame, frame_time: Optional[float]):
        """Stationary object with no person nearby for longer than the threshold.

        The perception layer's detect_abandonment() requires `owner_gone` as an
        input it refuses to infer. This rule supplies exactly that fact from
        detection geometry: an object is abandoned when it has not moved and no
        person has been within the proximity radius for the threshold period.
        """
        rule_config = section_to_dict(self.config.rules.get('abandoned_object'))
        if not rule_config.get('enabled', True):
            return

        watch_classes = rule_config.get(
            'classes', ['backpack', 'handbag', 'suitcase', 'bottle', 'laptop']
        )
        threshold_s = float(rule_config.get('threshold_seconds', 60))
        move_tolerance = float(rule_config.get('move_tolerance_px', 25))
        proximity = float(rule_config.get('owner_proximity_px', 150))
        now = frame_time if frame_time is not None else time.time()

        people = [
            self.inference_engine.get_bbox_center(d['bbox'])
            for d in detections if d.get('class_name') == 'person'
        ]

        for det in detections:
            cls = det.get('class_name')
            if cls not in watch_classes:
                continue
            track_id = det.get('track_id')
            if track_id is None:
                continue

            centre = self.inference_engine.get_bbox_center(det['bbox'])
            key = (camera_id, track_id)
            owner_near = any(
                ((centre[0] - p[0]) ** 2 + (centre[1] - p[1]) ** 2) ** 0.5 <= proximity
                for p in people
            )

            with self.lock:
                state = self._stationary.get(key)
                if state is None or owner_near:
                    # Reset while an owner is present: the clock only runs once
                    # the object is genuinely unattended.
                    self._stationary[key] = {
                        'since': now, 'centre': centre, 'last_seen': now,
                        'owner_last_near': now if owner_near else
                                           (state or {}).get('owner_last_near', now),
                    }
                    continue

                moved = (
                    (centre[0] - state['centre'][0]) ** 2
                    + (centre[1] - state['centre'][1]) ** 2
                ) ** 0.5
                state['last_seen'] = now
                if moved > move_tolerance:
                    state['since'] = now
                    state['centre'] = centre
                    continue

                unattended_for = now - max(state['since'], state.get('owner_last_near', state['since']))

            if unattended_for < threshold_s:
                continue

            event_hash = f"{camera_id}_abandoned_object_track_{track_id}"
            if self._is_duplicate_event(event_hash):
                continue

            snapshot_path = self._save_snapshot(
                frame, det['bbox'], camera_id, 'abandoned_object'
            )
            self.event_store.create_event(
                camera_id=camera_id,
                rule_type='abandoned_object',
                object_type=cls,
                confidence=det.get('confidence'),
                bbox=det['bbox'],
                snapshot_path=snapshot_path,
                priority=rule_config.get('priority', 'medium'),
                metadata={
                    'unattended_seconds': int(unattended_for),
                    'threshold_seconds': threshold_s,
                    'track_id': track_id,
                    'evidence': [
                        f"{cls} stationary within {move_tolerance:.0f}px",
                        f"no person within {proximity:.0f}px for "
                        f"{int(unattended_for)}s (threshold {int(threshold_s)}s)",
                    ],
                },
            )
            with self.lock:
                self.recent_events[event_hash] = datetime.now()
            logger.warning(
                f"Abandoned {cls}: unattended {int(unattended_for)}s on camera {camera_id}"
            )

    def _cleanup_stationary(self, now: Optional[float] = None):
        """Forget objects that have left the scene, and bound the speed history.

        Without this both dicts grow once per track for the process lifetime -
        the unbounded-state defect already fixed elsewhere in this codebase.

        `now` MUST be the frame clock, not the wall clock. Comparing a frame
        timestamp from replayed footage against time.time() produces an age of
        decades, so every entry looked stale and was evicted on the very frame
        it was created - the abandoned-object rule could never accumulate the
        history it needs. Falling back to time.time() is only correct when the
        caller supplied no frame time at all.
        """
        if now is None:
            now = time.time()
        with self.lock:
            stale = [
                k for k, v in self._stationary.items()
                if now - v.get('last_seen', now) > 120
            ]
            for k in stale:
                del self._stationary[k]
            if len(self._speed_history) > 512:
                for k in list(self._speed_history)[:128]:
                    del self._speed_history[k]

    def _check_tripwires(self, camera_id: int, zones: List[Dict], detections, frame):
        """Line-crossing zones, via the previously dormant zone_alerts module."""
        rule_config = section_to_dict(self.config.rules.get('line_crossing')) or {}
        if not rule_config.get('enabled', True):
            return

        line_zones = [z for z in zones if z.get('type') == 'line']
        if not line_zones:
            return

        try:
            signature = tuple(sorted(z['id'] for z in line_zones))
            if self._tripwire_zone_sig.get(camera_id) != signature:
                self.zone_alerts.load_zones(camera_id, line_zones)
                self._tripwire_zone_sig[camera_id] = signature

            for ev in self.zone_alerts.check_zone_crossings(camera_id, detections):
                event_hash = f"{camera_id}_{ev.zone_id}_line_crossing_{ev.track_id}"
                if self._is_duplicate_event(event_hash):
                    continue
                snapshot_path = self._save_snapshot(
                    frame, list(ev.bbox), camera_id, 'line_crossing'
                )
                self.event_store.create_event(
                    camera_id=camera_id,
                    rule_type='line_crossing',
                    object_type=ev.object_type,
                    confidence=ev.confidence,
                    bbox=list(ev.bbox),
                    snapshot_path=snapshot_path,
                    priority=rule_config.get('priority', 'high'),
                    metadata={
                        'zone_id': ev.zone_id,
                        'zone_name': ev.zone_name,
                        'track_id': ev.track_id,
                        'evidence': [
                            f"track {ev.track_id} crossed tripwire "
                            f"'{ev.zone_name}'",
                        ],
                    },
                )
                with self.lock:
                    self.recent_events[event_hash] = datetime.now()
                logger.info(
                    f"Line crossing: {ev.object_type} track {ev.track_id} "
                    f"crossed '{ev.zone_name}' (camera {camera_id})"
                )
        except Exception as exc:  # noqa: BLE001 - never break the frame loop
            logger.error(f"Tripwire check failed on camera {camera_id}: {exc}")

    def _speed_blockers(self) -> List[str]:
        """Speed needs at least one calibrated camera to be able to fire."""
        try:
            cameras = self.zone_manager.db.execute("SELECT id FROM cameras")
            ids = [r[0] if not isinstance(r, dict) else r['id'] for r in cameras]
        except Exception:  # noqa: BLE001
            ids = sorted(self._uncalibrated_warned)
        uncalibrated = [
            cid for cid in ids if not self.calibration.get(cid).is_calibrated
        ]
        if ids and len(uncalibrated) == len(ids):
            return [
                f"no camera has ground-plane calibration "
                f"(uncalibrated: {uncalibrated}); pixel motion cannot be "
                f"converted to a real speed, so no speed is claimed"
            ]
        if uncalibrated:
            return [f"cameras without ground-plane calibration: {uncalibrated}"]
        return []

    def _fall_blockers(self) -> List[str]:
        """Fall detection needs real pose keypoints, probed at source."""
        try:
            from backend.services.vision.pose_estimator import get_pose_estimator

            pose = get_pose_estimator()
            if not getattr(pose, 'enabled', False):
                return ["pose estimation is disabled in config"]
            if not getattr(pose, '_initialized', False):
                return [
                    "no pose keypoints available (MediaPipe not installed); the "
                    "bbox aspect-ratio fallback cannot distinguish a fall from "
                    "crouching or lying down, so no medical alert is raised"
                ]
        except Exception as exc:  # noqa: BLE001
            return [f"pose estimator could not be probed: {exc}"]
        return []

    def rule_status(self) -> Dict[str, Dict]:
        """Which rules are configured, implemented, and actually able to fire.

        Exposed through the API so a "rule enabled" claim can be verified rather
        than trusted.
        """
        implemented = {
            'intrusion', 'loitering', 'speed_violation',
            'fall_detection', 'abandoned_object', 'line_crossing',
        }
        out: Dict[str, Dict] = {}
        rules = section_to_dict(self.config.rules) or {}
        for name, raw in rules.items():
            cfg = section_to_dict(raw) or {}
            entry = {
                'configured_enabled': bool(cfg.get('enabled', False)),
                'implemented': name in implemented,
                'priority': cfg.get('priority', 'medium'),
                'description': cfg.get('description', ''),
                'blockers': [],
            }
            # Blockers are *probed*, not accumulated from runtime side effects.
            # Reporting can_fire=True until the first frame happens to reveal a
            # problem is the same "trust the config" defect this method exists
            # to expose.
            if name == 'speed_violation':
                entry['blockers'].extend(self._speed_blockers())
            if name == 'fall_detection':
                entry['blockers'].extend(self._fall_blockers())
            entry['can_fire'] = (
                entry['configured_enabled'] and entry['implemented']
                and not entry['blockers']
            )
            out[name] = entry
        return out

    def _save_snapshot(self, frame, bbox: List[int], camera_id: int, rule_type: str) -> str:
        """Save snapshot with bounding box overlay"""
        try:
            # Draw bounding box on frame
            frame_copy = frame.copy()
            x1, y1, x2, y2 = bbox
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add timestamp
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            cv2.putText(
                frame_copy, timestamp_str,
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2
            )

            # Generate filename
            filename = f"cam{camera_id}_{rule_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            filepath = self.snapshot_dir / filename
            
            # Save image
            cv2.imwrite(str(filepath), frame_copy)
            
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Error saving snapshot: {e}")
            return ""

    def _is_duplicate_event(self, event_hash: str) -> bool:
        """
        Suppress repeats of an event that is still ongoing.

        The window slides: every suppressed sighting pushes the expiry forward,
        so a subject who stays in a zone produces exactly one event no matter
        how long they linger. The event only re-arms after the subject has been
        absent for a full dedup window.

        A fixed (non-sliding) window re-fired the same alert every 5 seconds for
        as long as the condition held, which is what buried the operator's event
        feed under hundreds of duplicates per camera per day.
        """
        with self.lock:
            last_time = self.recent_events.get(event_hash)
            if last_time is not None and datetime.now() - last_time < self.dedup_window:
                # Still ongoing - extend the suppression rather than expiring it.
                self.recent_events[event_hash] = datetime.now()
                return True
            return False

    def _cleanup_zone_occupancy(self):
        """Remove old zone occupancy data"""
        current_time = datetime.now()
        timeout = timedelta(seconds=60)  # Clear if not seen for 60s
        
        with self.lock:
            for zone_id in list(self.zone_occupancy.keys()):
                for object_key in list(self.zone_occupancy[zone_id].keys()):
                    first_seen = self.zone_occupancy[zone_id][object_key]
                    if current_time - first_seen > timeout:
                        del self.zone_occupancy[zone_id][object_key]

    def _cleanup_recent_events(self):
        """Remove old events from deduplication cache"""
        current_time = datetime.now()
        
        with self.lock:
            expired_keys = [
                k for k, v in self.recent_events.items()
                if current_time - v > self.dedup_window
            ]
            for key in expired_keys:
                del self.recent_events[key]


# Global rules engine instance
_rules_engine = None


def get_rules_engine() -> RulesEngine:
    """Get global rules engine instance"""
    global _rules_engine
    if _rules_engine is None:
        _rules_engine = RulesEngine()
    return _rules_engine
