"""
Processing coordinator - ties together stream ingestion, inference, image enhancement,
face recognition, speed/height analysis, and rules engine

Refactored into a Decentralized Multi-Agent Swarm Consortium:
  - YOLO Detection Agent (primary) → posts context to broker
  - Consortium Broker → resolves resource bids
  - Face Recognition Agent → runs if humans detected
  - LPR Agent → runs if vehicles detected
  - Fallback: linear pipeline when consortium is disabled
"""
import asyncio
import logging
import threading
import time
import cv2
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from backend.services.management.stream_ingestion import get_stream_ingestion
from backend.services.core_engine.inference_engine import get_inference_engine
from backend.services.management.rules_engine import get_rules_engine
from backend.services.management.camera_manager import get_camera_manager
from backend.services.management.mqtt_publisher import get_mqtt_publisher
from backend.services.vision.image_enhancement import get_image_enhancement
from backend.services.vision.face_recognition import get_face_recognition
from backend.services.analytics.speed_height_analysis import get_speed_height_analyzer
from backend.services.vision.license_plate_recognition import get_license_plate_recognition
from backend.services.analytics.anomaly_detector import get_anomaly_detector
from backend.services.vision.pose_estimator import get_pose_estimator
from backend.services.core_engine.deep_tracker import get_deep_tracker
from backend.services.core_engine.evolutionary_engine import get_evolutionary_engine
from backend.services.core_engine.consortium_broker import get_consortium_broker
from backend.services.core_engine.yolo_detection_agent import get_yolo_detection_agent
from backend.services.vision.face_recognition_agent import get_face_recognition_agent
from backend.services.vision.lpr_agent import get_lpr_agent
from backend.services.management.user_attention_tracker import get_user_attention_tracker
from backend.services.core_engine.logic_mutator import get_logic_mutator
from backend.services.management.state_recovery_manager import get_state_recovery_manager
from backend.config.config import get_config

logger = logging.getLogger(__name__)


class ProcessingCoordinator:
    def __init__(self):
        self.stream_ingestion = get_stream_ingestion()
        self.inference_engine = get_inference_engine()
        self.rules_engine = get_rules_engine()
        self.camera_manager = get_camera_manager()
        self.mqtt_publisher = get_mqtt_publisher()
        self.image_enhancement = get_image_enhancement()
        self.face_recognition = get_face_recognition()
        self.speed_height_analyzer = get_speed_height_analyzer()
        self.license_plate_recognition = get_license_plate_recognition()
        self.anomaly_detector = get_anomaly_detector()
        self.pose_estimator = get_pose_estimator()
        self.deep_tracker = get_deep_tracker()
        self.evolutionary_engine = get_evolutionary_engine()
        self.config = get_config()

        # ── Swarm Consortium Components ──
        self.consortium_broker = get_consortium_broker()
        self.yolo_agent = get_yolo_detection_agent()
        self.face_agent = get_face_recognition_agent()
        self.lpr_agent = get_lpr_agent()
        self.user_attention_tracker = get_user_attention_tracker()
        self.logic_mutator = get_logic_mutator()
        self.state_recovery = get_state_recovery_manager()
        self._swarm_enabled = (
            self.config.consortium.enabled
            and self.config.yolo_agent.enabled
        )

        self.processing_threads: Dict[int, threading.Thread] = {}
        self.stop_flags: Dict[int, threading.Event] = {}
        self.running = False

        # Recovery freeze flag — set by StateRecoveryManager during rollback
        self._recovery_freeze = False

        # Store enhanced analysis results per camera
        self.camera_analysis: Dict[int, Dict] = {}
        self.analysis_lock = threading.Lock()

        # Previous frame for motion anomaly detection
        self.prev_frames: Dict[int, np.ndarray] = {}

        # Frame timing for evolutionary metrics
        self._frame_timestamps: Dict[int, float] = {}

        # Counter for periodic evolutionary vector application
        self._evolutionary_apply_counter: Dict[int, int] = {}

        logger.info(
            f"ProcessingCoordinator initialized "
            f"(swarm_mode={self._swarm_enabled})"
        )

    def start_camera_processing(self, camera_id: int):
        """Start processing pipeline for a camera (or webcam with camera_id 999)"""
        if camera_id in self.processing_threads and self.processing_threads[camera_id].is_alive():
            logger.warning(f"Processing already running for camera {camera_id}")
            return

        # Handle webcam mode (camera_id 999)
        if camera_id == 999:
            webcam_id = getattr(self, 'webcam_id', 0)
            rtsp_url = f"webcam://{webcam_id}"
        else:
            camera = self.camera_manager.get_camera(camera_id)
            if not camera:
                logger.error(f"Camera {camera_id} not found")
                return
            rtsp_url = camera['rtsp_url']

        # Start stream ingestion
        self.stream_ingestion.start_camera(camera_id, rtsp_url)

        # Start processing thread
        self.stop_flags[camera_id] = threading.Event()
        thread = threading.Thread(
            target=self._processing_loop,
            args=(camera_id,),
            daemon=True
        )
        self.processing_threads[camera_id] = thread
        thread.start()

        logger.info(f"Started processing for camera {camera_id}")

    def stop_camera_processing(self, camera_id: int):
        """Stop processing pipeline for a camera"""
        if camera_id in self.stop_flags:
            self.stop_flags[camera_id].set()
        self.stream_ingestion.stop_camera(camera_id)
        logger.info(f"Stopped processing for camera {camera_id}")

    # ── Swarm Processing Loop ───────────────────────────────────────────

    def _processing_loop(self, camera_id: int):
        """Main processing loop — swarm mode or fallback linear mode."""
        logger.info(f"Processing loop started for camera {camera_id} (swarm={self._swarm_enabled})")

        while not self.stop_flags[camera_id].is_set():
            # Check for recovery freeze — pause processing during rollback
            if self._recovery_freeze:
                time.sleep(0.1)
                continue

            try:
                frame_data = self.stream_ingestion.get_frame(camera_id, timeout=1.0)
                if frame_data is None:
                    continue

                frame, timestamp = frame_data
                frame_time = time.time()
                self._frame_timestamps[camera_id] = frame_time

                # Apply evolutionary optimization vectors
                ee_config = self.config.evolutionary_engine
                if ee_config and ee_config.enabled:
                    self._apply_evolutionary_vectors(camera_id)

                # Step 1: Image Enhancement (always runs)
                enhanced_frame = self.image_enhancement.enhance_frame(frame, mode="auto")

                if self._swarm_enabled:
                    # ── SWARM MODE: Asymmetric, event-driven ──
                    self._swarm_process_frame(camera_id, enhanced_frame, frame, frame_time)
                else:
                    # ── FALLBACK MODE: Original linear pipeline ──
                    self._linear_process_frame(camera_id, enhanced_frame, frame, frame_time)

                # Cleanup old tracking data periodically
                if int(frame_time) % 5 == 0:
                    self.speed_height_analyzer.cleanup_old_tracks()

                # Register heartbeat with state recovery manager
                self.state_recovery.register_heartbeat()

            except Exception as e:
                logger.error(f"Error in processing loop for camera {camera_id}: {e}")
                self.state_recovery.increment_error_count()
                time.sleep(1)

        logger.info(f"Processing loop stopped for camera {camera_id}")

    def _swarm_process_frame(
        self, camera_id: int, enhanced_frame: np.ndarray,
        original_frame: np.ndarray, frame_time: float
    ):
        """
        Asymmetric swarm processing:
        1. YOLO Agent runs (always primary)
        2. YOLO posts context to broker (human_detected, vehicle_detected)
        3. Broker resolves bids from all agents
        4. Face/LPR agents run concurrently based on allocations
        5. All results merged into rules engine
        """
        # ── Step 2: YOLO Agent (primary, always runs) ──
        detections = self.yolo_agent.process_frame(enhanced_frame, camera_id)

        # Apply deep tracking
        if self.deep_tracker.enabled:
            detections = self.deep_tracker.update(detections, enhanced_frame)

        # ── Apply dynamic logic mutation filter (sandboxed) ──
        try:
            detections = self.logic_mutator.apply_filter(detections)
        except Exception as e:
            logger.warning(f"Logic mutator filter error (sandboxed fallback): {e}")

        # ── Post context to broker blackboard ──
        if self.consortium_broker.enabled:
            # Detect humans and vehicles from YOLO results
            human_count = sum(1 for d in detections if d.get('class_name') == 'person')
            vehicle_count = sum(
                1 for d in detections
                if d.get('class_name') in ('car', 'truck', 'bus', 'motorcycle', 'bicycle')
            )
            crowd_detected = 1.0 if human_count > 5 else (human_count / 5.0 if human_count > 0 else 0.0)

            self.consortium_broker.post_context(
                self.yolo_agent.AGENT_ID, self.yolo_agent.DOMAIN,
                "human_detected", min(1.0, human_count / 3.0), ttl=2.0
            )
            self.consortium_broker.post_context(
                self.yolo_agent.AGENT_ID, self.yolo_agent.DOMAIN,
                "vehicle_detected", min(1.0, vehicle_count / 2.0), ttl=2.0
            )
            self.consortium_broker.post_context(
                self.yolo_agent.AGENT_ID, self.yolo_agent.DOMAIN,
                "crowd_detected", crowd_detected, ttl=2.0
            )

        # ── Submit bids and resolve ──
        if self.consortium_broker.enabled and self.consortium_broker.resource_bidding_enabled:
            self.yolo_agent.submit_bid_to_broker()
            self.face_agent.submit_bid_to_broker()
            self.lpr_agent.submit_bid_to_broker()
            allocations = self.consortium_broker.resolve_cycle()

            # Apply allocations to each agent
            yolo_alloc = allocations.get(self.yolo_agent.AGENT_ID)
            face_alloc = allocations.get(self.face_agent.AGENT_ID)
            lpr_alloc = allocations.get(self.lpr_agent.AGENT_ID)

            if yolo_alloc:
                self.yolo_agent.apply_allocation(yolo_alloc)
            if face_alloc:
                self.face_agent.apply_allocation(face_alloc)
            if lpr_alloc:
                self.lpr_agent.apply_allocation(lpr_alloc)

        # ── Step 3: Process detections through rules engine ──
        if detections:
            self.rules_engine.process_detections(camera_id, enhanced_frame, detections)

        # ── Step 4: Face Agent (runs if allocation allows) ──
        face_results = []
        face_allocation = self.consortium_broker.get_allocation(self.face_agent.AGENT_ID)
        if face_allocation.should_process:
            try:
                face_results = self.face_agent.process_frame(enhanced_frame, camera_id)
            except Exception as e:
                logger.warning(f"Face agent error (sandboxed): {e}")

        # ── Step 5: LPR Agent (runs if allocation allows) ──
        lpr_results = []
        lpr_allocation = self.consortium_broker.get_allocation(self.lpr_agent.AGENT_ID)
        if lpr_allocation.should_process:
            try:
                lpr_results = self.lpr_agent.process_frame(enhanced_frame, camera_id, detections)
            except Exception as e:
                logger.warning(f"LPR agent error (sandboxed): {e}")

        # ── Step 6: Pose Estimation (always runs, lightweight) ──
        pose_results = []
        if self.pose_estimator.enabled and self.pose_estimator._initialized:
            try:
                for det in detections:
                    if det.get('class_name') == 'person':
                        pose = self.pose_estimator.estimate_pose(enhanced_frame, det['bbox'])
                        if pose:
                            pose['track_id'] = det.get('track_id', det.get('object_id', 'unknown'))
                            pose['detection'] = det['bbox']
                            pose_results.append(pose)
            except Exception as e:
                logger.warning(f"Pose estimation error: {e}")

        # ── Step 7: Anomaly Detection ──
        anomalies = []
        prev_frame = self.prev_frames.get(camera_id)
        if self.anomaly_detector.enabled:
            try:
                motion_anomalies = self.anomaly_detector.detect_motion_anomalies(
                    enhanced_frame, prev_frame, detections
                )
                anomalies.extend(motion_anomalies)
            except Exception as e:
                logger.warning(f"Anomaly detection error: {e}")
        self.prev_frames[camera_id] = enhanced_frame.copy()

        # ── Step 8: Speed and height analysis ──
        analysis_results = self._run_speed_height_analysis(
            camera_id, detections, frame_time, enhanced_frame.shape[:2]
        )

        # Check for behavioral anomalies
        if self.anomaly_detector.enabled:
            for analysis in analysis_results:
                try:
                    behavior_anomaly = self.anomaly_detector.detect_behavior_anomalies(
                        analysis.get('object_id', ''), analysis
                    )
                    if behavior_anomaly:
                        anomalies.append(behavior_anomaly)
                except Exception as e:
                    logger.warning(f"Behavior anomaly error: {e}")

        # ── Step 9: Merge face results into analysis ──
        if face_results:
            for result in analysis_results:
                for face in face_results:
                    if self._bbox_overlap(result['bbox'], face['bbox']) > 0.3:
                        result['face_recognition'] = {
                            'person_name': face['person_name'],
                            'is_known': face['is_known'],
                            'face_confidence': face['confidence']
                        }

        # ── Store results ──
        self._store_analysis(
            camera_id, frame_time, detections, face_results, lpr_results,
            pose_results, anomalies, analysis_results, original_frame
        )

        # ── Record evolutionary metrics ──
        self._record_evolutionary_metrics(
            camera_id, frame_time, detections, anomalies
        )

    def _linear_process_frame(
        self, camera_id: int, enhanced_frame: np.ndarray,
        original_frame: np.ndarray, frame_time: float
    ):
        """
        Original linear pipeline — used when consortium is disabled.
        Full backward compatibility with zero-cost fallback.
        """
        # Step 2: Run inference
        detections = self.inference_engine.detect_objects(enhanced_frame)

        # Step 2b: Apply deep tracking
        if self.deep_tracker.enabled:
            detections = self.deep_tracker.update(detections, enhanced_frame)

        # Step 2c: Apply dynamic logic mutation filter (sandboxed)
        try:
            detections = self.logic_mutator.apply_filter(detections)
        except Exception as e:
            logger.warning(f"Logic mutator filter error (sandboxed fallback): {e}")

        # Step 3: Rules engine
        if detections:
            self.rules_engine.process_detections(camera_id, enhanced_frame, detections)

        # Step 4: Face recognition
        face_results = []
        if self.face_recognition.enabled and self.face_recognition.is_initialized():
            face_results = self.face_recognition.recognize_faces(enhanced_frame)

        # Step 5: License Plate Recognition
        lpr_results = []
        if self.license_plate_recognition.enabled:
            lpr_results = self.license_plate_recognition.detect_plates(
                enhanced_frame, detections
            )

        # Step 6: Pose Estimation
        pose_results = []
        if self.pose_estimator.enabled and self.pose_estimator._initialized:
            for det in detections:
                if det.get('class_name') == 'person':
                    pose = self.pose_estimator.estimate_pose(enhanced_frame, det['bbox'])
                    if pose:
                        pose['track_id'] = det.get('track_id', det.get('object_id', 'unknown'))
                        pose['detection'] = det['bbox']
                        pose_results.append(pose)

        # Step 7: Anomaly Detection
        anomalies = []
        prev_frame = self.prev_frames.get(camera_id)
        if self.anomaly_detector.enabled:
            motion_anomalies = self.anomaly_detector.detect_motion_anomalies(
                enhanced_frame, prev_frame, detections
            )
            anomalies.extend(motion_anomalies)
        self.prev_frames[camera_id] = enhanced_frame.copy()

        # Step 8: Speed and height analysis
        analysis_results = self._run_speed_height_analysis(
            camera_id, detections, frame_time, enhanced_frame.shape[:2]
        )

        if self.anomaly_detector.enabled:
            for analysis in analysis_results:
                behavior_anomaly = self.anomaly_detector.detect_behavior_anomalies(
                    analysis.get('object_id', ''), analysis
                )
                if behavior_anomaly:
                    anomalies.append(behavior_anomaly)

        # Step 9: Merge face results
        if face_results:
            for result in analysis_results:
                for face in face_results:
                    if self._bbox_overlap(result['bbox'], face['bbox']) > 0.3:
                        result['face_recognition'] = {
                            'person_name': face['person_name'],
                            'is_known': face['is_known'],
                            'face_confidence': face['confidence']
                        }

        # Store results
        self._store_analysis(
            camera_id, frame_time, detections, face_results, lpr_results,
            pose_results, anomalies, analysis_results, original_frame
        )

        # Record evolutionary metrics
        self._record_evolutionary_metrics(
            camera_id, frame_time, detections, anomalies
        )

    # ── Shared Helper Methods ───────────────────────────────────────────

    def _run_speed_height_analysis(
        self, camera_id: int, detections: List[Dict],
        frame_time: float, frame_shape: tuple
    ) -> List[Dict]:
        """Run speed and height analysis on detections."""
        analysis_results = []
        for detection in detections:
            object_id = f"cam{camera_id}_" + self.speed_height_analyzer.get_next_object_id()
            analysis = self.speed_height_analyzer.analyze_object(
                object_id=object_id,
                bbox=detection['bbox'],
                class_name=detection['class_name'],
                frame_time=frame_time,
                frame_shape=frame_shape
            )
            if 'track_id' in detection:
                analysis['track_id'] = detection['track_id']
            analysis['object_id'] = object_id
            analysis_results.append(analysis)
        return analysis_results

    def _store_analysis(
        self, camera_id: int, frame_time: float,
        detections: List[Dict], face_results: List[Dict],
        lpr_results: List[Dict], pose_results: List[Dict],
        anomalies: List[Dict], analysis_results: List[Dict],
        original_frame: np.ndarray
    ):
        """Store analysis results for API queries."""
        with self.analysis_lock:
            self.camera_analysis[camera_id] = {
                'timestamp': datetime.now().isoformat(),
                'frame_time': frame_time,
                'detections': detections,
                'face_results': face_results,
                'lpr_results': lpr_results,
                'pose_results': pose_results,
                'anomalies': anomalies,
                'analysis_results': analysis_results,
                'image_quality': self.image_enhancement.detect_quality_issues(original_frame),
            }

    def _record_evolutionary_metrics(
        self, camera_id: int, frame_time: float,
        detections: List[Dict], anomalies: List[Dict]
    ):
        """Record per-frame metrics for the evolutionary engine."""
        ee_config = self.config.evolutionary_engine
        if ee_config and ee_config.enabled:
            processing_time = time.time() - frame_time
            inference_time_ms = getattr(self.inference_engine, '_last_inference_time', 0.0)
            tracking_accuracy = 1.0 if detections else 0.0
            false_positives = sum(1 for d in detections if d.get('confidence', 1.0) < 0.3)
            fp_ratio = false_positives / max(len(detections), 1)
            rule_precision = 1.0 if not anomalies else max(0.0, 1.0 - (len(anomalies) * 0.1))

            self.evolutionary_engine.record_frame_metrics(
                camera_id=camera_id,
                inference_time_ms=inference_time_ms,
                tracking_accuracy=tracking_accuracy,
                false_positive_ratio=fp_ratio,
                num_detections=len(detections),
                kafka_latency_ms=None,
                rule_precision=rule_precision,
                processing_time_ms=processing_time * 1000,
            )

    def _apply_evolutionary_vectors(self, camera_id: int):
        """
        Hot-swap active pipeline thresholds based on the evolutionary engine's
        current best optimization vector. Runs every N frames without
        restarting camera streams.
        """
        try:
            if camera_id not in self._evolutionary_apply_counter:
                self._evolutionary_apply_counter[camera_id] = 0
            self._evolutionary_apply_counter[camera_id] += 1

            if self._evolutionary_apply_counter[camera_id] % 30 != 0:
                return

            vector_dict = self.evolutionary_engine.get_optimization_vector()
            if not vector_dict:
                return

            yolo_conf = vector_dict.get("yolo_conf_threshold")
            if yolo_conf is not None:
                self.inference_engine.confidence_threshold = yolo_conf

            iou_threshold = vector_dict.get("iou_threshold")
            if iou_threshold is not None:
                self.inference_engine.iou_threshold = iou_threshold

            track_buffer = vector_dict.get("tracking_history_buffer")
            if track_buffer is not None and hasattr(self.deep_tracker, 'track_buffer'):
                self.deep_tracker.track_buffer = int(track_buffer)

            cooldown = vector_dict.get("rules_engine_cooldown")
            if cooldown is not None and hasattr(self.rules_engine, 'dedup_window'):
                self.rules_engine.dedup_window = timedelta(seconds=cooldown)

        except Exception as e:
            logger.warning(f"Failed to apply evolutionary vectors for camera {camera_id}: {e}")

    def _bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate IoU (Intersection over Union) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0

    def start_all_cameras(self):
        """Start processing for all cameras in database"""
        cameras = self.camera_manager.get_all_cameras()
        for camera in cameras:
            self.start_camera_processing(camera['id'])

    def stop_all_cameras(self):
        """Stop processing for all cameras"""
        for camera_id in list(self.stop_flags.keys()):
            self.stop_camera_processing(camera_id)

    def get_processing_status(self) -> Dict:
        """Get status of all processing pipelines"""
        status = {}
        for camera_id in self.processing_threads.keys():
            analysis = {}
            with self.analysis_lock:
                if camera_id in self.camera_analysis:
                    analysis = self.camera_analysis[camera_id]

            status[camera_id] = {
                'thread_alive': self.processing_threads[camera_id].is_alive(),
                'queue_depth': self.stream_ingestion.get_queue_depth(camera_id),
                'swarm_mode': self._swarm_enabled,
                'latest_analysis': {
                    'detection_count': len(analysis.get('detections', [])),
                    'face_count': len(analysis.get('face_results', [])),
                    'tracked_objects': analysis.get('analysis_results', []),
                    'image_quality': analysis.get('image_quality', {}),
                    'last_update': analysis.get('timestamp', 'never'),
                },
            }
        return status

    def get_camera_analysis(self, camera_id: int) -> Optional[Dict]:
        """Get latest analysis data for a camera"""
        with self.analysis_lock:
            return self.camera_analysis.get(camera_id)

    def get_latest_frame(self, camera_id: int) -> Optional[tuple]:
        """
        Get the latest raw video frame for a camera along with its active detections.
        
        This method is called by the WebSocket streaming endpoints to retrieve
        the current frame and detection data for real-time frontend rendering.
        
        Args:
            camera_id: Camera identifier
            
        Returns:
            Tuple of (frame: np.ndarray, detections: List[Dict]) or None if unavailable.
            Detections are dicts with keys: track_id, class_name, confidence, bbox.
        """
        try:
            # 1. Get latest frame from the stream ingestion queue (non-blocking)
            frame_data = self.stream_ingestion.get_frame(camera_id, timeout=0.05)
            if frame_data is None:
                return None
            frame, _ = frame_data
            
            # 2. Get latest detections from analysis cache
            detections = []
            with self.analysis_lock:
                analysis = self.camera_analysis.get(camera_id)
                if analysis:
                    detections = analysis.get('detections', [])
            
            return (frame, detections)
        
        except Exception as e:
            logger.error(f"Error in get_latest_frame for camera {camera_id}: {e}")
            return None

    def get_speed_stats(self) -> Dict:
        """Get speed statistics across all cameras"""
        return self.speed_height_analyzer.get_speed_stats()

    def is_webcam_mode(self) -> bool:
        """Check if webcam mode is active"""
        return hasattr(self, 'webcam_mode') and self.webcam_mode

    def enable_webcam_mode(self, camera_id: int = 0) -> bool:
        """Enable webcam mode for testing"""
        try:
            import cv2
            cap = cv2.VideoCapture(camera_id)
            if cap.isOpened():
                cap.release()
                self.webcam_mode = True
                self.webcam_id = camera_id
                logger.info(f"Webcam mode enabled with camera {camera_id}")
                return True
            return False
        except Exception as e:
            logger.error(f"Could not enable webcam mode: {e}")
            return False

    def disable_webcam_mode(self):
        """Disable webcam mode"""
        self.webcam_mode = False
        self.webcam_id = None
        logger.info("Webcam mode disabled")


# Global processing coordinator instance
_processing_coordinator = None


def get_processing_coordinator() -> ProcessingCoordinator:
    """Get global processing coordinator instance"""
    global _processing_coordinator
    if _processing_coordinator is None:
        _processing_coordinator = ProcessingCoordinator()
    return _processing_coordinator