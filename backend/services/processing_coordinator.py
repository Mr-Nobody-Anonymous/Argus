"""
Processing coordinator - ties together stream ingestion, inference, image enhancement,
face recognition, speed/height analysis, and rules engine
"""
import logging
import threading
import time
import cv2
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime
from backend.services.stream_ingestion import get_stream_ingestion
from backend.services.inference_engine import get_inference_engine
from backend.services.rules_engine import get_rules_engine
from backend.services.camera_manager import get_camera_manager
from backend.services.mqtt_publisher import get_mqtt_publisher
from backend.services.image_enhancement import get_image_enhancement
from backend.services.face_recognition import get_face_recognition
from backend.services.speed_height_analysis import get_speed_height_analyzer
from backend.services.license_plate_recognition import get_license_plate_recognition
from backend.services.anomaly_detector import get_anomaly_detector
from backend.services.pose_estimator import get_pose_estimator
from backend.services.deep_tracker import get_deep_tracker
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
        self.config = get_config()

        self.processing_threads: Dict[int, threading.Thread] = {}
        self.stop_flags: Dict[int, threading.Event] = {}
        self.running = False
        
        # Store enhanced analysis results per camera
        self.camera_analysis: Dict[int, Dict] = {}
        self.analysis_lock = threading.Lock()
        
        # Previous frame for motion anomaly detection
        self.prev_frames: Dict[int, np.ndarray] = {}

    def start_camera_processing(self, camera_id: int):
        """Start processing pipeline for a camera (or webcam with camera_id 999)"""
        if camera_id in self.processing_threads and self.processing_threads[camera_id].is_alive():
            logger.warning(f"Processing already running for camera {camera_id}")
            return

        # Handle webcam mode (camera_id 999)
        if camera_id == 999:
            # Get webcam ID from webcam_mode setting
            webcam_id = getattr(self, 'webcam_id', 0)
            rtsp_url = f"webcam://{webcam_id}"
        else:
            # Get camera details
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

    def _processing_loop(self, camera_id: int):
        """Main processing loop for a camera with all features"""
        logger.info(f"Processing loop started for camera {camera_id}")

        while not self.stop_flags[camera_id].is_set():
            try:
                # Get next frame
                frame_data = self.stream_ingestion.get_frame(camera_id, timeout=1.0)

                if frame_data is None:
                    continue

                frame, timestamp = frame_data
                frame_time = time.time()

                # Step 1: Image Enhancement (improves detection quality)
                enhanced_frame = self.image_enhancement.enhance_frame(frame, mode="auto")

                # Step 2: Run inference on enhanced frame
                detections = self.inference_engine.detect_objects(enhanced_frame)

                # Step 2b: Apply deep tracking for persistent IDs
                if self.deep_tracker.enabled:
                    detections = self.deep_tracker.update(detections, enhanced_frame)

                # Step 3: Process detections through rules engine (zone-based events)
                if detections:
                    self.rules_engine.process_detections(camera_id, enhanced_frame, detections)

                # Step 4: Face recognition on detections that are persons
                face_results = []
                if self.face_recognition.enabled and self.face_recognition.is_initialized():
                    face_results = self.face_recognition.recognize_faces(enhanced_frame)

                # Step 5: License Plate Recognition on vehicles
                lpr_results = []
                if self.license_plate_recognition.enabled:
                    lpr_results = self.license_plate_recognition.detect_plates(
                        enhanced_frame, detections
                    )

                # Step 6: Pose Estimation on persons
                pose_results = []
                if self.pose_estimator.enabled and self.pose_estimator._initialized:
                    for det in detections:
                        if det.get('class_name') == 'person':
                            pose = self.pose_estimator.estimate_pose(
                                enhanced_frame, det['bbox']
                            )
                            if pose:
                                pose['track_id'] = det.get('track_id', det.get('object_id', 'unknown'))
                                pose['detection'] = det['bbox']
                                pose_results.append(pose)

                # Step 7: Anomaly Detection
                anomalies = []
                prev_frame = self.prev_frames.get(camera_id)
                if self.anomaly_detector.enabled:
                    # Motion anomalies
                    motion_anomalies = self.anomaly_detector.detect_motion_anomalies(
                        enhanced_frame, prev_frame, detections
                    )
                    anomalies.extend(motion_anomalies)

                # Update previous frame
                self.prev_frames[camera_id] = enhanced_frame.copy()

                # Step 8: Speed and height analysis for all detections
                analysis_results = []
                for detection in detections:
                    object_id = f"cam{camera_id}_" + self.speed_height_analyzer.get_next_object_id()
                    analysis = self.speed_height_analyzer.analyze_object(
                        object_id=object_id,
                        bbox=detection['bbox'],
                        class_name=detection['class_name'],
                        frame_time=frame_time,
                        frame_shape=frame.shape[:2]
                    )
                    # Add track ID if available
                    if 'track_id' in detection:
                        analysis['track_id'] = detection['track_id']
                    analysis_results.append(analysis)

                    # Check for behavioral anomalies
                    if self.anomaly_detector.enabled:
                        behavior_anomaly = self.anomaly_detector.detect_behavior_anomalies(
                            object_id, analysis
                        )
                        if behavior_anomaly:
                            anomalies.append(behavior_anomaly)

                # Step 9: Merge face recognition and pose into analysis results
                if face_results:
                    for result in analysis_results:
                        for face in face_results:
                            if self._bbox_overlap(result['bbox'], face['bbox']) > 0.3:
                                result['face_recognition'] = {
                                    'person_name': face['person_name'],
                                    'is_known': face['is_known'],
                                    'face_confidence': face['confidence']
                                }

                # Store analysis results for API queries
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
                        'image_quality': self.image_enhancement.detect_quality_issues(frame)
                    }

                # Cleanup old tracking data periodically
                if int(frame_time) % 5 == 0:  # Every ~5 seconds
                    self.speed_height_analyzer.cleanup_old_tracks()

            except Exception as e:
                logger.error(f"Error in processing loop for camera {camera_id}: {e}")
                time.sleep(1)  # Prevent tight error loop

        logger.info(f"Processing loop stopped for camera {camera_id}")

    def _bbox_overlap(self, bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate IoU (Intersection over Union) of two bounding boxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        # Intersection
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)

        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)

        # Union
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
                'latest_analysis': {
                    'detection_count': len(analysis.get('detections', [])),
                    'face_count': len(analysis.get('face_results', [])),
                    'tracked_objects': analysis.get('analysis_results', []),
                    'image_quality': analysis.get('image_quality', {}),
                    'last_update': analysis.get('timestamp', 'never')
                }
            }
        return status

    def get_camera_analysis(self, camera_id: int) -> Optional[Dict]:
        """Get latest analysis data for a camera"""
        with self.analysis_lock:
            return self.camera_analysis.get(camera_id)

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
            # Test if webcam is available
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
