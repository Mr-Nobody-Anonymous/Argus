"""
Adaptive Learning Service for SentinelSight
Self-learning behavior patterns, emotion recognition, and anomaly detection
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Any
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json
import threading

try:
    import numpy as np
except ImportError:
    np = None

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from backend.config.config import get_config, section_to_dict
from backend.database.db import get_db

logger = logging.getLogger(__name__)


class AdaptiveLearningEngine:
    """
    Self-learning AI engine that adapts to:
    - Normal behavior patterns for each location/time
    - Typical emotion expressions for known persons
    - Environmental changes (lighting, crowd patterns)
    - Anomaly thresholds based on historical data
    """

    def __init__(self):
        self.config = get_config()
        self.db = get_db()
        self._initialized = False
        self.lock = threading.Lock()

        # Behavior pattern learning
        self.behavior_profiles: Dict[str, Dict] = {}  # person_id -> behavior patterns
        self.location_patterns: Dict[int, Dict] = {}  # camera_id -> typical patterns
        self.time_patterns: Dict[str, Dict] = {}  # hour -> typical activities

        # Emotion learning for known persons
        self.emotion_baseline: Dict[str, Dict[str, float]] = {}  # person_id -> {emotion: avg_conf}

        # Adaptive thresholds
        self.anomaly_thresholds: Dict[str, float] = {
            'speed_variation': 2.0,
            'crowd_density': 5.0,
            'direction_changes': 3,
        }

        # Learning buffer
        self.learning_buffer: deque = deque(maxlen=10000)

        if np is not None:
            self._initialize()

    def _initialize(self):
        """Initialize learning models"""
        try:
            # Load existing patterns from database
            self._load_patterns()

            # Initialize ML models if sklearn available
            if SKLEARN_AVAILABLE:
                self.scaler = StandardScaler()
                self.clustering_model = KMeans(n_clusters=5, random_state=42)
                self.is_fitted = False

            self._initialized = True
            logger.info("Adaptive learning engine initialized")
        except Exception as e:
            logger.error(f"Error initializing adaptive learning: {e}")
            self._initialized = False

    def _load_patterns(self):
        """Load learned patterns from database"""
        try:
            # Load behavior profiles
            rows = self.db.fetchall("SELECT * FROM behavior_profiles")
            for row in rows:
                self.behavior_profiles[row['person_id']] = json.loads(row['patterns'])

            logger.info(f"Loaded {len(self.behavior_profiles)} behavior profiles")
        except Exception as e:
            logger.debug(f"No existing patterns found: {e}")

    # ==================== Behavior Pattern Learning ====================

    def learn_behavior(self, object_id: str, behavior_features: Dict):
        """
        Learn and update behavior patterns for an object.

        Features tracked:
        - Average speed
        - Typical directions
        - Common zones visited
        - Time of activity
        """
        with self.lock:
            self.learning_buffer.append({
                'object_id': object_id,
                'features': behavior_features,
                'timestamp': datetime.now().isoformat()
            })

            # Update behavior profile
            if object_id not in self.behavior_profiles:
                self.behavior_profiles[object_id] = {
                    'speeds': [],
                    'directions': [],
                    'zones': [],
                    'hours': [],
                    'total_observations': 0
                }

            profile = self.behavior_profiles[object_id]
            profile['total_observations'] += 1

            # Update speed history
            if 'speed_mps' in behavior_features:
                profile['speeds'].append(behavior_features['speed_mps'])
                if len(profile['speeds']) > 100:
                    profile['speeds'] = profile['speeds'][-100:]

            # Update direction history
            if 'direction' in behavior_features:
                profile['directions'].append(behavior_features['direction'])
                if len(profile['directions']) > 100:
                    profile['directions'] = profile['directions'][-100:]

            # Update time patterns
            hour = datetime.now().hour
            profile['hours'].append(hour)
            if len(profile['hours']) > 100:
                profile['hours'] = profile['hours'][-100:]

    def get_adaptive_threshold(self, object_id: str, metric: str) -> float:
        """Get adaptive threshold based on learned behavior"""
        if object_id in self.behavior_profiles:
            profile = self.behavior_profiles[object_id]
            if metric == 'speed' and len(profile['speeds']) > 10:
                mean_speed = np.mean(profile['speeds'])
                std_speed = np.std(profile['speeds'])
                return mean_speed + 2 * std_speed
            if metric == 'direction_changes' and len(profile['directions']) > 10:
                return len(profile['directions']) * 0.3

        return self.anomaly_thresholds.get(metric, 2.0)

    # ==================== Emotion Learning ====================

    def learn_emotion(self, person_id: str, emotions: Dict[str, float]):
        """
        Learn baseline emotions for a person.

        Creates adaptive thresholds for detecting unusual emotional states.
        """
        with self.lock:
            if person_id not in self.emotion_baseline:
                self.emotion_baseline[person_id] = {}

            baseline = self.emotion_baseline[person_id]

            for emotion, confidence in emotions.items():
                if emotion not in baseline:
                    baseline[emotion] = {'values': [], 'mean': 0, 'std': 0}

                baseline[emotion]['values'].append(confidence)
                if len(baseline[emotion]['values']) > 50:
                    baseline[emotion]['values'] = baseline[emotion]['values'][-50:]

                if len(baseline[emotion]['values']) > 5 and np is not None:
                    baseline[emotion]['mean'] = np.mean(baseline[emotion]['values'])
                    baseline[emotion]['std'] = np.std(baseline[emotion]['values'])

    def detect_unusual_emotion(self, person_id: str, current_emotions: Dict[str, float]) -> Optional[Dict]:
        """
        Detect emotions that deviate from learned baseline.

        Returns anomaly if person shows unusual emotional state.
        """
        if person_id not in self.emotion_baseline:
            return None

        anomalies = []
        baseline = self.emotion_baseline[person_id]

        for emotion, confidence in current_emotions.items():
            if emotion in baseline and baseline[emotion]['std'] > 0:
                mean = baseline[emotion]['mean']
                std = baseline[emotion]['std']

                # Check if current emotion is unusual (more than 2 std deviations)
                if abs(confidence - mean) > 2 * std:
                    anomalies.append({
                        'type': 'unusual_emotion',
                        'emotion': emotion,
                        'confidence': confidence,
                        'baseline_mean': mean,
                        'deviation': abs(confidence - mean) / std
                    })

        if anomalies:
            return {
                'type': 'emotion_anomaly',
                'confidence': 0.8,
                'person_id': person_id,
                'unusual_emotions': anomalies,
                'description': f'Unusual emotional state detected for person {person_id}'
            }

        return None

    # ==================== Pattern Recognition ====================

    def cluster_behavior_patterns(self) -> Dict[str, Any]:
        """
        Cluster behavior patterns to discover common activity types.

        Uses K-means clustering on behavior features.
        """
        if not SKLEARN_AVAILABLE or len(self.learning_buffer) < 50:
            return {}

        try:
            # Prepare feature vectors
            features = []
            for entry in self.learning_buffer:
                f = entry['features']
                features.append([
                    f.get('speed_mps', 0),
                    f.get('bbox_area', 0),
                ])

            X = np.array(features)

            # Fit clustering model
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            clusters = self.clustering_model.fit_predict(X_scaled)

            # Count cluster distribution
            cluster_counts = defaultdict(int)
            for c in clusters:
                cluster_counts[c] += 1

            return {
                'cluster_distribution': dict(cluster_counts),
                'total_patterns': len(features),
                'is_fitted': True
            }

        except Exception as e:
            logger.error(f"Error in pattern clustering: {e}")
            return {}

    def detect_anomalous_cluster(self, features: Dict) -> Optional[Dict]:
        """
        Detect if current features belong to an anomalous cluster.

        Returns anomaly if features don't match any learned cluster.
        """
        if not SKLEARN_AVAILABLE or not self.is_fitted:
            return None

        try:
            X = np.array([[
                features.get('speed_mps', 0),
                features.get('bbox_area', 0),
            ]])

            X_scaled = self.scaler.transform(X)
            cluster = self.clustering_model.predict(X_scaled)[0]
            distance = self.clustering_model.transform(X_scaled)[0][cluster]

            # If distance to cluster center is large, it's anomalous
            if distance > 5.0:
                return {
                    'type': 'cluster_anomaly',
                    'confidence': min(distance / 10.0, 1.0),
                    'cluster': int(cluster),
                    'distance': float(distance),
                    'description': 'Behavior pattern not matching learned clusters'
                }

            return None

        except Exception as e:
            logger.error(f"Error detecting cluster anomaly: {e}")
            return None

    # ==================== Time-based Pattern Learning ====================

    def learn_time_pattern(self, camera_id: int, hour: int, activity_count: int):
        """Learn typical activity patterns for each hour of day per camera"""
        if camera_id not in self.location_patterns:
            self.location_patterns[camera_id] = {'hourly': defaultdict(list)}

        self.location_patterns[camera_id]['hourly'][hour].append(activity_count)

    def detect_unusual_time_activity(self, camera_id: int, hour: int, current_count: int) -> Optional[Dict]:
        """Detect unusual activity for current time based on historical patterns"""
        if camera_id not in self.location_patterns:
            return None

        hourly_data = self.location_patterns[camera_id]['hourly'].get(hour, [])
        if len(hourly_data) < 10:
            return None

        try:
            mean = np.mean(hourly_data) if np is not None else 0
            std = np.std(hourly_data) if np is not None else 0

            if std > 0 and abs(current_count - mean) > 2 * std:
                return {
                    'type': 'unusual_time_activity',
                    'confidence': min(abs(current_count - mean) / (std + 1), 1.0),
                    'camera_id': camera_id,
                    'hour': hour,
                    'current_count': current_count,
                    'expected_range': f'{mean - std:.0f}-{mean + std:.0f}',
                    'description': f'Unusual activity count for hour {hour}'
                }
        except Exception as e:
            logger.error(f"Error in time pattern detection: {e}")

        return None

    def save_patterns(self):
        """Save learned patterns to database"""
        try:
            for person_id, patterns in self.behavior_profiles.items():
                # Update or insert behavior profile
                existing = self.db.fetchone(
                    "SELECT id FROM behavior_profiles WHERE person_id = ?", (person_id,)
                )

                if existing:
                    self.db.execute(
                        "UPDATE behavior_profiles SET patterns = ?, updated_at = ? WHERE person_id = ?",
                        (json.dumps(patterns), datetime.now(), person_id)
                    )
                else:
                    self.db.execute(
                        "INSERT INTO behavior_profiles (person_id, patterns, created_at) VALUES (?, ?, ?)",
                        (person_id, json.dumps(patterns), datetime.now())
                    )

            logger.info(f"Saved {len(self.behavior_profiles)} behavior profiles")
        except Exception as e:
            logger.warning(f"Could not save patterns (table may not exist yet): {e}")

    def get_learning_stats(self) -> Dict:
        """Get statistics about learned patterns"""
        return {
            'total_behavior_profiles': len(self.behavior_profiles),
            'total_emotion_baselines': len(self.emotion_baseline),
            'learning_buffer_size': len(self.learning_buffer),
            'location_patterns': len(self.location_patterns),
            'sklearn_available': SKLEARN_AVAILABLE,
            'is_fitted': self.is_fitted if SKLEARN_AVAILABLE else False
        }


# Global instance
_adaptive_learning = None


def get_adaptive_learning_engine() -> AdaptiveLearningEngine:
    """Get global adaptive learning engine instance"""
    global _adaptive_learning
    if _adaptive_learning is None:
        _adaptive_learning = AdaptiveLearningEngine()
    return _adaptive_learning