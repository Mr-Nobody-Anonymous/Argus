"""
Person Re-identification service for tracking persons across multiple non-overlapping cameras.
Integrates deep-person-reid and fast-reid approaches.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple
from collections import deque
from datetime import datetime
import json
from ...config.config import get_config, section_to_dict
from ...database.db import get_db

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import numpy as np
except ImportError:
    np = None

logger = logging.getLogger(__name__)


class PersonReID:
    """
    Person re-identification using deep feature embeddings.
    
    Features:
    - Extract person features using ResNet backbone
    - Compare features across cameras
    - Find same person in different camera views
    - Track person movements across camera network
    """

    def __init__(self):
        self.config = get_config()
        reid_config = section_to_dict(getattr(self.config, 'reid', {}))
        self.enabled = reid_config.get('enabled', False)
        self.similarity_threshold = reid_config.get('similarity_threshold', 0.7)
        self.feature_dim = reid_config.get('feature_dim', 2048)

        self.db = get_db()
        self._initialized = False
        self.feature_extractor = None

        # Track registered person features
        self.person_features: Dict[str, np.ndarray] = {}  # {person_id: feature_vector}
        self.camera_persons: Dict[int, List[Dict]] = {}  # {camera_id: [{person_id, features, bbox, time}]}
        
        # Matching history for timeline reconstruction
        self.matching_history: List[Dict] = []

        if self.enabled and cv2 is not None and np is not None:
            self._initialize()
        elif self.enabled:
            self.enabled = False
            logger.warning("Person ReID dependencies unavailable, ReID disabled")

    def _initialize(self):
        """Initialize re-identification model"""
        try:
            # Try to use torch-based feature extractor
            import torch
            from torchvision import transforms
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Use a pre-trained model for feature extraction
            # Using torchvision's ResNet as a base
            from torchvision.models import resnet50
            self.model = resnet50(weights='DEFAULT')
            self.model.fc = torch.nn.Identity()  # Remove final classification layer
            self.model.to(self.device)
            self.model.eval()
            
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((256, 128)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
            ])
            
            self._initialized = True
            logger.info(f"Person ReID initialized on {self.device}")

        except Exception as e:
            logger.error(f"Error initializing person reid: {e}")
            self._initialized = False

    def extract_features(self, person_img: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract feature embedding from a person image.
        
        Args:
            person_img: Person crop from frame
        
        Returns:
            Feature vector (numpy array) or None
        """
        if not self.enabled or not self._initialized:
            return None

        if cv2 is None or np is None:
            return None

        try:
            # Preprocessing
            if person_img.shape[-1] == 3:
                rgb_img = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
            else:
                rgb_img = person_img

            # Transform and extract features
            input_tensor = self.transform(rgb_img).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.model(input_tensor)
            
            # Normalize features
            features = features.cpu().numpy()
            features = features / np.linalg.norm(features)
            
            return features.flatten()

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def compute_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity between two feature vectors"""
        if feat1 is None or feat2 is None:
            return 0.0
        
        try:
            # Cosine similarity
            similarity = np.dot(feat1, feat2) / (np.linalg.norm(feat1) * np.linalg.norm(feat2))
            return float(max(0, similarity))  # Clamp to [0, 1]
        except Exception:
            return 0.0

    def find_match(
        self,
        query_features: np.ndarray,
        camera_id: int,
        timestamp: float,
        max_age_seconds: float = 30.0
    ) -> Optional[str]:
        """
        Find if a person was seen in another camera recently.
        
        Returns:
            person_id if match found, None otherwise
        """
        if not self.enabled or query_features is None:
            return None

        best_match = None
        best_similarity = 0

        try:
            for other_cam_id, persons in self.camera_persons.items():
                if other_cam_id == camera_id:
                    continue

                for person in persons:
                    # Check age
                    if timestamp - person['timestamp'] > max_age_seconds:
                        continue

                    similarity = self.compute_similarity(query_features, person['features'])
                    
                    if similarity > self.similarity_threshold and similarity > best_similarity:
                        best_match = person['person_id']
                        best_similarity = similarity

            return best_match

        except Exception as e:
            logger.error(f"Error finding match: {e}")
            return None

    def register_person(
        self,
        person_id: str,
        features: np.ndarray,
        camera_id: int,
        bbox: List[int],
        timestamp: float
    ):
        """Register a person's features at a camera location"""
        if camera_id not in self.camera_persons:
            self.camera_persons[camera_id] = []

        self.camera_persons[camera_id].append({
            'person_id': person_id,
            'features': features,
            'bbox': bbox,
            'timestamp': timestamp
        })

        # Cleanup old entries
        self.camera_persons[camera_id] = [
            p for p in self.camera_persons[camera_id]
            if timestamp - p['timestamp'] <= 60.0
        ]

    def get_person_trajectory(self, person_id: str) -> List[Dict]:
        """Get the trajectory of a person across cameras"""
        trajectory = []
        
        for record in self.matching_history:
            if record.get('person_id') == person_id:
                trajectory.append(record)

        return sorted(trajectory, key=lambda x: x.get('timestamp', 0))

    def init_database(self):
        """Initialize re-identification tables"""
        try:
            self.db.execute("""
                CREATE TABLE IF NOT EXISTS person_tracks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    person_id TEXT NOT NULL,
                    camera_id INTEGER,
                    bbox TEXT,
                    features BLOB,
                    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            logger.info("Person ReID database initialized")
        except Exception as e:
            logger.error(f"Error initializing Person ReID database: {e}")


# Global instance
_person_reid = None


def get_person_reid() -> PersonReID:
    """Get global person re-id instance"""
    global _person_reid
    if _person_reid is None:
        _person_reid = PersonReID()
    return _person_reid