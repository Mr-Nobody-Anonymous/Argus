"""
Face Recognition Agent — Autonomous SwarmAgent for face detection/recognition.

Wraps the existing FaceRecognition service as a self-managing SwarmAgent with
its own local evolutionary gene pool. Optimizes match_distance_threshold,
min_face_size_px, track_timeout_seconds, and frame_skip_cadence.

Lifecycle:
    __init__(): Load config, wrap FaceRecognition, init local gene pool
    start():    Start agent's background evolution loop
    stop():     Stop agent
    process_frame(): Run face recognition with current optimized parameters
    submit_bid_to_broker(): Submit resource bid based on broker context
    apply_allocation(): Adapt parameters based on broker allocation
"""

import asyncio
import copy
import logging
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.config import get_config
from .face_recognition import get_face_recognition
from ..core_engine.consortium_broker import (
    AgentBid,
    ConsortiumBroker,
    ResourceAllocation,
    get_consortium_broker,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Face Agent Gene Vector
# =============================================================================


@dataclass
class FaceGeneVector:
    """The evolvable gene vector for the face recognition agent."""

    match_distance_threshold: float = 0.6
    min_face_size_px: float = 50.0
    track_timeout_seconds: float = 5.0
    frame_skip_cadence: float = 3.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "match_distance_threshold": self.match_distance_threshold,
            "min_face_size_px": self.min_face_size_px,
            "track_timeout_seconds": self.track_timeout_seconds,
            "frame_skip_cadence": self.frame_skip_cadence,
        }

    def clone(self) -> "FaceGeneVector":
        return copy.deepcopy(self)

    def clamp_to_bounds(self, bounds: Dict[str, List[float]]):
        for attr_name, (lo, hi) in bounds.items():
            current = getattr(self, attr_name, None)
            if current is not None:
                clamped = max(lo, min(hi, current))
                setattr(self, attr_name, clamped)


# =============================================================================
# FaceRecognitionAgent
# =============================================================================


class FaceRecognitionAgent:
    """
    Autonomous SwarmAgent wrapping the FaceRecognition service.

    Responds to broker context (human_detected from YOLO agent) to dynamically
    adjust its processing intensity. Manages its own gene pool for local
    evolutionary optimization.
    """

    AGENT_ID = "face_agent"
    DOMAIN = "face"

    def __init__(self):
        self.config = get_config()
        self.agent_config = self.config.face_agent
        self.enabled = self.agent_config.enabled

        # Wrap the existing face recognition service
        self.face_recognition = get_face_recognition()

        # Broker reference
        self.broker: Optional[ConsortiumBroker] = None

        # Local gene pool
        self._gene_vector: FaceGeneVector = FaceGeneVector()
        self._fallback_vector: FaceGeneVector = FaceGeneVector()
        self._bounds = self._extract_bounds()

        # Local evolution state
        self._local_evolution_enabled = self.agent_config.local_evolution.enabled
        self._evaluation_window = self.agent_config.local_evolution.evaluation_window
        self._mutation_rate = self.agent_config.local_evolution.mutation_rate
        self._frame_count: int = 0
        self._fitness_history: List[float] = []
        self._face_counts: List[int] = []
        self._processing_times: List[float] = []

        # Frame skip control
        self._skip_counter: int = 0
        self._current_throttle: float = 1.0

        # Register with broker
        self._register_with_broker()

        logger.info(
            f"FaceRecognitionAgent initialized (enabled={self.enabled}, "
            f"local_evolution={self._local_evolution_enabled})"
        )

    def _register_with_broker(self):
        """Register this agent with the consortium broker."""
        try:
            self.broker = get_consortium_broker()
            if self.broker.enabled:
                self.broker.register_agent(self.AGENT_ID, self.DOMAIN)
        except Exception as e:
            logger.warning(f"Could not register with broker: {e}")

    def _extract_bounds(self) -> Dict[str, List[float]]:
        """Extract gene vector bounds from config."""
        b = self.agent_config.gene_vector_bounds
        return {
            "match_distance_threshold": b.match_distance_threshold,
            "min_face_size_px": b.min_face_size_px,
            "track_timeout_seconds": b.track_timeout_seconds,
            "frame_skip_cadence": b.frame_skip_cadence,
        }

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def start(self):
        """Start the agent."""
        if not self.enabled:
            logger.info("FaceRecognitionAgent is disabled (zero-cost fallback active)")
            return
        logger.info("FaceRecognitionAgent started")

    async def stop(self):
        """Stop the agent."""
        logger.info("FaceRecognitionAgent stopped")

    # ── Core Processing ─────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, camera_id: int) -> List[Dict]:
        """
        Run face recognition with current optimized parameters.

        Args:
            frame: Input video frame (BGR)
            camera_id: Camera identifier

        Returns:
            List of face recognition results with bbox, person_name, confidence
        """
        if not self.enabled or not self.face_recognition.enabled:
            return []

        # Apply frame skipping based on throttle and gene vector
        self._skip_counter += 1
        cadence = int(self._gene_vector.frame_skip_cadence * self._current_throttle)
        if self._skip_counter < max(1, cadence):
            return []
        self._skip_counter = 0

        start_time = time.time()

        # Apply evolved min face size to face detection
        # Modify the face_recognition's min face size behavior via a temporary override
        original_min_size = getattr(self.face_recognition, 'track_timeout', 5.0)
        self.face_recognition.track_timeout = self._gene_vector.track_timeout_seconds

        try:
            # Run recognition with detect_emotions=False for speed
            results = self.face_recognition.recognize_faces(frame, detect_emotions=False)
        except Exception as e:
            logger.warning(f"Face recognition error (sandboxed): {e}")
            results = []
        finally:
            self.face_recognition.track_timeout = original_min_size

        # Filter results by min face size
        if results:
            filtered = []
            for r in results:
                bbox = r.get("bbox", [0, 0, 0, 0])
                face_h = bbox[3] - bbox[1]
                face_w = bbox[2] - bbox[0]
                if face_h >= self._gene_vector.min_face_size_px and face_w >= self._gene_vector.min_face_size_px:
                    filtered.append(r)
            results = filtered

        # Track metrics
        elapsed = (time.time() - start_time) * 1000
        self._processing_times.append(elapsed)
        if len(self._processing_times) > 100:
            self._processing_times.pop(0)

        self._face_counts.append(len(results))
        if len(self._face_counts) > 100:
            self._face_counts.pop(0)

        # Run local evolution
        self._frame_count += 1
        self._run_local_evolution()

        return results

    # ── Broker Integration ──────────────────────────────────────────────

    def submit_bid_to_broker(self):
        """Compute urgency/relevance from broker context and submit bid."""
        if not self.enabled or not self.broker or not self.broker.enabled:
            return

        # Read contextual clues from broker blackboard
        human_detected = self.broker.read_context("human_detected", 0.0)
        crowd_detected = self.broker.read_context("crowd_detected", 0.0)

        # Base urgency: low by default (face is optional)
        urgency = 0.15

        # Boost urgency if humans are detected by YOLO
        if human_detected > 0.3:
            urgency = max(urgency, 0.7)
        if crowd_detected > 0.5:
            urgency = max(urgency, 0.85)

        # Contextual relevance tied to human presence
        contextual_relevance = max(0.0, human_detected)

        # Compute cost from recent processing times
        avg_cost = (
            sum(self._processing_times[-20:]) / max(len(self._processing_times[-20:]), 1)
            if self._processing_times
            else self.agent_config.compute_cost_per_frame
        )

        current_load = min(1.0, avg_cost / 33.0)

        bid = AgentBid(
            agent_id=self.AGENT_ID,
            domain=self.DOMAIN,
            urgency=urgency,
            compute_cost=avg_cost,
            contextual_relevance=contextual_relevance,
            current_load=current_load,
        )
        self.broker.submit_bid(bid)

    def apply_allocation(self, allocation: ResourceAllocation):
        """
        Adapt face recognition parameters based on broker allocation.

        When throttled: increase frame skip cadence, raise min face size.
        When boosted: process every frame, lower min face size.
        """
        if not self.enabled:
            return

        self._current_throttle = allocation.throttle_factor

        if allocation.throttle_factor < 0.5:
            # Conserve compute
            self._gene_vector.frame_skip_cadence = min(
                10.0, self._gene_vector.frame_skip_cadence + 1.0
            )
            self._gene_vector.min_face_size_px = min(
                100.0, self._gene_vector.min_face_size_px + 10.0
            )
        elif allocation.priority_boost > 1.5:
            # Boost quality
            self._gene_vector.frame_skip_cadence = max(
                1.0, self._gene_vector.frame_skip_cadence - 0.5
            )
            self._gene_vector.min_face_size_px = max(
                30.0, self._gene_vector.min_face_size_px - 5.0
            )

        self._gene_vector.clamp_to_bounds(self._bounds)

    # ── Local Evolution ─────────────────────────────────────────────────

    def _run_local_evolution(self):
        """Run local evolutionary optimization on the gene vector."""
        if not self._local_evolution_enabled:
            return
        if self._frame_count < self._evaluation_window:
            return
        if self._frame_count % self._evaluation_window != 0:
            return

        fitness = self._compute_fitness()
        self._fitness_history.append(fitness)
        if len(self._fitness_history) > 10:
            self._fitness_history.pop(0)

        if len(self._fitness_history) >= 3:
            recent = self._fitness_history[-3:]
            if max(recent) - min(recent) < 0.05 or fitness < 0.5:
                self._mutate_vector()

    def _compute_fitness(self) -> float:
        """Compute fitness based on processing speed and face detection rate."""
        if not self._processing_times:
            return 0.5

        avg_time = sum(self._processing_times[-50:]) / max(len(self._processing_times[-50:]), 1)
        speed_fitness = max(0.0, 1.0 - (avg_time / 100.0))

        avg_faces = (
            sum(self._face_counts[-50:]) / max(len(self._face_counts[-50:]), 1)
            if self._face_counts
            else 0
        )
        quality_fitness = min(1.0, avg_faces / 3.0)

        return max(0.0, min(1.0, 0.5 * speed_fitness + 0.5 * quality_fitness))

    def _mutate_vector(self):
        """Apply mutation to the gene vector."""
        for attr_name in vars(self._gene_vector).keys():
            lo, hi = self._bounds.get(attr_name, (0.0, 1.0))
            if random.random() < self._mutation_rate:
                current = getattr(self._gene_vector, attr_name)
                range_size = hi - lo
                delta = random.gauss(0, range_size * 0.1)
                mutated = current + delta
                clamped = max(lo, min(hi, mutated))
                setattr(self._gene_vector, attr_name, clamped)

    # ── Public Accessors ────────────────────────────────────────────────

    def get_gene_vector(self) -> Dict:
        """Get the current best gene vector."""
        return self._gene_vector.to_dict()

    def get_status(self) -> Dict:
        """Get agent health and stats."""
        avg_time = (
            sum(self._processing_times[-50:]) / max(len(self._processing_times[-50:]), 1)
            if self._processing_times
            else 0
        )
        return {
            "enabled": self.enabled,
            "agent_id": self.AGENT_ID,
            "domain": self.DOMAIN,
            "frame_count": self._frame_count,
            "avg_processing_time_ms": round(avg_time, 2),
            "current_throttle": round(self._current_throttle, 2),
            "gene_vector": self._gene_vector.to_dict(),
            "local_evolution_enabled": self._local_evolution_enabled,
        }


# =============================================================================
# Global Instance
# =============================================================================

_face_recognition_agent: Optional[FaceRecognitionAgent] = None


def get_face_recognition_agent() -> FaceRecognitionAgent:
    """Get global FaceRecognitionAgent instance (singleton)."""
    global _face_recognition_agent
    if _face_recognition_agent is None:
        _face_recognition_agent = FaceRecognitionAgent()
    return _face_recognition_agent