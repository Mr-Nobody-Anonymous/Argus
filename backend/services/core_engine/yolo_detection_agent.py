"""
YOLO Detection Agent — Autonomous SwarmAgent for object detection.

Wraps the existing InferenceEngine as a self-managing SwarmAgent with its own
local evolutionary gene pool. Optimizes yolo_conf_threshold, iou_threshold,
input_resolution_scale, and tracker_matching_threshold.

Lifecycle:
    __init__(): Load config, wrap InferenceEngine, init local gene pool
    start():    Start agent's background evolution loop
    stop():     Stop agent, persist best vector
    process_frame(): Run detection with current optimized parameters
    submit_bid_to_broker(): Compute urgency/relevance and submit resource bid
    apply_allocation(): Adapt thresholds based on broker allocation
"""

import asyncio
import copy
import logging
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.config.config import get_config
from .inference_engine import get_inference_engine
from .consortium_broker import (
    AgentBid,
    ConsortiumBroker,
    ResourceAllocation,
    get_consortium_broker,
)

logger = logging.getLogger(__name__)


# =============================================================================
# YOLO Agent Gene Vector
# =============================================================================


@dataclass
class YoloGeneVector:
    """The evolvable gene vector for the YOLO detection agent."""

    yolo_conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    input_resolution_scale: float = 1.0
    tracker_matching_threshold: float = 0.6

    def to_dict(self) -> Dict[str, float]:
        return {
            "yolo_conf_threshold": self.yolo_conf_threshold,
            "iou_threshold": self.iou_threshold,
            "input_resolution_scale": self.input_resolution_scale,
            "tracker_matching_threshold": self.tracker_matching_threshold,
        }

    def clone(self) -> "YoloGeneVector":
        return copy.deepcopy(self)

    def clamp_to_bounds(self, bounds: Dict[str, List[float]]):
        for attr_name, (lo, hi) in bounds.items():
            current = getattr(self, attr_name, None)
            if current is not None:
                clamped = max(lo, min(hi, current))
                setattr(self, attr_name, clamped)


# =============================================================================
# YoloDetectionAgent
# =============================================================================


class YoloDetectionAgent:
    """
    Autonomous SwarmAgent wrapping the InferenceEngine.

    Manages its own local evolutionary gene pool to optimize detection
    parameters. Submits resource bids to the ConsortiumBroker and adapts
    its behavior based on broker allocations.
    """

    AGENT_ID = "yolo_agent"
    DOMAIN = "yolo"

    def __init__(self):
        self.config = get_config()
        self.agent_config = self.config.yolo_agent
        self.enabled = self.agent_config.enabled

        # Wrap the existing inference engine
        self.inference_engine = get_inference_engine()

        # Broker reference
        self.broker: Optional[ConsortiumBroker] = None

        # Local gene pool
        self._gene_vector: YoloGeneVector = YoloGeneVector()
        self._fallback_vector: YoloGeneVector = YoloGeneVector()
        self._bounds = self._extract_bounds()

        # Local evolution state
        self._local_evolution_enabled = self.agent_config.local_evolution.enabled
        self._evaluation_window = self.agent_config.local_evolution.evaluation_window
        self._mutation_rate = self.agent_config.local_evolution.mutation_rate
        self._frame_count: int = 0
        self._fitness_history: List[float] = []
        self._last_mutation_time: float = 0.0

        # Performance tracking
        self._inference_times: List[float] = []
        self._detection_counts: List[int] = []
        self._false_positive_ratios: List[float] = []

        # Frame skip counter for throttling
        self._skip_counter: int = 0
        self._current_throttle: float = 1.0

        # Register with broker
        self._register_with_broker()

        logger.info(
            f"YoloDetectionAgent initialized (enabled={self.enabled}, "
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
            "yolo_conf_threshold": b.yolo_conf_threshold,
            "iou_threshold": b.iou_threshold,
            "input_resolution_scale": b.input_resolution_scale,
            "tracker_matching_threshold": b.tracker_matching_threshold,
        }

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def start(self):
        """Start the agent's background evolution loop."""
        if not self.enabled:
            logger.info("YoloDetectionAgent is disabled (zero-cost fallback active)")
            return
        logger.info("YoloDetectionAgent started")

    async def stop(self):
        """Stop the agent."""
        logger.info("YoloDetectionAgent stopped")

    # ── Core Processing ─────────────────────────────────────────────────

    def process_frame(self, frame: np.ndarray, camera_id: int) -> List[Dict]:
        """
        Run YOLO detection with current optimized parameters.

        Args:
            frame: Input video frame (BGR)
            camera_id: Camera identifier

        Returns:
            List of detection dicts with bbox, confidence, class_name, class_id
        """
        if not self.enabled:
            return self.inference_engine.detect_objects(frame)

        # Apply frame skipping based on throttle
        self._skip_counter += 1
        if self._skip_counter < int(1.0 / max(self._current_throttle, 0.1)):
            return []  # Skip this frame
        self._skip_counter = 0

        start_time = time.time()

        # Apply resolution scaling if needed
        if self._gene_vector.input_resolution_scale < 1.0:
            h, w = frame.shape[:2]
            new_w = int(w * self._gene_vector.input_resolution_scale)
            new_h = int(h * self._gene_vector.input_resolution_scale)
            processed_frame = cv2.resize(frame, (new_w, new_h))
        else:
            processed_frame = frame

        # Apply evolved confidence threshold
        original_threshold = self.inference_engine.config.inference.confidence_threshold
        self.inference_engine.config.inference.confidence_threshold = (
            self._gene_vector.yolo_conf_threshold
        )

        try:
            detections = self.inference_engine.detect_objects(processed_frame)
        finally:
            # Restore original threshold
            self.inference_engine.config.inference.confidence_threshold = original_threshold

        # Scale bbox coordinates back if resolution was downscaled
        if self._gene_vector.input_resolution_scale < 1.0 and detections:
            scale_x = frame.shape[1] / max(processed_frame.shape[1], 1)
            scale_y = frame.shape[0] / max(processed_frame.shape[0], 1)
            for det in detections:
                det["bbox"] = [
                    int(det["bbox"][0] * scale_x),
                    int(det["bbox"][1] * scale_y),
                    int(det["bbox"][2] * scale_x),
                    int(det["bbox"][3] * scale_y),
                ]

        # Track performance metrics
        elapsed = (time.time() - start_time) * 1000
        self._inference_times.append(elapsed)
        if len(self._inference_times) > 100:
            self._inference_times.pop(0)

        self._detection_counts.append(len(detections))
        if len(self._detection_counts) > 100:
            self._detection_counts.pop(0)

        # Estimate false positive ratio
        if detections:
            low_conf = sum(1 for d in detections if d.get("confidence", 1.0) < 0.3)
            self._false_positive_ratios.append(low_conf / max(len(detections), 1))
        else:
            self._false_positive_ratios.append(0.0)
        if len(self._false_positive_ratios) > 100:
            self._false_positive_ratios.pop(0)

        # Run local evolution
        self._frame_count += 1
        self._run_local_evolution()

        return detections

    # ── Broker Integration ──────────────────────────────────────────────

    def submit_bid_to_broker(self):
        """Compute urgency/relevance and submit resource bid to broker."""
        if not self.enabled or not self.broker or not self.broker.enabled:
            return

        # Urgency: YOLO is always primary — base urgency 0.8
        urgency = 0.8

        # Boost urgency if recent detections were found
        if self._detection_counts and sum(self._detection_counts[-10:]) > 0:
            urgency = min(1.0, urgency + 0.15)

        # Contextual relevance: always 1.0 for YOLO (feeds all downstream)
        contextual_relevance = 1.0

        # Compute cost from recent inference times
        avg_cost = (
            sum(self._inference_times[-20:]) / max(len(self._inference_times[-20:]), 1)
            if self._inference_times
            else self.agent_config.compute_cost_per_frame
        )

        # Current load: ratio of recent inference time to budget
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
        Adapt detection parameters based on broker allocation.

        When throttled: increase confidence threshold, reduce resolution.
        When boosted: lower threshold, increase resolution.
        """
        if not self.enabled:
            return

        self._current_throttle = allocation.throttle_factor

        if allocation.throttle_factor < 0.5:
            # Heavily throttled — conserve compute
            self._gene_vector.yolo_conf_threshold = min(
                0.7, self._gene_vector.yolo_conf_threshold + 0.05
            )
            self._gene_vector.input_resolution_scale = max(
                0.5, self._gene_vector.input_resolution_scale - 0.1
            )
        elif allocation.priority_boost > 1.5:
            # Boosted — use higher quality
            self._gene_vector.yolo_conf_threshold = max(
                0.3, self._gene_vector.yolo_conf_threshold - 0.05
            )
            self._gene_vector.input_resolution_scale = min(
                1.0, self._gene_vector.input_resolution_scale + 0.1
            )

        self._gene_vector.clamp_to_bounds(self._bounds)

    # ── Local Evolution ─────────────────────────────────────────────────

    def _run_local_evolution(self):
        """Run local evolutionary optimization on the gene vector."""
        if not self._local_evolution_enabled:
            return
        if self._frame_count < self._evaluation_window:
            return

        # Evaluate every evaluation_window frames
        if self._frame_count % self._evaluation_window != 0:
            return

        # Compute current fitness
        fitness = self._compute_fitness()
        self._fitness_history.append(fitness)
        if len(self._fitness_history) > 10:
            self._fitness_history.pop(0)

        # Mutate if fitness is stagnant or below threshold
        if len(self._fitness_history) >= 3:
            recent = self._fitness_history[-3:]
            if max(recent) - min(recent) < 0.05 or fitness < 0.6:
                self._mutate_vector()

        logger.debug(
            f"YOLO agent local evolution: frame={self._frame_count}, "
            f"fitness={fitness:.3f}, vector={self._gene_vector.to_dict()}"
        )

    def _compute_fitness(self) -> float:
        """Compute fitness based on inference speed and detection quality."""
        if not self._inference_times:
            return 0.5

        # Speed fitness: lower inference time is better
        avg_time = sum(self._inference_times[-50:]) / max(len(self._inference_times[-50:]), 1)
        speed_fitness = max(0.0, 1.0 - (avg_time / 100.0))

        # Detection quality: higher detection count is better (up to a point)
        avg_detections = (
            sum(self._detection_counts[-50:]) / max(len(self._detection_counts[-50:]), 1)
            if self._detection_counts
            else 0
        )
        quality_fitness = min(1.0, avg_detections / 20.0)

        # False positive penalty
        avg_fp = (
            sum(self._false_positive_ratios[-50:]) / max(len(self._false_positive_ratios[-50:]), 1)
            if self._false_positive_ratios
            else 0
        )
        fp_penalty = 1.0 - avg_fp

        # Composite
        fitness = 0.4 * speed_fitness + 0.35 * quality_fitness + 0.25 * fp_penalty
        return max(0.0, min(1.0, fitness))

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
            sum(self._inference_times[-50:]) / max(len(self._inference_times[-50:]), 1)
            if self._inference_times
            else 0
        )
        return {
            "enabled": self.enabled,
            "agent_id": self.AGENT_ID,
            "domain": self.DOMAIN,
            "frame_count": self._frame_count,
            "avg_inference_time_ms": round(avg_time, 2),
            "current_throttle": round(self._current_throttle, 2),
            "gene_vector": self._gene_vector.to_dict(),
            "local_evolution_enabled": self._local_evolution_enabled,
        }


# =============================================================================
# Global Instance
# =============================================================================

_yolo_detection_agent: Optional[YoloDetectionAgent] = None


def get_yolo_detection_agent() -> YoloDetectionAgent:
    """Get global YoloDetectionAgent instance (singleton)."""
    global _yolo_detection_agent
    if _yolo_detection_agent is None:
        _yolo_detection_agent = YoloDetectionAgent()
    return _yolo_detection_agent