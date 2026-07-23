"""
License Plate Recognition Agent — Autonomous SwarmAgent for LPR.

Wraps the existing LicensePlateRecognition service as a self-managing
SwarmAgent with its own local evolutionary gene pool. Optimizes
segmentation_threshold, min_plate_height_px, resolution_downscale,
detection_confidence, and ocr_beam_width.

Lifecycle:
    __init__(): Load config, wrap LPR service, init local gene pool
    start():    Start agent
    stop():     Stop agent
    process_frame(): Run LPR with current optimized parameters
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
from .license_plate_recognition import get_license_plate_recognition
from ..core_engine.consortium_broker import (
    AgentBid,
    ConsortiumBroker,
    ResourceAllocation,
    get_consortium_broker,
)

logger = logging.getLogger(__name__)


# =============================================================================
# LPR Agent Gene Vector
# =============================================================================


@dataclass
class LprGeneVector:
    """The evolvable gene vector for the license plate recognition agent."""

    segmentation_threshold: float = 0.5
    min_plate_height_px: float = 25.0
    resolution_downscale: float = 0.8
    detection_confidence: float = 0.6
    ocr_beam_width: float = 3.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "segmentation_threshold": self.segmentation_threshold,
            "min_plate_height_px": self.min_plate_height_px,
            "resolution_downscale": self.resolution_downscale,
            "detection_confidence": self.detection_confidence,
            "ocr_beam_width": self.ocr_beam_width,
        }

    def clone(self) -> "LprGeneVector":
        return copy.deepcopy(self)

    def clamp_to_bounds(self, bounds: Dict[str, List[float]]):
        for attr_name, (lo, hi) in bounds.items():
            current = getattr(self, attr_name, None)
            if current is not None:
                clamped = max(lo, min(hi, current))
                setattr(self, attr_name, clamped)


# =============================================================================
# LprAgent
# =============================================================================


class LprAgent:
    """
    Autonomous SwarmAgent wrapping the LicensePlateRecognition service.

    Responds to broker context (vehicle_detected from YOLO agent) to
    dynamically adjust its processing intensity. Manages its own gene pool
    for local evolutionary optimization.
    """

    AGENT_ID = "lpr_agent"
    DOMAIN = "lpr"

    def __init__(self):
        self.config = get_config()
        self.agent_config = self.config.lpr_agent
        self.enabled = self.agent_config.enabled

        # Wrap the existing LPR service
        self.lpr_service = get_license_plate_recognition()

        # Broker reference
        self.broker: Optional[ConsortiumBroker] = None

        # Local gene pool
        self._gene_vector: LprGeneVector = LprGeneVector()
        self._fallback_vector: LprGeneVector = LprGeneVector()
        self._bounds = self._extract_bounds()

        # Local evolution state
        self._local_evolution_enabled = self.agent_config.local_evolution.enabled
        self._evaluation_window = self.agent_config.local_evolution.evaluation_window
        self._mutation_rate = self.agent_config.local_evolution.mutation_rate
        self._frame_count: int = 0
        self._fitness_history: List[float] = []
        self._plate_counts: List[int] = []
        self._processing_times: List[float] = []

        # Frame skip control
        self._skip_counter: int = 0
        self._current_throttle: float = 1.0

        # Register with broker
        self._register_with_broker()

        logger.info(
            f"LprAgent initialized (enabled={self.enabled}, "
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
            "segmentation_threshold": b.segmentation_threshold,
            "min_plate_height_px": b.min_plate_height_px,
            "resolution_downscale": b.resolution_downscale,
            "detection_confidence": b.detection_confidence,
            "ocr_beam_width": b.ocr_beam_width,
        }

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def start(self):
        """Start the agent."""
        if not self.enabled:
            logger.info("LprAgent is disabled (zero-cost fallback active)")
            return
        logger.info("LprAgent started")

    async def stop(self):
        """Stop the agent."""
        logger.info("LprAgent stopped")

    # ── Core Processing ─────────────────────────────────────────────────

    def process_frame(
        self, frame: np.ndarray, camera_id: int, detections: List[Dict]
    ) -> List[Dict]:
        """
        Run license plate recognition with current optimized parameters.

        Args:
            frame: Input video frame (BGR)
            camera_id: Camera identifier
            detections: Object detections from YOLO (used to find vehicle ROIs)

        Returns:
            List of LPR results with plate_text, confidence, bbox
        """
        if not self.enabled or not self.lpr_service.enabled:
            return []

        # Apply frame skipping based on throttle
        self._skip_counter += 1
        if self._skip_counter < int(1.0 / max(self._current_throttle, 0.1)):
            return []
        self._skip_counter = 0

        start_time = time.time()

        # Apply resolution downscale to plate crops if needed
        original_min_confidence = getattr(self.lpr_service, 'min_confidence', 0.6)
        self.lpr_service.min_confidence = self._gene_vector.detection_confidence

        try:
            results = self.lpr_service.detect_plates(frame, detections)
        except Exception as e:
            logger.warning(f"LPR error (sandboxed): {e}")
            results = []
        finally:
            self.lpr_service.min_confidence = original_min_confidence

        # Filter results by min plate height
        if results:
            filtered = []
            for r in results:
                bbox = r.get("bbox", [0, 0, 0, 0])
                plate_h = bbox[3] - bbox[1]
                if plate_h >= self._gene_vector.min_plate_height_px:
                    filtered.append(r)
            results = filtered

        # Track metrics
        elapsed = (time.time() - start_time) * 1000
        self._processing_times.append(elapsed)
        if len(self._processing_times) > 100:
            self._processing_times.pop(0)

        self._plate_counts.append(len(results))
        if len(self._plate_counts) > 100:
            self._plate_counts.pop(0)

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
        vehicle_detected = self.broker.read_context("vehicle_detected", 0.0)

        # Base urgency: very low by default (LPR is niche)
        urgency = 0.1

        # Boost urgency if vehicles are detected by YOLO
        if vehicle_detected > 0.3:
            urgency = max(urgency, 0.75)
        if vehicle_detected > 0.7:
            urgency = max(urgency, 0.9)

        # Contextual relevance tied to vehicle presence
        contextual_relevance = max(0.0, vehicle_detected)

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
        Adapt LPR parameters based on broker allocation.

        When throttled: increase downscale, raise detection confidence.
        When boosted: reduce downscale, lower detection threshold.
        """
        if not self.enabled:
            return

        self._current_throttle = allocation.throttle_factor

        if allocation.throttle_factor < 0.5:
            # Conserve compute
            self._gene_vector.resolution_downscale = max(
                0.5, self._gene_vector.resolution_downscale - 0.1
            )
            self._gene_vector.detection_confidence = min(
                0.8, self._gene_vector.detection_confidence + 0.05
            )
            self._gene_vector.ocr_beam_width = max(
                1.0, self._gene_vector.ocr_beam_width - 0.5
            )
        elif allocation.priority_boost > 1.5:
            # Boost quality
            self._gene_vector.resolution_downscale = min(
                1.0, self._gene_vector.resolution_downscale + 0.1
            )
            self._gene_vector.detection_confidence = max(
                0.4, self._gene_vector.detection_confidence - 0.05
            )
            self._gene_vector.ocr_beam_width = min(
                5.0, self._gene_vector.ocr_beam_width + 0.5
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
        """Compute fitness based on processing speed and plate detection rate."""
        if not self._processing_times:
            return 0.5

        avg_time = sum(self._processing_times[-50:]) / max(len(self._processing_times[-50:]), 1)
        speed_fitness = max(0.0, 1.0 - (avg_time / 100.0))

        avg_plates = (
            sum(self._plate_counts[-50:]) / max(len(self._plate_counts[-50:]), 1)
            if self._plate_counts
            else 0
        )
        quality_fitness = min(1.0, avg_plates / 2.0)

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

_lpr_agent: Optional[LprAgent] = None


def get_lpr_agent() -> LprAgent:
    """Get global LprAgent instance (singleton)."""
    global _lpr_agent
    if _lpr_agent is None:
        _lpr_agent = LprAgent()
    return _lpr_agent