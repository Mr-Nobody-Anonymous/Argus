"""
Generative-Evolutionary Cognition Engine

A self-evolving, generative AI service that continuously optimizes the Argus
video pipeline through evolutionary algorithms. Blends generative intelligence
(dynamic code/logic generation) with evolutionary algorithms (continuous
self-mutation, fitness evaluation, and architectural survival selection).

Components:
  - GenerativeSynthesizer: Dynamically generates pipeline logic variants
  - EvolutionaryEvaluator: Tracks real-world pipeline fitness metrics
  - MutationOrchestrator: Prunes low-performers, breeds optimized variants
  - EvolutionaryEngine: Main service class with lifecycle methods
"""

import asyncio
import copy
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from ...config.config import get_config
from ..management.event_store import get_event_store

logger = logging.getLogger(__name__)

# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class OptimizationVector:
    """The 'gene' — a unified set of pipeline parameters that the engine evolves."""

    yolo_conf_threshold: float = 0.5
    iou_threshold: float = 0.5
    tracking_history_buffer: int = 30
    frame_skipping_cadence: int = 1
    rules_engine_cooldown: float = 5.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "yolo_conf_threshold": self.yolo_conf_threshold,
            "iou_threshold": self.iou_threshold,
            "tracking_history_buffer": float(self.tracking_history_buffer),
            "frame_skipping_cadence": float(self.frame_skipping_cadence),
            "rules_engine_cooldown": self.rules_engine_cooldown,
        }

    def clone(self) -> "OptimizationVector":
        return copy.deepcopy(self)

    def clamp_to_bounds(self, bounds: Dict[str, List[float]]):
        """Clamp all vector values to their defined bounds."""
        for attr_name, (lo, hi) in bounds.items():
            current = getattr(self, attr_name, None)
            if current is not None:
                clamped = max(lo, min(hi, current))
                # Round int params
                if attr_name in ("tracking_history_buffer", "frame_skipping_cadence"):
                    clamped = int(round(clamped))
                setattr(self, attr_name, clamped)


@dataclass
class PipelineVariant:
    """A single variant in the gene pool with its fitness score and lineage."""

    variant_id: str
    vector: OptimizationVector
    fitness_score: float = 0.0
    generation: int = 0
    parent_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    error_count: int = 0
    is_alive: bool = True


@dataclass
class FrameMetrics:
    """Per-frame metrics recorded for fitness evaluation."""

    camera_id: int
    timestamp: float
    inference_time_ms: float
    tracking_accuracy: float  # 0.0 to 1.0
    false_positive_ratio: float  # 0.0 to 1.0
    num_detections: int
    kafka_latency_ms: Optional[float] = None
    rule_precision: Optional[float] = None
    processing_time_ms: float = 0.0


@dataclass
class RuleSuggestion:
    """A dynamically generated rule suggestion from the synthesizer."""

    rule_name: str
    condition: str
    priority: str
    parameters: Dict
    confidence: float


# =============================================================================
# GenerativeSynthesizer
# =============================================================================


class GenerativeSynthesizer:
    """
    Dynamically generates pipeline logic variants, rule modifications,
    and frame-skipping strategies based on deep pipeline context.

    Uses a heuristic generative approach: analyzes recent metrics patterns
    and intelligently mutates variants rather than purely random mutations.
    """

    def __init__(self, config):
        self.config = config.synthesizer
        self.enabled = self.config.enabled
        self.context_window = self.config.context_window
        self.min_generation_interval_s = self.config.min_generation_interval_s
        self.max_generated_rules = self.config.max_generated_rules
        self._last_generation_time = 0.0
        self._metrics_history: List[Dict] = []

    def feed_metrics(self, metrics_snapshot: Dict):
        """Feed a metrics snapshot into the context window."""
        self._metrics_history.append(metrics_snapshot)
        # Trim to context window
        if len(self._metrics_history) > self.context_window:
            self._metrics_history = self._metrics_history[-self.context_window :]

    def can_generate(self) -> bool:
        """Check if enough time has elapsed since last generation."""
        if not self.enabled:
            return False
        elapsed = time.time() - self._last_generation_time
        return elapsed >= self.min_generation_interval_s

    def generate_pipeline_variant(
        self, base_vector: OptimizationVector, bounds: Dict[str, List[float]], mutation_rate: float
    ) -> OptimizationVector:
        """
        Generate a new pipeline variant by intelligently mutating the base vector.

        Uses context from recent metrics to bias mutations toward areas
        that need improvement (e.g., if inference is slow, prioritize
        mutating confidence threshold to reduce load).
        """
        new_vector = base_vector.clone()

        if not self.enabled:
            return new_vector

        # Analyze recent metrics to determine which params need attention
        attention_weights = self._compute_attention_weights()

        # Mutate each parameter based on attention-weighted probability
        for attr_name in vars(new_vector).keys():
            if attr_name == "variant_id":
                continue
            lo, hi = bounds.get(attr_name, (0.0, 1.0))
            # Weighted mutation probability
            base_prob = mutation_rate
            attention = attention_weights.get(attr_name, 1.0)
            mutation_prob = min(1.0, base_prob * attention)

            if random.random() < mutation_prob:
                current = getattr(new_vector, attr_name)
                # Gaussian mutation scaled to the range
                range_size = hi - lo
                delta = random.gauss(0, range_size * 0.15)
                mutated = current + delta
                # Clamp
                clamped = max(lo, min(hi, mutated))
                if attr_name in ("tracking_history_buffer", "frame_skipping_cadence"):
                    clamped = int(round(clamped))
                setattr(new_vector, attr_name, clamped)

        self._last_generation_time = time.time()
        return new_vector

    def _compute_attention_weights(self) -> Dict[str, float]:
        """
        Analyze recent metrics to determine which params need mutation attention.

        Returns a dict mapping param names to attention multipliers (0.5-2.0).
        """
        weights = {
            "yolo_conf_threshold": 1.0,
            "iou_threshold": 1.0,
            "tracking_history_buffer": 1.0,
            "frame_skipping_cadence": 1.0,
            "rules_engine_cooldown": 1.0,
        }

        if len(self._metrics_history) < 10:
            return weights

        # Look at recent metrics trends
        recent = self._metrics_history[-10:]
        avg_inference_time = np.mean([m.get("inference_time_ms", 0) for m in recent])
        avg_false_positives = np.mean([m.get("false_positive_ratio", 0) for m in recent])

        # If inference is slow, bias toward higher threshold (fewer detections)
        if avg_inference_time > 50:  # ms
            weights["yolo_conf_threshold"] = 2.0
            weights["iou_threshold"] = 1.5

        # If false positives are high, bias toward threshold increases
        if avg_false_positives > 0.3:
            weights["yolo_conf_threshold"] = max(weights["yolo_conf_threshold"], 1.8)
            weights["rules_engine_cooldown"] = 1.5

        # If tracking is poor, bias toward larger buffer
        avg_accuracy = np.mean([m.get("tracking_accuracy", 1.0) for m in recent])
        if avg_accuracy < 0.7:
            weights["tracking_history_buffer"] = 2.0

        return weights

    def generate_edge_case_rule(self, anomaly_pattern: Dict) -> Optional[RuleSuggestion]:
        """
        Synthesize a new rule suggestion based on an anomaly pattern.

        This is a generative heuristic that creates rule suggestions
        when novel anomaly patterns are detected.
        """
        if not self.enabled:
            return None

        try:
            pattern_type = anomaly_pattern.get("type", "unknown")
            confidence = anomaly_pattern.get("confidence", 0.5)

            if confidence < 0.3:
                return None

            if pattern_type == "crowd_density":
                return RuleSuggestion(
                    rule_name="crowd_density_alert",
                    condition="num_detections > threshold",
                    priority="medium",
                    parameters={"threshold": 15, "zone": anomaly_pattern.get("zone", "any")},
                    confidence=confidence,
                )
            elif pattern_type == "rapid_movement":
                return RuleSuggestion(
                    rule_name="rapid_movement_alert",
                    condition="speed > threshold",
                    priority="high",
                    parameters={
                        "threshold_kmh": 50,
                        "zone": anomaly_pattern.get("zone", "any"),
                    },
                    confidence=confidence,
                )
            elif pattern_type == "stationary_object":
                return RuleSuggestion(
                    rule_name="stationary_object_extended",
                    condition="object_stationary_duration > threshold",
                    priority="medium",
                    parameters={
                        "threshold_seconds": 120,
                        "zone": anomaly_pattern.get("zone", "any"),
                    },
                    confidence=confidence,
                )
            elif pattern_type == "night_activity":
                return RuleSuggestion(
                    rule_name="night_activity_alert",
                    condition="is_night AND detection_confidence > threshold",
                    priority="high",
                    parameters={
                        "threshold": 0.7,
                        "confidence": confidence,
                    },
                    confidence=confidence,
                )

        except Exception as e:
            logger.warning(f"GenerativeSynthesizer edge case rule error: {e}")

        return None


# =============================================================================
# EvolutionaryEvaluator
# =============================================================================


class EvolutionaryEvaluator:
    """
    Tracks real-world pipeline fitness metrics across all cameras.

    Maintains a sliding window of FrameMetrics per camera and computes
    composite fitness scores using weighted aggregation.
    """

    def __init__(self, config):
        self.config = config.evaluator
        self.enabled = self.config.enabled
        self.evaluation_window = self.config.evaluation_window
        self.fitness_weights = self.config.fitness_weights
        self.survival_threshold = self.config.survival_threshold

        # Per-camera sliding window of metrics
        self._metrics: Dict[int, List[FrameMetrics]] = defaultdict(list)
        self._frame_counts: Dict[int, int] = defaultdict(int)
        self._fitness_history: List[float] = []

    def record_metrics(self, camera_id: int, metrics: FrameMetrics):
        """Record per-frame metrics for a camera."""
        if not self.enabled:
            return

        window = self._metrics[camera_id]
        window.append(metrics)
        self._frame_counts[camera_id] += 1

        # Trim window
        if len(window) > self.evaluation_window:
            window.pop(0)

    def get_fitness_snapshot(self) -> Dict:
        """Return current fitness landscape across all cameras."""
        if not self.enabled or not self._metrics:
            return {
                "overall_fitness": 0.0,
                "per_camera": {},
                "component_scores": {},
                "frame_count": 0,
            }

        per_camera = {}
        for cam_id, window in self._metrics.items():
            if not window:
                continue
            per_camera[cam_id] = self._compute_camera_fitness(window)

        overall = (
            np.mean([v["composite"] for v in per_camera.values()]) if per_camera else 0.0
        )

        # Aggregate component scores
        component_scores = {}
        for key in self.fitness_weights:
            vals = [
                v["components"].get(key, 0.0)
                for v in per_camera.values()
                if key in v.get("components", {})
            ]
            component_scores[key] = float(np.mean(vals)) if vals else 0.0

        total_frames = sum(self._frame_counts.values())

        return {
            "overall_fitness": float(overall),
            "per_camera": per_camera,
            "component_scores": component_scores,
            "frame_count": total_frames,
            "camera_count": len(per_camera),
        }

    def compute_variant_fitness(self, variant: PipelineVariant, snapshot: Dict) -> float:
        """
        Compute how well a given variant performs based on the current
        fitness landscape. Variants are scored relative to the baseline.
        """
        if not self.enabled or not snapshot.get("per_camera"):
            return 0.5  # Default mid-range fitness

        overall = snapshot.get("overall_fitness", 0.5)

        # Adjust based on variant's deviation from baseline
        # Variants that are too extreme get lower scores (exploration penalty)
        default_vector = OptimizationVector()
        deviation = self._compute_vector_deviation(variant.vector, default_vector)
        exploration_bonus = min(0.15, deviation * 0.05)

        # Combine: base fitness + exploration bonus for variety
        fitness = overall + exploration_bonus

        # Penalize variants with errors
        if variant.error_count > 0:
            fitness *= max(0.1, 1.0 - (variant.error_count * 0.2))

        return float(max(0.0, min(1.0, fitness)))

    def get_frame_count(self) -> int:
        """Get total frames evaluated across all cameras."""
        return sum(self._frame_counts.values())

    def _compute_camera_fitness(self, window: List[FrameMetrics]) -> Dict:
        """Compute composite fitness for a single camera's metrics window."""
        if not window:
            return {"composite": 0.0, "components": {}}

        components = {}

        # 1. Inference speed fitness (lower is better, target <30ms)
        avg_inference = np.mean([m.inference_time_ms for m in window])
        speed_fitness = max(0.0, 1.0 - (avg_inference / 100.0))
        components["inference_speed"] = speed_fitness

        # 2. Tracking accuracy (higher is better)
        avg_tracking = np.mean([m.tracking_accuracy for m in window])
        components["tracking_accuracy"] = float(avg_tracking)

        # 3. False positive ratio (lower is better)
        avg_fp = np.mean([m.false_positive_ratio for m in window])
        fp_fitness = max(0.0, 1.0 - avg_fp)
        components["false_positive_ratio"] = fp_fitness

        # 4. Kafka latency (lower is better)
        latencies = [m.kafka_latency_ms for m in window if m.kafka_latency_ms is not None]
        if latencies:
            avg_latency = np.mean(latencies)
            latency_fitness = max(0.0, 1.0 - (avg_latency / 500.0))
        else:
            latency_fitness = 1.0  # No latency data = no penalty
        components["kafka_latency_ms"] = latency_fitness

        # 5. Rule precision (higher is better)
        precisions = [m.rule_precision for m in window if m.rule_precision is not None]
        if precisions:
            components["rule_precision"] = float(np.mean(precisions))
        else:
            components["rule_precision"] = 1.0

        # Weighted composite
        composite = sum(
            components.get(key, 0.0) * self.fitness_weights.get(key, 0.0)
            for key in self.fitness_weights
        )

        # Normalize to [0, 1]
        total_weight = sum(self.fitness_weights.values())
        if total_weight > 0:
            composite /= total_weight

        self._fitness_history.append(composite)

        return {"composite": float(composite), "components": components}

    @staticmethod
    def _compute_vector_deviation(v1: OptimizationVector, v2: OptimizationVector) -> float:
        """Compute normalized deviation between two optimization vectors."""
        deviations = []
        for attr in vars(v1).keys():
            val1 = getattr(v1, attr, 0)
            val2 = getattr(v2, attr, 0)
            max_val = max(abs(val1), abs(val2), 1.0)
            deviations.append(abs(val1 - val2) / max_val)
        return float(np.mean(deviations)) if deviations else 0.0


# =============================================================================
# MutationOrchestrator
# =============================================================================


class MutationOrchestrator:
    """
    Manages the gene pool. Prunes low-performing variants, breeds crossovers
    between top performers, and scales mutation rates adaptively when fitness
    plateaus.
    """

    def __init__(self, config):
        self.config = config.mutation
        self.enabled = self.config.enabled
        self.initial_mutation_rate = self.config.initial_mutation_rate
        self.adaptive_rate = self.config.adaptive_rate
        self.max_mutation_attempts = self.config.max_mutation_attempts
        self.population_cap = self.config.population_cap
        self.prune_interval_s = self.config.prune_interval_s
        self.crossover_rate = self.config.crossover_rate

        # Gene pool: list of PipelineVariant
        self._gene_pool: List[PipelineVariant] = []
        self._current_mutation_rate = self.initial_mutation_rate
        self._generation = 0
        self._last_prune_time = 0.0
        self._plateau_count = 0
        self._last_fitness_values: List[float] = []

        # Create default variant
        default_variant = PipelineVariant(
            variant_id="baseline",
            vector=OptimizationVector(),
            fitness_score=0.5,
            generation=0,
        )
        self._gene_pool.append(default_variant)

    @property
    def current_mutation_rate(self) -> float:
        return self._current_mutation_rate

    @property
    def generation(self) -> int:
        return self._generation

    def get_gene_pool(self) -> List[PipelineVariant]:
        """Get a copy of the current gene pool (alive variants only)."""
        return [v for v in self._gene_pool if v.is_alive]

    def get_optimal_vector(self) -> OptimizationVector:
        """Return the highest-fitness variant's vector for active use."""
        alive = self.get_gene_pool()
        if not alive:
            return OptimizationVector()
        best = max(alive, key=lambda v: v.fitness_score)
        return best.vector.clone()

    def add_variant(self, variant: PipelineVariant):
        """Add a new variant to the gene pool."""
        if len(self._gene_pool) >= self.population_cap:
            # Prune the weakest before adding
            self._prune_weakest()
        variant.generation = self._generation
        self._gene_pool.append(variant)

    def prune_population(self, fitness_scores: Dict[str, float]):
        """
        Prune low-performing variants based on fitness scores.

        Args:
            fitness_scores: Dict mapping variant_id -> fitness_score
        """
        if not self.enabled:
            return

        now = time.time()
        if now - self._last_prune_time < self.prune_interval_s:
            return

        for variant in self._gene_pool:
            if variant.variant_id in fitness_scores:
                variant.fitness_score = fitness_scores[variant.variant_id]

        # Keep baseline always alive
        alive = [v for v in self._gene_pool if v.variant_id == "baseline"]
        candidates = [v for v in self._gene_pool if v.variant_id != "baseline" and v.is_alive]

        # Sort by fitness (ascending)
        candidates.sort(key=lambda v: v.fitness_score)

        # Prune bottom percentage
        prune_count = int(len(candidates) * self.survival_threshold)
        for variant in candidates[:prune_count]:
            variant.is_alive = False
            logger.debug(
                f"Pruned variant {variant.variant_id} with fitness {variant.fitness_score:.3f}"
            )

        alive.extend([v for v in candidates if v.is_alive])
        self._gene_pool = alive
        self._last_prune_time = now

    def breed_next_generation(self, bounds: Dict[str, List[float]]) -> List[PipelineVariant]:
        """
        Breed the next generation using crossover and mutation.

        Returns a list of new PipelineVariant objects.
        """
        if not self.enabled:
            return []

        self._generation += 1
        new_variants: List[PipelineVariant] = []

        alive = self.get_gene_pool()
        if len(alive) < 2:
            # Not enough variants to breed; mutate baseline
            baseline = self._get_baseline()
            if baseline:
                new_vec = self._mutate_vector(baseline.vector, bounds)
                new_variants.append(
                    PipelineVariant(
                        variant_id=f"gen{self._generation}_mut0",
                        vector=new_vec,
                        generation=self._generation,
                        parent_id="baseline",
                    )
                )
            return new_variants

        # Select elite pool (top 30%)
        alive.sort(key=lambda v: v.fitness_score, reverse=True)
        elite_count = max(2, int(len(alive) * 0.3))
        elite = alive[:elite_count]

        # Detect fitness plateau
        self._detect_plateau()

        # Breed new variants
        attempts = 0
        while len(new_variants) < self.population_cap // 2 and attempts < self.max_mutation_attempts * 10:
            attempts += 1

            # Crossover: pick two elite parents
            if len(elite) >= 2 and random.random() < self.crossover_rate:
                parent1 = random.choice(elite)
                parent2 = random.choice([e for e in elite if e.variant_id != parent1.variant_id] or elite)
                child_vec = self._crossover(parent1.vector, parent2.vector, bounds)
            else:
                # Mutation: pick one elite parent
                parent = random.choice(elite)
                child_vec = self._mutate_vector(parent.vector, bounds)

            variant_id = f"gen{self._generation}_v{len(new_variants)}"
            new_variants.append(
                PipelineVariant(
                    variant_id=variant_id,
                    vector=child_vec,
                    generation=self._generation,
                    parent_id=parent.variant_id if not (len(elite) >= 2 and random.random() < self.crossover_rate) else f"{parent1.variant_id}+{parent2.variant_id}",
                )
            )

        logger.info(
            f"Bred {len(new_variants)} new variants at generation {self._generation} "
            f"(mutation_rate={self._current_mutation_rate:.3f})"
        )
        return new_variants

    def _crossover(
        self,
        v1: OptimizationVector,
        v2: OptimizationVector,
        bounds: Dict[str, List[float]],
    ) -> OptimizationVector:
        """Perform uniform crossover between two vectors."""
        child = OptimizationVector()
        for attr_name in vars(child).keys():
            # 50/50 chance from either parent
            if random.random() < 0.5:
                setattr(child, attr_name, copy.deepcopy(getattr(v1, attr_name)))
            else:
                setattr(child, attr_name, copy.deepcopy(getattr(v2, attr_name)))
        child.clamp_to_bounds(bounds)
        return child

    def _mutate_vector(
        self,
        vector: OptimizationVector,
        bounds: Dict[str, List[float]],
    ) -> OptimizationVector:
        """Apply mutation to a vector."""
        new_vec = vector.clone()
        for attr_name in vars(new_vec).keys():
            lo, hi = bounds.get(attr_name, (0.0, 1.0))
            if random.random() < self._current_mutation_rate:
                current = getattr(new_vec, attr_name)
                range_size = hi - lo
                delta = random.gauss(0, range_size * 0.1)
                mutated = current + delta
                clamped = max(lo, min(hi, mutated))
                if attr_name in ("tracking_history_buffer", "frame_skipping_cadence"):
                    clamped = int(round(clamped))
                setattr(new_vec, attr_name, clamped)
        return new_vec

    def _detect_plateau(self):
        """Detect if fitness has plateaued and adjust mutation rate."""
        if not self.adaptive_rate:
            return

        # Collect recent average fitness values
        alive = self.get_gene_pool()
        if len(alive) < 3:
            return

        avg_fitness = np.mean([v.fitness_score for v in alive])
        self._last_fitness_values.append(avg_fitness)

        # Keep window of 10
        if len(self._last_fitness_values) > 10:
            self._last_fitness_values.pop(0)

        if len(self._last_fitness_values) < 5:
            return

        # Check if fitness has plateaued (low variance in last 5 values)
        recent = self._last_fitness_values[-5:]
        variance = np.var(recent)

        if variance < 0.001:
            self._plateau_count += 1
            # Increase mutation rate to break out of plateau
            self._current_mutation_rate = min(0.5, self._current_mutation_rate * 1.5)
            logger.debug(
                f"Fitness plateau detected (var={variance:.6f}), "
                f"mutation rate increased to {self._current_mutation_rate:.3f}"
            )
        else:
            self._plateau_count = 0
            # Gradually decay mutation rate back toward initial
            self._current_mutation_rate = max(
                self.initial_mutation_rate,
                self._current_mutation_rate * 0.98,
            )

    def _prune_weakest(self):
        """Remove the single weakest variant from the gene pool."""
        alive = [v for v in self._gene_pool if v.variant_id != "baseline" and v.is_alive]
        if alive:
            weakest = min(alive, key=lambda v: v.fitness_score)
            weakest.is_alive = False
            logger.debug(f"Pruned weakest variant {weakest.variant_id}")

    def _get_baseline(self) -> Optional[PipelineVariant]:
        """Get the baseline variant."""
        for v in self._gene_pool:
            if v.variant_id == "baseline":
                return v
        return None

    @staticmethod
    def _compute_survival_threshold(config) -> float:
        """Compute the survival threshold from config."""
        return config.evaluator.survival_threshold if hasattr(config, 'evaluator') else 0.4


# =============================================================================
# EvolutionaryEngine — Main Service Class
# =============================================================================


class EvolutionaryEngine:
    """
    Main service class for the Generative-Evolutionary Cognition Engine.

    Lifecycle:
        __init__(): Load config, instantiate sub-components, init gene pool
        start():    Start background evolution loop
        stop():     Stop evolution loop, persist best variant
    """

    def __init__(self):
        self.config = get_config()
        self.ee_config = self.config.evolutionary_engine
        self.enabled = self.ee_config.enabled

        # Sub-components
        self.synthesizer = GenerativeSynthesizer(self.ee_config)
        self.evaluator = EvolutionaryEvaluator(self.ee_config)
        self.orchestrator = MutationOrchestrator(self.ee_config)

        # Evolution loop control
        self._evolution_task: Optional[asyncio.Task] = None
        self._running = False
        self._last_breed_time = 0.0

        # Active optimization vector (hot-swappable)
        self._active_vector: OptimizationVector = OptimizationVector()

        # Event store for persisting best variants
        self.event_store = get_event_store()

        # Zero-cost fallback vector
        self._fallback_vector: OptimizationVector = OptimizationVector()

        logger.info("EvolutionaryEngine initialized")

    @property
    def active_vector(self) -> OptimizationVector:
        """Get the currently active optimization vector (thread-safe)."""
        return self._active_vector

    @active_vector.setter
    def active_vector(self, vector: OptimizationVector):
        """Set the active optimization vector."""
        self._active_vector = vector

    async def start(self):
        """Start the evolutionary engine background loop."""
        if not self.enabled:
            logger.info("EvolutionaryEngine is disabled (zero-cost fallback active)")
            self._active_vector = self._fallback_vector
            return

        if self._running:
            logger.warning("EvolutionaryEngine already running")
            return

        self._running = True
        self._evolution_task = asyncio.create_task(self._evolution_loop())
        logger.info("EvolutionaryEngine started")

    async def stop(self):
        """Stop the evolutionary engine and persist best variant."""
        if not self._running:
            return

        self._running = False
        if self._evolution_task:
            self._evolution_task.cancel()
            try:
                await self._evolution_task
            except asyncio.CancelledError:
                pass
            self._evolution_task = None

        # Persist best variant metadata
        best = self.orchestrator.get_optimal_vector()
        logger.info(
            f"EvolutionaryEngine stopped. Best vector: "
            f"conf={best.yolo_conf_threshold:.3f}, "
            f"iou={best.iou_threshold:.3f}, "
            f"buffer={best.tracking_history_buffer}, "
            f"skip={best.frame_skipping_cadence}, "
            f"cooldown={best.rules_engine_cooldown:.1f}s"
        )
        self._save_best_variant(best)

    def record_frame_metrics(
        self,
        camera_id: int,
        inference_time_ms: float,
        tracking_accuracy: float,
        false_positive_ratio: float,
        num_detections: int,
        kafka_latency_ms: Optional[float] = None,
        rule_precision: Optional[float] = None,
        processing_time_ms: float = 0.0,
    ):
        """Record per-frame metrics into the evaluator."""
        if not self.enabled:
            return

        metrics = FrameMetrics(
            camera_id=camera_id,
            timestamp=time.time(),
            inference_time_ms=inference_time_ms,
            tracking_accuracy=tracking_accuracy,
            false_positive_ratio=false_positive_ratio,
            num_detections=num_detections,
            kafka_latency_ms=kafka_latency_ms,
            rule_precision=rule_precision,
            processing_time_ms=processing_time_ms,
        )
        self.evaluator.record_metrics(camera_id, metrics)

    def get_optimization_vector(self) -> Dict:
        """
        Get the current best optimization vector for pipeline hot-swapping.

        Falls back to default values if engine is disabled or has no data.
        """
        if not self.enabled:
            return self._fallback_vector.to_dict()

        try:
            best = self.orchestrator.get_optimal_vector()
            self._active_vector = best
            return best.to_dict()
        except Exception as e:
            logger.error(f"Error getting optimization vector: {e}")
            return self._fallback_vector.to_dict()

    def get_status(self) -> Dict:
        """Get full status of the evolutionary engine."""
        if not self.enabled:
            return {
                "enabled": False,
                "status": "disabled",
                "message": "Zero-cost fallback active — using static config values",
            }

        snapshot = self.evaluator.get_fitness_snapshot()
        alive_variants = len(self.orchestrator.get_gene_pool())
        optimal = self.orchestrator.get_optimal_vector()

        return {
            "enabled": True,
            "running": self._running,
            "generation": self.orchestrator.generation,
            "mutation_rate": self.orchestrator.current_mutation_rate,
            "gene_pool_size": alive_variants,
            "population_cap": self.orchestrator.population_cap,
            "total_frames_evaluated": snapshot.get("frame_count", 0),
            "overall_fitness": snapshot.get("overall_fitness", 0.0),
            "component_scores": snapshot.get("component_scores", {}),
            "camera_count": snapshot.get("camera_count", 0),
            "active_vector": optimal.to_dict(),
            "fallback_active": False,
        }

    def _save_best_variant(self, vector: OptimizationVector):
        """Log the best variant for reference across restarts."""
        try:
            self.event_store.create_event(
                camera_id=0,
                rule_type="evolutionary_snapshot",
                object_type="system",
                confidence=1.0,
                bbox=[0, 0, 0, 0],
                snapshot_path="",
                priority="low",
                metadata={
                    "vector": vector.to_dict(),
                    "generation": self.orchestrator.generation,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.warning(f"Could not persist best variant: {e}")

    async def _evolution_loop(self):
        """Main background evolution loop — runs as an asyncio task."""
        logger.info("Evolution loop started")

        try:
            while self._running:
                await asyncio.sleep(2.0)  # Check every 2 seconds

                if not self.enabled:
                    continue

                # 1. Feed metrics snapshot to synthesizer
                snapshot = self.evaluator.get_fitness_snapshot()
                self.synthesizer.feed_metrics(snapshot)

                # 2. Check if we should generate new variants
                frame_count = self.evaluator.get_frame_count()
                min_frames = self.ee_config.evaluator.evaluation_window

                if frame_count < min_frames:
                    continue  # Not enough data yet

                if not self.synthesizer.can_generate():
                    continue

                # 3. Score all existing variants
                bounds = self._get_vector_bounds()
                fitness_scores: Dict[str, float] = {}

                for variant in self.orchestrator.get_gene_pool():
                    try:
                        score = self.evaluator.compute_variant_fitness(variant, snapshot)
                        fitness_scores[variant.variant_id] = score
                    except Exception as e:
                        # Sandboxed execution: failed variant gets zero fitness
                        logger.warning(
                            f"Variant {variant.variant_id} threw error, zeroing fitness: {e}"
                        )
                        fitness_scores[variant.variant_id] = 0.0
                        variant.error_count += 1
                        variant.is_alive = False

                # 4. Prune low performers
                self.orchestrator.prune_population(fitness_scores)

                # 5. Breed next generation
                new_variants = self.orchestrator.breed_next_generation(bounds)
                for variant in new_variants:
                    # Sanity-check the variant vector before adding
                    try:
                        variant.vector.clamp_to_bounds(bounds)
                        self.orchestrator.add_variant(variant)
                    except Exception as e:
                        logger.warning(f"Failed to add new variant, dropping: {e}")

                # 6. Update active vector to best performer
                try:
                    best = self.orchestrator.get_optimal_vector()
                    self._active_vector = best
                    logger.info(
                        f"Evolution cycle complete — generation {self.orchestrator.generation}, "
                        f"pool size {len(self.orchestrator.get_gene_pool())}, "
                        f"best fitness: {max(fitness_scores.values()) if fitness_scores else 0.0:.3f}"
                    )
                except Exception as e:
                    logger.error(f"Error updating active vector: {e}")

        except asyncio.CancelledError:
            logger.info("Evolution loop cancelled")
        except Exception as e:
            logger.error(f"Evolution loop error: {e}")
            self._running = False

    def _get_vector_bounds(self) -> Dict[str, List[float]]:
        """Extract vector bounds from config."""
        bounds = self.ee_config.mutation.vector_bounds
        return {
            "yolo_conf_threshold": bounds.yolo_conf_threshold,
            "iou_threshold": bounds.iou_threshold,
            "tracking_history_buffer": bounds.tracking_history_buffer,
            "frame_skipping_cadence": bounds.frame_skipping_cadence,
            "rules_engine_cooldown": bounds.rules_engine_cooldown,
        }


# =============================================================================
# Global Instance
# =============================================================================

_evolutionary_engine: Optional[EvolutionaryEngine] = None


def get_evolutionary_engine() -> EvolutionaryEngine:
    """Get global EvolutionaryEngine instance (singleton)."""
    global _evolutionary_engine
    if _evolutionary_engine is None:
        _evolutionary_engine = EvolutionaryEngine()
    return _evolutionary_engine