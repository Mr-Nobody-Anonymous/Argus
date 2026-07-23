"""
Consortium Attention Board — Decentralized Multi-Agent Swarm Blackboard

Acts as a zero-latency, thread-safe shared memory matrix where autonomous
SwarmAgents post real-time inference confidence, workload intensity, and
hardware utilization. The broker resolves resource contention via a
proportional bidding algorithm, allocating compute budgets per frame.

Architecture:
  - Thread-safe blackboard cache with read/write locks
  - Proportional resource-bidding algorithm
  - Rapid async post/read cycle with configurable sync interval
  - Zero-cost fallback when consortium is disabled
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backend.config.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class AgentBid:
    """A single agent's resource bid for the current frame cycle."""

    agent_id: str
    domain: str  # "yolo", "face", "lpr"
    urgency: float  # 0.0-1.0 — how critical is this domain right now
    compute_cost: float  # estimated ms per frame
    contextual_relevance: float  # 0.0-1.0 — relevance to current scene
    current_load: float  # 0.0-1.0 — current utilization
    bid_time: float = field(default_factory=time.time)


@dataclass
class ResourceAllocation:
    """Allocation result for a single agent after bidding resolution."""

    agent_id: str
    allocated_budget_ms: float  # time budget per frame
    priority_boost: float  # 0.5-2.0 multiplier on thresholds
    throttle_factor: float  # 0.0-1.0 — frame skipping multiplier
    should_process: bool = True  # whether agent should run this cycle


@dataclass
class ContextPost:
    """A contextual clue posted to the blackboard by an agent."""

    agent_id: str
    domain: str
    key: str  # e.g., "vehicle_detected", "human_detected"
    value: float  # confidence or count
    timestamp: float = field(default_factory=time.time)
    ttl_seconds: float = 2.0  # auto-expire after this time


# =============================================================================
# ConsortiumBroker
# =============================================================================


class ConsortiumBroker:
    """
    Thread-safe shared blackboard and resource bidding engine.

    Agents post contextual clues and submit bids. The broker resolves
    allocations using a proportional scoring algorithm that accounts for
    urgency, contextual relevance, and compute cost.
    """

    def __init__(self):
        self.config = get_config()
        self.consortium_config = self.config.consortium
        self.enabled = self.consortium_config.enabled
        self.sync_interval_ms = self.consortium_config.sync_interval_ms
        self.bidding_strategy = self.consortium_config.bidding_strategy
        self.max_agents = self.consortium_config.max_agents
        self.resource_bidding_enabled = self.consortium_config.resource_bidding_enabled

        # Thread-safe blackboard state
        self._lock = threading.RLock()
        self._board: Dict[str, ContextPost] = {}  # key -> ContextPost
        self._bids: Dict[str, AgentBid] = {}  # agent_id -> AgentBid
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._last_sync_time: float = 0.0
        self._cycle_count: int = 0

        # Agent registry
        self._registered_agents: Dict[str, str] = {}  # agent_id -> domain

        # Default total budget per frame (ms) — split across agents
        self._total_budget_ms: float = 33.0  # ~30 FPS target

        # Telemetry monitor reference (lazy init — avoids circular imports)
        self._telemetry_monitor = None
        # Attention tracker reference (lazy init — avoids circular imports)
        self._attention_tracker = None

        logger.info(
            f"ConsortiumBroker initialized (enabled={self.enabled}, "
            f"strategy={self.bidding_strategy}, max_agents={self.max_agents})"
        )

    # ── Agent Registration ──────────────────────────────────────────────

    def register_agent(self, agent_id: str, domain: str) -> bool:
        """Register an agent with the broker."""
        with self._lock:
            if len(self._registered_agents) >= self.max_agents:
                logger.warning(f"Max agents ({self.max_agents}) reached, cannot register {agent_id}")
                return False
            self._registered_agents[agent_id] = domain
            logger.debug(f"Agent {agent_id} ({domain}) registered with broker")
            return True

    def unregister_agent(self, agent_id: str):
        """Unregister an agent from the broker."""
        with self._lock:
            self._registered_agents.pop(agent_id, None)
            self._bids.pop(agent_id, None)
            self._allocations.pop(agent_id, None)
            # Remove all posts from this agent
            keys_to_remove = [
                k for k, v in self._board.items() if v.agent_id == agent_id
            ]
            for k in keys_to_remove:
                del self._board[k]

    # ── Blackboard Context Operations ───────────────────────────────────

    def post_context(self, agent_id: str, domain: str, key: str, value: float, ttl: float = 2.0):
        """
        Post a contextual clue to the blackboard.

        Args:
            agent_id: The posting agent's ID
            domain: Agent domain ("yolo", "face", "lpr")
            key: Context key (e.g., "vehicle_detected", "human_detected")
            value: Numeric value (confidence, count, etc.)
            ttl: Time-to-live in seconds before auto-expiry
        """
        if not self.enabled:
            return

        with self._lock:
            self._board[key] = ContextPost(
                agent_id=agent_id,
                domain=domain,
                key=key,
                value=value,
                timestamp=time.time(),
                ttl_seconds=ttl,
            )

    def read_context(self, key: str, default: float = 0.0) -> float:
        """
        Read a contextual clue from the blackboard.

        Args:
            key: Context key to read
            default: Default value if key not found or expired

        Returns:
            The context value, or default if not present/expired
        """
        if not self.enabled:
            return default

        with self._lock:
            post = self._board.get(key)
            if post is None:
                return default
            # Check expiry
            if time.time() - post.timestamp > post.ttl_seconds:
                del self._board[key]
                return default
            return post.value

    def read_all_context(self) -> Dict[str, float]:
        """Read all non-expired context from the blackboard."""
        if not self.enabled:
            return {}

        with self._lock:
            now = time.time()
            result = {}
            expired_keys = []
            for key, post in self._board.items():
                if now - post.timestamp > post.ttl_seconds:
                    expired_keys.append(key)
                else:
                    result[key] = post.value
            for k in expired_keys:
                del self._board[k]
            return result

    # ── Bidding Operations ──────────────────────────────────────────────

    def submit_bid(self, bid: AgentBid):
        """
        Submit a resource bid for the current frame cycle.

        Args:
            bid: The AgentBid containing urgency, cost, relevance, load
        """
        if not self.enabled or not self.resource_bidding_enabled:
            return

        with self._lock:
            self._bids[bid.agent_id] = bid

    def resolve_cycle(self) -> Dict[str, ResourceAllocation]:
        """
        Run the proportional resource-bidding algorithm.

        Returns:
            Dict mapping agent_id -> ResourceAllocation for this cycle
        """
        if not self.enabled:
            return {}

        with self._lock:
            self._cycle_count += 1
            now = time.time()

            # Check sync interval
            if now - self._last_sync_time < (self.sync_interval_ms / 1000.0):
                return dict(self._allocations)

            # Clean expired context
            self._expire_context()

            if not self._bids:
                # No bids — give all agents equal share
                return self._default_allocations()

            # Compute scores using proportional bidding algorithm
            scores: Dict[str, float] = {}
            for agent_id, bid in self._bids.items():
                if bid.compute_cost <= 0:
                    score = 1.0
                else:
                    # score = urgency * (1 + contextual_relevance) / compute_cost
                    score = bid.urgency * (1.0 + bid.contextual_relevance) / max(bid.compute_cost, 0.1)
                scores[agent_id] = score

            # ── Ingest Real-Time Hardware Telemetry ──
            # Apply Dynamic Stress Multiplier to all scores
            telemetry = self._get_telemetry_monitor()
            stress_multiplier = telemetry.get_stress_multiplier()
            if stress_multiplier < 1.0:
                for agent_id in scores:
                    scores[agent_id] *= stress_multiplier
                logger.info(
                    f"Dynamic Stress Multiplier active: {stress_multiplier:.3f} "
                    f"(system under load — all agent bids scaled down)"
                )

            # ── Ingest User Attention Multiplier ──
            # Scale scores based on viewport focus (human attention)
            attention_tracker = self._get_attention_tracker()
            if attention_tracker is not None:
                all_attention = attention_tracker.get_all_attention_multipliers()
                if all_attention:
                    max_attention = max(all_attention.values())
                    for agent_id in scores:
                        scores[agent_id] *= max_attention
                    if max_attention > 1.0:
                        logger.debug(
                            f"User Attention Multiplier active: {max_attention:.2f} "
                            f"(viewport focus boosting agent bids)"
                        )

            # Normalize scores
            total_score = sum(scores.values())
            if total_score <= 0:
                return self._default_allocations()

            normalized_scores = {
                aid: s / total_score for aid, s in scores.items()
            }

            # Compute mean score for priority boost baseline
            mean_score = total_score / max(len(scores), 1)

            # Allocate budget proportionally
            new_allocations: Dict[str, ResourceAllocation] = {}
            for agent_id, norm_score in normalized_scores.items():
                allocated_budget = self._total_budget_ms * norm_score
                priority_boost = max(0.5, min(2.0, scores[agent_id] / max(mean_score, 0.01)))
                bid = self._bids[agent_id]

                # Throttle factor: if allocated budget < compute cost, throttle
                if bid.compute_cost > 0 and allocated_budget < bid.compute_cost:
                    throttle = allocated_budget / bid.compute_cost
                else:
                    throttle = 1.0

                # Apply stress multiplier to throttle as well for aggressive downscale
                throttle *= stress_multiplier

                new_allocations[agent_id] = ResourceAllocation(
                    agent_id=agent_id,
                    allocated_budget_ms=allocated_budget,
                    priority_boost=priority_boost,
                    throttle_factor=max(0.1, min(1.0, throttle)),
                    should_process=throttle > 0.05,
                )

            self._allocations = new_allocations
            self._last_sync_time = now

            logger.debug(
                f"Broker cycle {self._cycle_count}: resolved {len(new_allocations)} allocations"
            )
            return dict(self._allocations)

    def get_allocation(self, agent_id: str) -> ResourceAllocation:
        """Get the current allocation for a specific agent."""
        with self._lock:
            alloc = self._allocations.get(agent_id)
            if alloc is None:
                return ResourceAllocation(
                    agent_id=agent_id,
                    allocated_budget_ms=self._total_budget_ms / max(len(self._registered_agents), 1),
                    priority_boost=1.0,
                    throttle_factor=1.0,
                    should_process=True,
                )
            return alloc

    # ── Status & Monitoring ─────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Get full status of the consortium broker."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "cycle_count": self._cycle_count,
                "registered_agents": dict(self._registered_agents),
                "active_bids": len(self._bids),
                "active_context_posts": len(self._board),
                "allocations": {
                    aid: {
                        "budget_ms": round(alloc.allocated_budget_ms, 2),
                        "priority_boost": round(alloc.priority_boost, 2),
                        "throttle": round(alloc.throttle_factor, 2),
                        "should_process": alloc.should_process,
                    }
                    for aid, alloc in self._allocations.items()
                },
                "total_budget_ms": self._total_budget_ms,
                "bidding_strategy": self.bidding_strategy,
            }

    def set_total_budget(self, budget_ms: float):
        """Set the total per-frame budget in milliseconds."""
        with self._lock:
            self._total_budget_ms = max(1.0, budget_ms)

    def clear_volatile_state(self):
        """Wipe bids, allocations, and context posts. Called during recovery."""
        with self._lock:
            self._bids.clear()
            self._allocations.clear()
            self._board.clear()
            self._last_sync_time = 0.0
        logger.info("ConsortiumBroker volatile state cleared (recovery)")

    # ── Telemetry Integration ───────────────────────────────────────────

    def _get_telemetry_monitor(self):
        """Lazy-init telemetry monitor reference (avoids circular imports)."""
        if self._telemetry_monitor is None:
            try:
                from ..management.telemetry_monitor import get_telemetry_monitor as _g_tm
                self._telemetry_monitor = _g_tm()
            except Exception as e:
                logger.warning(f"Could not init telemetry monitor: {e}")
                # Try with full backend path
                try:
                    from backend.services.management.telemetry_monitor import get_telemetry_monitor as _g_tm2
                    self._telemetry_monitor = _g_tm2()
                except Exception as e2:
                    logger.warning(f"Could not init telemetry monitor (backend path): {e2}")
                    return None
        return self._telemetry_monitor

    def _get_attention_tracker(self):
        """Lazy-init user attention tracker reference (avoids circular imports)."""
        if self._attention_tracker is None:
            try:
                from ..management.user_attention_tracker import get_user_attention_tracker as _g_uat
                self._attention_tracker = _g_uat()
            except Exception as e:
                logger.warning(f"Could not init attention tracker: {e}")
                # Try with full backend path
                try:
                    from backend.services.management.user_attention_tracker import get_user_attention_tracker as _g_uat2
                    self._attention_tracker = _g_uat2()
                except Exception as e2:
                    logger.warning(f"Could not init attention tracker (backend path): {e2}")
                    return None
        return self._attention_tracker

    # ── Internal Helpers ────────────────────────────────────────────────

    def _default_allocations(self) -> Dict[str, ResourceAllocation]:
        """Generate equal-share allocations when no bids are present."""
        agent_count = max(len(self._registered_agents), 1)
        equal_budget = self._total_budget_ms / agent_count

        allocs = {}
        for agent_id in self._registered_agents:
            allocs[agent_id] = ResourceAllocation(
                agent_id=agent_id,
                allocated_budget_ms=equal_budget,
                priority_boost=1.0,
                throttle_factor=1.0,
                should_process=True,
            )
        self._allocations = allocs
        return dict(allocs)

    def _expire_context(self):
        """Remove expired context posts from the blackboard."""
        now = time.time()
        expired = [
            k for k, v in self._board.items()
            if now - v.timestamp > v.ttl_seconds
        ]
        for k in expired:
            del self._board[k]


# =============================================================================
# Global Instance
# =============================================================================

_consortium_broker: Optional[ConsortiumBroker] = None


def get_consortium_broker() -> ConsortiumBroker:
    """Get global ConsortiumBroker instance (singleton)."""
    global _consortium_broker
    if _consortium_broker is None:
        _consortium_broker = ConsortiumBroker()
    return _consortium_broker