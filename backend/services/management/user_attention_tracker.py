"""
User Interaction & Viewport Attention Matrix

Tracks user viewport focus and interaction patterns to dynamically redirect
compute budgets toward human-focused areas of interest.

Features:
  - Thread-safe per-camera attention state lookup table
  - WebSocket connection counting for active viewport tracking
  - Interaction event logging (clicks, expands, overrides)
  - Attention decay over time when users disengage
  - Zero-cost fallback when disabled

Lifecycle:
    __init__(): Load config, init thread-safe state
    start():    Start background attention decay thread
    stop():     Stop decay thread
"""

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, List

from ...config.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class CameraAttentionState:
    """Per-camera attention state — immutable on read."""

    camera_id: int
    active_viewers: int = 0
    is_expanded: bool = False
    last_interaction_time: float = 0.0
    interaction_count: int = 0
    base_multiplier: float = 1.0
    last_updated: float = field(default_factory=time.time)


@dataclass
class UserInteraction:
    """A single user interaction event."""

    camera_id: int
    action_type: str  # "view", "click", "expand", "override"
    timestamp: float = field(default_factory=time.time)
    metadata: Optional[Dict] = None


# =============================================================================
# UserAttentionTracker
# =============================================================================


class UserAttentionTracker:
    """
    Tracks user viewport focus and interaction patterns.

    Hook methods (called by WebSocket/REST route handlers):
        register_active_stream(camera_id)   — WebSocket connected
        unregister_active_stream(camera_id) — WebSocket disconnected
        log_user_interaction(camera_id, type) — user clicked/expanded
        set_expanded_view(camera_id, bool)  — full-screen toggle

    Query methods:
        get_attention_multiplier(camera_id) — 0.3-2.0 per camera
        get_all_attention_multipliers()     — all cameras at once
    """

    def __init__(self):
        self.config = get_config()
        self.ua_config = self.config.user_attention
        self.enabled = self.ua_config.enabled

        # Config values
        self.active_view_mult = self.ua_config.active_view_multiplier
        self.click_boost = self.ua_config.click_interaction_boost
        self.expand_mult = self.ua_config.expanded_view_multiplier
        self.click_duration = self.ua_config.click_boost_duration_s
        self.decay_rate = self.ua_config.decay_rate_seconds
        self.unattended_mult = self.ua_config.unattended_camera_multiplier

        # Thread-safe state
        self._lock = threading.RLock()
        self._camera_states: Dict[int, CameraAttentionState] = {}
        self._interaction_history: List[UserInteraction] = []

        # Decay thread
        self._decay_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        logger.info(f"UserAttentionTracker initialized (enabled={self.enabled})")

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def start(self):
        """Start background attention decay thread."""
        if not self.enabled:
            logger.info("UserAttentionTracker is disabled (zero-cost fallback)")
            return

        if self._decay_thread and self._decay_thread.is_alive():
            return

        self._stop_event.clear()
        self._decay_thread = threading.Thread(
            target=self._decay_loop,
            daemon=True,
            name="attention-decay",
        )
        self._decay_thread.start()
        logger.info("UserAttentionTracker started")

    async def stop(self):
        """Stop background decay thread."""
        self._stop_event.set()
        if self._decay_thread:
            self._decay_thread.join(timeout=2.0)
            self._decay_thread = None
        logger.info("UserAttentionTracker stopped")

    # ── Hook Methods (called by routes) ──

    def register_active_stream(self, camera_id: int):
        """
        Called when a WebSocket connection opens for a camera stream.
        Increments the active_viewers counter.
        """
        if not self.enabled:
            return

        with self._lock:
            state = self._get_or_create_state(camera_id)
            state.active_viewers += 1
            state.last_updated = time.time()
            logger.debug(f"Stream registered for camera {camera_id} (viewers={state.active_viewers})")

    def unregister_active_stream(self, camera_id: int):
        """
        Called when a WebSocket connection closes.
        Decrements the active_viewers counter.
        """
        if not self.enabled:
            return

        with self._lock:
            state = self._camera_states.get(camera_id)
            if state and state.active_viewers > 0:
                state.active_viewers -= 1
                state.last_updated = time.time()
                logger.debug(
                    f"Stream unregistered for camera {camera_id} (viewers={state.active_viewers})"
                )

    def log_user_interaction(self, camera_id: int, action_type: str, metadata: Optional[Dict] = None):
        """
        Called when a user clicks a detection, expands a view, or overrides a rule.

        Args:
            camera_id: The camera the user interacted with
            action_type: "click", "expand", "override", "focus"
            metadata: Optional dict with additional context
        """
        if not self.enabled:
            return

        interaction = UserInteraction(
            camera_id=camera_id,
            action_type=action_type,
            metadata=metadata,
        )

        with self._lock:
            self._interaction_history.append(interaction)
            # Keep history bounded
            if len(self._interaction_history) > 1000:
                self._interaction_history = self._interaction_history[-500:]

            state = self._get_or_create_state(camera_id)
            state.last_interaction_time = time.time()
            state.interaction_count += 1

            # Apply specific boosts based on action type
            if action_type == "expand":
                state.is_expanded = True
            elif action_type == "click":
                pass  # Timer-based boost handled in multiplier computation

            state.last_updated = time.time()

        logger.debug(f"User interaction logged: camera={camera_id}, action={action_type}")

    def set_expanded_view(self, camera_id: int, is_expanded: bool):
        """
        Called when a user expands a camera to full-screen or collapses it.

        Args:
            camera_id: The camera being expanded/collapsed
            is_expanded: True for expansion, False for collapse
        """
        if not self.enabled:
            return

        with self._lock:
            state = self._get_or_create_state(camera_id)
            state.is_expanded = is_expanded
            if is_expanded:
                state.last_interaction_time = time.time()
            state.last_updated = time.time()

        logger.debug(f"Camera {camera_id} expanded={is_expanded}")

    # ── Query Methods ──

    def get_attention_multiplier(self, camera_id: int) -> float:
        """
        Get current attention multiplier for a camera (0.3-2.0).

        Formula:
            base = 1.0
            if active_viewers > 0: base *= active_view_mult
            if is_expanded:        base *= expanded_view_mult
            if recent_interaction: base *= click_interaction_boost
            if active_viewers == 0: base *= unattended_camera_mult

            Apply decay over time since last interaction.

        Returns:
            Multiplier clamped to [0.3, 2.0]
        """
        if not self.enabled:
            return 1.0

        with self._lock:
            state = self._camera_states.get(camera_id)
            if state is None:
                return self.unattended_mult  # Unknown camera = minimal attention

            # Start with baseline
            multiplier = state.base_multiplier  # 1.0

            # Active viewers boost
            if state.active_viewers > 0:
                multiplier *= self.active_view_mult
            else:
                multiplier *= self.unattended_mult

            # Expanded view boost
            if state.is_expanded:
                multiplier *= self.expand_mult

            # Recent interaction boost (timed)
            now = time.time()
            elapsed_since_interaction = now - state.last_interaction_time

            if elapsed_since_interaction < self.click_duration and state.interaction_count > 0:
                # Full boost during click duration window
                boost = self.click_boost
                # Scale boost down as time passes within the window
                progress = elapsed_since_interaction / self.click_duration
                boost = 1.0 + (boost - 1.0) * (1.0 - progress)
                multiplier *= boost
            elif elapsed_since_interaction < self.click_duration + self.decay_rate:
                # Decay phase: linearly decrease from boost back to baseline
                decay_progress = (elapsed_since_interaction - self.click_duration) / self.decay_rate
                multiplier = 1.0 + (multiplier - 1.0) * (1.0 - decay_progress)

            return max(0.3, min(2.0, multiplier))

    def get_all_attention_multipliers(self) -> Dict[int, float]:
        """Get attention multipliers for all known cameras."""
        with self._lock:
            return {
                cid: self.get_attention_multiplier(cid)
                for cid in list(self._camera_states.keys())
            }

    def get_status(self) -> Dict:
        """Return full attention state for monitoring API."""
        with self._lock:
            return {
                "enabled": self.enabled,
                "active_cameras": len(self._camera_states),
                "total_interactions": len(self._interaction_history),
                "decay_thread_alive": self._decay_thread is not None and self._decay_thread.is_alive(),
                "cameras": {
                    cid: {
                        "active_viewers": s.active_viewers,
                        "is_expanded": s.is_expanded,
                        "interaction_count": s.interaction_count,
                        "multiplier": self.get_attention_multiplier(cid),
                        "last_updated": s.last_updated,
                    }
                    for cid, s in self._camera_states.items()
                },
            }

    # ── Internal ──

    def _get_or_create_state(self, camera_id: int) -> CameraAttentionState:
        """Get existing state or create a new one."""
        if camera_id not in self._camera_states:
            self._camera_states[camera_id] = CameraAttentionState(camera_id=camera_id)
        return self._camera_states[camera_id]

    def _decay_loop(self):
        """
        Background loop that gradually resets interaction counts.

        Runs every viewport_poll_interval_s seconds. After decay_rate_seconds
        of no interactions, resets the interaction counter for that camera.
        """
        interval = self.ua_config.viewport_poll_interval_s

        while not self._stop_event.is_set():
            try:
                now = time.time()
                with self._lock:
                    for camera_id, state in self._camera_states.items():
                        # If no interaction for decay_rate + click_duration, reset counter
                        idle_time = now - state.last_interaction_time
                        if idle_time > (self.click_duration + self.decay_rate) and state.interaction_count > 0:
                            state.interaction_count = 0
                            logger.debug(f"Attention decayed to baseline for camera {camera_id}")

                # Sleep with stop-event checking
                for _ in range(int(interval * 2)):
                    if self._stop_event.is_set():
                        return
                    time.sleep(0.5)

            except Exception as e:
                logger.warning(f"Attention decay error: {e}")
                time.sleep(1.0)


# =============================================================================
# Global Instance
# =============================================================================

_user_attention_tracker: Optional[UserAttentionTracker] = None


def get_user_attention_tracker() -> UserAttentionTracker:
    """Get global UserAttentionTracker instance (singleton)."""
    global _user_attention_tracker
    if _user_attention_tracker is None:
        _user_attention_tracker = UserAttentionTracker()
    return _user_attention_tracker