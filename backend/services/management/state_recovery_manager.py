"""
Byzantine Fault-Tolerant State Recovery Manager

Maintains an immutable ledger of system snapshots and runs an isolated
watchdog loop that monitors execution vitality. Automatically rolls back
to the last known stable state if a stall or metric collapse is detected.

Architecture:
  - Immutable ring buffer (deque) of SystemSnapshot objects
  - Isolated background watchdog thread that polls ConsortiumBroker and
    ProcessingCoordinator for execution vitality
  - Three-tier stall detection: heartbeat timeout, error collapse, metric collapse
  - Full recovery workflow: pipeline freeze → volatile wipe → snapshot reinjection

Lifecycle:
    __init__(): Load config, init ledger, lazy-init dependencies
    start():    Start watchdog background loop
    stop():     Stop watchdog loop
"""

import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

from config.config import get_config

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class SystemSnapshot:
    """Immutable snapshot of the entire system state at a point in time.

    Captures the complete evolutionary state needed to fully restore the
    swarm to a known-good configuration:
      - Active OptimizationVector values from the EvolutionaryEngine
      - Compiled code strings from the LogicMutator
      - ConsortiumBroker resource allocation state
      - Baseline performance metrics from the ProcessingCoordinator
    """

    snapshot_id: str
    timestamp: float
    optimization_vector: Dict[str, float] = field(default_factory=dict)
    logic_mutator_variants: Dict[str, str] = field(default_factory=dict)
    active_variant_id: Optional[str] = None
    consortium_broker_state: Dict = field(default_factory=dict)
    performance_metrics: Dict = field(default_factory=dict)
    error_count: int = 0
    is_stable: bool = False


@dataclass
class WatchdogStatus:
    """Live watchdog monitoring status — mutable, lock-protected."""

    last_heartbeat: float = field(default_factory=time.time)
    consecutive_errors: int = 0
    is_stall_detected: bool = False
    last_recovery_time: Optional[float] = None
    total_recoveries: int = 0


# =============================================================================
# StateRecoveryManager
# =============================================================================


class StateRecoveryManager:
    """
    Byzantine fault-tolerant state recovery manager.

    - Maintains an immutable ring buffer of system snapshots (ledger)
    - Runs an isolated watchdog loop that monitors execution vitality
    - Automatically rolls back to the last stable snapshot on stall
    - Logs all recovery events to data/recovery_log.json

    The watchdog polls the ConsortiumBroker and ProcessingCoordinator to
    verify execution vitality. Three detection mechanisms:
      1. Heartbeat timeout — no frame cycle heartbeat within threshold
      2. Error collapse — consecutive errors exceed configured limit
      3. Metric collapse — performance metrics degraded beyond threshold
    """

    def __init__(self):
        self.config = get_config()
        self.sr_config = self.config.state_recovery
        self.enabled = self.sr_config.enabled

        # Config values
        self.heartbeat_timeout_ms = self.sr_config.heartbeat_timeout_ms
        self.max_allowed_errors = self.sr_config.max_allowed_consecutive_errors
        self.ledger_history_limit = self.sr_config.ledger_history_limit
        self.watchdog_poll_interval_ms = self.sr_config.watchdog_poll_interval_ms
        self.auto_rollback_enabled = self.sr_config.auto_rollback_enabled
        self.log_recovery_events = self.sr_config.log_recovery_events

        # Immutable ledger (ring buffer) — stores complete system snapshots
        self._ledger: deque = deque(maxlen=self.ledger_history_limit)
        self._ledger_lock = threading.RLock()

        # Watchdog state
        self._watchdog_status = WatchdogStatus()
        self._status_lock = threading.RLock()

        # Lazy-init service references (avoids circular imports at construction)
        self._consortium_broker: Optional[Any] = None
        self._logic_mutator: Optional[Any] = None
        self._processing_coordinator: Optional[Any] = None
        self._evolutionary_engine: Optional[Any] = None

        # Background loop
        self._watchdog_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Recovery flag to prevent concurrent rollbacks
        self._recovering = False
        self._recovery_lock = threading.Lock()

        # Data directory for recovery logs
        self._data_dir = self._get_data_dir()

        logger.info(
            f"StateRecoveryManager initialized (enabled={self.enabled}, "
            f"heartbeat_timeout={self.heartbeat_timeout_ms}ms, "
            f"ledger_limit={self.ledger_history_limit})"
        )

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def start(self):
        """Start background watchdog monitoring loop."""
        if not self.enabled:
            logger.info("StateRecoveryManager is disabled (zero-cost fallback)")
            return

        if self._watchdog_thread and self._watchdog_thread.is_alive():
            logger.warning("StateRecoveryManager watchdog already running")
            return

        self._stop_event.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            daemon=True,
            name="state-recovery-watchdog",
        )
        self._watchdog_thread.start()
        logger.info("StateRecoveryManager watchdog started")

    async def stop(self):
        """Stop watchdog loop."""
        self._stop_event.set()
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=3.0)
            self._watchdog_thread = None
        logger.info("StateRecoveryManager watchdog stopped")

    # ── Snapshot Management ─────────────────────────────────────────────

    def capture_snapshot(self, force_stable: bool = False) -> SystemSnapshot:
        """Capture a complete system state snapshot.

        A snapshot includes:
          - The active OptimizationVector values from the EvolutionaryEngine
          - The compiled code strings from the LogicMutator
          - Baseline performance metrics from the ProcessingCoordinator
          - ConsortiumBroker resource allocation state

        Snapshots are appended to the immutable ledger (ring buffer).
        If the ledger is full, the oldest snapshot is evicted.
        """
        snapshot_id = f"snap_{int(time.time() * 1000)}"

        # Get optimization vector from evolutionary engine
        opt_vector: Dict[str, float] = {}
        try:
            if self._evolutionary_engine is None:
                from services.core_engine.evolutionary_engine import get_evolutionary_engine
                self._evolutionary_engine = get_evolutionary_engine()
            if self._evolutionary_engine:
                opt_vector = self._evolutionary_engine.get_optimization_vector()
        except Exception as e:
            logger.debug(f"Could not capture optimization vector: {e}")

        # Get logic mutator variants (compiled code strings)
        variants: Dict[str, str] = {}
        active_variant_id: Optional[str] = None
        try:
            if self._logic_mutator is None:
                from services.core_engine.logic_mutator import get_logic_mutator
                self._logic_mutator = get_logic_mutator()
            if self._logic_mutator:
                with self._logic_mutator._lock:
                    for vid, v in self._logic_mutator._variants.items():
                        variants[vid] = v.code_string
                    active_variant_id = self._logic_mutator._active_variant_id
        except Exception as e:
            logger.debug(f"Could not capture logic mutator state: {e}")

        # Get consortium broker state
        broker_state: Dict = {}
        try:
            if self._consortium_broker is None:
                from services.core_engine.consortium_broker import get_consortium_broker
                self._consortium_broker = get_consortium_broker()
            if self._consortium_broker:
                broker_state = {
                    "registered_agents": dict(self._consortium_broker._registered_agents),
                    "total_budget_ms": self._consortium_broker._total_budget_ms,
                    "cycle_count": self._consortium_broker._cycle_count,
                }
        except Exception as e:
            logger.debug(f"Could not capture broker state: {e}")

        # Get performance metrics from processing coordinator
        metrics: Dict = {}
        try:
            if self._processing_coordinator is None:
                from services.core_engine.processing_coordinator import get_processing_coordinator
                self._processing_coordinator = get_processing_coordinator()
            if self._processing_coordinator:
                status = self._processing_coordinator.get_processing_status()
                metrics = {"processing_status": status}
        except Exception as e:
            logger.debug(f"Could not capture performance metrics: {e}")

        # Determine stability — a snapshot is stable if no errors and no stall
        with self._status_lock:
            error_count = self._watchdog_status.consecutive_errors
            is_stall = self._watchdog_status.is_stall_detected

        is_stable = force_stable or (error_count == 0 and not is_stall)

        snapshot = SystemSnapshot(
            snapshot_id=snapshot_id,
            timestamp=time.time(),
            optimization_vector=opt_vector,
            logic_mutator_variants=variants,
            active_variant_id=active_variant_id,
            consortium_broker_state=broker_state,
            performance_metrics=metrics,
            error_count=error_count,
            is_stable=is_stable,
        )

        with self._ledger_lock:
            self._ledger.append(snapshot)

        logger.debug(f"Snapshot captured: {snapshot_id} (stable={is_stable})")
        return snapshot

    def get_last_stable_snapshot(self) -> Optional[SystemSnapshot]:
        """Get the most recent stable snapshot from the ledger.

        Iterates the ledger in reverse (newest first) and returns the
        first snapshot marked as stable. Returns None if no stable
        snapshot exists.
        """
        with self._ledger_lock:
            for snapshot in reversed(self._ledger):
                if snapshot.is_stable:
                    return snapshot
            return None

    # ── Heartbeat & Error Tracking ──────────────────────────────────────

    def register_heartbeat(self):
        """Called by ProcessingCoordinator after each frame cycle.

        Updates the last heartbeat timestamp and clears any stall flag.
        This is the primary liveness signal for the watchdog.
        """
        with self._status_lock:
            self._watchdog_status.last_heartbeat = time.time()
            self._watchdog_status.is_stall_detected = False

    def increment_error_count(self) -> int:
        """Increment the consecutive error counter.

        Called by ProcessingCoordinator when a sandboxed error occurs
        in the processing loop. Returns the new error count.
        """
        with self._status_lock:
            self._watchdog_status.consecutive_errors += 1
            return self._watchdog_status.consecutive_errors

    # ── Watchdog Monitoring ─────────────────────────────────────────────

    def _watchdog_loop(self):
        """Background loop that monitors execution vitality.

        Runs at the configured watchdog_poll_interval_ms. On each cycle:
          1. Capture a periodic system snapshot
          2. Check for heartbeat stall
          3. Check for error collapse
          4. Check for metric collapse

        If any condition is met, triggers the full recovery workflow.
        """
        while not self._stop_event.is_set():
            try:
                self._check_and_recover()
            except Exception as e:
                logger.warning(f"Watchdog check error: {e}")

            # Sleep with stop-event checking
            poll_interval_s = self.watchdog_poll_interval_ms / 1000.0
            for _ in range(int(poll_interval_s / 0.1)):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

    def _check_and_recover(self):
        """Check system vitality and trigger recovery if needed."""
        if not self.auto_rollback_enabled:
            return

        # Capture periodic snapshot (adds to ledger ring buffer)
        self.capture_snapshot()

        # Check for stall (no heartbeat within timeout)
        if self._check_stall():
            self._execute_recovery("heartbeat_timeout")
            return

        # Check for error collapse (consecutive errors exceed threshold)
        if self._check_error_collapse():
            self._execute_recovery("error_collapse")
            return

        # Check for metric collapse (performance degradation)
        if self._check_metric_collapse():
            self._execute_recovery("metric_collapse")
            return

    def _check_stall(self) -> bool:
        """Check if the system has stalled (no heartbeat within timeout).

        The heartbeat is updated by ProcessingCoordinator.register_heartbeat()
        after each frame cycle. If the elapsed time since the last heartbeat
        exceeds heartbeat_timeout_ms, a stall is detected.
        """
        with self._status_lock:
            elapsed_ms = (time.time() - self._watchdog_status.last_heartbeat) * 1000
            if elapsed_ms > self.heartbeat_timeout_ms:
                self._watchdog_status.is_stall_detected = True
                logger.warning(
                    f"Stall detected: {elapsed_ms:.0f}ms since last heartbeat "
                    f"(timeout={self.heartbeat_timeout_ms}ms)"
                )
                return True
        return False

    def _check_error_collapse(self) -> bool:
        """Check if error count has exceeded the allowed threshold.

        Consecutive errors accumulate when ProcessingCoordinator encounters
        sandboxed exceptions. If they exceed max_allowed_consecutive_errors,
        a rollback is triggered.
        """
        with self._status_lock:
            if self._watchdog_status.consecutive_errors >= self.max_allowed_errors:
                logger.warning(
                    f"Error collapse detected: {self._watchdog_status.consecutive_errors} "
                    f"consecutive errors (max={self.max_allowed_errors})"
                )
                return True
        return False

    def _check_metric_collapse(self) -> bool:
        """Check if performance metrics have degraded beyond acceptable bounds.

        Compares the most recent snapshot against the previous one in the
        ledger. Triggers recovery if:
          - Error count increased by more than 3 (rapid error spike)
          - Detection count dropped by more than 50% (pipeline not processing)
          - Processing time increased by more than 100% (severe slowdown)
        """
        with self._ledger_lock:
            if len(self._ledger) < 2:
                return False

            recent = self._ledger[-1]
            previous = self._ledger[-2]

            # Check if error count increased significantly
            if recent.error_count > previous.error_count + 3:
                logger.warning(
                    f"Metric collapse: error count jumped from "
                    f"{previous.error_count} to {recent.error_count}"
                )
                return True

            # Check for detection count collapse (pipeline not processing frames)
            recent_metrics = recent.performance_metrics
            prev_metrics = previous.performance_metrics
            if recent_metrics and prev_metrics:
                recent_status = recent_metrics.get("processing_status", {})
                prev_status = prev_metrics.get("processing_status", {})

                # Compare total detection counts across all cameras
                recent_detections = 0
                prev_detections = 0
                for cam_data in recent_status.values():
                    latest = cam_data.get("latest_analysis", {})
                    recent_detections += len(latest.get("detections", []))
                for cam_data in prev_status.values():
                    latest = cam_data.get("latest_analysis", {})
                    prev_detections += len(latest.get("detections", []))

                # If we had detections before but now have none, that's a collapse
                if prev_detections > 0 and recent_detections == 0:
                    logger.warning(
                        f"Metric collapse: detections dropped from "
                        f"{prev_detections} to 0"
                    )
                    return True

                # If detections dropped by more than 50%, flag as collapse
                if prev_detections > 5 and recent_detections < prev_detections * 0.5:
                    logger.warning(
                        f"Metric collapse: detections dropped from "
                        f"{prev_detections} to {recent_detections} (>50% drop)"
                    )
                    return True

        return False

    # ── Recovery Workflow ───────────────────────────────────────────────

    def _execute_recovery(self, reason: str):
        """Execute the full recovery workflow.

        Steps:
          1. Freeze the pipeline (prevent new frame processing)
          2. Wipe volatile state (ConsortiumBroker + LogicMutator)
          3. Re-inject last stable snapshot from ledger
          4. Reset error counters and stall flags
          5. Log critical self-healing event to data/recovery_log.json

        Uses a non-blocking lock to prevent concurrent recoveries.
        """
        # Prevent concurrent recoveries
        if not self._recovery_lock.acquire(blocking=False):
            logger.warning("Recovery already in progress, skipping")
            return

        try:
            start_time = time.time()
            logger.critical(f"Recovery triggered: {reason}")

            # 1. Freeze the pipeline — prevent new frame processing
            self._freeze_pipeline()

            # 2. Wipe volatile memory structures of broker and mutator
            self._wipe_volatile_state()

            # 3. Re-inject last verified stable snapshot from ledger
            snapshot = self.get_last_stable_snapshot()
            if snapshot:
                self._reinject_snapshot(snapshot)
            else:
                logger.warning("No stable snapshot available, using defaults")

            # 4. Reset error counters and stall flags
            with self._status_lock:
                self._watchdog_status.consecutive_errors = 0
                self._watchdog_status.is_stall_detected = False
                self._watchdog_status.last_recovery_time = time.time()
                self._watchdog_status.total_recoveries += 1

            # 5. Log critical self-healing event
            recovery_duration = (time.time() - start_time) * 1000
            if self.log_recovery_events:
                self._log_recovery_event({
                    "event_type": "auto_rollback",
                    "reason": reason,
                    "snapshot_id": snapshot.snapshot_id if snapshot else "none",
                    "recovery_duration_ms": round(recovery_duration, 2),
                    "total_recoveries": self._watchdog_status.total_recoveries,
                })

            logger.info(f"Recovery complete: {reason} ({recovery_duration:.1f}ms)")

        except Exception as e:
            logger.error(f"Recovery failed: {e}", exc_info=True)
        finally:
            self._recovery_lock.release()

    def _freeze_pipeline(self):
        """Freeze all processing threads temporarily.

        Sets a recovery freeze flag on the ProcessingCoordinator. The
        processing loop checks this flag and pauses frame processing
        until the flag is cleared by _reinject_snapshot().
        """
        try:
            if self._processing_coordinator is None:
                from services.core_engine.processing_coordinator import get_processing_coordinator
                self._processing_coordinator = get_processing_coordinator()
            if self._processing_coordinator:
                self._processing_coordinator._recovery_freeze = True
                logger.info("Pipeline frozen for recovery")
        except Exception as e:
            logger.warning(f"Could not freeze pipeline: {e}")

    def _wipe_volatile_state(self):
        """Clear volatile memory structures of broker and mutator.

        - ConsortiumBroker.clear_volatile_state() — wipes bids, allocations,
          and context posts from the blackboard
        - LogicMutator.reset_to_baseline() — wipes all variants, resets
          to default pass-through filter
        """
        # Wipe consortium broker
        try:
            if self._consortium_broker is None:
                from services.core_engine.consortium_broker import get_consortium_broker
                self._consortium_broker = get_consortium_broker()
            if self._consortium_broker:
                self._consortium_broker.clear_volatile_state()
                logger.info("Consortium broker volatile state wiped")
        except Exception as e:
            logger.warning(f"Could not wipe broker state: {e}")

        # Wipe logic mutator
        try:
            if self._logic_mutator is None:
                from services.core_engine.logic_mutator import get_logic_mutator
                self._logic_mutator = get_logic_mutator()
            if self._logic_mutator:
                self._logic_mutator.reset_to_baseline()
                logger.info("Logic mutator volatile state wiped")
        except Exception as e:
            logger.warning(f"Could not wipe mutator state: {e}")

    def _reinject_snapshot(self, snapshot: SystemSnapshot):
        """Re-inject a stable snapshot's state into all services.

        - Re-injects the OptimizationVector into the EvolutionaryEngine
        - Re-injects compiled code variants into the LogicMutator
        - Resets mutation rates to baseline
        - Unfreezes the pipeline for normal operation
        """
        # Re-inject optimization vector into evolutionary engine
        try:
            if self._evolutionary_engine is None:
                from services.core_engine.evolutionary_engine import get_evolutionary_engine
                self._evolutionary_engine = get_evolutionary_engine()
            if self._evolutionary_engine and snapshot.optimization_vector:
                # Set the active vector from the snapshot
                from services.core_engine.evolutionary_engine import OptimizationVector
                restored_vector = OptimizationVector()
                for key, value in snapshot.optimization_vector.items():
                    if hasattr(restored_vector, key):
                        setattr(restored_vector, key, value)
                self._evolutionary_engine._active_vector = restored_vector

                # Reset mutation rates to baseline
                if hasattr(self._evolutionary_engine, 'orchestrator'):
                    self._evolutionary_engine.orchestrator.initial_mutation_rate = 0.1
                    self._evolutionary_engine.orchestrator._current_mutation_rate = 0.1
                    logger.info(
                        f"Mutation rates reset to baseline "
                        f"(rate={self._evolutionary_engine.orchestrator._current_mutation_rate})"
                    )

                logger.info(f"Optimization vector reinjected from {snapshot.snapshot_id}")
        except Exception as e:
            logger.warning(f"Could not reinject optimization vector: {e}")

        # Re-inject logic mutator variants (compiled code strings)
        try:
            if self._logic_mutator and snapshot.logic_mutator_variants:
                from services.core_engine.logic_mutator import CodeVariant
                with self._logic_mutator._lock:
                    self._logic_mutator._variants.clear()
                    for vid, code in snapshot.logic_mutator_variants.items():
                        try:
                            compiled = self._logic_mutator.compile_heuristic_filter(code)
                            variant = CodeVariant(
                                variant_id=vid,
                                code_string=code,
                                compiled_func=compiled,
                            )
                            self._logic_mutator._variants[vid] = variant
                        except Exception:
                            logger.debug(f"Could not recompile variant {vid}, skipping")
                    self._logic_mutator._active_variant_id = snapshot.active_variant_id
                logger.info(
                    f"Logic mutator variants reinjected from {snapshot.snapshot_id} "
                    f"({len(snapshot.logic_mutator_variants)} variants)"
                )
        except Exception as e:
            logger.warning(f"Could not reinject mutator variants: {e}")

        # Unfreeze pipeline — resume normal operation
        try:
            if self._processing_coordinator:
                self._processing_coordinator._recovery_freeze = False
                logger.info("Pipeline unfrozen after recovery")
        except Exception as e:
            logger.warning(f"Could not unfreeze pipeline: {e}")

    def _log_recovery_event(self, event: Dict):
        """Log a recovery event to data/recovery_log.json.

        Appends the event to a JSON array, keeping only the last 100
        events to prevent unbounded growth.
        """
        try:
            log_path = self._data_dir / "recovery_log.json"
            events: List[Dict] = []
            if log_path.exists():
                try:
                    with open(log_path, 'r') as f:
                        events = json.load(f)
                except (json.JSONDecodeError, IOError):
                    events = []

            event_entry = {
                "timestamp": datetime.now().isoformat(),
                **event,
            }
            events.append(event_entry)

            # Keep only last 100 events
            events = events[-100:]

            with open(log_path, 'w') as f:
                json.dump(events, f, indent=2)

            logger.info(f"Recovery event logged to {log_path}")
        except Exception as e:
            logger.warning(f"Could not log recovery event: {e}")

    def _get_data_dir(self) -> Path:
        """Get the data directory path (project-root-relative).

        Resolves to <project_root>/data/ where project_root is determined
        by traversing up from this file's location.
        """
        try:
            project_root = Path(__file__).resolve().parent.parent.parent.parent
            data_dir = project_root / "data"
            data_dir.mkdir(parents=True, exist_ok=True)
            return data_dir
        except Exception:
            return Path("data")

    # ── Status & Monitoring ─────────────────────────────────────────────

    def get_status(self) -> Dict:
        """Return full recovery manager status for monitoring API."""
        with self._ledger_lock, self._status_lock:
            return {
                "enabled": self.enabled,
                "ledger_size": len(self._ledger),
                "ledger_limit": self.ledger_history_limit,
                "watchdog_active": self._watchdog_thread is not None and self._watchdog_thread.is_alive(),
                "last_heartbeat": self._watchdog_status.last_heartbeat,
                "consecutive_errors": self._watchdog_status.consecutive_errors,
                "is_stall_detected": self._watchdog_status.is_stall_detected,
                "total_recoveries": self._watchdog_status.total_recoveries,
                "last_recovery_time": self._watchdog_status.last_recovery_time,
                "snapshots": [
                    {
                        "snapshot_id": s.snapshot_id,
                        "timestamp": s.timestamp,
                        "is_stable": s.is_stable,
                        "error_count": s.error_count,
                    }
                    for s in self._ledger
                ],
            }


# =============================================================================
# Global Instance
# =============================================================================

_state_recovery_manager: Optional[StateRecoveryManager] = None


def get_state_recovery_manager() -> StateRecoveryManager:
    """Get global StateRecoveryManager instance (singleton).

    Uses absolute package imports originating from the root of backend:
        from services.management.state_recovery_manager import StateRecoveryManager
    """
    global _state_recovery_manager
    if _state_recovery_manager is None:
        _state_recovery_manager = StateRecoveryManager()
    return _state_recovery_manager
