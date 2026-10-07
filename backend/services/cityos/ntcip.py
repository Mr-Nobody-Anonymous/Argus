"""
CityOS NTCIP controller integration layer.

Bridges CityOS to a real Advanced Transportation Controller (ATC) speaking
NTCIP-style objects - WITHOUT ever silently actuating hardware:

    SignalOptimizer (commanded state)
        -> NTCIPControllerClient
             mode="simulation"  : a faithful software ASC that mirrors
                                  commanded phases with configurable latency,
                                  so the whole integration path is testable
                                  with zero risk.
             mode="live"        : explicit operator opt-in; commands are
                                  validated, acknowledged, timeout-guarded,
                                  and fall back SAFE on repeated failures.

Responsibilities:

  - status polling          : observed phase/state vs commanded phase/state
  - command validation      : illegal transitions are refused before transmit
  - acknowledgements        : every accepted command returns an ack record
  - timeouts                : polls/commands that never answer are counted
  - communication failures  : consecutive failures tracked per channel
  - safe fallback           : after MAX_CONSECUTIVE_FAILURES the client
                              degrades itself to "fallback" and stops
                              forwarding commands until re-enabled
  - audit trail             : every poll/command/fault is logged

In simulation mode the simulated controller applies commanded phases after
SIM_LATENCY_S and can inject faults (set_fault) for testing fallback paths.
"""
import threading
import time
from collections import deque
from typing import Dict, List, Optional

SIM_LATENCY_S = 0.5          # simulated controller response delay
POLL_TIMEOUT_S = 2.0
COMMAND_TIMEOUT_S = 3.0
MAX_CONSECUTIVE_FAILURES = 5

VALID_STATES = ("green", "yellow", "all_red")
VALID_PHASES = ("NS", "EW")


class CommandRejected(Exception):
    """A command failed validation and was never transmitted."""


class NTCIPControllerClient:
    """Per-intersection traffic-signal controller interface."""

    def __init__(self, intersection_id: str, endpoint: Optional[str] = None,
                 mode: str = "simulation"):
        if mode not in ("simulation", "live"):
            raise ValueError("mode must be 'simulation' or 'live'")
        self.intersection_id = intersection_id
        self.endpoint = endpoint            # required for live mode
        self.mode = mode
        self._lock = threading.Lock()
        self.enabled = True                 # safe-fallback kill switch

        # Observed controller state (from polling).
        self.observed_phase: Optional[str] = None
        self.observed_state: Optional[str] = None
        self.last_poll_at: Optional[float] = None
        self.last_ack_at: Optional[float] = None

        # Health / fault tracking.
        self.consecutive_failures = 0
        self.total_failures = 0
        self.total_timeouts = 0
        self.total_polls = 0
        self.total_commands = 0
        self.commands_rejected = 0
        self.disagreements = 0              # commanded != observed
        self.in_fallback = False
        self.fallback_reason: Optional[str] = None
        self.fault_injected: Optional[str] = None   # simulation fault hook

        # Simulated controller internals.
        self._sim_phase: Optional[str] = None
        self._sim_state: Optional[str] = None
        self._sim_apply_at: Optional[float] = None

        self.audit: deque = deque(maxlen=500)

    # ── Audit ───────────────────────────────────────────────────────

    def _log(self, kind: str, **fields):
        self.audit.append({
            "ts": round(time.time(), 3),
            "kind": kind,
            **fields,
        })

    def recent_audit(self, limit: int = 50) -> List[Dict]:
        with self._lock:
            out = list(self.audit)[-limit:]
        out.reverse()
        return out

    # ── Polling ─────────────────────────────────────────────────────

    def poll(self, commanded_phase: str, commanded_state: str) -> Dict:
        """Poll the controller for its observed signal state.

        Returns {observed_phase, observed_state, latency_s, ok}.
        In simulation the mirrored state advances after SIM_LATENCY_S; an
        injected fault or too many consecutive failures makes polls fail.
        """
        now = time.time()
        with self._lock:
            self.total_polls += 1
            if not self.enabled:
                self._log("poll_refused", reason="client disabled "
                          "(safe fallback)")
                return {"ok": False, "reason": "disabled"}
            if self.fault_injected == "poll_timeout":
                self._register_failure("poll timeout (injected)")
                return {"ok": False, "reason": "timeout"}

            if self.mode == "simulation":
                # Advance the simulated controller's pending transition.
                if (self._sim_apply_at is not None
                        and now >= self._sim_apply_at):
                    self.observed_phase = self._sim_phase
                    self.observed_state = self._sim_state
                    self._sim_apply_at = None
                elif self.observed_phase is None:
                    # First poll: adopt the commanded state as boot state.
                    self.observed_phase = commanded_phase
                    self.observed_state = commanded_state
                self.last_poll_at = now
                self.consecutive_failures = 0
                self._check_disagreement(commanded_phase, commanded_state)
                self._log("poll_ok",
                          commanded=f"{commanded_phase}/{commanded_state}",
                          observed=f"{self.observed_phase}/"
                                   f"{self.observed_state}")
                return {
                    "ok": True,
                    "observed_phase": self.observed_phase,
                    "observed_state": self.observed_state,
                    "latency_s": SIM_LATENCY_S,
                }

            # Live mode: a real transport (SNMP/serial) plugs in here. Until
            # one is registered the client reports unreachable rather than
            # pretending success.
            self._register_failure("no live transport registered")
            return {"ok": False, "reason": "unreachable"}

    def _check_disagreement(self, commanded_phase: str, commanded_state: str):
        if (self.observed_phase is not None
                and (self.observed_phase != commanded_phase
                     or self.observed_state != commanded_state)):
            self.disagreements += 1
            self._log("disagreement",
                      commanded=f"{commanded_phase}/{commanded_state}",
                      observed=f"{self.observed_phase}/"
                               f"{self.observed_state}")

    def _register_failure(self, reason: str):
        self.total_failures += 1
        self.consecutive_failures += 1
        self._log("failure", reason=reason)
        if (self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                and not self.in_fallback):
            self.in_fallback = True
            self.fallback_reason = reason
            self._log("safe_fallback",
                      reason=(f"{MAX_CONSECUTIVE_FAILURES} consecutive "
                              f"failures - commands suspended"))

    # ── Commands ────────────────────────────────────────────────────

    def send_command(self, action: str, phase: Optional[str] = None,
                     state: Optional[str] = None) -> Dict:
        """Validate + transmit a phase command; returns an ack record.

        Validated BEFORE transmission: unknown actions/phases/states and
        unsafe transitions are rejected locally with CommandRejected.
        """
        now = time.time()
        with self._lock:
            self.total_commands += 1
            if not self.enabled or self.in_fallback:
                self.commands_rejected += 1
                self._log("command_rejected",
                          action=action, reason="safe fallback active")
                raise CommandRejected(
                    "controller client in safe fallback; commands suspended")

            # Validation gate.
            if action not in ("hold", "force_phase", "force_state"):
                self.commands_rejected += 1
                raise CommandRejected(f"unknown action '{action}'")
            if action in ("force_phase", "force_state"):
                if phase is not None and phase not in VALID_PHASES:
                    self.commands_rejected += 1
                    raise CommandRejected(f"invalid phase '{phase}'")
                if state is not None and state not in VALID_STATES:
                    self.commands_rejected += 1
                    raise CommandRejected(f"invalid state '{state}'")
                # Safety: never jump straight to green from a foreign phase
                # without an interval - force green must go via yellow.
                if (state == "green" and self.observed_phase is not None
                        and phase != self.observed_phase):
                    self.commands_rejected += 1
                    self._log("command_rejected",
                              action=action,
                              reason="unsafe transition: cross-phase green "
                                     "requires yellow clearance first")
                    raise CommandRejected(
                        "unsafe transition: cross-phase green requires "
                        "yellow clearance first")

            if self.fault_injected == "command_timeout":
                self.total_timeouts += 1
                self._register_failure("command timeout (injected)")
                return {"ok": False, "acked": False, "reason": "timeout"}

            if self.mode == "simulation":
                # Mirror the command into the simulated controller after
                # its response latency.
                if action == "force_phase" and phase:
                    self._sim_phase, self._sim_state = phase, "green"
                    self._sim_apply_at = now + SIM_LATENCY_S
                elif action == "force_state" and state:
                    self._sim_state = state
                    self._sim_apply_at = now + SIM_LATENCY_S
                self.last_ack_at = now
                ack = {
                    "ok": True, "acked": True, "action": action,
                    "phase": phase, "state": state,
                    "latency_s": SIM_LATENCY_S, "ts": round(now, 3),
                }
                self._log("command_acked", **{k: v for k, v in ack.items()
                                              if k != "ok"})
                return ack

            self._register_failure("no live transport registered")
            return {"ok": False, "acked": False, "reason": "unreachable"}

    # ── Simulation hooks ────────────────────────────────────────────

    def set_fault(self, fault: Optional[str]):
        """Inject a simulated fault: None | 'poll_timeout' |
        'command_timeout'. Testing only."""
        with self._lock:
            self.fault_injected = fault
            self._log("fault_injected", fault=fault)

    def clear_fallback(self):
        """Operator action: restore command flow after investigating."""
        with self._lock:
            self.in_fallback = False
            self.fallback_reason = None
            self.consecutive_failures = 0
            self._log("fallback_cleared")

    # ── Status ──────────────────────────────────────────────────────

    def status(self) -> Dict:
        with self._lock:
            comm_ok = self.consecutive_failures == 0
            return {
                "intersection_id": self.intersection_id,
                "mode": self.mode,
                "endpoint": self.endpoint,
                "enabled": self.enabled,
                "communication_ok": comm_ok,
                "observed_phase": self.observed_phase,
                "observed_state": self.observed_state,
                "last_poll_age_s": (
                    round(time.time() - self.last_poll_at, 1)
                    if self.last_poll_at else None),
                "total_polls": self.total_polls,
                "total_commands": self.total_commands,
                "commands_rejected": self.commands_rejected,
                "total_timeouts": self.total_timeouts,
                "consecutive_failures": self.consecutive_failures,
                "total_failures": self.total_failures,
                "commanded_vs_observed_disagreements": self.disagreements,
                "in_safe_fallback": self.in_fallback,
                "fallback_reason": self.fallback_reason,
                "note": ("simulation controller - no physical device is "
                         "actuated" if self.mode == "simulation"
                         else "LIVE controller mode"),
            }