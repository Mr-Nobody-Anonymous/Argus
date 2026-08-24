"""
CityOS Signal Optimiser.

A traffic-signal control interface in the spirit of NTCIP / SDLC controller
integrations:

  - models the intersection as two phases (north-south, east-west) with
    green -> yellow -> all-red transitions
  - FIXED mode runs configured timings
  - ADAPTIVE mode recommends green extensions/splits from live per-approach
    demand measured by the perception engine
  - MANUAL mode lets an operator hold a phase (with an audit trail)

The optimiser RECOMMENDS; it never actuates a real controller. Applying a
recommendation updates the internal model and emits a `signal_command`
record that a real NTCIP bridge could consume.
"""
import threading
import time
from typing import Dict, Optional

PHASES = ("NS", "EW")
MIN_GREEN_S = 8.0
MAX_GREEN_S = 45.0
YELLOW_S = 3.0
ALL_RED_S = 1.5

# Pedestrian clearance runs through yellow + all-red of the parallel phase.
PED_CLEARANCE_S = YELLOW_S + ALL_RED_S + 4.0

MODES = ("fixed", "adaptive", "manual")

# Approach -> its signal axis.
APPROACH_AXIS = {"north": "NS", "south": "NS", "east": "EW", "west": "EW"}


class SignalOptimizer:
    """Per-intersection signal phase model + adaptive recommendations."""

    def __init__(self, intersection_id: str,
                 fixed_green_s: float = 20.0):
        self.intersection_id = intersection_id
        # RLock, not Lock: tick() holds the lock across _advance() and then
        # calls status(), which also locks. A plain Lock deadlocks there -
        # found by tests/test_cityos.py hanging mid-run.
        self._lock = threading.RLock()
        self.mode = "adaptive"
        self.phase = "NS"
        self.state = "green"          # green | yellow | all_red
        self.fixed_green_s = fixed_green_s
        self._state_started = time.time()
        self.command_log: list = []
        self.recommendations_applied = 0

    # ── Core state machine ──────────────────────────────────────────────

    def _enter(self, phase: str, state: str):
        self.phase = phase
        self.state = state
        self._state_started = time.time()

    def _advance(self, demand: Optional[Dict[str, float]]):
        """Progress the phase machine one tick."""
        elapsed = time.time() - self._state_started
        if self.state == "yellow" and elapsed >= YELLOW_S:
            self._enter(self.phase, "all_red")
        elif self.state == "all_red" and elapsed >= ALL_RED_S:
            next_phase = "EW" if self.phase == "NS" else "NS"
            self._enter(next_phase, "green")
        elif self.state == "green":
            if self.mode == "fixed":
                # Fixed timing: clamp the configured green into legal bounds.
                green_needed = min(MAX_GREEN_S, max(MIN_GREEN_S, self.fixed_green_s))
                if elapsed >= green_needed:
                    self._enter(self.phase, "yellow")
            else:
                # Adaptive: extend while this phase's demand dominates and we
                # have not hit max green; switch early when the cross phase
                # clearly needs service.
                cur_demand = demand.get(
                    "north" if self.phase == "NS" else "east", 0)
                if elapsed >= MAX_GREEN_S or (
                    elapsed >= MIN_GREEN_S and cur_demand < 0.5
                    and demand and max(demand.values()) > cur_demand * 2
                ):
                    self._enter(self.phase, "yellow")

    @staticmethod
    def _adaptive_green(demand: Dict[str, float]) -> float:
        """Green split proportional to this phase's share of total demand."""
        ns = demand.get("north", 0) + demand.get("south", 0)
        ew = demand.get("east", 0) + demand.get("west", 0)
        total = ns + ew
        if total <= 0:
            return MIN_GREEN_S
        share = (ns if True else ew) / total
        return min(MAX_GREEN_S, max(MIN_GREEN_S, share * (MAX_GREEN_S + MIN_GREEN_S)))

    def tick(self, demand: Optional[Dict[str, float]] = None) -> Dict:
        """Advance the model; returns the current signal status."""
        with self._lock:
            self._advance(demand)
            return self.status()

    def status(self) -> Dict:
        with self._lock:
            return {
                "intersection_id": self.intersection_id,
                "mode": self.mode,
                "phase": self.phase,
                "state": self.state,
                "seconds_in_state": round(time.time() - self._state_started, 1),
                "min_green_s": MIN_GREEN_S,
                "max_green_s": MAX_GREEN_S,
                "yellow_s": YELLOW_S,
                "all_red_s": ALL_RED_S,
                "recommendations_applied": self.recommendations_applied,
            }

    # ── Per-approach & pedestrian signal intelligence ───────────────────

    def approach_state(self, approach: str) -> str:
        """green / yellow / red for one approach, derived from the phase."""
        axis = APPROACH_AXIS.get(approach)
        if axis is None:
            return "red"
        with self._lock:
            if self.phase != axis:
                return "red"
            return {"green": "green", "yellow": "yellow"}.get(
                self.state, "red")

    def ped_states(self) -> Dict[str, str]:
        """Pedestrian signal per phase: walk / flashing / dont_walk.

        Walk during the parallel vehicle green; flashing don't-walk through
        yellow + all-red (the clearance interval); solid don't-walk otherwise.
        """
        with self._lock:
            out = {}
            for ph in PHASES:
                if ph == self.phase and self.state == "green":
                    out[ph] = "walk"
                elif ph == self.phase and self.state in ("yellow", "all_red"):
                    out[ph] = "flashing"
                else:
                    out[ph] = "dont_walk"
            return out

    # ── Adaptive recommendation ─────────────────────────────────────────

    def recommend(self, demand: Dict[str, float]) -> Dict:
        """What the optimiser would do given live approach demand.

        Pure function of demand - safe to call for display without mutating
        the phase machine.
        """
        ns = demand.get("north", 0) + demand.get("south", 0)
        ew = demand.get("east", 0) + demand.get("west", 0)
        total = ns + ew
        if total <= 0:
            return {
                "action": "hold",
                "reason": "no measurable demand",
                "ns_share": None,
                "suggested_green_s": self.fixed_green_s,
            }
        ns_share = ns / total
        suggested = min(MAX_GREEN_S, max(MIN_GREEN_S, ns_share * 60.0))
        if self.phase == "NS":
            action = "extend" if ns >= ew else "terminate_early"
        else:
            action = "extend" if ew > ns else "terminate_early"
        return {
            "action": action,
            "reason": f"NS demand {ns:.1f} vs EW demand {ew:.1f}",
            "ns_share": round(ns_share, 3),
            "suggested_green_s": round(suggested, 1),
        }

    def apply_recommendation(self, recommendation: Dict) -> Dict:
        """Apply an adaptive decision to the internal model.

        In MANUAL mode external commands are refused unless forced, so an
        operator holding a phase is never silently overridden.
        """
        with self._lock:
            if self.mode == "manual":
                cmd = {
                    "ts": time.time(),
                    "applied": False,
                    "reason": "manual mode holds operator override",
                    **recommendation,
                }
                self.command_log.append(cmd)
                return cmd
            action = recommendation.get("action")
            if action == "extend" and self.state == "green":
                # Reset the state timer so the green continues up to max.
                self._state_started = time.time()
            elif action == "terminate_early" and self.state == "green":
                self._enter(self.phase, "yellow")
            self.recommendations_applied += 1
            cmd = {"ts": time.time(), "applied": True, **recommendation}
            self.command_log.append(cmd)
            if len(self.command_log) > 200:
                del self.command_log[:-200]
            return cmd

    # ── Operator controls ───────────────────────────────────────────────

    def set_mode(self, mode: str) -> Dict:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        with self._lock:
            self.mode = mode
            self.command_log.append({
                "ts": time.time(), "applied": True,
                "action": "set_mode", "mode": mode,
            })
            return self.status()

    def force_phase(self, phase: str) -> Dict:
        """Manual operator override: jump straight to a phase's green."""
        if phase not in PHASES:
            raise ValueError(f"phase must be one of {PHASES}")
        with self._lock:
            self._enter(phase, "green")
            self.command_log.append({
                "ts": time.time(), "applied": True,
                "action": "force_phase", "phase": phase,
            })
            return self.status()

    def recent_commands(self, limit: int = 20) -> list:
        with self._lock:
            cmds = list(self.command_log)[-limit:]
            cmds.reverse()
            return cmds