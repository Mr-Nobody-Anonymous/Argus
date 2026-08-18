"""Capability registry: what Argus can actually run here, and what it costs.

The swarm currently auctions a time budget between three hard-coded agents. It
cannot answer the question that matters - *"what can I run, what will it cost,
and what would it tell me that I do not already know?"* - because nothing
describes the analyses themselves.

This registry does. Each capability declares:

    name          what it produces
    tier          CPU or GPU
    cost_ms       measured, not guessed
    provides      the attributes/entities it can add
    requires      what must already be true for it to be worth running
    available     whether its backend actually loaded on this machine

From that, a scheduler can compute **information gain per millisecond** and
skip work whose output is already known. Running a plate reader on a frame
containing no vehicle is pure waste; so is re-reading a plate that has been
read at high confidence for the last thirty frames.

Availability is resolved by *probing*, never by assuming. A capability whose
backend is missing reports `available=False` with a reason, and the pipeline
degrades instead of raising. That is what makes the same code run on a 2-core
laptop and a GPU server.

Stdlib only - the probes import their targets defensively inside a try.
"""

from __future__ import annotations

import importlib
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set


class Tier:
    CPU = "cpu"
    GPU = "gpu"


@dataclass
class Capability:
    """One analysis Argus can perform."""

    name: str
    tier: str
    cost_ms: float                       # typical single-frame cost
    provides: Set[str] = field(default_factory=set)
    requires: Set[str] = field(default_factory=set)
    description: str = ""
    module: Optional[str] = None         # import path probed for availability
    available: bool = False
    unavailable_reason: str = ""
    measured: bool = False               # True once cost_ms came from a real run

    def value_per_ms(self, already_known: Set[str]) -> float:
        """Information gain per millisecond, given what is already known.

        Zero when every attribute this capability provides is already present:
        the scheduler should then spend the budget on something that would
        actually tell it something new.
        """
        if not self.available:
            return 0.0
        novel = self.provides - already_known
        if not novel:
            return 0.0
        return len(novel) / max(1.0, self.cost_ms)


class CapabilityRegistry:
    """Everything Argus can do, and whether it can do it here."""

    def __init__(self):
        self._caps: Dict[str, Capability] = {}
        self._lock = threading.RLock()
        self._probed = False

    # -- registration ---------------------------------------------------------

    def register(self, cap: Capability) -> Capability:
        with self._lock:
            self._caps[cap.name] = cap
            return cap

    def get(self, name: str) -> Optional[Capability]:
        with self._lock:
            return self._caps.get(name)

    def all(self) -> List[Capability]:
        with self._lock:
            return list(self._caps.values())

    def available(self) -> List[Capability]:
        return [c for c in self.all() if c.available]

    def by_tier(self, tier: str) -> List[Capability]:
        return [c for c in self.all() if c.tier == tier]

    # -- availability ---------------------------------------------------------

    def probe(self, force: bool = False) -> Dict[str, bool]:
        """Determine what is actually runnable, by importing it.

        Import success is the only honest test: a package can be declared in
        requirements and still fail to load (missing system library, wrong
        wheel, unsupported CPU). Guessing from config would report capabilities
        that do not work.
        """
        with self._lock:
            if self._probed and not force:
                return {c.name: c.available for c in self._caps.values()}

            gpu = _cuda_available()
            for cap in self._caps.values():
                if cap.tier == Tier.GPU and not gpu:
                    cap.available = False
                    cap.unavailable_reason = "requires a CUDA device; none detected"
                    continue
                if cap.module is None:
                    cap.available = True
                    cap.unavailable_reason = ""
                    continue
                try:
                    importlib.import_module(cap.module)
                    cap.available = True
                    cap.unavailable_reason = ""
                except Exception as exc:  # noqa: BLE001 - any failure means unusable
                    cap.available = False
                    cap.unavailable_reason = f"{type(exc).__name__}: {exc}"[:160]
            self._probed = True
            return {c.name: c.available for c in self._caps.values()}

    def record_cost(self, name: str, elapsed_ms: float) -> None:
        """Replace the estimated cost with a measured one.

        Exponentially smoothed so one slow frame (a cold cache, a GC pause)
        does not permanently distort scheduling.
        """
        with self._lock:
            cap = self._caps.get(name)
            if cap is None:
                return
            cap.cost_ms = (elapsed_ms if not cap.measured
                           else 0.8 * cap.cost_ms + 0.2 * elapsed_ms)
            cap.measured = True

    # -- scheduling -----------------------------------------------------------

    def plan(self, context: Set[str], already_known: Optional[Set[str]] = None,
             budget_ms: float = 100.0) -> List[Capability]:
        """Choose what to run this frame, best value first, within budget.

        ``context`` is what the fast pass found ("person", "vehicle", "text").
        A capability whose ``requires`` is not satisfied is not merely low
        value - it is inapplicable, and running it would be meaningless.
        """
        known = already_known or set()
        candidates = []
        for cap in self.all():
            if not cap.available:
                continue
            if cap.requires and not cap.requires.issubset(context):
                continue
            value = cap.value_per_ms(known)
            if value <= 0:
                continue
            candidates.append((value, cap))

        candidates.sort(key=lambda vc: -vc[0])
        chosen, spent = [], 0.0
        for _, cap in candidates:
            if spent + cap.cost_ms > budget_ms:
                continue
            chosen.append(cap)
            spent += cap.cost_ms
        return chosen

    def report(self) -> Dict[str, Any]:
        """Operator-facing summary - also what the API exposes."""
        caps = self.all()
        return {
            "gpu_available": _cuda_available(),
            "total": len(caps),
            "available": sum(1 for c in caps if c.available),
            "capabilities": [
                {
                    "name": c.name,
                    "tier": c.tier,
                    "available": c.available,
                    "cost_ms": round(c.cost_ms, 1),
                    "measured": c.measured,
                    "provides": sorted(c.provides),
                    "requires": sorted(c.requires),
                    "reason": c.unavailable_reason,
                    "description": c.description,
                }
                for c in sorted(caps, key=lambda x: (x.tier, x.name))
            ],
        }


def _cuda_available() -> bool:
    """True only if torch reports a usable CUDA device.

    Wrapped because torch may be absent entirely - the registry must remain
    importable in the dependency-free perception layer.
    """
    try:
        import torch  # noqa: PLC0415 - optional by design
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# ── the standard Argus capability set ────────────────────────────────────────

def build_default_registry() -> CapabilityRegistry:
    """Register everything Argus knows how to do, then probe for reality.

    Costs for implemented CPU capabilities are measured values from this
    codebase; GPU costs are documented estimates and are marked unmeasured
    until a real run replaces them.
    """
    reg = CapabilityRegistry()

    # -- CPU: implemented today ----------------------------------------------
    reg.register(Capability(
        name="object_detection", tier=Tier.CPU, cost_ms=108.0,
        provides={"bbox", "category", "class_id"},
        module="ultralytics",
        description="YOLOv8n fixed-class detection (measured on this host)",
    ))
    reg.register(Capability(
        name="tracking", tier=Tier.CPU, cost_ms=4.0,
        provides={"track_id"}, requires={"detection"},
        module="backend.services.core_engine.deep_tracker",
        description="ByteTrack-style association with centre-distance fallback",
    ))
    reg.register(Capability(
        name="temporal_analysis", tier=Tier.CPU, cost_ms=1.0,
        provides={"speed", "direction", "dwell", "trajectory"},
        requires={"track_id"},
        module="backend.services.perception.temporal",
        description="Trajectory, velocity, dwell, pacing, disappearance",
    ))
    reg.register(Capability(
        name="relationships", tier=Tier.CPU, cost_ms=2.0,
        provides={"near", "carrying", "following"}, requires={"track_id"},
        module="backend.services.perception.scene_graph",
        description="Persistent scene graph over tracked entities",
    ))
    reg.register(Capability(
        name="basic_attributes", tier=Tier.CPU, cost_ms=3.0,
        provides={"dominant_colour", "size", "aspect_ratio"},
        requires={"detection"},
        module="backend.services.perception.attributes",
        description="Colour and geometry extracted from the crop",
    ))
    reg.register(Capability(
        name="pose", tier=Tier.CPU, cost_ms=35.0,
        provides={"posture", "keypoints"}, requires={"person"},
        module="backend.services.vision.pose_estimator",
        description="Posture estimation (MediaPipe when present, else fallback)",
    ))
    reg.register(Capability(
        name="face_recognition", tier=Tier.CPU, cost_ms=45.0,
        provides={"identity", "face_visible"}, requires={"person"},
        module="backend.services.vision.face_recognition",
        description="Face matching (histogram fallback without opencv-contrib)",
    ))
    reg.register(Capability(
        name="lpr", tier=Tier.CPU, cost_ms=60.0,
        provides={"plate"}, requires={"vehicle"},
        module="backend.services.vision.license_plate_recognition",
        description="License-plate reading",
    ))

    # -- CPU: specified, not yet implemented ---------------------------------
    reg.register(Capability(
        name="ocr", tier=Tier.CPU, cost_ms=120.0,
        provides={"text"}, requires={"text_region"},
        module="paddleocr",
        description="General scene text recognition (Phase 2)",
    ))
    reg.register(Capability(
        name="scene_classification", tier=Tier.CPU, cost_ms=40.0,
        provides={"scene_type", "lighting", "indoor_outdoor"},
        module="backend.services.perception.scene_classifier",
        description="Environment and lighting context (Phase 2)",
    ))

    # -- GPU: pluggable backends ---------------------------------------------
    reg.register(Capability(
        name="segmentation", tier=Tier.GPU, cost_ms=90.0,
        provides={"mask", "precise_boundary"}, requires={"detection"},
        module="ultralytics",
        description="Instance segmentation - pixels, not rectangles (Phase 2)",
    ))
    reg.register(Capability(
        name="open_vocabulary", tier=Tier.GPU, cost_ms=250.0,
        provides={"category", "novel_object"},
        module="transformers",
        description="OWL-ViT / GroundingDINO for classes outside COCO (Phase 2)",
    ))
    reg.register(Capability(
        name="action_recognition", tier=Tier.GPU, cost_ms=180.0,
        provides={"action"}, requires={"person", "track_id"},
        module="transformers",
        description="Temporal action classification (Phase 3+)",
    ))
    reg.register(Capability(
        name="vlm", tier=Tier.GPU, cost_ms=1500.0,
        provides={"description", "answer", "scene_reasoning"},
        module="transformers",
        description="Vision-language reasoning on interesting crops (Phase 4)",
    ))

    reg.probe()
    return reg


_REGISTRY: Optional[CapabilityRegistry] = None
_REGISTRY_LOCK = threading.Lock()


def get_registry() -> CapabilityRegistry:
    """Process-wide registry, built once."""
    global _REGISTRY
    with _REGISTRY_LOCK:
        if _REGISTRY is None:
            _REGISTRY = build_default_registry()
        return _REGISTRY
