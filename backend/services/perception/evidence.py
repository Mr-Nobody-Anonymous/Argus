"""Evidence API: "why does Argus believe this?"

Every inferred claim must be able to justify itself. As the system gains more
powerful models this stops being a nicety and becomes the only thing standing
between a surveillance platform and an unaccountable one: a VLM that asserts
"person appears to be casing the building" is worthless - and dangerous -
unless a human can see precisely what produced it.

An `EvidenceChain` answers that question in a fixed shape:

    Assessment   what is claimed, in plain language
    Confidence   how sure, and how that number was reached
    Evidence[]   the individual measurements, each with its own source
    Sources[]    which analysers contributed
    Provenance   when it was first and last supported, and by how many looks

The rule the whole layer is built around:

    **Argus never silently turns an inference into an observation.**

`EvidenceItem.is_measurement` records that distinction per item. A chain built
entirely from inferences is explicitly marked `derived`, so a downstream
consumer can refuse to treat it as fact. `supporting_measurements()` exists so
a reviewer can strip the interpretation away and look only at what was actually
seen.

Stdlib only, like the rest of the perception layer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .observation import LOW_CONFIDENCE, Observation, Source


def _now() -> float:
    return time.time()


@dataclass
class EvidenceItem:
    """One fact supporting a claim.

    ``is_measurement`` separates "this was observed" from "this was concluded".
    Chains that blur the two produce circular justifications - an inference
    citing another inference, with no measurement anywhere underneath.
    """

    description: str
    source: str = Source.RULE.value
    confidence: float = 1.0
    is_measurement: bool = True
    value: Any = None
    observed_at: float = field(default_factory=_now)

    def __post_init__(self) -> None:
        if isinstance(self.source, Source):
            self.source = self.source.value
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceChain:
    """A claim, its confidence, and everything that supports it."""

    assessment: str
    confidence: float
    kind: str = "assessment"
    subject_track_ids: List[int] = field(default_factory=list)
    items: List[EvidenceItem] = field(default_factory=list)
    first_observed: Optional[float] = None
    last_observed: Optional[float] = None
    observation_count: int = 0
    created_at: float = field(default_factory=_now)

    def __post_init__(self) -> None:
        self.confidence = max(0.0, min(1.0, float(self.confidence)))

    # -- construction ---------------------------------------------------------

    def add(self, description: str, source: str = Source.RULE.value,
            confidence: float = 1.0, is_measurement: bool = True,
            value: Any = None) -> "EvidenceChain":
        self.items.append(EvidenceItem(
            description=description, source=source, confidence=confidence,
            is_measurement=is_measurement, value=value))
        return self

    # -- inspection -----------------------------------------------------------

    @property
    def sources(self) -> List[str]:
        """Which analysers contributed, in first-seen order."""
        seen: List[str] = []
        for item in self.items:
            if item.source not in seen:
                seen.append(item.source)
        return seen

    def supporting_measurements(self) -> List[EvidenceItem]:
        """Only the things actually observed, with interpretation stripped out."""
        return [i for i in self.items if i.is_measurement]

    def supporting_inferences(self) -> List[EvidenceItem]:
        return [i for i in self.items if not i.is_measurement]

    @property
    def is_grounded(self) -> bool:
        """True when at least one real measurement underlies the claim.

        A chain of inferences citing other inferences justifies nothing, and
        this is the check that catches it.
        """
        return any(i.is_measurement for i in self.items)

    @property
    def is_actionable(self) -> bool:
        """Confidence alone never makes a claim actionable.

        It must also be grounded in a measurement. This mirrors
        ``Observation.is_actionable`` so the two cannot drift apart.
        """
        return (self.confidence >= LOW_CONFIDENCE
                and bool(self.items)
                and self.is_grounded)

    @property
    def status(self) -> str:
        if not self.items:
            return "unsupported"
        if not self.is_grounded:
            return "derived"          # inference only - treat with suspicion
        if self.confidence < LOW_CONFIDENCE:
            return "weak"
        return "supported"

    # -- output ---------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": self.kind,
            "assessment": self.assessment,
            "confidence": round(self.confidence, 3),
            "status": self.status,
            "actionable": self.is_actionable,
            "grounded": self.is_grounded,
            "subject_track_ids": list(self.subject_track_ids),
            "sources": self.sources,
            "evidence": [i.to_dict() for i in self.items],
            "measurements": len(self.supporting_measurements()),
            "inferences": len(self.supporting_inferences()),
            "first_observed": self.first_observed,
            "last_observed": self.last_observed,
            "observation_count": self.observation_count,
        }

    def explain(self) -> str:
        """Human-readable justification - what an operator sees on 'why?'."""
        lines = [
            f"Assessment: {self.assessment}",
            f"Confidence: {self.confidence:.2f} ({self.status})",
            "Evidence:",
        ]
        for item in self.items:
            tag = "measured" if item.is_measurement else "inferred"
            lines.append(f"  - [{tag}] {item.description} ({item.source})")
        if self.sources:
            lines.append(f"Sources: {', '.join(self.sources)}")
        if self.observation_count:
            lines.append(f"Support: {self.observation_count} observations "
                         f"over {(self.last_observed or 0) - (self.first_observed or 0):.1f}s")
        return "\n".join(lines)


# ── builders from existing structures ────────────────────────────────────────

def chain_from_observation(obs: Observation) -> EvidenceChain:
    """Turn a temporal Observation into an explainable chain.

    The strings in ``Observation.evidence`` are measurements (durations,
    distances, frame counts) produced by the temporal engine, so they are
    marked as such; the summary itself is the inference drawn from them.
    """
    chain = EvidenceChain(
        assessment=obs.summary,
        confidence=obs.confidence,
        kind=obs.kind,
        subject_track_ids=list(obs.track_ids),
        first_observed=obs.timestamp,
        last_observed=obs.timestamp,
    )
    for text in obs.evidence:
        chain.add(text, source=obs.source, is_measurement=True)
    return chain


def chain_from_edge(edge, subject_category: str = "entity",
                    object_category: str = "entity") -> EvidenceChain:
    """Explain a scene-graph relationship.

    The persistence facts (observation count, duration) are measurements; the
    predicate itself is the inference those measurements support.
    """
    chain = EvidenceChain(
        assessment=(f"{subject_category} {edge.subject_id} {edge.predicate} "
                    f"{object_category} {edge.object_id}"),
        confidence=edge.confidence,
        kind=f"relationship:{edge.predicate}",
        subject_track_ids=[edge.subject_id, edge.object_id],
        first_observed=edge.first_seen,
        last_observed=edge.last_seen,
        observation_count=edge.observation_count,
    )
    chain.add(f"{edge.observation_count} consecutive observations",
              source=Source.TEMPORAL.value, value=edge.observation_count)
    chain.add(f"sustained for {edge.duration:.1f}s",
              source=Source.TEMPORAL.value, value=round(edge.duration, 1))

    if edge.distances:
        mean = sum(edge.distances) / len(edge.distances)
        chain.add(f"mean separation {mean:.0f}px",
                  source=Source.TEMPORAL.value, value=round(mean, 1))
    trend = edge.trend()
    if trend:
        # A trend is computed from measured distances but is itself a reading
        # of them, so it is recorded as an inference.
        chain.add(f"distance trend: {trend}", source=Source.TEMPORAL.value,
                  is_measurement=False, value=trend)

    for key, value in (edge.evidence or {}).items():
        if key in ("observations", "duration_s", "mean_distance_px", "trend"):
            continue  # already stated above
        chain.add(f"{key}: {value}", source=Source.RULE.value, value=value)
    return chain


def chain_from_track(track) -> EvidenceChain:
    """Summarise what is directly measured about a track.

    Deliberately contains no conclusions: this is the raw observational
    record a reviewer starts from.
    """
    chain = EvidenceChain(
        assessment=track.describe(),
        confidence=0.9 if track.frame_count >= 5 else 0.5,
        kind="track_summary",
        subject_track_ids=[track.track_id],
        first_observed=track.first_seen,
        last_observed=track.last_seen,
        observation_count=track.frame_count,
    )
    chain.add(f"observed in {track.frame_count} frames",
              source=Source.TRACKER.value, value=track.frame_count)
    chain.add(f"tracked for {track.duration:.1f}s",
              source=Source.TRACKER.value, value=round(track.duration, 1))
    if track.trajectory:
        chain.add(f"travelled {track.path_length():.0f}px "
                  f"(net {track.displacement():.0f}px)",
                  source=Source.TRACKER.value)
    if len(track.cameras_seen) > 1:
        chain.add(f"seen by cameras {track.cameras_seen}",
                  source=Source.TRACKER.value, value=list(track.cameras_seen))

    for name, attr in track.attributes.items():
        # Every attribute already carries its own provenance; a low-confidence
        # attribute stays in the chain but is visibly low-confidence rather
        # than being quietly dropped.
        chain.add(f"{name} = {attr.value}", source=attr.source,
                  confidence=attr.confidence, value=attr.value)
    return chain


def explain_track(track, graph=None, tracks=None) -> Dict[str, Any]:
    """The full justification for everything known about one entity.

    Returns measurements and inferences in separate keys, so a caller cannot
    accidentally present a conclusion as an observation.
    """
    # The track's own chain mixes both kinds, so it is split here rather than
    # handed over whole under a key called "measured". A dict labelled
    # "measured" that contains inferences is exactly the conflation this API
    # exists to prevent.
    track_chain = chain_from_track(track)
    measured: List[Dict[str, Any]] = [i.to_dict() for i
                                      in track_chain.supporting_measurements()]
    inferred: List[Dict[str, Any]] = [i.to_dict() for i
                                      in track_chain.supporting_inferences()]

    for obs in track.observations:
        inferred.append(chain_from_observation(obs).to_dict())

    relationships: List[Dict[str, Any]] = []
    if graph is not None:
        for edge in graph.edges_for(track.track_id):
            other_id = (edge.object_id if edge.subject_id == track.track_id
                        else edge.subject_id)
            other = tracks.get(other_id) if tracks is not None else None
            relationships.append(chain_from_edge(
                edge,
                subject_category=track.category if edge.subject_id == track.track_id
                else (other.category if other else "entity"),
                object_category=(other.category if other else "entity")
                if edge.subject_id == track.track_id else track.category,
            ).to_dict())

    return {
        "track_id": track.track_id,
        "assessment": track_chain.assessment,
        "confidence": track_chain.confidence,
        "status": track_chain.status,
        "is_grounded": track_chain.is_grounded,
        "is_actionable": track_chain.is_actionable,
        "measured": measured,
        "inferred": inferred,
        "relationships": relationships,
        "note": ("'measured' contains only direct observations; everything in "
                 "'inferred' and 'relationships' is a conclusion drawn from "
                 "them and must not be presented as fact."),
    }
