# Argus Perception Architecture

**From "a set of detectors" to a visual perception and reasoning platform.**

This document turns the proposed vision into an executable plan. It states what
exists, what each phase adds, what it costs, and how you know it worked.

Phase 1 is **implemented and tested** (`backend/services/perception/`). Phases
2–7 are specified, not built.

---

## 0. The premise, stated honestly

No camera system can "analyse everything it can see". Resolution, occlusion,
motion blur, frame rate, viewpoint and model accuracy all bound what is
knowable. A system that hides those bounds will confidently invent facts, and
in a surveillance context invented facts have consequences for real people.

The engineering goal is therefore **maximum useful observation with explicit
confidence and evidence** — never the appearance of certainty.

Three rules follow, and they are enforced in code rather than left to
discipline:

| Rule | Enforcement |
|---|---|
| Every claim carries a confidence **and a source** | `Attribute(value, confidence, source)` — no bare values exist in the model |
| **Unobserved ≠ absent** | `entity.observed(name)` is distinct from `entity.get(name) is None`; tested |
| A conclusion without evidence is **not actionable** | `Observation.is_actionable` requires non-empty `evidence`, regardless of confidence |

The third rule is why the reasoning layer must never emit *"criminal activity
detected"*. It emits an assessment, a confidence, and the evidence that
produced it — so a human can disagree.

---

## 1. The problem this solves

Today each analyser invents its own shape:

| Module | Emits |
|---|---|
| `inference_engine` | `{class_id, class_name, confidence, bbox}` |
| `license_plate_recognition` | row: `plate_text, frame_id, vehicle_bbox` |
| `face_recognition` | encoding + name |
| `pose_estimator` | keypoints |
| `anomaly_detector` | row: `type, confidence, description` |

Nothing shares an identity, so nothing can be joined. The system physically
cannot answer *"what is this person carrying?"* or *"where has this plate been
seen?"* — the facts exist, in unrelated tables, with no key between them.

**The single most important change is one canonical representation.** Every
other item on the roadmap depends on it, which is why it is Phase 1.

---

## 2. The canonical model (Phase 1 — implemented)

```
Scene
├── camera_id / frame_id / timestamp
├── environment{}      lighting, weather, occupancy   (Attribute)
├── Entity[]
│   ├── kind           person | vehicle | animal | object | text | unknown
│   ├── category       free-form fine label ("delivery van")
│   ├── track_id       stable across frames
│   ├── bbox / mask_rle
│   ├── confidence / source
│   └── attributes{}   every property, each with confidence + source
├── Relationship[]     subject —predicate→ object, + evidence
└── Observation[]      durable statement, + evidence
```

**`kind` is a small enum; `category` is a free string.** This is deliberate: an
open-vocabulary detector must be able to report "traffic cone" without anyone
editing an enum, while `kind` stays small enough to route processing (a PERSON
gets pose, a VEHICLE gets plate reading).

### Conflict resolution is by source authority, not raw confidence

When two models claim the same attribute, comparing their confidences is
meaningless — they are not calibrated against each other. A face model at 0.55
is worth more than a generic detector at 0.99 *on identity*. So claims are
ranked by source:

```
human 100 > lpr/face 60 > ocr 55 > pose/segmenter 50 > vlm 45
      > open_vocab/detector 40 > appearance 35 > tracker 30 > rule 20
```

Confidence breaks ties within a rank. A human confirmation always wins.

### Why the layer imports no models

`backend/services/perception/` imports **only the standard library**. It is
pure data. That means the schema stays testable on any machine — no GPU, no
weights, no torch — and a CI job can validate it in seconds. A test enforces
this by blocking `torch`, `cv2`, `numpy`, `sklearn`, `scipy`, `onnxruntime` and
`PIL` via `sys.meta_path` and importing the package anyway.

### Verified against real output

| Check | Result |
|---|---|
| Real `InferenceEngine.detect_objects()` on a street photo | 16 detections → **16 entities, lossless** |
| Correct routing | 15 person, 1 object (handbag) |
| Relationship inferred | *person carrying handbag* |
| `Scene.to_dict()` → `from_dict()` | **byte-identical**, provenance preserved |
| Import with all ML deps blocked | **passes** |

---

## 3. Roadmap

Effort is for one engineer. "GPU" means it will not run on the current 2-core /
2.1 GB / no-CUDA host (measured: YOLOv8n alone is **108 ms/frame** and **568 MB
RSS**; a second model would not fit).

### Phase 1 — Universal observation model ✅ **done**

`Scene`, `Entity`, `Attribute`, `Relationship`, `Observation`, `BBox`, plus
adapters from the current detector/LPR/face/pose output and single-frame
spatial relationship inference. 15 tests, 2 mutation-verified.

**Acceptance:** real detector output converts losslessly; round-trip is exact;
malformed input is skipped not fatal; source authority beats confidence. *All
met.*

### Phase 2 — Perception expansion · ~3 weeks · GPU for two items

| Capability | Model | Runs here? |
|---|---|---|
| General scene OCR | PaddleOCR / EasyOCR | CPU, slow — batch on crops only |
| Colour + attribute extraction | HSV histogram in-region | **yes**, negligible cost |
| Scene classification (indoor/outdoor, day/night) | Places365 MobileNet | **yes**, ~40 ms |
| Instance segmentation | YOLOv8-seg or SAM | **GPU** |
| Open-vocabulary detection | OWL-ViT / GroundingDINO | **GPU** |

**Acceptance:** every capability writes `Attribute`s with its own `Source`;
disabling any one degrades the scene gracefully rather than erroring; a text
entity appears for a legible sign at 720p.

> **OCR ≠ LPR.** Keep license-plate reading specialised (it has regional
> formats, aspect priors and a plate detector). General OCR is a separate
> `Source.OCR` producing `TEXT` entities that may belong to no other object.

### Phase 3 — Temporal intelligence ✅ **done** (`temporal.py`, `scene_graph.py`)

`Track` accumulates across frames: trajectory (bounded ring), velocity,
direction, path-vs-displacement, attribute history, zones, cameras, status
(`active` → `stale` → `lost`).

`SceneGraph` makes relationships first-class and *persistent*, keyed by
`track_id` rather than the per-frame `entity_id` — an entity-keyed graph could
never accumulate anything. Edge confidence grows logarithmically with sustained
observation and decays linearly once unsupported, so *"was true once"* never
reads as *"is true"*.

Detectors implemented: **dwell** (requires duration *and* stillness — a long
walk is not loitering), **pacing** (path-to-displacement ratio), **following**
(both moving, headings agreeing, one behind the other), **approach/separation**
(distance trend), **disappearance** (established tracks only).

**Two bugs found by testing, both fixed:**

1. **Aging used the wall clock**, so replaying archived footage marked every
   track `lost` on arrival — silently disabling all temporal analysis on
   recorded video. Ages are now measured in *stream time*.
2. Relationship edges had the same flaw and decayed to zero instantly on
   replay.

**Measured on the live demo clip:** 47 tracks, 22 active, 47 relationships,
colour attributes and following edges accumulated — at **0.33 ms/frame for 15
entities** against 108 ms for detection (0.3% overhead). Camera FPS unchanged
at 14.78.

**Acceptance:** dwell fires exactly **once**, not once per frame ✓; trajectory
memory is bounded ✓; replayed footage still ages correctly ✓.

### Phase 4 — VLM integration · ~2 weeks · **GPU required**

A vision-language model behind the scheduler, triggered on interest — never per
frame. Answers "what is the person holding", "is the door open", "describe the
unusual activity".

**Acceptance:** VLM runs on <1% of frames; every VLM claim is stored with
`Source.VLM` and a crop reference; the pipeline's FPS is unchanged when the VLM
is disabled.

### Phase 5 — Reasoning · ~2 weeks · CPU  *(scene graph now done in Phase 3)*

The temporal scene graph exists; what remains is the reasoning layer that composes observations
into assessments **with evidence lists**. Output shape is fixed:

```
Assessment: unusual behaviour        Confidence: 0.71
Evidence:
  - dwell 143 s (baseline 22 s)
  - 3 zone transitions in 40 s
  - vehicle arrival within 12 s
```

**Acceptance:** no assessment can be emitted with an empty evidence list
(already enforced by `Observation.is_actionable`).

### Phase 6 — Memory · ~2 weeks · CPU

Route storage by purpose:

| Store | Holds |
|---|---|
| SQLite → PostgreSQL | structured events, entities, relationships |
| Object storage | snapshots, video evidence |
| **Qdrant** (in compose + a config schema, but no client code) | appearance + text embeddings |
| Time-series | telemetry, FPS, queue depth |

**Acceptance:** "find this person across cameras" returns ranked results by
embedding similarity; "what happened near the loading bay yesterday" answers
from stored observations.

### Phase 7 — Autonomous analysis · ~2 weeks · CPU (schedules GPU work)

The scheduler decides what to run from what it sees — the point where the
existing swarm/consortium broker earns its keep.

```
        frame
          ↓
   fast detection (always, 108 ms)
          ↓
    what is visible?
    ├── vehicle  → LPR
    ├── sign     → OCR
    ├── person   → pose → action
    └── unknown  → open-vocab → VLM → human confirmation
```

**Acceptance:** measured cost per frame drops versus running every analyser
unconditionally, with no loss in recall on a labelled clip.

---

## 3b. The capability registry ✅ **implemented** (`capabilities.py`)

The swarm auctions a time budget between three hard-coded agents. It cannot
answer the question that matters — *"what can I run, what will it cost, and
what would it tell me that I do not already know?"* — because nothing described
the analyses themselves.

Each capability now declares its tier, cost, what it **provides**, what it
**requires**, and whether its backend actually loaded:

```
GET /api/v1/perception/capabilities   ->  8/14 available on this host, gpu=false

  OK  cpu  object_detection      108.0 ms   provides bbox, category
  OK  cpu  tracking                4.0 ms   requires detection
  OK  cpu  temporal_analysis       1.0 ms   provides speed, direction, dwell
  OK  cpu  relationships           2.0 ms   provides near, carrying, following
  OK  cpu  basic_attributes        3.0 ms   provides dominant_colour, size
  OK  cpu  pose / face / lpr     35-60 ms
  --  cpu  ocr                            ModuleNotFoundError: paddleocr
  --  gpu  segmentation / open_vocabulary / action_recognition / vlm
                                          requires a CUDA device; none detected
```

**Availability is probed by importing, never assumed from config** — a package
can be declared in `requirements.txt` and still fail to load (missing system
library, wrong wheel, unsupported CPU). Guessing would advertise capabilities
that do not work.

From this the scheduler computes **information gain per millisecond**:

```python
registry.plan(context={"person", "detection", "track_id"}, budget_ms=100)
```

- a capability whose `requires` is unmet is **inapplicable** — running a plate
  reader on a frame with no vehicle is pure waste, and it is excluded, not
  merely deprioritised;
- a capability whose `provides` is **already known** scores zero, so the budget
  goes to something that would actually add information;
- `record_cost()` replaces estimates with measured values, exponentially
  smoothed so one slow frame does not distort scheduling.

This is what turns the swarm from a resource auction into something that
reasons about information gain. Wiring `plan()` into the consortium broker is
Phase 7.

---

## 4. The unknown-object pathway

A closed detector can only say "person / car / bicycle". An open-world system
must be able to say **"I see something I do not confidently understand"** —
otherwise the system is permanently trapped inside whatever class list it was
trained on.

```
detection
   ├── known + confident        → normal processing
   └── unknown or low-confidence
          → crop → open-vocab → VLM → candidate descriptions
          → store as UNKNOWN with candidates
          → human confirmation when it matters
```

`EntityKind.UNKNOWN` and `LOW_CONFIDENCE = 0.4` exist in Phase 1 for exactly
this. An unknown entity is recorded, not discarded — discarding it is how a
system stays blind to anything novel.

---

## 5. Compute reality

Measured on this host: **2 cores, 2.1 GB RAM, no CUDA.** YOLOv8n = 108 ms/frame,
568 MB RSS with one model loaded.

| Tier | Fits here | Needs a GPU |
|---|---|---|
| Detection, tracking, colour, geometry, relationships, temporal logic | ✅ | |
| Scene classification, OCR on crops | ✅ (batched) | |
| Segmentation, open-vocabulary, VLM, Re-ID at scale | | ✅ |

**Design consequence:** heavy models are *pluggable backends behind an
interface*, disabled by default and reported as unavailable — not hard
dependencies. The canonical layer never imports them, so absence degrades a
scene's richness instead of breaking the pipeline.

---

## 6. Privacy and retention

Richer perception means more identifiable data. Existing controls extend to the
new fields:

- Attributes that identify a person (`identity`, `plate`, appearance embeddings)
  inherit the retention windows in `config/config.yaml`.
- The audit trail records who queried them; the RBAC model already gates the
  routes.
- Face embeddings remain deliberately excluded from automatic deletion (an
  explicit, documented decision in `retention.py`).
- **Before Phase 2 ships**, extend retention to cover appearance embeddings and
  OCR text — both are re-identifying, and neither is currently in scope.

---

## 7. Suggested order

1. **Phase 1** ✅ — nothing else is joinable without it
2. **Phase 3** ✅ (temporal + scene graph) — done; 0.3% frame cost
3. **Phase 2** CPU items — colour ✅ done; scene classification and OCR remain
4. **Phase 6** (memory) — makes everything retrievable
5. **Phase 5** (reasoning) — needs 3 + 6 to be meaningful
6. **Phase 2** GPU items + **Phase 4** (VLM) — when hardware exists
7. **Phase 7** (autonomous scheduling) — once there is enough to schedule

Phases 3 and 6 deliver the most value on the hardware you actually have.
