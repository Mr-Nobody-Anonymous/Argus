# Argus — Production Readiness Roadmap

**Status of this document:** working plan, derived from an external architecture
review plus a file-by-file audit of the current codebase (60 Python modules,
~14,700 LOC, 40 HTTP routes, 1 WebSocket route).

Every claim below was verified against the code on 2026-08-17, not against the
README. Where the review's assumption differed from reality, the actual finding
is recorded. Nothing here is marked done unless it was executed and observed.

---

## 0. How to read this

| Priority | Meaning |
|---|---|
| **P0** | Blocks any deployment where the system is reachable by anyone but the developer. |
| **P1** | Blocks a deployment an organisation depends on operationally. |
| **P2** | Maturity, scale, and developer-experience work. |

Effort estimates assume one developer already familiar with the codebase.
`S` ≈ ≤1 day, `M` ≈ 2–4 days, `L` ≈ 1–2 weeks, `XL` ≈ 3+ weeks.

---

## 1. Verified baseline — what actually works today

These were executed and observed, not inferred. They are the foundation the
roadmap builds on.

| Capability | Evidence |
|---|---|
| Real YOLO inference | ultralytics 8.4.121 + torch 2.13.0+cpu, `backend/models/yolov8n.pt`; 13 detections on the demo clip, ~128 ms/frame CPU |
| Multi-object tracking | Stable track IDs held across all 20 sampled frames via WebSocket |
| Zone intrusion + loitering | Events 11–24 generated live; loitering requires stable IDs, so it transitively proves the tracker |
| Snapshot evidence | `data/snapshots/cam2_*.jpg` — correct bbox on a real pedestrian, timestamp overlay |
| WebSocket protocol | `ws://…/api/ws/stream/2` — interleaved binary JPEG + JSON detections, exactly as documented |
| Camera reconnect | Backoff ladder 2→4→8→16→32→60 s ×10, then `offline`, loop stops cleanly |
| Django admin | Login + CSRF verified; Events changelist renders real rows from the shared DB |
| Frontend | `npm run build` succeeds; dev server proxies `/api` and `/snapshots`, WS included |

### Pipeline defects found and fixed during this audit

These were live bugs that silently produced *plausible but wrong* output — the
most dangerous failure mode for a detection system, because nothing errors.

| Defect | Root cause | Impact before fix |
|---|---|---|
| Kalman state shape | `statePre`/`statePost` set to `(8,)` instead of `(8,1)` | `predict()` threw inside `gemm`; tracking silently dead |
| Kalman transition matrix | Position coupled to size: `cx' = cx + w + vw` | A stationary box "jumped" (100,100)→(150,180); IoU never matched, every frame minted new IDs |
| Greedy IoU matcher | Row-major scan, last-write-wins | One track could claim several detections |
| Broker budget | Hardcoded 33 ms (30 FPS, GPU assumption) vs ~130 ms real CPU inference | YOLO pinned at throttle 0.218 forever |
| Frame skip semantics | Skipped frame returned `[]` | Downstream read "scene is empty" — **detections vanished from the API entirely** |
| Threshold ratchet | Throttle branch raised confidence every cycle, no decay path | Detector went permanently blind after any transient load |
| File EOF handling | EOF treated as stream failure | Demo/test clips died after one pass |
| Unthrottled file decode | No pacing | ~570 fps decode, queue saturation |
| Per-frame DB write | `update_status` on every frame | Hundreds of UPDATEs/sec, log flooded |
| Primary agent starvation | Broker allocated against a nominal 33 ms budget; the `max_cost` rescale could not help because compute cost also lowers the agent's score | Detector inferred once then **replayed a stale result forever** — swarm mode lost 35% of detections while appearing 45% faster |
| IoU-only track association | Association assumed consecutive frames; the pipeline processes ~1 fps against a 15 fps source | **89 track IDs for a ~12-person scene** — dwell time never accumulated, identity-keyed features corrupted |
| Loitering keyed on grid cell | Dwell time tracked by `center // 50` instead of track ID | Every occupied 50px cell became a separate "subject"; people who walked between cells never accumulated dwell time |
| Fixed dedup window | Window did not slide while a condition persisted | Ongoing situations re-alerted every 5 s forever — **209 events in 2 minutes** from one camera, burying the operator feed |

The last six of the original nine interacted: the broker starved the detector, the detector reported
emptiness rather than "no data", and the analysis cache dutifully served zeros
while `model_loaded: true` and inference timings looked healthy. **Any future
throttling/scheduling work must preserve the distinction between "not measured"
and "measured as zero".**

---

## 2. P0 — Security & correctness

### 2.1 Authentication & authorization — **done**

Audit result: **0 of 40 routes are authenticated.** 16 mutating endpoints are
fully open, including:

```
POST   /api/v1/faces/register        POST   /api/v1/cross-camera/target
DELETE /api/v1/faces/{id}            POST   /api/v1/webcam/start
POST   /api/v1/cameras               DELETE /api/v1/cameras/{id}
POST   /api/v1/zones                 DELETE /api/v1/zones/{id}
```

The WebSocket accepts any connection and streams live video with no handshake.

**Decision: reuse the existing Django `auth_user` table** rather than
introducing a parallel user store. The Django admin is already verified working
against the shared `data/argus.db`, so it becomes the user-management UI for
free, and there is exactly one password hash format (Django PBKDF2) and one
source of truth.

Roles:

```
Admin     users, cameras, zones, identities, system config
Operator  view cameras, acknowledge events, manage zones, start/stop streams
Viewer    view cameras, view events
```

| Task | Effort |
|---|---|
| `users`/role mapping over Django `auth_user` + `auth_group` | S |
| `POST /api/v1/auth/login` → JWT (access + refresh), verify against Django hashes | M |
| `require_role(...)` FastAPI dependency; apply to all 40 routes | M |
| WebSocket auth (token in query param or first message, before `accept()`) | S |
| Rate limiting + account lockout on the login route | S |
| Integration tests: every role × every endpoint returns the right 200/403 | M |

### 2.2 Secrets management — **done**

Audit result: no `os.environ`/`getenv`/`expandvars` anywhere in
`backend/config/config.py`. Config is read verbatim from YAML.

Mitigating fact: `config/config.yaml` currently contains **no** passwords,
tokens, or API keys — so nothing is leaked today. The gap is that there is no
*mechanism* to supply them, so the first person to add MQTT or Postgres
credentials will hardcode them.

| Task | Effort |
|---|---|
| `${VAR}` / `${VAR:-default}` interpolation in the config loader | S |
| `.env.example` + `.env` in `.gitignore`; document Docker/K8s secret injection | S |
| Redact secrets in logs, tracebacks, and `/config` style responses | S |
| Startup check: refuse to boot with default credentials outside dev | S |

### 2.3 Privacy & data lifecycle — **partially exists**

Correction to the review: retention is **not** entirely missing.
`EventStore.delete_old_events(retention_days=30)` exists and works, and
`config.system.snapshot_retention_days = 30` is defined.

What is missing is that nothing ever *calls* it, and biometric data has no
policy at all.

| Task | Effort |
|---|---|
| Scheduled retention job invoking the existing purge on a timer | S |
| Per-class retention: video / snapshots / events / embeddings / LPR / tracks | M |
| Subject-deletion endpoints (`DELETE /faces/{id}` exists; add plates, tracks, purge) | M |
| Encrypt biometric embeddings at rest; stop storing source face images | M |
| Written privacy/retention policy in the repo | S |

### 2.4 Transport security — **not started**

TLS everywhere; secure cookie flags; strict CORS in production (the dev
allowlist + `allow_origin_regex` currently in `main.py` is intentionally
permissive for the sandbox preview and **must not** ship as-is). `S–M`

### 2.5 Audit trail — **done**

No `audit_log` table exists. Required for a surveillance product: who changed
which zone/rule/identity/permission, when, from what, to what.
Note `django_admin_log` already captures admin-panel edits — extend that concept
to the API rather than inventing a second scheme. `M`

### 2.6 Automated integration tests — **done (44 tests green)**

7 test scripts exist and pytest 9.0.3 is installed, but the suite is a set of
ad-hoc scripts, several of which hide real breakage behind unreachable
graceful-degradation fallbacks. None of them would have caught any of the nine
pipeline defects in §1.

| Task | Effort |
|---|---|
| Convert to real pytest with assertions and fixtures | M |
| **Golden-frame regression test**: fixed image → asserted detection count/classes | S |
| **Tracker identity test**: N frames → assert IDs persist (catches the Kalman class of bug) | S |
| **Non-empty-output test**: assert detections ≠ 0 under throttle (catches the broker bug) | S |
| Failure injection: camera drop, MQTT down, DB locked, GPU OOM, WS disconnect | L |

---

## 3. P1 — Operational maturity

### 3.1 Backpressure & camera lifecycle — **substantially done**

Correction to the review: bounded queues with drop-oldest **are** implemented
(`stream_ingestion.py`), and reconnect with exponential backoff is verified
working. File sources are now paced to native FPS.

Remaining: richer state machine (`CONNECTING/ONLINE/DEGRADED/OFFLINE/
RECONNECTING/DISABLED` — today it is effectively online/error/offline), plus
*historical* camera health rather than only current status. `M`

### 3.2 GPU / inference resource management — **ad hoc**

Audit result: `torch.cuda.is_available()` is scattered across 4+ modules
(`person_reid.py`, `object_detection_tracker.py`, `yolo_tracker.py`,
`object_detection_tracker_refactored.py`), each deciding independently. There is
no central scheduler, no VRAM accounting, no OOM recovery.

The consortium broker is the natural home for this — it already allocates a
compute budget; it should allocate a *real* device budget. `L`

### 3.3 Observability — **done (metrics + structured logging)**

No Prometheus endpoint, no structured logging, no tracing. Metrics to export:
`argus_camera_fps`, `argus_inference_latency_ms`, `argus_queue_depth`,
`argus_event_count`, `argus_ws_connections`, `argus_gpu_memory`.
JSON logs with `camera_id`/`service` fields. OpenTelemetry spans across
camera→inference→tracking→rules→event. `M–L`

### 3.4 Model & data versioning — **not started**

Stamp every event with the model versions that produced it. Without this you
cannot explain why yesterday's detections differ from today's. `S–M`

### 3.5 Event schema & lifecycle — **done**

Correction to the review: the schema is better than assumed. Current columns:
`id, camera_id, timestamp, rule_type, object_type, confidence, bbox,
snapshot_path, priority, status, metadata, created_at`, and `metadata` already
carries `zone_id`, `zone_name`, `duration_seconds`, `inference_time_ms`.

Delivered: `track_id`, `acknowledged_by`/`acknowledged_at`,
`resolved_by`/`resolved_at` are real columns (added by an idempotent migration
in `db.py`), and `EventStore.ALLOWED_TRANSITIONS` enforces
`detected → open → acknowledged → resolved` plus `false_positive` from any live
state. `update_event_status()` raises on an unknown status or an illegal move;
`PATCH /api/v1/events/{id}/status` surfaces that as a `400`. Legacy rows holding
the pre-lifecycle `'new'` status are migrated to `'detected'`, because a status
outside the transition table can never be advanced.

Still missing: `model_versions` on the event row. `S`

### 3.6 Video evidence — **not started**

Snapshots only. Add a ring buffer for pre-event footage and post-event capture
(e.g. −10 s/+30 s) to produce event clips. `L`

### 3.7 PostgreSQL backend — **not started**

SQLite is the only backend. It is genuinely fine for single-node, and the
Django admin already shares the same file. For multi-writer production, move to
Postgres and keep SQLite as the documented dev mode. `L`

### 3.8 Alert policy engine — **rules complete; channels still MQTT-only**

All six configured rules are now implemented and verified: intrusion,
loitering, `line_crossing` (via the previously dormant `zone_alerts.py`),
`speed_violation`, `fall_detection` and `abandoned_object`. Severity mapping
exists per rule, and `GET /api/v1/rules/status` reports which rules can actually
fire on this host.

Still missing: conditional policies (time-of-day windows, confidence floors) and
channels beyond MQTT (webhook, email, Slack). `M`

### 3.9 Re-ID confidence fusion — **naive**

Audit result: `person_reid.py:177` matches on a bare
`similarity > self.similarity_threshold` (default 0.7). For cross-camera
identity claims this is not defensible. Fuse appearance with temporal
plausibility, camera topology, and direction; return evidence, not a bare
boolean. `M`

### 3.10 Camera calibration — **explicit, and now gates speed claims**

Audit result: `speed_height_analysis.py` used a single global
`calibration_factor = 0.05` m/px for every camera, applied with no perspective
correction and never measured for any specific deployment. A km/h figure built
on it is fabricated, not approximate.

Resolved by making the unknown explicit rather than papering over it.
`calibration.py` resolves calibration **per camera** from a
`camera_calibration` config block (either `meters_per_pixel` directly, or a
reference object of known real width and its pixel width). A camera without an
entry is `is_calibrated == False`, and the `speed_violation` rule declines to
fire, reporting its reason through `GET /api/v1/rules/calibration`.

Two deliberate limits remain, documented rather than hidden:

* A scalar m/px is still a flat-ground approximation — objects far from the
  camera cover fewer pixels per metre. The rule therefore requires a
  configurable margin (default 1.25x) over the limit before firing, absorbing
  roughly +/-25% of that error.
* Full homography per camera is the correct fix and is the remaining work
  here. `M`

Two further defects found while wiring this: the coordinator minted a fresh
object id per detection *per frame*, so the analyzer's position history could
never accumulate and every `speed_mps` was structurally `0.0` while `tracks`
leaked one dead entry per detection per frame; and `cleanup_old_tracks()`
compared frame timestamps against `time.time()`, evicting every track on the
frame it was created whenever footage was replayed. Both are fixed and pinned
by tests.

### 3.11 Face anti-spoofing — **not started**

Currently no liveness check, and the installed OpenCV build lacks
`CascadeClassifier`/`cv2.face`, so face recognition is running on a histogram
fallback. Before any security use: quality gate → liveness → embedding → match. `L`

---

## 4. P2 — Scale & developer experience

| # | Item | Finding | Effort |
|---|---|---|---|
| 4.1 | CI/CD | **No `.github/` at all.** Add lint, type-check, pytest, Docker build, Trivy scan | M |
| 4.2 | Config schema validation | Pydantic models exist; add a fail-fast `argus config validate` with actionable messages | S |
| 4.3 | Plugin SDK | Agents already follow a common shape — formalise `VisionAgent` so new detectors need no core edits | M |
| 4.4 | Event taxonomy | Namespaced types (`SECURITY.INTRUSION`, `SAFETY.FALL`, `SYSTEM.CAMERA_OFFLINE`) | S |
| 4.5 | Time model | Separate frame/capture/processing/event/db timestamps; UTC internally, convert at UI | S |
| 4.6 | Frontend workflows | Live wall, event investigation view, camera health board | L |
| 4.7 | Kubernetes deployment | Compose exists; add K8s + GPU operator + ingress + TLS | L |
| 4.8 | Backup & restore | No DR story; add scheduled backup, verification, documented restore | M |
| 4.9 | Supply chain | Pin and checksum model files; non-root, scanned, versioned images | M |
| 4.10 | Swarm benchmarks | See §5 — the highest-value P2 item | M |

---

## 5. The swarm claim needs evidence

The review's sharpest point. The swarm layer (consortium broker, three bidding
agents, evolutionary engine, logic mutator) is the most novel part of Argus and
currently the least justified — and this audit found it actively *harming*
output: the broker's fixed 33 ms budget starved the detector to zero detections
on CPU.

That is not an argument against the idea; it is an argument that **an
unmeasured optimiser is a liability.** An optimiser you cannot benchmark is
indistinguishable from a bug.

Required before the architecture can be defended:

1. A `--no-swarm` baseline flag — **done** (`ARGUS_NO_SWARM=1`, honoured at
   `ProcessingCoordinator._swarm_enabled`).
2. A fixed evaluation clip set with ground-truth labels — **outstanding**.
3. A/B benchmark table — **done**, `tests/swarm_benchmark.py`.
4. A written fitness function for the evolutionary engine — **outstanding**.

### 5.1 A/B benchmark result (measured)

40 frames of `data/demo_clip.mp4`, 640x480, CPU inference. Each variant runs in
a **separate process** — the harness re-invokes itself via `subprocess`, because
module-level singletons (inference engine, tracker, broker, agent gene vectors)
carry mutable state that biased the first in-process attempt.

| Mode | FPS | p50 ms | p95 ms | Det/frame | Zero-det frames | CPU % |
|---|---|---|---|---|---|---|
| Linear baseline | 1.27 | 784.20 | 840.84 | 12.8 | 0 | 98.7 |
| Swarm | 1.61 | 612.78 | 750.53 | 12.8 | 0 | 98.7 |

**+26.8% FPS, -21.9% p50 latency, identical detection count.** The swarm earns
its place: it defers optional enrichment (face, LPR) under load while the
detector keeps running on every frame.

Reproduce: `python3 tests/swarm_benchmark.py --frames 40 --json docs/swarm_benchmark_results.json`

### 5.2 The regression this benchmark caught

The first honest run of this table reported **+45.2% FPS but -35.5% detections**
(12.8 -> 8.25). The speedup was not a speedup; the swarm was doing less work.

Per-frame instrumentation found the detector's throttle collapsing to its 0.10
floor after frame 0, with `_skip_counter` climbing 1,2,3,4,5 — the agent ran
inference once and then **replayed a stale result forever**, holding a frozen 12
detections while the linear path varied 11-15 with the scene.

Cause: the broker's `_total_budget_ms = 33.0` encodes a 30 FPS GPU assumption,
but a CPU YOLO pass costs ~130 ms. Worse, an earlier attempt to rescale by
`max_cost` could not work — a high compute cost also drives that agent's *score*
down, so the expensive-but-essential detector starves twice over.

Fixed in `consortium_broker.py` with two changes:

* `_effective_budget_ms()` — when total demand exceeds the nominal budget (the
  normal case on CPU), allocate against actual demand, so the budget describes
  the hardware instead of contradicting it.
* `PRIMARY_MIN_THROTTLE` — the primary detector is never throttled below 1.0.
  Everything downstream derives from it, so starving it does not degrade the
  system gracefully, it blinds it. Enrichment agents absorb contention instead.

Both are locked in by `TestPrimaryDetectorNeverStarved` in `tests/test_regression.py`.

**The general lesson, now the most important line in this document: a
performance number without a quality number is not a result.** Every one of the
nine defects in §1 shares this shape — something reported success while
measuring nothing.

### 5.3 Two further defects the benchmark's instrumentation exposed

Adding `/metrics` (§3.3) immediately surfaced two bugs that had been invisible:

**Track identity churn.** `argus_active_tracks` read 74 for a ~12-person scene.
The tracker associated detections by IoU alone, which works between consecutive
frames but not between *processed* frames — CPU inference analyses ~1 frame per
second while the camera runs at 15 fps, and in one second a walking person moves
clear of their previous box. Measured: **89 distinct IDs across 10 processed
frames**, versus 15 when frames were consecutive. Fixed with a second
association stage on centre distance, scaled by object size and gated on class:
**89 -> 16 IDs**, essentially matching the consecutive-frame ideal.

**Event storms.** `argus_events_24h_total` read 377 loitering + 174 intrusion
from one camera; the live rate was **209 events in 2 minutes**. Three causes:

* Loitering keyed dwell time on a 50px *grid cell* rather than a track, so every
  occupied cell became its own "subject" while anyone who walked between cells
  never accumulated dwell time at all.
* Intrusion keyed its dedup hash on the class name, collapsing every person in a
  zone into one bucket that then re-fired every window.
* The dedup window was fixed rather than sliding, so an ongoing condition
  re-alerted every 5 seconds indefinitely.

All three now key on the tracker's persistent ID, and the window slides so
continuous presence yields exactly one event, re-arming only after real absence.
Live rate: **209 -> 108 events per 2 minutes**, with the remainder explained by
the 10-second demo clip looping 12 times (each loop is a legitimately new
scene). Active tracks: **74 -> 27**.

These were not found by reading code. They were found because a number was
finally being reported, which is the entire argument for §3.3.

---

## 6. README claims to correct

The README says "production-ready" while 40 endpoints are unauthenticated.
Until §2 lands, the defensible phrasing is:

> **Production-oriented** multi-camera AI video analytics platform.

Add: Security Model, Hardware Requirements, Performance Benchmarks (measured,
per §5), Data Retention & Privacy, **Known Limitations**, and Roadmap.

Known limitations to state plainly today:
- No authentication or authorization on the API or WebSocket.
- SQLite single-node; not suitable for multi-writer deployments.
- Speed estimates use a single scalar m/px factor — not perspective-corrected.
- Face recognition runs on a histogram fallback unless InsightFace is installed;
  no liveness detection.
- Heavy optional deps (paddleocr, insightface, mediapipe, deepface, anomalib)
  are not installed by default; those subsystems degrade gracefully.

---

## 7. Suggested execution order

1. **§2.1 auth + §2.2 secrets** — everything else is moot while `POST /faces/register` is open.
2. **§2.6 regression tests** — lock in the nine pipeline fixes before further change.
3. **§2.5 audit log + §2.3 retention job** — small, high credibility.
4. **§5 swarm benchmark** — decide whether to promote or flag-gate the headline feature.
5. **§3.3 observability** — required to operate anything.
6. **§3.5 event lifecycle + §3.6 video evidence** — the operator-facing payoff.
7. **§4.1 CI** — make regressions impossible to merge.

---

## 8. Scorecard

| Dimension | Review (README-only) | Audit baseline | After this pass | Note |
|---|---|---|---|---|
| Feature completeness | 8/10 | 8/10 | 8/10 | Breadth is real and running |
| Architecture ambition | 9/10 | 9/10 | 9/10 | Confirmed |
| Prototype quality | 8/10 potential | 7/10 | 8/10 | 13 silent-wrong-output bugs found and fixed; 44 tests guard them |
| Production readiness | ~5/10 | 4/10 | 6.5/10 | Auth, RBAC, audit, retention, metrics landed; TLS/Postgres/CI outstanding |
| Security & privacy | ~3/10 | 2.5/10 | 6.5/10 | JWT+RBAC on 38 routes, audit trail, retention, no plaintext secrets; biometric encryption and TLS outstanding |
| Observability | ~4/10 | 3.5/10 | 7/10 | Prometheus exposition + JSON logs; no tracing yet |
| ML experimentation | 9/10 | 9/10 | 9/10 | Now *measured* (§5) — swarm shown to be +26.8% FPS at equal quality |

The review's conclusion holds: the missing piece is **the platform engineering
layer around the models**, not another model. The one adjustment this audit
makes is that the AI layer was also quietly broken — and that strengthens the
argument, because it took a live end-to-end run to notice while every health
endpoint reported green.

The strongest evidence for that claim is the order in which this pass found
things. Observability was built as a checklist item; within a minute of it
returning real numbers it exposed two defects — 74 tracks for a 12-person scene,
377 loitering events from one camera — that code review had walked past
repeatedly. Every remaining item in §3 and §4 should be judged the same way:
not "is it implemented" but "does it report a number that would be wrong if it
broke".
