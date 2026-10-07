# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

> No Git tags or GitHub Releases are currently published. Version-labelled sections below are project notes, not evidence of a published release.

## Unreleased

### Security
- Reject legacy MD5 password hashes; accounts using them must reset their passwords.
- Restrict retention and perception-memory SQL to fixed query templates, and document the existing allowlists for dynamic update statements.
- Require webhook URLs to be absolute HTTP(S) URLs without embedded credentials.
- Clarify that the Python filter compiler constrains trusted generated code but is not a sandbox for hostile source.

### Dependencies
- Refresh vulnerable frontend dependencies, including the React Router v7 security fixes.
- Align React 19, React Router 7, MUI 9, Framer Motion 14, Recharts 3, Vite 8, and their peer dependencies; regenerate the frontend lockfile.
- Update the minimum versions for the core Python packages and optional Elasticsearch/Qdrant clients from the merged Dependabot branches.

### Fixed
- Import `os` in the database adapter before reading `DATABASE_URL`, so schema initialization succeeds.

- Load service exports lazily so importing the perception layer does not require optional ML and geometry packages.
- Refresh the documented API route and authentication counts to match the registered FastAPI operations.

## Draft v0.2.0 notes (not yet released; updated 2026-09-30)

### Security
- **Purged Default Credentials**: Removed hardcoded `admin/admin123` across all scripts, Docker entrypoints, and documentation. Added secure random password generation on first run and CLI utility `python argus.py create-admin`.
- **Biometric Encryption at Rest**: Implemented application-level authenticated encryption using **AES-256-GCM** with per-record nonces in `backend/security/crypto.py`. New face embeddings are encrypted at rest; legacy plaintext embeddings are upgraded when recognition is explicitly enabled.
- **Cryptographic Audit Log**: Implemented SHA-256 hash chaining on all audit trail events in `backend/services/management/audit_log.py` with tamper-detection verification endpoint (`GET /api/v1/audit/integrity`).
- **Sliding-Window Rate Limiting**: Added `RateLimitMiddleware` protecting sensitive authentication endpoints (`/api/v1/auth/*`) against brute-force and credential stuffing.
- **Security Headers & CORS**: Injected defensive headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, `X-XSS-Protection`, `Referrer-Policy`) and restricted CORS handling.
- **Added SECURITY.md**: Established a responsible vulnerability-reporting route and severity triage guidance without promising a response-time SLA.

### Added
- **PostgreSQL Database Adapter & Migration Tool**: Added `backend/database/postgres.py` with automatic connection routing via `DATABASE_URL` and `backend/scripts/migrate_sqlite_to_pg.py` for SQLite-to-PostgreSQL migration.
- **Automated Security CI Workflows**: Added `.github/workflows/security.yml` (running `pip-audit`, `bandit`, `npm audit`), `.github/workflows/docker.yml`, and `.github/dependabot.yml`.
- **Pre-commit Hooks**: Added `.pre-commit-config.yaml` with trailing whitespace, YAML validation, Ruff, and Bandit checks.
- **Evaluation Harness**: Added a metrics engine for Precision, Recall, F1, and ID-switch measurements. A public labeled ground-truth dataset is not included yet.
- **Reproducible Benchmark Suite**: Added `benchmarks/benchmark.py` recording hardware environment metadata, latency, and throughput FPS.
- **Production TLS Configurations**: Added reference configurations for Caddy (`infrastructure/caddy/Caddyfile`) and Nginx (`infrastructure/nginx/nginx.conf`).
- **GitHub Contribution Templates**: Added `.github/ISSUE_TEMPLATE/` (bug report, feature request, security report) and `.github/PULL_REQUEST_TEMPLATE.md`.

## CityOS — intersection intelligence layer

A geometry-only traffic-intelligence layer inspired by Aeva CityOS, built on
the existing detection pipeline. It consumes ONLY detection geometry (track
ids, classes, boxes, speeds) — no face embeddings, plate text or imagery enter
it, so the intersection model it builds is privacy-preserving by construction.

- `backend/services/cityos/` (new package):
  - **Digital Perception Engine** (`perception_engine.py`) — road-user objects
    with stable ids, category classification (vehicle/truck/bus/motorcycle/
    cyclist/pedestrian), normalised position, velocity, compass heading and
    bounded trajectory history; stale objects retire into completed trips.
  - **Safety analytics** (`safety_analytics.py`) — wrong-way detection against
    learned dominant flow per approach (or configured legal headings),
    near-miss detection via time-to-collision between converging vehicles,
    and VRU conflict detection (vehicle vs pedestrian/cyclist). All events
    deduplicated with cooldowns.
  - **Traffic flow** (`traffic_flow.py`) — per-minute volume buckets by
    category, turning-movement matrix from completed trips, average/85th-
    percentile speeds, live demand-per-approach.
  - **Signal optimiser** (`signal_optimizer.py`) — NS/EW phase machine with
    FIXED / ADAPTIVE / MANUAL modes, demand-proportional green recommendations,
    operator force-phase override, audit-logged command log. It recommends; it
    never actuates a real controller.
  - **Engine** (`engine.py`) — per-intersection orchestration, camera→
    intersection registry (scales 1 → N intersections), digital-twin snapshot
    and merged alert feed.
- Wired into `ProcessingCoordinator._store_analysis()` so every processed frame
  feeds it in both swarm and linear modes; disable with `ARGUS_NO_CITYOS=1`.
  Failures are swallowed — CityOS must never break a frame.
- 9 new role-guarded endpoints under `/api/v1/cityos/*` (status, twin, alerts,
  flow, signal status/mode/phase, bind).
- New **CityOS dashboard page** (`/cityos`, Traffic icon in the nav): live
  digital-twin canvas (top-down intersection, colour-coded road users,
  trajectory trails, velocity vectors, signal-state indicator), category/VRU
  counters, safety stat cards, alert feed, traffic-volume chart, turning
  movements, and a signal panel with adaptive recommendation + manual override.
- `tests/test_cityos.py`: 15 tests covering every analyser; found and fixed a
  real deadlock in `SignalOptimizer.tick()` during development.

### Also fixed in this change
- Unresolved git merge-conflict markers in `backend/api/stream_ws.py` that made
  the entire application unimportable.
- `tests/test_regression.py` failed to import on Python 3.10 (`tomllib`);
  added the standard `tomli` fallback.

## Alert delivery and video evidence — events that leave the database

### Every alert Argus produced was delivered nowhere
`MQTTPublisher.publish_event()` was fully implemented and `mqtt.enabled` was
`true` in `config.yaml`, but **nothing in the codebase ever called it**. Events
were written to SQLite and that was the end of them; no endpoint revealed the
gap. This is the same defect class as the three configured-but-unimplemented
rules: a capability advertised in config that no code path delivers.

- `backend/services/management/notifications.py`: MQTT and webhook transports,
  an alert policy, a bounded queue with a worker thread, and counters for
  everything it drops. The webhook uses stdlib `urllib` — no new dependency.
- Dispatch is hooked into `EventStore.create_event`, the single choke point
  every event passes through, and wrapped so a failing transport can never
  prevent an event being recorded. Recording is the guarantee; delivery is
  best-effort and separately reported.
- **Availability is probed live, never inferred from the config flag.** With
  `mqtt.enabled: true` and no broker running, `GET /api/v1/notifications/status`
  reports the channel unavailable *and gives the reason*. A delivery that was
  never attempted is never reported as sent.
- Policy: priority floor, allow/deny lists, quiet hours (including windows that
  wrap midnight), and a per-`(camera, rule)` rate limit so one flapping camera
  cannot exhaust a pager. Every suppression is counted and surfaced.
- `POST /api/v1/notifications/test` sends a synthetic alert through every
  channel and reports the real per-transport outcome, so a misconfigured
  webhook is found during setup rather than during an incident.

Verified live: a 60-second run on real footage delivered **34 alerts over real
HTTP** to a local receiver, with honest counters — 34 webhook successes, 34
MQTT failures (no broker present), 20 rate-limited.

### Pre-event video evidence
A snapshot shows the instant a rule fired, not the approach — usually the part
an investigator needs. `backend/services/management/evidence_clips.py` keeps a
per-camera ring buffer and exports an mp4 when a high-value rule fires.

- **Frames are buffered JPEG-encoded, not raw.** Measured on 480p: raw costs
  0.88 MB/frame, so 10 s at 10 fps is 88 MB *per camera* — 352 MB across four
  cameras, which does not fit beside the detection model on a 2 GB host.
  JPEG q=80 costs 0.5–20 MB for the same window at ~2 ms/frame encode, against
  an existing ~108 ms/frame YOLO budget.
- **Bounded in bytes as well as frames.** Frame size varies ~40× with scene
  content, so a frame count alone is not a memory bound. Both limits are
  enforced and the byte ceiling is reported.
- Clip export is opt-in per rule (`evidence_clips.clip_rules`); writing one
  costs a decode per frame plus disk.
- A clip that cannot be written says so. No codec, no frames, or an unwritable
  path each return `written: false` with a reason — never a path to a file that
  does not exist.
- Served from `GET /api/v1/events/{event_id}/clip`: `404` with the reason when
  there is no clip, `410` once retention has removed it.
- Retention: `clips_days` plus a `clips_max_mb` ceiling, oldest evicted first.
  New files on disk with no expiry is a disk-exhaustion bug.

Verified live: a real intrusion event produced a playable 40-frame, 6.3-second
mp4 of actual footage, attached to the event and readable back with OpenCV.

### Three latent bugs found while testing this work
- `backend/api/main.py` called `json.loads` with **no `json` import** — a
  runtime 500 waiting for the first caller.
- `backend/services/perception/capabilities.py` discarded a verifier's
  explanation when the module was missing, leaving operators a bare
  `ModuleNotFoundError` instead of naming the half of the feature that still
  works. OCR now reports: *"pytesseract unusable: ModuleNotFoundError. Text
  regions are still detected by the text_regions capability."*
- The SPA test reloaded `backend.api.main` with a built UI present and left it
  in `sys.modules`, leaking a different route table into every subsequent test.
  The README route-count test passed in isolation and failed in a full run —
  a trap for any future route addition.

Mutation-verified: 14 mutations of the new guarantees, 14 caught. Two initially
survived — a webhook that silently claims success, and a deleted MQTT liveness
probe — because one test skipped itself when the channel looked available and
one asserted on source text that the `import` line alone satisfied. Both tests
were rewritten to drive real behaviour.

## Analysis coverage — closing the gap between what Argus computes and what it reports

### Perception observations now reach operators
- `backend/services/management/observation_events.py`: promotes actionable
  perception observations (dwell, pacing, abandonment, occupancy anomaly,
  disappearance, scene change) into the `events` table with their evidence,
  confidence and provenance. Previously these were computed, logged and stored
  in their own table but never surfaced in the event feed — Argus was analysing
  far more than it reported. Verified live: 92 events across 4 kinds from a
  75-second run that previously produced zero.
- Only grounded, actionable observations are promoted; bookkeeping kinds
  (`object_appeared`, `object_left_frame`, `track_summary`) are suppressed by
  declared policy and remain queryable via `/memory/recall`.
- Per-(camera, kind, subject) cooldown stops a persisting condition flooding the
  feed. The dedup table is bounded.

### Three configured rules were never implemented
`config.yaml` declared `speed_violation`, `fall_detection` and
`abandoned_object` as `enabled: true` while the engine implemented only
intrusion and loitering. All three are now implemented, plus `line_crossing`.

- **`speed_violation`** requires per-camera ground-plane calibration
  (`backend/services/management/calibration.py`). An uncalibrated camera reports
  its blocker rather than converting pixels to km/h with a global guess.
- **`fall_detection`** requires real pose keypoints; the bbox aspect-ratio
  fallback cannot distinguish a fall from crouching and is refused.
- **`abandoned_object`** derives owner-absence from detection geometry.
- **`line_crossing`** wires up `zone_alerts.py`, which nothing had ever
  imported. `zone_manager.is_point_in_zone()` has no branch for type `line`, so
  tripwire zones were accepted by the API and silently never evaluated.
- `GET /api/v1/rules/status` reports configured vs implemented vs *can actually
  fire here*, with the specific blocker. Blockers are probed, not accumulated
  from runtime side effects.

### Defects found and fixed while wiring the above
- **Speed was structurally always 0.0.** The coordinator called
  `get_next_object_id()` per detection per frame, so the analyzer's position
  history never accumulated, and `tracks` leaked one dead single-point entry per
  detection per frame (~100k/hour on one busy camera). Now keyed on the
  tracker's persistent id.
- **`cleanup_old_tracks()` compared frame timestamps to `time.time()`**,
  evicting every track on the frame it was created when replaying footage.
- **`RuleConfig` silently discarded unknown keys.** Per-rule tuning
  (`classes`, `move_tolerance_px`, `violation_margin`) was parsed, dropped, and
  replaced by hard-coded defaults with no warning. Now `extra="allow"`.
- **`owner_gone` was never computed**, so `detect_abandonment()` was dead code
  everywhere in the system.
- **`zone_alerts.load_zones()` dropped the zone `id`**, so every tripwire event
  would have reported `zone_id: 0` and been unattributable.
- **`adaptive_learning.py` was never called by anything**, and
  `/stats/learning` relabelled cross-camera tracker counts as "behaviour
  profiles" — reporting a subsystem that had learned nothing. Now fed from the
  coordinator and reporting its own counters.
- **A snapshot-cap test read the shared snapshot directory**, so it passed on CI
  and failed on any machine that had actually run Argus.
- **`/api/v1/events/lifecycle` was shadowed by `/events/{event_id}`** and
  returned 422; literal routes must be registered before parameterised ones.

### Event lifecycle
- `detected → open → acknowledged → resolved`, plus `false_positive` from any
  live state, enforced by `EventStore.ALLOWED_TRANSITIONS`. `track_id`,
  `acknowledged_by/at` and `resolved_by/at` added by an idempotent migration.
- `PATCH /api/v1/events/{id}/status` (operator role) records the acting user;
  illegal transitions return 400.

### Frontend
- New **Memory Explorer** page: search observations with their evidence, and a
  panel naming any rule that cannot fire here.
- Event Feed gained Acknowledge / Resolve / False-positive actions.

### Documentation
- Fixed 9 stale module paths across 3 docs left behind by the restructure.
- Conditional artefacts (`yolov8m.pt`, `data/qdrant/`, `data/kafka/`,
  `data/streams/`) are now labelled as such instead of implying they exist.
- New test asserts every path named in any `.md` resolves.

### Tests
- 236 passing (was 208). 14 new mutation-verified guarantees, each confirmed to
  fail when the behaviour it protects is broken.


## [Unreleased]

### Security and operations
- Default local and development bindings to loopback; require explicit configuration to expose services on a network.
- Default face recognition to opt-in, stop persisting new source face crops, and encrypt legacy plaintext embeddings when recognition is enabled.
- Harden the admin setup command against shell/code injection and require stronger passwords.
- Make dependency audit failures visible in CI and verify the Docker image runs as a non-root user.
- Correct the security policy and roadmap to avoid unsupported response-time promises and stale status claims.

### Added
- CI check that fails when a package is declared in `requirements.txt` but
  imported nowhere (with an explicit allowlist for framework plugins and
  runtime backends that are legitimately never imported by our source).
- Six regression tests covering zone-alert payload shapes and per-track memory
  bounds (mutation-verified: each fails when its bug is reintroduced).
- **Continuous integration** (`.github/workflows/ci.yml`). Nothing ran the test
  suite before. Four jobs: lint/static checks, secret and gitignore hygiene,
  the full suite on Python 3.11 and 3.13 plus a real server boot, and a
  frontend build.
- `pytest.ini` — restricts collection to `test_*.py`, registers the `slow` and
  `integration` markers, and enables `asyncio_mode = auto`. A bare `pytest` now
  collects exactly the 44 maintained tests instead of also picking up the
  legacy diagnostic scripts (one of which never terminates).
- `.dockerignore` — the build context previously included `.git`,
  `node_modules/`, `data/`, and the model weights.
- `CONTRIBUTING.md` and this changelog.
- The **Adaptive Learning dashboard** is now reachable at `/learning` with a
  sidebar entry. The 19 KB page existed and its endpoints
  (`/stats/learning`, `/cross-camera/*`) were implemented, but it was never
  imported or routed, so it was dead code.
- `docker-compose.yml` now passes `ARGUS_JWT_SECRET`, `ARGUS_CORS_ORIGINS`,
  `ARGUS_LOG_FORMAT`, and `ARGUS_LOG_LEVEL` through to the backend, and refuses
  to start without a signing key.
- `requests` added to `requirements.txt`.
- `run_admin.py --setup-only` — creates the Argus and Django auth tables and
  exits, instead of only doing so as a side effect of starting a blocking
  server. CI uses it, and then fails if more than two security tests skip: on
  an unseeded database 21 of the 22 would skip and the run would go green
  having verified nothing.
- `backend/models/.gitkeep` — git omits empty directories, so `backend/models/`
  did not exist in a fresh clone and the documented weights copy failed with
  "Not a directory".

### Added
- **A hard disk ceiling for snapshots** (`retention.snapshots_max_mb`, default
  2048). Time-based expiry cannot bound disk usage inside its own window: at the
  measured event rate one camera writes ~4 GB of snapshots per day, so a 30-day
  policy only reclaims space after ~130 GB (~2.7 TB across 20 cameras). Every
  retention pass now evicts oldest-first until the directory fits.

### Changed
- Six modules that nothing imports (`yolo_tracker.py`, `object_detection_tracker.py`,
  `object_detection_tracker_refactored.py`, `video_pipeline.py`,
  `multistream_pipeline.py`, `model_optimizer.py`) are now labelled **DORMANT**
  in `FOLDER_STRUCTURE.md` rather than described as live components. In
  particular `yolo_tracker.py` was documented as the "fallback if deep_tracker
  is disabled" - no such fallback exists in code. The modules are kept, and a
  test now fails if one is wired up (or orphaned) without the docs following.

### Removed
- Nine declared-but-unimported dependencies: `scikit-image` (35 MB on disk),
  `statsmodels`, `torchmetrics`, `deap`, `pydantic-settings`, `httpx`,
  `python-dateutil`, `black`, `pylint`. 32 declared packages down to 23.
  Verified by blocking each module at import time: the app and all 49 tests
  still pass, and a live server serves authenticated traffic including the
  multipart upload path.

### Security
- **33 orphaned role grants, eleven of them `admin`, were sitting in
  `auth_user_groups`.** SQLite enables foreign keys per connection and defaults
  to OFF, so deleting a user left its group membership behind pointing at a dead
  id. Demonstrated the consequence: a freshly created non-superuser that reuses
  such an id resolves to `admin`. `AUTOINCREMENT` makes reuse unlikely today,
  which is why this had gone unnoticed - not a reason to leave it.
  - The security-test fixture was the source, leaking three rows per run: it
    deleted `auth_user` but never `auth_user_groups`. Fixed in both setup and
    teardown, and verified flat across three consecutive runs.
  - `backend/api/auth.py` now enables `PRAGMA foreign_keys = ON`, which makes
    SQLite reject orphan inserts outright (verified), and role lookup re-joins
    `auth_user` with `is_active = 1`.
  - Existing orphans purged; `PRAGMA foreign_key_check` reports 0 violations.
  - Two new tests fail if any orphaned or duplicate grant reappears.

### Fixed
- **README and FOLDER_STRUCTURE claimed a "DEAP-based genetic algorithm".**
  `evolutionary_engine.py` never imported DEAP - it implements its own GA with
  elitism, crossover and Gaussian mutation on top of `random`.
- **Zone checking crashed on API-shaped detections.** `_get_center_from_dict_or_obj`
  indexed `bbox` as a list, but every serialised detection (WebSocket, REST) uses
  `{"x1":..,"y1":..,"x2":..,"y2":..}`, so feeding one back raised `KeyError: 0`
  and aborted the zone check rather than degrading. Bboxes are now coerced from
  either shape, and the class is read from `class_name` *or* `class`.
- **Unbounded memory growth in `ZoneAlerts`.** `zone_triggers` was keyed by
  `track_id` and never pruned, so a 24/7 feed leaked an entry per track for the
  process lifetime (50,000 tracks measured as 50,000 retained entries). Now an
  LRU-bounded 4096 entries, with `loitering_triggers` pruned in step.
- Bare `except:` in the frame-drop path of `stream_ingestion.py` also swallowed
  `KeyboardInterrupt`/`SystemExit`, interfering with clean shutdown. Narrowed to
  `(queue.Empty, queue.Full)`.
- Removed three byte-identical 355 KB copies of the logo PNG (none referenced by
  any code) and wired the existing 1.3 KB SVG up as the previously-missing
  favicon.
- **The Docker healthcheck could never pass.** It ran
  `python -c "import requests; ..."` while `requests` was not a declared
  dependency, so the backend container was reported `unhealthy` even when the
  API was serving correctly, and any `depends_on: condition: service_healthy`
  would block forever. Rewritten to use `urllib` from the standard library, and
  it now checks the HTTP status code instead of discarding the response.
- **The containerised backend signed tokens with an ephemeral key.** Compose set
  no `ARGUS_*` variables, so every restart invalidated every issued token and
  logged the ephemeral-secret warning.
- **`django` was declared in both requirements files** with conflicting ranges
  (`>=5.0,<6.0` core versus `>=4.2,<6.0` optional), silently downgrading Django
  depending on install order. It is now declared once, in the core file. CI
  fails on any future overlap.

## Historical notes for version 2.1.0 (no published tag or GitHub release)

### Added
- JWT authentication and role-based access control (admin / operator / viewer)
  over the existing Django `auth_user` table, applied to 40 of 43 API routes.
- Audit logging of all mutating requests, with retention policies.
- Prometheus metrics at `GET /metrics` and optional JSON structured logging.
- Login page and token refresh handling in the frontend.
- `PRODUCTION_ROADMAP.md` covering the 30 outstanding production-readiness items.
- `data/demo_clip.mp4` fixture plus `backend/scripts/make_demo_clip.py`, so the
  pipeline is testable without a camera.

### Fixed
- **Track identity churn at realistic frame rates.** IoU-only association fails
  near 1 FPS, where successive boxes do not overlap: 89 identities were created
  for 16 people. Added a centre-distance matching stage gated on class and size
  ratio — 89 → 16 IDs.
- **Event storms** from unstable track IDs: 209 → 108 events per two minutes.
- **Primary detector starvation** under swarm contention, via
  `_effective_budget_ms()` and a throttle floor for the primary agent.
- WebSocket route conflicts, and timestamps normalised to UTC ISO-8601 with `Z`.
- CORS misconfiguration (`allow_origins=["*"]` together with credentials).
- Static `/snapshots` mount bypassing route authentication.
- Repository size reduced from 117 MB to 13 MB; runtime data removed from git.

[Unreleased]: https://github.com/Mr-Nobody-Anonymous/Argus/commits/main/
