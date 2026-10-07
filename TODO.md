# Argus — Work Log & Outstanding Work

Two sections: what has been **completed** (kept as a record, oldest first) and
what is still **outstanding**. The full production gap analysis with acceptance
criteria and effort estimates lives in [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md).

---

# ✅ Completed

## Phase 1 — Project restructuring

## ✅ Step 1: Create destination directories at root level
## ✅ Step 2: Move files from `core/` to root (using PowerShell)
## ✅ Step 3: Fix `backend/services/__init__.py` imports
## ✅ Step 4: Fix `backend/api/main.py` imports
## ✅ Step 5: Fix `backend/api/stream_routes.py` imports
## ✅ Step 6: Fix `backend/api/stream_ws.py` imports
## ✅ Step 7: Fix `backend/scripts/*.py` imports
## ✅ Step 8: Fix `backend/config/config.py` path resolution
## ✅ Step 9: Fix `backend/database/db.py` path resolution
## ✅ Step 10: Fix Dockerfiles (paths)
## ✅ Step 11: Fix docker-compose files (volume mounts, context)
## ✅ Step 12: Fix run_app.bat and run_webcam.bat
## ✅ Step 13: Fix Django admin settings.py paths
## ✅ Step 14: Fix internal service imports (processing_coordinator, rules_engine, stream_ingestion, video_pipeline)
## ✅ Step 15: Update README.md structure diagram
## ✅ Step 16: Remove empty `core/` directory
## ✅ Step 17: Clean up and verify - remove core/ if still exists, verify all files

---

## Phase 2 — Import fixes

### Fixed Broken Relative Imports
| # | File | Original Import | Fixed Import |
|---|------|----------------|--------------|
| ✅ | `backend/services/management/telemetry_monitor.py` | `from config.config import get_config` | `from ...config.config import get_config` (with fallback) |
| ✅ | `backend/services/management/state_recovery_manager.py` | `from config.config import get_config` | `from ...config.config import get_config` |
| ✅ | `backend/services/management/user_attention_tracker.py` | `from config.config import get_config` | `from ...config.config import get_config` |
| ✅ | `backend/services/core_engine/evolutionary_engine.py` | `from config.config import get_config` + `from management.event_store import get_event_store` | `from ...config.config import get_config` + `from ..management.event_store import get_event_store` |
| ✅ | `backend/services/core_engine/logic_mutator.py` | `from config.config import get_config` | `from ...config.config import get_config` |

All service modules now use absolute `backend.*` imports. Six missing
`__init__.py` files were added so every package is importable.

---

## Phase 3 — Making the pipeline actually work

The system started, reported `healthy`, and produced **zero detections**.
Thirteen defects were found; each produced *plausible but wrong* output rather
than an error, which is why none of them surfaced as a crash.

| # | Defect | Fix |
|---|---|---|
| ✅ 1 | Kalman `statePre`/`statePost` shaped `(8,)` instead of `(8,1)` | `predict()` threw inside `gemm`; tracking was silently dead |
| ✅ 2 | Kalman transition matrix coupled position to size (`cx' = cx + w + vw`) | Hand-recomputed; a stationary box no longer "jumps" |
| ✅ 3 | Greedy IoU matcher used a row-major scan (last write wins) | Sort candidate pairs by descending IoU, consume each once |
| ✅ 4 | `errorCovPost` left zeroed by OpenCV | Explicitly initialised |
| ✅ 5 | Broker's fixed 33 ms budget (a 30 FPS GPU assumption) | Allocate against measured demand |
| ✅ 6 | Skipped frames returned `[]` — "not measured" read as "measured as zero" | Return the previous result, marked stale |
| ✅ 7 | One-way confidence ratchet with no decay path | Detector no longer goes permanently blind after transient load |
| ✅ 8 | EOF on a video file treated as stream failure | Files now loop |
| ✅ 9 | Unthrottled file decode (~570 fps) saturated the queue | Paced to the file's native FPS |
| ✅ 10 | Per-frame DB status write | Throttled to 1/s |
| ✅ 11 | Primary agent starved into replaying a stale result forever | `PRIMARY_MIN_THROTTLE` floor; the detector is never throttled |
| ✅ 12 | IoU-only track association failed at the real ~1 fps processing rate | Added a size-scaled, class-gated centre-distance stage: **89 → 16 IDs** |
| ✅ 13 | Loitering keyed on a 50px grid cell; dedup window did not slide | Both key on persistent track IDs; window slides: **209 → 108 events / 2 min** |

Also fixed: an `RLock` deadlock in `cross_camera_tracker.py` and
`rules_engine.py`; a pydantic-vs-dict config bug (`config.rules` holds models,
so `.get()` raised); blocking calls in async handlers that throttled the
WebSocket to ~4 msg/s.

---

## Phase 4 — Security & operations

| Area | Status |
|---|---|
| ✅ JWT auth (HS256) reusing the Django `auth_user` table | No second user store to drift out of sync |
| ✅ Three-role RBAC (viewer/operator/admin) on all 40 protected routes | Roles re-read from the DB per request, so a forged claim is inert |
| ✅ Brute-force lockout | 5 attempts → 300 s; correct password also refused while locked |
| ✅ WebSocket auth before `accept()` | Unauthenticated upgrades get 403 at the handshake |
| ✅ Audit trail | Every mutation + auth attempt, with `denied` outcomes; writes never raise |
| ✅ Data retention scheduler | Per-type schedules, surfaced in `/api/v1/health` |
| ✅ `${VAR}` config interpolation + startup validation | Refuses to boot on unresolved secret placeholders |
| ✅ CORS allowlist | Replaced the browser-rejected `*` + credentials combination |
| ✅ Prometheus `/metrics` + JSON logging | Found two live bugs within a minute of going live |
| ✅ Frontend login, token refresh, WS `?token=` | Dashboard works against the authenticated API |
| ✅ Automated pytest suites | Regression, API security, CityOS, and sensor suites are collected by CI; check the latest Actions run for pass status.
| ✅ Swarm A/B sample | One 40-frame run showed +26.8% FPS and equal detection counts; it did not measure ground-truth accuracy.

---

# ⬜ Outstanding

Ordered by priority. See [PRODUCTION_ROADMAP.md](PRODUCTION_ROADMAP.md) for
acceptance criteria.

## P0 — Blocks any real deployment
- [ ] **TLS termination** (§2.4) — tokens currently travel in cleartext unless
      you front the service with a reverse proxy.
- [x] **Encrypt face embeddings at rest** (§2.3) — new embeddings are encrypted and recognition is off by default. Legacy plaintext crop files still require operator review and cleanup; production keys should be stored separately from the database.

## P1 — Operational maturity
- [ ] Camera state machine with explicit lifecycle transitions (§3.1)
- [ ] GPU/inference scheduler with memory accounting (§3.2)
- [ ] OpenTelemetry tracing to complement the metrics (§3.3)
- [ ] Model versioning recorded on every event (§3.4)
- [ ] Event lifecycle `DETECTED → OPEN → ACKNOWLEDGED → RESOLVED` (§3.5)
- [ ] Video ring buffer for pre/post-event evidence clips (§3.6)
- [ ] PostgreSQL backend — SQLite is single-writer (§3.7)
- [ ] Alert policy engine: routing, escalation, suppression windows (§3.8)
- [ ] Re-ID confidence fusion — replace the bare `similarity > 0.7` at
      `person_reid.py:43/177` (§3.9)
- [ ] Homography camera calibration — `speed_height_analysis.py:39` uses a
      frame-global `calibration_factor=0.05` m/px, which is wrong at any depth (§3.10)
- [ ] Face anti-spoofing / liveness (§3.11)

## P2 — Scale & developer experience
- [x] **CI/CD workflows configured** — push and pull-request workflows cover backend checks, frontend build, security scans, and Docker verification. Confirm the first pushed runs pass in GitHub Actions.
- [ ] Labelled evaluation clips + a written fitness function (§5) — the
      evolutionary engine currently optimises against no ground truth, so it can
      converge on something meaningless.
- [ ] Config-validation CLI
- [ ] `VisionAgent` plugin SDK
- [ ] Formal event taxonomy and a single time model (the WS stream, REST API,
      and audit log historically used three different timestamp formats; the WS
      stream has been normalised to ISO 8601 UTC, the others still differ)
- [ ] SOC operator workflows in the frontend (acknowledge, assign, resolve)
- [ ] Kubernetes manifests, backup/recovery runbook, supply-chain pinning
- [ ] Deduplicate device selection, currently repeated in `person_reid.py:70`,
      `object_detection_tracker.py:42`, `yolo_tracker.py:45`,
      `object_detection_tracker_refactored.py:80`

## Known test debt
- [ ] Seven older scripts in `tests/` (`ai_pipeline_test.py`,
      `buffer_monitor.py`, `infrastructure_healthcheck.py`, `run_all_tests.py`,
      `stream_simulator.py`, `websocket_stress_tester.py`, `resource_monitor.sh`)
      predate authentication and send unauthenticated requests, so they now fail
      against a secured API. They are diagnostic tools rather than assertions;
      they need updating or retiring.
