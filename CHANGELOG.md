# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Removed
- Nine declared-but-unimported dependencies: `scikit-image` (35 MB on disk),
  `statsmodels`, `torchmetrics`, `deap`, `pydantic-settings`, `httpx`,
  `python-dateutil`, `black`, `pylint`. 32 declared packages down to 23.
  Verified by blocking each module at import time: the app and all 49 tests
  still pass, and a live server serves authenticated traffic including the
  multipart upload path.

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

## [2.1.0]

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

[Unreleased]: https://github.com/Mr-Nobody-Anonymous/Argus/compare/v2.1.0...HEAD
[2.1.0]: https://github.com/Mr-Nobody-Anonymous/Argus/releases/tag/v2.1.0
