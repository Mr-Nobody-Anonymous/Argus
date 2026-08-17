# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
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

### Fixed
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
