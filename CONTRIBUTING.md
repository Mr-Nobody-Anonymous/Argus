# Contributing to Argus

## Setting up

```bash
git clone <your-fork>
cd Argus

python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Generate a signing key and paste it into ARGUS_JWT_SECRET:
openssl rand -base64 48
```

A fresh clone deliberately does **not** contain the model weights, the database,
or snapshots. See [What a fresh clone does *not* include](README.md#5-what-a-fresh-clone-does-not-include)
in the README for how to obtain each.

```bash
python backend/scripts/init_db.py                   # Argus tables
python backend/scripts/run_admin.py --setup-only   # Django auth tables + admin/admin123
uvicorn backend.api.main:app --reload

cd frontend && npm install && npm run dev
```

## Running the tests

```bash
pytest                    # 74 tests, ~14s
pytest -m "not slow"      # skip the slower pipeline tests
```

`pytest.ini` restricts collection to `test_*.py`. The other files in `tests/`
are hand-run diagnostic scripts, not automated tests — some of them open
sockets and `buffer_monitor.py` never terminates on its own. Do not add them to
the default run.

If a test needs the database it should **skip** when the DB is absent rather
than fail: CI and fresh clones have no `data/argus.db`.

Run `run_admin.py --setup-only` before the security suite. Without Django's
`auth_user` table, 21 of its 22 tests skip themselves and the run goes green
having verified essentially nothing — so CI seeds the schema first and then
**fails if more than two security tests skip**.

## What CI enforces

Every push and pull request runs [`.github/workflows/ci.yml`](.github/workflows/ci.yml):

| Job | Checks |
| --- | --- |
| `lint` | Byte-compiles all modules, rejects a package declared in both requirements files, validates every YAML file |
| `security` | Fails on plaintext credentials, asserts `data/` and `*.pt` stay gitignored while `data/demo_clip.mp4` stays tracked |
| `test` | Runs both suites on Python 3.11 and 3.13, then boots the real server and asserts `/health` works, an unauthenticated request gets `401`, and `/metrics` emits `argus_` lines |
| `frontend` | `npm ci && npm run build`, and asserts `dist/index.html` exists |

## House rules

These come from bugs that actually shipped in this repo.

**Never commit runtime data.** `data/` holds the database, snapshots, and
enrolled face images. The one tracked exception is `data/demo_clip.mp4`, the
test fixture. Note the two `.gitignore` traps involved: git cannot re-include a
file inside an excluded *directory* (so the rule is `data/*`, not `data/`), and
a negation only wins if no later pattern re-excludes it (so `!data/demo_clip.mp4`
sits at the end of the file, after `*.mp4`). Verify with
`git check-ignore -v <path>`.

**Declare every import you use.** The Docker healthcheck called `requests` for
months while it was not in `requirements.txt`, so the backend container was
permanently `unhealthy`. Prefer the standard library for infrastructure glue.

**Declare a package in exactly one requirements file.** `django` was once in
both with different ranges, silently downgrading whichever installed last. CI
now fails on any overlap.

**Config takes no secrets.** `config/config.yaml` supports `${VAR}` and
`${VAR:-default}` interpolation. Put credentials in `.env`.

**Verify with a real clone, not by reading the tree.** An untracked
`Login.jsx` that every local build used was missing for anyone else:

```bash
git clone . /tmp/clonetest && cd /tmp/clonetest
cp <path>/backend/models/yolov8n.pt backend/models/
pytest tests/test_regression.py
```

**Frames are not optional.** A skipped frame must never be reported as an empty
detection list or a silent replay of the previous result — both look like a
working detector that sees nothing.

**Association at low frame rates is not IoU.** Live cameras run near 1 FPS,
where boxes for the same person do not overlap between frames. Test tracking at
realistic strides; consecutive-frame footage hides the bug (15 IDs in test, 89
in production).

## Pull requests

- One logical change per PR; explain *why*, not just what.
- Add a regression test for any bug fix.
- Update the README when you change routes, env vars, or the project layout.
- Add a `CHANGELOG.md` entry under `[Unreleased]`.
- Keep all four CI jobs green.
