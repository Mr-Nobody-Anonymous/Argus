#!/bin/sh
# ─────────────────────────────────────────────────────────────────────────────
# Argus container entrypoint.
#
# Bridges the gap between "a checked-out repo a developer runs by hand" and
# "an immutable image a platform starts with no human present". It performs the
# three things a first boot needs and a second boot must not repeat:
#
#   1. Bind to the port the platform assigns.
#   2. Create the schema if the data volume is empty.
#   3. Create the initial admin account, without ever baking a default password
#      into the image.
#
# Everything here is idempotent: restarts and rolling deploys re-run it.
# ─────────────────────────────────────────────────────────────────────────────
set -eu

# ── 1. Port ──────────────────────────────────────────────────────────────────
# Render, Railway, Fly, Heroku, Cloud Run and App Runner all inject $PORT and
# health-check that exact port. Hardcoding 8000 makes the deploy hang until the
# platform times out, which presents as "deploy failed" with a healthy app in
# the logs - one of the most confusing failures in the whole list.
PORT="${PORT:-8000}"
HOST="${ARGUS_HOST:-0.0.0.0}"

# ── 2. Writable state ────────────────────────────────────────────────────────
# ARGUS_DATA_DIR is interpolated into config.yaml (snapshot dir + sqlite path).
# On a container filesystem this must point at a mounted disk or every restart
# silently discards users, cameras and recorded events.
ARGUS_DATA_DIR="${ARGUS_DATA_DIR:-/app/data}"
export ARGUS_DATA_DIR
mkdir -p "$ARGUS_DATA_DIR/snapshots" "$ARGUS_DATA_DIR/clips"

if [ ! -w "$ARGUS_DATA_DIR" ]; then
    echo "FATAL: ARGUS_DATA_DIR ($ARGUS_DATA_DIR) is not writable." >&2
    echo "       Mount a volume there, or set ARGUS_DATA_DIR to a writable path." >&2
    exit 1
fi

# Warn loudly when state is ephemeral. This is not fatal - a kick-the-tyres
# deploy on a free tier is a legitimate use - but silently losing the database
# on every redeploy is not something anyone should discover in production.
case "$ARGUS_DATA_DIR" in
    /app/data|/tmp/*)
        if [ ! -f "$ARGUS_DATA_DIR/.persistence-checked" ]; then
            echo "WARNING: $ARGUS_DATA_DIR looks like container-local storage."
            echo "         Attach a persistent volume and point ARGUS_DATA_DIR at it,"
            echo "         or the database is lost on every restart."
        fi
        ;;
esac

# Seed the demo clip on first boot only. It is baked at /app/seed because
# $ARGUS_DATA_DIR is a mounted volume at runtime: anything written there during
# the build is hidden the moment the volume is mounted over it. Copying only
# when absent means a user who deletes the demo camera does not get it back on
# every restart.
if [ -f /app/seed/demo_clip.mp4 ] && [ ! -f "$ARGUS_DATA_DIR/demo_clip.mp4" ]; then
    cp /app/seed/demo_clip.mp4 "$ARGUS_DATA_DIR/demo_clip.mp4" || true
fi

# ── 3. Secrets ───────────────────────────────────────────────────────────────
# Without a stable signing key the app generates an ephemeral one, so every
# restart invalidates all sessions. Refuse rather than degrade: the failure is
# otherwise intermittent and blamed on the browser.
if [ -z "${ARGUS_JWT_SECRET:-}" ] && [ "${ARGUS_DISABLE_AUTH:-0}" != "1" ]; then
    echo "FATAL: ARGUS_JWT_SECRET is not set (needs >=32 chars)." >&2
    echo "       Generate one with:  openssl rand -base64 48" >&2
    echo "       Set it in your platform's environment/secrets settings." >&2
    exit 1
fi

# ── 4. Schema + admin bootstrap ──────────────────────────────────────────────
python backend/scripts/init_db.py
python backend/scripts/run_admin.py --setup-only

# ARGUS_ADMIN_PASSWORD sets the first admin password from the platform's secret
# store. It is applied only when the account still has its default password, so
# a later manual change is never silently reverted on redeploy.
if [ -n "${ARGUS_ADMIN_PASSWORD:-}" ]; then
    python - <<'PYBOOTSTRAP' || echo "WARNING: admin password bootstrap failed; the default remains in effect."
import os
import django

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "backend.django_admin.settings")
django.setup()
from django.contrib.auth.models import User

username = os.environ.get("ARGUS_ADMIN_USERNAME", "admin")
password = os.environ["ARGUS_ADMIN_PASSWORD"]
user = User.objects.filter(username=username).first()
if user is None:
    User.objects.create_superuser(username=username, email="", password=password)
    print(f"Created admin user '{username}' from ARGUS_ADMIN_PASSWORD.")
elif user.check_password("admin123"):
    user.set_password(password)
    user.save(update_fields=["password"])
    print(f"Replaced the default password for '{username}' from ARGUS_ADMIN_PASSWORD.")
else:
    print(f"Admin '{username}' already has a non-default password; left unchanged.")
PYBOOTSTRAP
fi

touch "$ARGUS_DATA_DIR/.persistence-checked" 2>/dev/null || true

# ── 5. Serve ─────────────────────────────────────────────────────────────────
# Single worker, deliberately, and not a value worth "tuning up".
#
# This process is not stateless. It holds the processing coordinator, the
# tracker state, the pre-event frame buffers and a retention thread in memory,
# and it writes to SQLite. A second worker would get its own coordinator and
# its own retention scheduler competing over the same files, and SQLite would
# start returning "database is locked" under concurrent writes. The symptom is
# events that vanish depending on which worker served the request - which reads
# as data loss, not as a config mistake.
#
# Scale by running one Argus per camera group behind a load balancer, or move
# to Postgres and externalise the coordinator first.
if [ "${ARGUS_WORKERS:-1}" != "1" ]; then
    echo "WARNING: ARGUS_WORKERS=${ARGUS_WORKERS} ignored - Argus holds camera state"
    echo "         and a retention thread in-process and writes to SQLite."
    echo "         Multiple workers duplicate both and corrupt event ordering."
fi

# exec so uvicorn becomes PID 1 and receives SIGTERM directly. Without it the
# shell holds PID 1, never forwards the signal, and every deploy waits out the
# platform's kill timeout before dying uncleanly.
#
# --proxy-headers + --forwarded-allow-ips make the app trust X-Forwarded-Proto
# from the platform's TLS terminator. Without it every generated URL is http://
# on an https:// site and the browser blocks it as mixed content.
echo "Starting Argus on ${HOST}:${PORT} (data: ${ARGUS_DATA_DIR})"
exec python -m uvicorn backend.api.main:app \
    --host "$HOST" \
    --port "$PORT" \
    --proxy-headers \
    --forwarded-allow-ips "${ARGUS_FORWARDED_ALLOW_IPS:-*}"
