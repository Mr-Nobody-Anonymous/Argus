# Deploying Argus

Argus runs anywhere that can run a container. This guide covers the common
targets and, just as importantly, the one popular target that **cannot** host
it and why.

---

## Read this first: what kind of application Argus is

Most deployment guides can skip this. This one cannot, because Argus breaks the
assumptions behind serverless and autoscaling platforms:

| Property | Consequence for hosting |
|---|---|
| Holds camera connections, tracker state and pre-event frame buffers **in memory** | Cannot be scaled horizontally without losing state; cannot be suspended to zero |
| Runs a **retention thread** inside the process | A second replica runs a second thread over the same files |
| Serves **WebSocket** video streams | Needs a long-lived process, not request/response invocations |
| Writes to **SQLite** on local disk | Needs a persistent volume; concurrent writers corrupt ordering |
| Depends on **torch + OpenCV** (~955 MB installed) | Exceeds serverless bundle limits; needs ≥2 GB RAM |

In short: Argus needs **one always-on container with a disk**. Every
recommendation below follows from that.

---

## Quick reference

| Target | Hosts | Command |
|---|---|---|
| [Local machine](#0-just-run-it-locally) | Everything | `python argus.py start` / `stop` |
| [Docker (any machine)](#1-any-machine-with-docker) | Everything | `docker compose -f docker-compose.prod.yml up -d` |
| [Render](#2-render) | Everything | Blueprint from `render.yaml` |
| [Railway](#3-railway) | Everything | Detects `railway.json` |
| [Fly.io](#4-flyio) | Everything | `fly deploy` |
| [Prebuilt image](#5-prebuilt-image-from-ghcr) | Everything | `docker run ghcr.io/OWNER/argus` |
| [Vercel](#6-vercel-dashboard-only) | **Dashboard only** | Needs an API elsewhere |
| [VPS without Docker](#7-vps-without-docker-systemd) | Everything | systemd unit |

---

## Configuration

Every target uses the same environment variables.

| Variable | Required | Purpose |
|---|---|---|
| `ARGUS_JWT_SECRET` | **Yes** | Token signing key, ≥32 chars. Generate: `openssl rand -base64 48`. Without a stable value every restart signs all operators out. The container refuses to start without it. |
| `ARGUS_ADMIN_PASSWORD` | Strongly advised | Sets the initial admin password. If omitted, a random secure password is generated on first boot and printed once to the console. |
| `ARGUS_BIOMETRIC_KEY` | Strongly advised | 32-byte key for AES-256-GCM encryption of biometric face embeddings at rest. If omitted, a key is auto-generated in `data/.biometric.key`. |
| `ARGUS_DATA_DIR` | Yes, on any container | Where the database, snapshots and clips are written. **Must point at a mounted volume** or all of it is lost on restart. |
| `PORT` | Auto | Injected by the platform; the entrypoint binds it. |
| `ARGUS_CORS_ORIGINS` | Only when split | Comma-separated browser origins. Leave empty when the API serves the dashboard. |
| `ARGUS_LOG_FORMAT` | No | `json` (default in the image) or `text`. |

> **The single most common deployment mistake** is mounting a volume at a path
> that is not `ARGUS_DATA_DIR`. The app then writes to the container filesystem
> while an empty disk sits beside it, and everything looks fine until the first
> restart. The provided `render.yaml` and `fly.toml` already match.

---

## 0. Just run it locally

Not deploying to a server, only running Argus on a machine in front of you?
Skip this guide:

```bash
python argus.py start     # sets up everything and runs it
python argus.py stop      # stops everything
```

or `make start` / `make stop`, or double-click `start.command` / `start.bat`.
The launcher creates the virtualenv, installs dependencies, seeds the database,
builds the dashboard and waits for health — and picks Docker automatically if
the daemon is running. The rest of this guide is for hosting Argus somewhere
nobody is logged in.

---

## 1. Any machine with Docker

The baseline path: a VPS, a homelab box, an on-prem server, an EC2 instance.

```bash
git clone https://github.com/OWNER/Argus.git && cd Argus

cp .env.example .env
echo "ARGUS_JWT_SECRET=$(openssl rand -base64 48)" >> .env
echo "ARGUS_ADMIN_PASSWORD=choose-a-strong-one"    >> .env

docker compose -f docker-compose.prod.yml up -d
```

Open <http://localhost:8000> and sign in as `admin`.

By default the port is published on `127.0.0.1` only. To expose it on the
network, set `ARGUS_BIND=0.0.0.0` in `.env` — but put TLS in front of it first
(see [Putting it behind HTTPS](#putting-it-behind-https)). Argus streams
identifiable footage; plain HTTP on a LAN is a poor idea and on the open
internet an unacceptable one.

Useful commands:

```bash
docker compose -f docker-compose.prod.yml logs -f      # follow logs
docker compose -f docker-compose.prod.yml pull && \
docker compose -f docker-compose.prod.yml up -d        # upgrade
docker run --rm -v argus_argus-data:/d -v "$PWD":/b alpine \
  tar czf /b/argus-backup.tgz -C /d .                  # back up the volume
```

---

## 2. Render

1. Push the repo to GitHub.
2. Render → **New** → **Blueprint** → select the repo. `render.yaml` is detected.
3. Set `ARGUS_ADMIN_PASSWORD` when prompted. `ARGUS_JWT_SECRET` is generated and
   kept stable across deploys.
4. Deploy. First build takes ~10 minutes (torch + the UI build).

The blueprint requests the **standard** plan (2 GB). The 512 MB starter plan
cannot load torch and YOLOv8 — the container is OOM-killed during startup, which
appears in the log as an abrupt exit with no traceback.

A 10 GB disk is mounted at `/var/lib/argus`, matching `ARGUS_DATA_DIR`.

---

## 3. Railway

1. Railway → **New Project** → **Deploy from GitHub repo**. `railway.json`
   selects the Dockerfile automatically.
2. **Variables** → add `ARGUS_JWT_SECRET` and `ARGUS_ADMIN_PASSWORD`.
3. **Settings → Volumes** → add a volume, then set `ARGUS_DATA_DIR` to that
   mount path (Railway does not infer it).

Railway injects `PORT`; the entrypoint honours it.

---

## 4. Fly.io

```bash
fly launch --no-deploy --copy-config          # edit `app` in fly.toml first
fly volumes create argus_data --size 10       # same region as the app
fly secrets set ARGUS_JWT_SECRET="$(openssl rand -base64 48)"
fly secrets set ARGUS_ADMIN_PASSWORD='choose-a-strong-one'
fly deploy
```

`fly.toml` sets `auto_stop_machines = false` deliberately. Suspending to zero
drops every camera connection and discards the rolling pre-event buffer, so an
alert firing shortly after a wake-up would have no footage behind it — the clip
is cut from frames that no longer exist.

---

## 5. Prebuilt image from GHCR

`.github/workflows/deploy.yml` builds and publishes on every push to the default
branch and on `v*` tags, after smoke-testing that the image boots, answers
`/api/v1/health`, **and** serves the dashboard.

```bash
docker run -d -p 8000:8000 \
  -e ARGUS_JWT_SECRET="$(openssl rand -base64 48)" \
  -e ARGUS_ADMIN_PASSWORD='choose-a-strong-one' \
  -v argus-data:/app/data \
  ghcr.io/OWNER/argus:latest
```

Images are `linux/amd64`. On Apple Silicon or ARM servers, build locally
(`docker build -t argus .`) — cross-building torch under QEMU is impractically
slow.

---

## 6. Vercel (dashboard only)

**Vercel cannot host the Argus backend.** This is not a configuration problem;
four independent limits each rule it out:

| Requirement | Vercel |
|---|---|
| ~955 MB of Python dependencies | 500 MB max function bundle |
| WebSocket video streaming | Serverless functions can't hold sockets (the beta pins a socket to one invocation and closes it at the duration limit) |
| Always-on retention thread + in-memory buffers | Functions terminate after responding |
| Persistent SQLite database | Filesystem is ephemeral; only `/tmp`, wiped between invocations |

What *does* work is hosting the **dashboard** on Vercel against an API deployed
elsewhere (Render, Fly, your own server):

1. Deploy the backend first using any section above; note its URL.
2. Import the repo into Vercel. `vercel.json` builds `frontend/` as a static site.
3. Add an environment variable:
   `VITE_API_ORIGIN = https://your-api-host.example.com`
   This is a **build-time** variable — Vite inlines it, so changing it requires
   a redeploy, not just a restart.
4. On the backend, allow the Vercel origin:
   `ARGUS_CORS_ORIGINS=https://your-project.vercel.app`

The dashboard then calls the API cross-origin and opens its WebSocket directly
against the API host (`wss://`), bypassing Vercel entirely for video.

Honestly: unless you specifically want Vercel's CDN, serving the UI from the API
container is simpler, has no CORS surface, and is one deployment instead of two.

---

## 7. VPS without Docker (systemd)

```bash
sudo apt install -y python3.11-venv libgl1 libglib2.0-0 nodejs npm
git clone https://github.com/OWNER/Argus.git /opt/argus && cd /opt/argus

python3 -m venv .venv && . .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

npm ci --prefix frontend && npm run build --prefix frontend
python backend/scripts/init_db.py
python backend/scripts/run_admin.py --setup-only
```

`/etc/systemd/system/argus.service`:

```ini
[Unit]
Description=Argus surveillance
After=network.target

[Service]
Type=exec
User=argus
WorkingDirectory=/opt/argus
Environment=ARGUS_DATA_DIR=/var/lib/argus
Environment=ARGUS_JWT_SECRET=<openssl rand -base64 48>
Environment=ARGUS_LOG_FORMAT=json
ExecStart=/opt/argus/.venv/bin/python -m uvicorn backend.api.main:app \
          --host 0.0.0.0 --port 8000 --proxy-headers
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now argus && sudo journalctl -u argus -f
```

---

## Putting it behind HTTPS

Argus does not terminate TLS. Front it with Caddy (simplest — automatic
certificates):

```caddyfile
argus.example.com {
    reverse_proxy 127.0.0.1:8000
}
```

Caddy proxies WebSockets without extra configuration. With nginx, the
`Upgrade`/`Connection` headers must be forwarded explicitly or the video wall
silently fails to connect while the rest of the UI works — a confusing partial
failure:

```nginx
location / {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_set_header X-Forwarded-Proto $scheme;
    proxy_read_timeout 3600s;   # video streams are long-lived
}
```

`X-Forwarded-Proto` matters: without it the app builds `http://` URLs on an
`https://` page and the browser blocks them as mixed content. The container
already passes `--proxy-headers`.

---

## Verifying a deployment

```bash
curl -fsS https://your-host/api/v1/health          # subsystem status
curl -s -o /dev/null -w '%{http_code}\n' https://your-host/   # 200 = UI served
```

Then in the browser: sign in, open the video wall, and confirm a tile shows
`LIVE` rather than `NO SIGNAL`. A tile stuck on `NO SIGNAL` with a healthy API
almost always means the WebSocket upgrade is not being proxied.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| Deploy times out; logs show the app started | Platform's `$PORT` ignored. Use the provided entrypoint. |
| Signed out after every deploy | `ARGUS_JWT_SECRET` unset or rotating. |
| Users/cameras/events gone after restart | No volume, or it is not mounted at `ARGUS_DATA_DIR`. |
| Container killed during startup, no traceback | Out of memory. Argus needs ≥2 GB. |
| `libGL.so.1: cannot open shared object file` | Missing OpenCV system libs. Install `libgl1` and `libglib2.0-0`. |
| UI loads, video tiles show `NO SIGNAL` | WebSocket upgrade not proxied, or `VITE_API_ORIGIN` wrong on a split deploy. |
| UI loads, every API call fails with a CORS error | Add the dashboard's origin to `ARGUS_CORS_ORIGINS`. |
| Snapshots broken but events listed | On a split deploy, the UI was built without `VITE_API_ORIGIN`. |
| `database is locked` | More than one worker/replica. Argus runs single-process by design. |
