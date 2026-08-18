# ─────────────────────────────────────────────────────────────────────────────
# Argus - single production image (API + built dashboard)
#
# One container, not two. The API already serves frontend/dist as a SPA, so a
# separate nginx container only adds a second thing to configure, a second
# thing to scale, and a CORS boundary that does not need to exist. Platforms
# that host exactly one container per service (Render, Railway, Fly, Cloud Run,
# App Runner) can run this image directly.
#
# Build:  docker build -t argus .
# Run:    docker run -p 8000:8000 -e ARGUS_JWT_SECRET=... -v argus-data:/app/data argus
#
# docker/Dockerfile.backend and docker/Dockerfile.frontend remain for the
# two-container development compose setup.
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: build the dashboard ─────────────────────────────────────────────
FROM node:20-alpine AS ui

WORKDIR /ui

# Copy manifests first so this layer is cached until dependencies actually
# change. `npm ci` needs the lockfile and installs exactly what it pins.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./

# VITE_API_ORIGIN is empty here on purpose: this image serves the UI and the API
# from the same origin, so relative URLs are correct. It is only set when the
# dashboard is deployed to a static host separate from the API (see vercel.json).
ARG VITE_API_ORIGIN=""
ENV VITE_API_ORIGIN=$VITE_API_ORIGIN

# Vite's default heap is generous but the bundle pulls in MUI + charts; cap it
# so the build fails loudly on a small builder instead of being OOM-killed
# halfway with a truncated dist/ that looks like a successful build.
ENV NODE_OPTIONS=--max-old-space-size=2048
RUN npm run build && test -f dist/index.html


# ── Stage 2: python dependencies ─────────────────────────────────────────────
# Kept separate from the runtime stage so build toolchains never reach the
# final image.
FROM python:3.11-slim AS deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .

# Install into a venv so stage 3 can copy one self-contained directory.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# The CPU wheel index keeps torch at ~200 MB instead of pulling the ~2.5 GB
# CUDA build. Nothing in a container without a GPU can use those kernels, and
# on most hosts the CUDA image exceeds the image size limit outright.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt


# ── Stage 3: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

# libGL/libglib are OpenCV's runtime shared libraries. opencv-python-headless
# still links them; without these the first `import cv2` fails at startup with
# "libGL.so.1: cannot open shared object file".
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        curl \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ARGUS_DATA_DIR=/app/data \
    ARGUS_LOG_FORMAT=json

COPY --from=deps /opt/venv /opt/venv

WORKDIR /app

COPY backend/ ./backend/
COPY config/ ./config/
COPY argus.py ./
COPY docker/entrypoint.sh ./docker/entrypoint.sh
COPY --from=ui /ui/dist ./frontend/dist

# The 965 KB demo clip ships so a fresh deployment has a working camera to show
# instead of an empty video wall. It is copied to a seed location rather than
# into /app/data, because that path is a mounted volume at runtime and the mount
# would hide anything baked underneath it. The entrypoint copies it across on
# first boot only.
COPY data/demo_clip.mp4 /app/seed/demo_clip.mp4

RUN chmod +x docker/entrypoint.sh

# Run as a non-root user. A surveillance system holding identifiable footage is
# a poor candidate for a root container: a process escape would own the host.
RUN useradd --create-home --uid 10001 argus \
    && mkdir -p /app/data/snapshots /app/data/clips \
    && chown -R argus:argus /app
USER argus

# Model weights are NOT baked in: ultralytics downloads yolov8n.pt on first use
# and caches it under the user's home. Baking it would add ~6 MB and pin the
# version inside the image.
ENV YOLO_CONFIG_DIR=/app/data/.ultralytics

EXPOSE 8000

# The platform overrides $PORT; the healthcheck must follow it rather than
# assume 8000, or it reports unhealthy on every host that assigns a port.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8000}/api/v1/health" || exit 1

ENTRYPOINT ["./docker/entrypoint.sh"]
