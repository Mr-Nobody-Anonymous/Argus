#!/usr/bin/env python3
"""
Generate `data/demo_clip.mp4`, the bundled test fixture.

Why this script exists: `data/` is gitignored (it holds the database, snapshots,
and known faces - none of which belong in version control), so a fresh clone has
no demo clip. But `tests/test_regression.py` and `tests/swarm_benchmark.py` both
depend on one, and the README tells you to point a camera at it. This regenerates
it deterministically.

The clip must contain **real people**, not drawn shapes: YOLO returns zero
detections on synthetic rectangles and circles, so a shape-based fixture would
look like a broken detector rather than a working one. We therefore tile and pan
a real photograph to synthesise camera motion across a static scene.

Usage:
    python backend/scripts/make_demo_clip.py
    python backend/scripts/make_demo_clip.py --source path/to/photo.jpg --seconds 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "demo_clip.mp4"

WIDTH, HEIGHT = 640, 480
FPS = 15


def _synthesise_street_scene(width: int, height: int) -> np.ndarray:
    """
    Fallback source image when no photograph is supplied.

    This produces a crude but *photographic-ish* scene: gradient sky, textured
    ground, and several human-proportioned figures with head/torso/legs and
    soft edges. It will not detect as reliably as a real photo - if you need a
    dependable fixture, pass --source with an actual street photograph.
    """
    img = np.zeros((height, width, 3), dtype=np.uint8)

    # Sky gradient (top) and ground (bottom)
    for y in range(height):
        t = y / height
        if t < 0.55:
            img[y, :] = (int(180 - 40 * t), int(170 - 30 * t), int(160 - 20 * t))
        else:
            g = int(90 + 40 * (t - 0.55))
            img[y, :] = (g, g - 5, g - 10)

    rng = np.random.default_rng(7)
    # Ground texture so the detector has something other than flat colour
    noise = rng.integers(-12, 12, size=(height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Human-proportioned figures at varying depths
    people = [
        (90, 300, 1.0), (200, 315, 0.9), (310, 330, 1.05),
        (430, 300, 0.85), (540, 320, 0.95), (150, 265, 0.7),
        (380, 262, 0.65), (610, 290, 0.8),
    ]
    for cx, base_y, scale in people:
        h = int(150 * scale)
        w = max(6, int(h * 0.22))
        skin = (150, 165, 190)
        shirt = tuple(int(c) for c in rng.integers(60, 200, 3))
        trousers = tuple(int(c) for c in rng.integers(40, 110, 3))

        head_r = max(4, int(h * 0.11))
        head_c = (cx, base_y - h + head_r)
        cv2.circle(img, head_c, head_r, skin, -1, lineType=cv2.LINE_AA)

        torso_top = base_y - h + 2 * head_r
        torso_bot = base_y - int(h * 0.42)
        cv2.rectangle(img, (cx - w, torso_top), (cx + w, torso_bot), shirt, -1)
        cv2.rectangle(img, (cx - w, torso_bot), (cx - 1, base_y), trousers, -1)
        cv2.rectangle(img, (cx + 1, torso_bot), (cx + w, base_y), trousers, -1)

    return cv2.GaussianBlur(img, (3, 3), 0)


def build_clip(source: Path | None, out: Path, seconds: int) -> Path:
    if source is not None:
        frame_src = cv2.imread(str(source))
        if frame_src is None:
            raise SystemExit(f"Could not read source image: {source}")
    else:
        frame_src = _synthesise_street_scene(WIDTH * 2, HEIGHT)

    # Scale so the source is wider than the output, leaving room to pan.
    target_h = HEIGHT
    scale = target_h / frame_src.shape[0]
    new_w = max(WIDTH + 80, int(frame_src.shape[1] * scale))
    frame_src = cv2.resize(frame_src, (new_w, target_h), interpolation=cv2.INTER_AREA)

    total_frames = seconds * FPS
    max_shift = frame_src.shape[1] - WIDTH

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    if not writer.isOpened():
        raise SystemExit("OpenCV could not open an mp4 writer (missing codec?)")

    for i in range(total_frames):
        # Smooth back-and-forth pan so tracks persist and then genuinely change.
        phase = np.sin(2 * np.pi * i / total_frames) * 0.5 + 0.5
        x = int(phase * max_shift)
        writer.write(frame_src[:, x:x + WIDTH].copy())

    writer.release()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", type=Path, default=None,
                    help="Photograph to pan across. Strongly recommended: a real "
                         "street scene with people. Without it a synthetic scene "
                         "is used, which detects far less reliably.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--seconds", type=int, default=10)
    args = ap.parse_args()

    path = build_clip(args.source, args.out, args.seconds)
    size_kb = path.stat().st_size / 1024
    print(f"Wrote {path} ({size_kb:.0f} KB, {args.seconds}s @ {FPS}fps, {WIDTH}x{HEIGHT})")

    # Report detections so you know immediately whether the fixture is usable.
    try:
        from backend.services.core_engine.inference_engine import get_inference_engine
        cap = cv2.VideoCapture(str(path))
        ok, frame = cap.read()
        cap.release()
        if ok:
            n = len(get_inference_engine().detect_objects(frame))
            print(f"First frame yields {n} detections.")
            if n == 0:
                print("  WARNING: zero detections. Pass --source with a real "
                      "photograph of a street scene; YOLO does not detect "
                      "drawn shapes.")
    except Exception as exc:  # noqa: BLE001 - reporting only
        print(f"(Skipped detection check: {exc})")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(PROJECT_ROOT))
    raise SystemExit(main())
