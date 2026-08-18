"""Pre-event video evidence: what happened in the seconds *before* the alert.

A snapshot shows the instant a rule fired. It cannot show the person walking up
to the bag, or the direction a vehicle came from - which is usually the part an
investigator needs. Roadmap 3.6 asked for a ring buffer of recent footage so an
event can be exported as a clip rather than a still.

Design decisions, all driven by measurement on this host rather than intuition:

**Frames are buffered JPEG-encoded, not raw.** Measured on 480p:

    raw BGR frame        0.88 MB   ->  10 s @ 10 fps = 88 MB *per camera*
    JPEG q=80 (noise)    0.20 MB   ->  10 s @ 10 fps = 20 MB  (worst case)
    JPEG q=80 (flat)     0.005 MB  ->  10 s @ 10 fps = 0.5 MB (best case)

Raw buffering costs 352 MB across four cameras on a 2 GB box - it would kill
the process. Encoding costs ~2 ms/frame against an existing ~108 ms/frame YOLO
budget, which is under 2% overhead for a 4-40x memory saving.

**The buffer is bounded in bytes, not only in frames.** A frame count alone is
not a memory bound, because frame size varies by an order of magnitude with
scene content. Both limits are enforced and the byte ceiling is reported.

**Clip export is opt-in per rule.** Writing a clip costs a decode per frame
(~3.4 ms) plus disk. Exporting one for every `scene_change` on a busy camera
would be pure waste, so `clip_rules` names the kinds worth the cost.

**A clip that cannot be written says so.** No OpenCV writer, no codec, buffer
too short: each returns a `ClipResult` with `written=False` and the reason. It
never returns a path to a file that does not exist.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_SECONDS = 10.0
DEFAULT_FPS = 10.0
DEFAULT_QUALITY = 80
# Hard ceiling per camera. 24 MB is roughly the worst-case 10 s of high-noise
# 480p; typical footage uses a small fraction of it.
DEFAULT_MAX_MB = 24.0


@dataclass
class ClipResult:
    written: bool
    path: Optional[str] = None
    frames: int = 0
    seconds: float = 0.0
    reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "written": self.written,
            "path": self.path,
            "frames": self.frames,
            "seconds": round(self.seconds, 2),
            "reason": self.reason,
        }


class FrameRingBuffer:
    """The last N seconds of one camera, JPEG-encoded and byte-bounded."""

    def __init__(self, seconds: float = DEFAULT_SECONDS, fps: float = DEFAULT_FPS,
                 quality: int = DEFAULT_QUALITY, max_mb: float = DEFAULT_MAX_MB):
        self.seconds = seconds
        self.fps = fps
        self.quality = int(quality)
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.max_frames = max(1, int(seconds * fps))
        self._frames: Deque[Tuple[float, bytes]] = deque()
        self._bytes = 0
        self._lock = threading.RLock()
        self.dropped_for_size = 0

    def append(self, frame, timestamp: Optional[float] = None) -> bool:
        """Encode and store a frame. Returns False if it could not be stored."""
        try:
            import cv2
        except Exception:  # noqa: BLE001
            return False
        if frame is None:
            return False
        try:
            ok, encoded = cv2.imencode(
                ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality]
            )
            if not ok:
                return False
            payload = encoded.tobytes()
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Ring buffer encode failed: {exc}")
            return False

        ts = time.time() if timestamp is None else timestamp
        with self._lock:
            self._frames.append((ts, payload))
            self._bytes += len(payload)
            # Frame-count bound.
            while len(self._frames) > self.max_frames:
                _, old = self._frames.popleft()
                self._bytes -= len(old)
            # Byte bound. Frame size varies ~40x with scene content, so a
            # frame count alone is not a memory guarantee.
            while self._bytes > self.max_bytes and len(self._frames) > 1:
                _, old = self._frames.popleft()
                self._bytes -= len(old)
                self.dropped_for_size += 1
        return True

    def snapshot(self) -> List[Tuple[float, bytes]]:
        with self._lock:
            return list(self._frames)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            frames = len(self._frames)
            span = (
                self._frames[-1][0] - self._frames[0][0] if frames > 1 else 0.0
            )
            return {
                "frames": frames,
                "seconds_buffered": round(span, 2),
                "bytes": self._bytes,
                "mb": round(self._bytes / 1048576, 2),
                "max_mb": round(self.max_bytes / 1048576, 2),
                "dropped_for_size": self.dropped_for_size,
            }


class EvidenceClipService:
    """Keeps a ring buffer per camera and writes clips on demand."""

    def __init__(self, enabled: bool = True, seconds: float = DEFAULT_SECONDS,
                 fps: float = DEFAULT_FPS, quality: int = DEFAULT_QUALITY,
                 max_mb: float = DEFAULT_MAX_MB,
                 clip_rules: Optional[List[str]] = None,
                 output_dir: Optional[Path] = None):
        self.enabled = enabled
        self.seconds = seconds
        self.fps = fps
        self.quality = quality
        self.max_mb = max_mb
        # Only these rule types are worth the decode + disk cost.
        self.clip_rules = set(clip_rules or [
            "intrusion", "line_crossing", "abandoned_object", "fall_detection",
            "speed_violation",
        ])
        self._buffers: Dict[int, FrameRingBuffer] = {}
        self._lock = threading.RLock()
        self._output_dir = output_dir
        self.counters = {"clips_written": 0, "clips_failed": 0, "clips_skipped": 0}

    def _dir(self) -> Path:
        if self._output_dir is None:
            from backend.config.config import get_config, resolve_path

            base = resolve_path(get_config().system.snapshot_dir).parent
            self._output_dir = base / "clips"
        self._output_dir.mkdir(parents=True, exist_ok=True)
        return self._output_dir

    def buffer_for(self, camera_id: int) -> FrameRingBuffer:
        with self._lock:
            buf = self._buffers.get(camera_id)
            if buf is None:
                buf = FrameRingBuffer(
                    seconds=self.seconds, fps=self.fps,
                    quality=self.quality, max_mb=self.max_mb,
                )
                self._buffers[camera_id] = buf
            return buf

    def record(self, camera_id: int, frame, timestamp: Optional[float] = None) -> bool:
        """Add a frame to this camera's pre-event buffer."""
        if not self.enabled:
            return False
        try:
            return self.buffer_for(camera_id).append(frame, timestamp)
        except Exception as exc:  # noqa: BLE001 - never break the frame loop
            logger.debug(f"Evidence buffer append failed: {exc}")
            return False

    def wants_clip(self, rule_type: str) -> bool:
        return self.enabled and rule_type in self.clip_rules

    def write_clip(self, camera_id: int, rule_type: str,
                   event_id: Optional[int] = None) -> ClipResult:
        """Write the buffered pre-event footage for this camera to an mp4."""
        if not self.enabled:
            return ClipResult(False, reason="evidence clips are disabled")
        if not self.wants_clip(rule_type):
            self.counters["clips_skipped"] += 1
            return ClipResult(
                False,
                reason=f"rule {rule_type!r} is not in clip_rules; "
                       f"clip not worth the decode and disk cost",
            )

        frames = self.buffer_for(camera_id).snapshot()
        if len(frames) < 2:
            self.counters["clips_failed"] += 1
            return ClipResult(
                False, frames=len(frames),
                reason="fewer than 2 buffered frames - nothing to write yet",
            )

        try:
            import cv2
            import numpy as np
        except Exception as exc:  # noqa: BLE001
            self.counters["clips_failed"] += 1
            return ClipResult(False, reason=f"OpenCV unavailable: {exc}")

        try:
            first = cv2.imdecode(
                np.frombuffer(frames[0][1], dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if first is None:
                self.counters["clips_failed"] += 1
                return ClipResult(False, reason="could not decode buffered frame")
            height, width = first.shape[:2]

            name = (
                f"cam{camera_id}_{rule_type}"
                f"{'_ev' + str(event_id) if event_id else ''}"
                f"_{datetime.now():%Y%m%d_%H%M%S}.mp4"
            )
            path = self._dir() / name
            writer = cv2.VideoWriter(
                str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
            )
            if not writer.isOpened():
                self.counters["clips_failed"] += 1
                return ClipResult(
                    False, reason="no mp4v encoder available in this OpenCV build"
                )

            written = 0
            for _, payload in frames:
                image = cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR
                )
                if image is None:
                    continue
                if image.shape[:2] != (height, width):
                    image = cv2.resize(image, (width, height))
                writer.write(image)
                written += 1
            writer.release()

            # Never report a path unless the file exists and has content.
            if not path.exists() or path.stat().st_size == 0:
                self.counters["clips_failed"] += 1
                return ClipResult(
                    False, reason="writer produced no output (missing codec?)"
                )

            span = frames[-1][0] - frames[0][0]
            self.counters["clips_written"] += 1
            return ClipResult(True, str(path), written, span, "ok")
        except Exception as exc:  # noqa: BLE001
            self.counters["clips_failed"] += 1
            return ClipResult(False, reason=str(exc))

    def status(self) -> Dict[str, Any]:
        with self._lock:
            buffers = {
                cid: buf.stats() for cid, buf in self._buffers.items()
            }
        total_mb = sum(b["mb"] for b in buffers.values())
        return {
            "enabled": self.enabled,
            "pre_event_seconds": self.seconds,
            "fps": self.fps,
            "jpeg_quality": self.quality,
            "max_mb_per_camera": self.max_mb,
            "clip_rules": sorted(self.clip_rules),
            "buffers": buffers,
            "total_buffered_mb": round(total_mb, 2),
            "counters": dict(self.counters),
            "note": (
                "Frames are buffered JPEG-encoded, not raw: raw 480p costs "
                "88 MB per camera for 10 s, which does not fit alongside the "
                "detection model."
            ),
        }


def _build_from_config() -> EvidenceClipService:
    from backend.config.config import get_config, section_to_dict

    section = section_to_dict(getattr(get_config(), "evidence_clips", {})) or {}
    return EvidenceClipService(
        enabled=bool(section.get("enabled", True)),
        seconds=float(section.get("pre_event_seconds", DEFAULT_SECONDS)),
        fps=float(section.get("fps", DEFAULT_FPS)),
        quality=int(section.get("jpeg_quality", DEFAULT_QUALITY)),
        max_mb=float(section.get("max_mb_per_camera", DEFAULT_MAX_MB)),
        clip_rules=list(section.get("clip_rules") or []) or None,
    )


_service: Optional[EvidenceClipService] = None
_lock = threading.Lock()


def get_evidence_service() -> EvidenceClipService:
    global _service
    if _service is None:
        with _lock:
            if _service is None:
                try:
                    _service = _build_from_config()
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Evidence clip config failed, using defaults: {exc}")
                    _service = EvidenceClipService()
    return _service


def reset_evidence_service() -> None:
    global _service
    with _lock:
        _service = None
