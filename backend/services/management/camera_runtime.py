"""In-memory camera liveness state.

Whether a camera is currently delivering frames, at what rate, and when its last
frame arrived are properties of the *running process*. They were previously
written to SQLite roughly once per second per camera, which was both wasteful
and wrong:

* Volume: 86,400 durable writes per camera per day - 3.15 billion per year at
  100 cameras - for values that are meaningless once the process stops.
* Correctness: a clean shutdown reset the row to 'offline', but SIGKILL (crash,
  OOM-kill, power loss) bypassed that hook. Measured on a live server, the
  database kept reporting ``('online', 14.73)`` after the process was killed, so
  the API served a dead camera as live.

Keeping this state in memory fixes both. A camera that the current process is
not actively ingesting is offline *by definition*, which is the correct answer
after a crash without needing any recovery logic.

Camera identity (id, name, rtsp_url, location_tag) is operator-managed
configuration and stays in the database; only liveness lives here.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Dict, Optional

# Values a camera row carries for liveness. A camera with no runtime entry is
# reported with these, i.e. offline - the correct default for "this process is
# not ingesting it".
_DEFAULT_STATE: Dict[str, object] = {
    "status": "offline",
    "fps": 0.0,
    "last_frame_time": None,
}

_lock = threading.RLock()
_state: Dict[int, Dict[str, object]] = {}


def set_status(camera_id: int, status: str, fps: float = 0.0) -> None:
    """Record the current liveness of a camera. Never touches disk."""
    with _lock:
        _state[int(camera_id)] = {
            "status": status,
            "fps": float(fps or 0.0),
            # Only a live camera has a meaningful last-frame timestamp.
            "last_frame_time": datetime.now() if status == "online" else None,
        }


def get_status(camera_id: int) -> Dict[str, object]:
    """Liveness for one camera, defaulting to offline when unknown."""
    with _lock:
        return dict(_state.get(int(camera_id), _DEFAULT_STATE))


def clear(camera_id: int) -> None:
    """Forget a camera - it is then reported offline."""
    with _lock:
        _state.pop(int(camera_id), None)


def clear_all() -> None:
    """Drop all runtime state (used on shutdown and by tests)."""
    with _lock:
        _state.clear()


def apply_to(camera: Optional[Dict]) -> Optional[Dict]:
    """Overlay live state onto a camera row loaded from the database.

    Every read path goes through here, so callers keep receiving the same
    ``status`` / ``fps`` / ``last_frame_time`` keys they always did - they are
    just served from memory instead of from a stale disk row.
    """
    if not camera:
        return camera
    merged = dict(camera)
    merged.update(get_status(merged.get("id")))
    return merged


def snapshot() -> Dict[int, Dict[str, object]]:
    """Copy of all tracked runtime state (diagnostics/tests)."""
    with _lock:
        return {cid: dict(v) for cid, v in _state.items()}
