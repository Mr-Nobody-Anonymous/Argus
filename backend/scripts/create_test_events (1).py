#!/usr/bin/env python
"""
Generate synthetic events (intrusion / loitering) for development and UI testing.

Creates a camera to attach the events to (unless one already exists), writes a
placeholder snapshot, and inserts a batch of events into the event store.

Usage:
    python backend/scripts/create_test_events.py
    python backend/scripts/create_test_events.py --count 10 --camera-id 1
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np

from backend.config.config import get_config, resolve_path
from backend.services.management.camera_manager import get_camera_manager
from backend.services.management.event_store import get_event_store


def ensure_camera(camera_id: int = None) -> int:
    """Return a valid camera id, creating a demo camera when necessary.

    Events have a FOREIGN KEY on cameras(id), so inserting against a
    non-existent camera fails - the original script hardcoded camera_id=5.
    """
    manager = get_camera_manager()

    if camera_id is not None:
        if manager.get_camera(camera_id):
            return camera_id
        raise SystemExit(
            f"Camera {camera_id} does not exist. Create it first or omit --camera-id."
        )

    cameras = manager.get_all_cameras()
    if cameras:
        return cameras[0]["id"]

    demo = manager.get_camera_by_url("rtsp://demo/test-events")
    if demo:
        return demo["id"]

    created = manager.create_camera(
        name="Test Events Camera",
        rtsp_url="rtsp://demo/test-events",
        location_tag="synthetic",
    )
    print(f"Created demo camera #{created['id']} ({created['name']})")
    return created["id"]


def write_placeholder_snapshot() -> str:
    """Write a placeholder snapshot image and return its path."""
    snapshot_dir = resolve_path(get_config().system.snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    filepath = snapshot_dir / "test_event.jpg"

    # A real BGR numpy image - cv2.imwrite cannot accept nested Python lists.
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (0, 128, 0)  # solid green (BGR)
    cv2.putText(img, "ARGUS TEST EVENT", (60, 250),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 255), 2)

    if not cv2.imwrite(str(filepath), img):
        raise SystemExit(f"Failed to write snapshot to {filepath}")

    print(f"Wrote placeholder snapshot: {filepath}")
    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Create synthetic Argus events")
    parser.add_argument("--count", type=int, default=5, help="Number of events to create")
    parser.add_argument("--camera-id", type=int, default=None,
                        help="Existing camera id to attach events to")
    args = parser.parse_args()

    camera_id = ensure_camera(args.camera_id)
    snapshot_path = write_placeholder_snapshot()

    store = get_event_store()
    for i in range(args.count):
        event = store.create_event(
            camera_id=camera_id,
            rule_type="intrusion" if i % 2 == 0 else "loitering",
            object_type="person",
            confidence=round(min(0.99, 0.85 + i * 0.02), 4),
            bbox=[100 + i * 10, 100, 200 + i * 10, 200],
            snapshot_path=snapshot_path,
            priority="high" if i < 2 else "medium",
            metadata={"test": True, "index": i},
        )
        print(f"Created event {event['id']}: {event['rule_type']} (camera {camera_id})")

    print(f"\n✅ Created {args.count} test events on camera {camera_id}. "
          f"Check the Events page in your browser.")


if __name__ == "__main__":
    main()
