"""
Data retention enforcement.

Argus stores identifiable material: event records, snapshot images of people,
face embeddings, and licence-plate reads. Keeping that indefinitely is a
privacy liability and, in many jurisdictions, unlawful.

``EventStore.delete_old_events()`` already existed but nothing ever called it,
so retention was defined and never enforced. This module runs the purge on a
schedule and extends it to the artefacts on disk.

Per-class retention (days), configurable under ``retention:`` in config.yaml::

    retention:
      enabled: true
      interval_hours: 6
      events_days: 90
      snapshots_days: 30
      anomalies_days: 30
      plates_days: 30
      audit_days: 365      # audit trails are usually kept longest

Face embeddings are deliberately NOT auto-purged: they are enrolled
deliberately and must be removed explicitly via DELETE /api/v1/faces/{id},
otherwise a scheduled job would silently break recognition.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional

from backend.config.config import get_config, resolve_path, section_to_dict

logger = logging.getLogger(__name__)

_DEFAULTS: Dict[str, Any] = {
    "enabled": True,
    "interval_hours": 6,
    "events_days": 90,
    "snapshots_days": 30,
    "anomalies_days": 30,
    "plates_days": 30,
    "audit_days": 365,
    # Time-based retention alone cannot bound disk usage: at the measured event
    # rate a single camera writes ~4 GB of snapshots per day, so the 30-day
    # window is reached only after ~130 GB. This is a hard ceiling enforced on
    # every pass, oldest evicted first. 0 disables it.
    "snapshots_max_mb": 2048,
    # Evidence clips. Video is far larger per item than a snapshot, so this
    # window is short and the byte ceiling does the real work.
    "clips_days": 14,
    "clips_max_mb": 4096,
    # Phase 6 perception memory. Descriptors are re-identifying, so their
    # window is deliberately much shorter than the observations'.
    "perception_observations_days": 60,
    "perception_appearances_days": 7,
}


def _policy() -> Dict[str, Any]:
    cfg = section_to_dict(getattr(get_config(), "retention", None))
    merged = dict(_DEFAULTS)
    merged.update({k: v for k, v in cfg.items() if v is not None})
    return merged


class RetentionPolicyError(RuntimeError):
    """A retention policy cannot be enforced as written.

    Raised rather than logged, because a purge that silently does nothing
    leaves data on disk that operators and privacy notices both claim was
    deleted.
    """


def _db_path() -> str:
    return str(resolve_path("data/argus.db"))


def _purge_table(conn: sqlite3.Connection, table: str, column: str, days: int) -> int:
    """Delete rows older than `days`. Returns rows removed (0 if absent)."""
    if days <= 0:
        return 0
    try:
        exists = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not exists:
            return 0
        cutoff = (datetime.now() - timedelta(days=days)).isoformat(sep=" ")
        cursor = conn.execute(f"DELETE FROM {table} WHERE {column} < ?", (cutoff,))
        return cursor.rowcount or 0
    except sqlite3.Error as exc:
        # A missing table is legitimate (an optional feature was never used),
        # but a missing COLUMN means this policy has silently never run. That
        # was a real defect here: `anomalies` and `license_plates` were purged
        # on a `timestamp` column neither table has, so plate reads - the most
        # privacy-sensitive rows in the database - accumulated forever behind
        # a log line nobody reads. Retention that fails quietly is worse than
        # no retention, because it is believed.
        message = str(exc)
        if "no such column" in message:
            raise RetentionPolicyError(
                f"Retention is misconfigured for '{table}': {message}. "
                f"Rows in this table are NOT being deleted.") from exc
        logger.error(f"Retention purge failed for {table}: {exc}")
        return 0


def purge_snapshots(days: int) -> int:
    """Delete snapshot images older than `days`. Returns files removed."""
    if days <= 0:
        return 0
    try:
        snapshot_dir = resolve_path(get_config().system.snapshot_dir)
        if not snapshot_dir.exists():
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        for path in snapshot_dir.glob("*.jpg"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning(f"Could not delete snapshot {path.name}: {exc}")
        return removed
    except Exception as exc:  # noqa: BLE001 - retention must never crash the app
        logger.error(f"Snapshot retention failed: {exc}")
        return 0


def purge_clips(days: int) -> int:
    """Delete evidence clips older than `days`. Returns files removed."""
    if days <= 0:
        return 0
    try:
        clip_dir = resolve_path(get_config().system.snapshot_dir).parent / "clips"
        if not clip_dir.exists():
            return 0
        cutoff = time.time() - days * 86400
        removed = 0
        for path in clip_dir.glob("*.mp4"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError as exc:
                logger.warning(f"Could not delete clip {path.name}: {exc}")
        return removed
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Clip retention failed: {exc}")
        return 0


def enforce_clip_size_cap(max_mb: int) -> int:
    """Hard byte ceiling on the clip directory, oldest evicted first.

    Video dwarfs stills: one 10 s 480p clip measured ~4 MB against ~200 KB for
    a snapshot. Without a ceiling a busy site fills the disk long before the
    time window expires.
    """
    if max_mb <= 0:
        return 0
    try:
        clip_dir = resolve_path(get_config().system.snapshot_dir).parent / "clips"
        if not clip_dir.exists():
            return 0
        files = []
        total = 0
        for path in clip_dir.glob("*.mp4"):
            try:
                stat = path.stat()
            except OSError:
                continue
            files.append((stat.st_mtime, stat.st_size, path))
            total += stat.st_size
        budget = max_mb * 1024 * 1024
        if total <= budget:
            return 0
        files.sort()
        removed = 0
        for _, size, path in files:
            if total <= budget:
                break
            try:
                path.unlink()
                total -= size
                removed += 1
            except OSError as exc:
                logger.warning(f"Could not evict clip {path.name}: {exc}")
        if removed:
            logger.info(f"Clip size cap evicted {removed} files")
        return removed
    except Exception as exc:  # noqa: BLE001
        logger.error(f"Clip size cap failed: {exc}")
        return 0


def enforce_snapshot_size_cap(max_mb: int) -> int:
    """Evict oldest snapshots until the directory fits under `max_mb`.

    Time-based expiry cannot prevent a disk filling up inside the retention
    window, so this runs as a second, independent bound.
    """
    if max_mb <= 0:
        return 0
    try:
        snapshot_dir = resolve_path(get_config().system.snapshot_dir)
        if not snapshot_dir.exists():
            return 0

        budget = max_mb * 1024 * 1024
        files = []
        total = 0
        for path in snapshot_dir.glob("*.jpg"):
            try:
                st = path.stat()
            except OSError:
                continue
            files.append((st.st_mtime, st.st_size, path))
            total += st.st_size

        if total <= budget:
            return 0

        files.sort()  # oldest first
        removed = 0
        for _mtime, size, path in files:
            if total <= budget:
                break
            try:
                path.unlink()
                total -= size
                removed += 1
            except OSError as exc:
                logger.warning(f"Could not evict snapshot {path.name}: {exc}")
        if removed:
            logger.warning(
                f"Snapshot directory exceeded {max_mb} MB - evicted {removed} "
                f"oldest file(s) to stay under the cap"
            )
        return removed
    except Exception as exc:  # noqa: BLE001 - retention must never crash the app
        logger.error(f"Snapshot size cap failed: {exc}")
        return 0


def run_retention_once() -> Dict[str, int]:
    """Execute one full retention pass. Returns per-class deletion counts."""
    policy = _policy()
    results: Dict[str, int] = {}

    try:
        with sqlite3.connect(_db_path(), timeout=15) as conn:
            results["events"] = _purge_table(conn, "events", "created_at", policy["events_days"])
            results["anomalies"] = _purge_table(conn, "anomalies", "detected_at", policy["anomalies_days"])
            results["license_plates"] = _purge_table(conn, "license_plates", "detected_at", policy["plates_days"])
            results["audit_log"] = _purge_table(conn, "audit_log", "timestamp", policy["audit_days"])
            # Perception memory. Appearance descriptors are re-identifying:
            # they must expire on their own, shorter clock, and they must not
            # be able to outlive this policy by living in a separate module
            # that nobody remembered to wire up.
            results["perception_observations"] = _purge_table(
                conn, "perception_observations", "timestamp",
                policy["perception_observations_days"])
            results["perception_appearances"] = _purge_table(
                conn, "perception_appearances", "last_seen",
                policy["perception_appearances_days"])
            results["perception_tracks"] = _purge_table(
                conn, "perception_tracks", "last_seen",
                policy["perception_appearances_days"])
            conn.commit()
    except sqlite3.Error as exc:
        logger.error(f"Retention database pass failed: {exc}")

    results["snapshots"] = purge_snapshots(policy["snapshots_days"])
    results["snapshots_over_cap"] = enforce_snapshot_size_cap(
        policy.get("snapshots_max_mb", 0)
    )
    results["clips"] = purge_clips(policy.get("clips_days", 0))
    results["clips_over_cap"] = enforce_clip_size_cap(
        policy.get("clips_max_mb", 0)
    )

    total = sum(results.values())
    if total:
        logger.info(f"Retention pass removed {total} items: {results}")
    else:
        logger.debug("Retention pass: nothing to remove")
    return results


class RetentionScheduler:
    """Background thread running the retention policy at a fixed interval."""

    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self.last_run: Optional[str] = None
        self.last_results: Dict[str, int] = {}

    def start(self) -> bool:
        policy = _policy()
        if not policy["enabled"]:
            logger.info("Retention scheduler disabled by config")
            return False
        if self._thread and self._thread.is_alive():
            return True

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True, name="retention")
        self._thread.start()
        logger.info(
            f"Retention scheduler started (every {policy['interval_hours']}h; "
            f"events={policy['events_days']}d snapshots={policy['snapshots_days']}d)"
        )
        return True

    def stop(self) -> None:
        self._stop.set()

    def _loop(self) -> None:
        interval = max(1, int(_policy()["interval_hours"])) * 3600
        # Delay the first pass so startup is not competing with a bulk delete.
        if self._stop.wait(60):
            return
        while not self._stop.is_set():
            try:
                self.last_results = run_retention_once()
                self.last_run = datetime.now().isoformat()
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Retention scheduler error: {exc}")
            if self._stop.wait(interval):
                break

    def status(self) -> Dict[str, Any]:
        return {
            "running": bool(self._thread and self._thread.is_alive()),
            "last_run": self.last_run,
            "last_results": self.last_results,
            "policy": _policy(),
        }


_scheduler: Optional[RetentionScheduler] = None


def get_retention_scheduler() -> RetentionScheduler:
    global _scheduler
    if _scheduler is None:
        _scheduler = RetentionScheduler()
    return _scheduler
