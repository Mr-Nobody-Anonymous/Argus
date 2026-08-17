"""
Audit trail for privileged actions.

Distinct from the event store: events record what the *cameras* saw, the audit
log records what *people* did. For a surveillance product this is a compliance
requirement — every change to a zone, camera, identity, rule or permission must
be attributable.

Recorded for each action: who, what, when, from where, and the before/after
values. Reads are never audited (too noisy, no integrity impact); every
mutation is.

Note ``django_admin_log`` already captures edits made through the Django admin
panel. This table covers the API surface using the same concept, and both can
be reconciled on ``user_id``.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

from backend.config.config import resolve_path, redact_secrets

logger = logging.getLogger(__name__)


class AuditLog:
    """Append-only record of privileged actions."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(resolve_path(db_path or "data/argus.db"))
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        try:
            with self._connect() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS audit_log (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp   TEXT NOT NULL,
                        user_id     INTEGER,
                        username    TEXT NOT NULL,
                        role        TEXT,
                        action      TEXT NOT NULL,
                        resource    TEXT NOT NULL,
                        resource_id TEXT,
                        outcome     TEXT NOT NULL DEFAULT 'success',
                        client_ip   TEXT,
                        details     TEXT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_username ON audit_log(username)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_resource ON audit_log(resource, resource_id)"
                )
                conn.commit()
            logger.info("Audit log initialized")
        except sqlite3.Error as exc:
            logger.error(f"Failed to initialise audit log: {exc}")

    def record(
        self,
        *,
        username: str,
        action: str,
        resource: str,
        user_id: Optional[int] = None,
        role: Optional[str] = None,
        resource_id: Optional[str] = None,
        outcome: str = "success",
        client_ip: Optional[str] = None,
        before: Optional[Dict[str, Any]] = None,
        after: Optional[Dict[str, Any]] = None,
        detail: Optional[str] = None,
    ) -> Optional[int]:
        """
        Append an audit entry.

        Never raises: a failure to audit must not break the request, but it is
        logged at ERROR so the gap is visible.
        """
        details: Dict[str, Any] = {}
        if before is not None:
            details["before"] = redact_secrets(before)
        if after is not None:
            details["after"] = redact_secrets(after)
        if detail:
            details["detail"] = detail

        try:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO audit_log
                        (timestamp, user_id, username, role, action, resource,
                         resource_id, outcome, client_ip, details)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        datetime.now().isoformat(),
                        user_id,
                        username,
                        role,
                        action,
                        resource,
                        str(resource_id) if resource_id is not None else None,
                        outcome,
                        client_ip,
                        json.dumps(details) if details else None,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as exc:
            logger.error(f"Failed to write audit entry ({action} {resource}): {exc}")
            return None

    def query(
        self,
        *,
        username: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        outcome: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[List[Dict[str, Any]], int]:
        """Return (entries, total) most recent first."""
        clauses: List[str] = []
        params: List[Any] = []
        if username:
            clauses.append("username = ?")
            params.append(username)
        if resource:
            clauses.append("resource = ?")
            params.append(resource)
        if action:
            clauses.append("action = ?")
            params.append(action)
        if outcome:
            clauses.append("outcome = ?")
            params.append(outcome)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""

        try:
            with self._connect() as conn:
                total = conn.execute(
                    f"SELECT COUNT(*) FROM audit_log {where}", params
                ).fetchone()[0]
                rows = conn.execute(
                    f"SELECT * FROM audit_log {where} ORDER BY id DESC LIMIT ? OFFSET ?",
                    (*params, limit, offset),
                ).fetchall()

            entries = []
            for row in rows:
                entry = dict(row)
                if entry.get("details"):
                    try:
                        entry["details"] = json.loads(entry["details"])
                    except json.JSONDecodeError:
                        pass
                entries.append(entry)
            return entries, total
        except sqlite3.Error as exc:
            logger.error(f"Audit query failed: {exc}")
            return [], 0


_audit_log: Optional[AuditLog] = None


def get_audit_log() -> AuditLog:
    global _audit_log
    if _audit_log is None:
        _audit_log = AuditLog()
    return _audit_log
