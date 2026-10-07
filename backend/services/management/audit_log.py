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
                        details     TEXT,
                        prev_hash   TEXT,
                        entry_hash  TEXT
                    )
                    """
                )
                # Migrate existing databases if prev_hash/entry_hash are missing
                existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(audit_log)")}
                if "prev_hash" not in existing_cols:
                    conn.execute("ALTER TABLE audit_log ADD COLUMN prev_hash TEXT")
                if "entry_hash" not in existing_cols:
                    conn.execute("ALTER TABLE audit_log ADD COLUMN entry_hash TEXT")

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
            logger.info("Audit log initialized with cryptographic hash chaining")
        except sqlite3.Error as exc:
            logger.error(f"Failed to initialise audit log: {exc}")

    def _compute_hash(self, prev_hash: str, timestamp: str, user_id: Optional[int],
                      username: str, role: Optional[str], action: str, resource: str,
                      resource_id: Optional[str], outcome: str, client_ip: Optional[str],
                      details_json: Optional[str]) -> str:
        import hashlib
        payload = (
            f"{prev_hash}|{timestamp}|{user_id}|{username}|{role}|{action}|"
            f"{resource}|{resource_id}|{outcome}|{client_ip}|{details_json}"
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

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
        Append a tamper-evident audit entry into the hash chain.

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

        now_str = datetime.now().isoformat()
        res_id_str = str(resource_id) if resource_id is not None else None
        details_json = json.dumps(details, sort_keys=True) if details else None

        try:
            with self._connect() as conn:
                # Fetch last entry_hash to chain
                last_row = conn.execute("SELECT entry_hash FROM audit_log ORDER BY id DESC LIMIT 1").fetchone()
                prev_hash = last_row[0] if (last_row and last_row[0]) else "GENESIS_0000000000000000000000000000000000000000000000000000000000000000"

                entry_hash = self._compute_hash(
                    prev_hash, now_str, user_id, username, role, action, resource,
                    res_id_str, outcome, client_ip, details_json
                )

                cursor = conn.execute(
                    """
                    INSERT INTO audit_log
                        (timestamp, user_id, username, role, action, resource,
                         resource_id, outcome, client_ip, details, prev_hash, entry_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        now_str,
                        user_id,
                        username,
                        role,
                        action,
                        resource,
                        res_id_str,
                        outcome,
                        client_ip,
                        details_json,
                        prev_hash,
                        entry_hash,
                    ),
                )
                conn.commit()
                return cursor.lastrowid
        except sqlite3.Error as exc:
            logger.error(f"Failed to write audit entry ({action} {resource}): {exc}")
            return None

    def verify_integrity(self) -> Dict[str, Any]:
        """
        Traverse the audit log and verify cryptographic integrity of the hash chain.
        Returns a verification summary indicating whether tampering was detected.
        """
        try:
            with self._connect() as conn:
                rows = conn.execute(
                    """
                    SELECT id, timestamp, user_id, username, role, action, resource,
                           resource_id, outcome, client_ip, details, prev_hash, entry_hash
                    FROM audit_log ORDER BY id ASC
                    """
                ).fetchall()

                if not rows:
                    return {"verified": True, "entries_checked": 0, "status": "empty"}

                expected_prev = "GENESIS_0000000000000000000000000000000000000000000000000000000000000000"
                for idx, row in enumerate(rows):
                    row_id = row["id"]
                    stored_prev = row["prev_hash"]
                    stored_entry = row["entry_hash"]

                    # If entries predate hash chaining, initialize baseline
                    if not stored_entry:
                        continue

                    if stored_prev != expected_prev and idx != 0:
                        return {
                            "verified": False,
                            "entries_checked": idx,
                            "broken_at_id": row_id,
                            "error": f"Hash chain broken at ID {row_id}: previous hash mismatch",
                        }

                    calculated = self._compute_hash(
                        stored_prev, row["timestamp"], row["user_id"], row["username"],
                        row["role"], row["action"], row["resource"], row["resource_id"],
                        row["outcome"], row["client_ip"], row["details"]
                    )
                    if calculated != stored_entry:
                        return {
                            "verified": False,
                            "entries_checked": idx,
                            "broken_at_id": row_id,
                            "error": f"Tampering detected at ID {row_id}: content hash mismatch",
                        }

                    expected_prev = stored_entry

                return {
                    "verified": True,
                    "entries_checked": len(rows),
                    "status": "valid",
                    "latest_hash": expected_prev,
                }
        except sqlite3.Error as exc:
            return {"verified": False, "error": str(exc)}

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
                    f"SELECT COUNT(*) FROM audit_log {where}",  # nosec B608: `where` uses fixed predicates above; values are bound.
                    params
                ).fetchone()[0]
                rows = conn.execute(
                    f"SELECT * FROM audit_log {where} ORDER BY id DESC LIMIT ? OFFSET ?",  # nosec B608: `where` uses fixed predicates; values use placeholders.
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
