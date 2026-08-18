"""
Event storage and retrieval service
"""
import logging
import json
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from backend.database.db import get_db

logger = logging.getLogger(__name__)


class EventStore:
    def __init__(self):
        self.db = get_db()

    def create_event(
        self,
        camera_id: int,
        rule_type: str,
        timestamp: datetime = None,
        object_type: str = None,
        confidence: float = None,
        bbox: List[int] = None,
        snapshot_path: str = None,
        priority: str = "medium",
        metadata: dict = None
    ) -> Dict:
        """Create a new event"""
        try:
            if timestamp is None:
                timestamp = datetime.now()
            
            bbox_json = json.dumps(bbox) if bbox else None
            metadata_json = json.dumps(metadata) if metadata else None
            
            cursor = self.db.execute(
                """
                INSERT INTO events (
                    camera_id, timestamp, rule_type, object_type, confidence,
                    bbox, snapshot_path, priority, status, metadata, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'detected', ?, ?)
                """,
                (
                    camera_id, timestamp, rule_type, object_type, confidence,
                    bbox_json, snapshot_path, priority, metadata_json, datetime.now()
                )
            )
            event_id = cursor.lastrowid
            logger.info(f"Created event: {rule_type} for camera {camera_id} (ID: {event_id})")

            # Export pre-event footage for rules worth the cost. The clip path
            # is recorded in metadata rather than a column so no migration is
            # needed, and a failure to write one never blocks the event: the
            # event is the guarantee, the clip is supporting evidence.
            try:
                from backend.services.management.evidence_clips import (
                    get_evidence_service,
                )

                evidence = get_evidence_service()
                if evidence.wants_clip(rule_type):
                    clip = evidence.write_clip(camera_id, rule_type, event_id)
                    payload = dict(metadata or {})
                    payload["clip"] = clip.to_dict()
                    self.db.execute(
                        "UPDATE events SET metadata = ? WHERE id = ?",
                        (json.dumps(payload), event_id),
                    )
                    if not clip.written:
                        logger.warning(
                            "No evidence clip for event %s: %s",
                            event_id, clip.reason,
                        )
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Evidence clip failed for event {event_id}: {exc}")

            event = self.get_event(event_id)

            # Deliver the alert. Every event in the system passes through this
            # method, so it is the one place worth hooking - and until now
            # nothing did: MQTTPublisher.publish_event() was fully implemented,
            # mqtt.enabled was true in config, and no caller existed. Every
            # event Argus ever produced was recorded and delivered nowhere.
            #
            # notify() applies policy, is non-blocking, and never raises, so a
            # broken transport cannot stop an event being persisted. Recording
            # the event is the guarantee; delivering it is best-effort and
            # separately reported.
            try:
                from backend.services.management.notifications import (
                    get_notification_service,
                )

                get_notification_service().notify(event)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Alert dispatch failed for event {event_id}: {exc}")

            return event
        except Exception as e:
            logger.error(f"Error creating event: {e}")
            raise

    def get_event(self, event_id: int) -> Optional[Dict]:
        """Get event by ID"""
        row = self.db.fetchone("SELECT * FROM events WHERE id = ?", (event_id,))
        if row:
            event = dict(row)
            if event['bbox']:
                event['bbox'] = json.loads(event['bbox'])
            if event['metadata']:
                event['metadata'] = json.loads(event['metadata'])
            return event
        return None

    def query_events(
        self,
        camera_id: int = None,
        from_time: datetime = None,
        to_time: datetime = None,
        rule_type: str = None,
        priority: str = None,
        status: str = None,
        limit: int = 100,
        offset: int = 0
    ) -> tuple[List[Dict], int]:
        """Query events with filters"""
        query = "SELECT * FROM events WHERE 1=1"
        count_query = "SELECT COUNT(*) as count FROM events WHERE 1=1"
        params = []
        
        if camera_id is not None:
            query += " AND camera_id = ?"
            count_query += " AND camera_id = ?"
            params.append(camera_id)
        
        if from_time:
            query += " AND timestamp >= ?"
            count_query += " AND timestamp >= ?"
            params.append(from_time)
        
        if to_time:
            query += " AND timestamp <= ?"
            count_query += " AND timestamp <= ?"
            params.append(to_time)
        
        if rule_type:
            query += " AND rule_type = ?"
            count_query += " AND rule_type = ?"
            params.append(rule_type)
        
        if priority:
            query += " AND priority = ?"
            count_query += " AND priority = ?"
            params.append(priority)
        
        if status:
            query += " AND status = ?"
            count_query += " AND status = ?"
            params.append(status)
        
        # Get total count
        count_row = self.db.fetchone(count_query, tuple(params))
        total = count_row['count'] if count_row else 0
        
        # Get paginated results
        query += " ORDER BY timestamp DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])
        
        rows = self.db.fetchall(query, tuple(params))
        events = []
        for row in rows:
            event = dict(row)
            if event['bbox']:
                event['bbox'] = json.loads(event['bbox'])
            if event['metadata']:
                event['metadata'] = json.loads(event['metadata'])
            events.append(event)
        
        return events, total

    # Event lifecycle. An event moves DETECTED -> OPEN -> ACKNOWLEDGED ->
    # RESOLVED, and may be dismissed as a false positive from any live state.
    # The transition table is enforced rather than advisory: update_event_status
    # previously wrote whatever string it was handed, so a typo silently created
    # a new status that no filter would ever match again, and an event could
    # jump straight from DETECTED to RESOLVED with no one having looked at it.
    STATUS_DETECTED = "detected"
    STATUS_OPEN = "open"
    STATUS_ACKNOWLEDGED = "acknowledged"
    STATUS_RESOLVED = "resolved"
    STATUS_FALSE_POSITIVE = "false_positive"

    ALLOWED_TRANSITIONS = {
        STATUS_DETECTED: {STATUS_OPEN, STATUS_ACKNOWLEDGED, STATUS_FALSE_POSITIVE},
        STATUS_OPEN: {STATUS_ACKNOWLEDGED, STATUS_FALSE_POSITIVE},
        STATUS_ACKNOWLEDGED: {STATUS_RESOLVED, STATUS_FALSE_POSITIVE},
        # Terminal states.
        STATUS_RESOLVED: set(),
        STATUS_FALSE_POSITIVE: set(),
    }

    def update_event_status(self, event_id: int, status: str,
                            actor: Optional[str] = None) -> Optional[Dict]:
        """Move an event to `status`, enforcing the lifecycle.

        Raises ValueError on an unknown status or an illegal transition, so a
        bad call fails loudly instead of corrupting the feed's state machine.
        """
        status = (status or "").strip().lower()
        if status not in self.ALLOWED_TRANSITIONS:
            raise ValueError(
                f"unknown event status {status!r}; "
                f"valid: {sorted(self.ALLOWED_TRANSITIONS)}"
            )

        current = self.get_event(event_id)
        if current is None:
            return None

        present = (current.get("status") or self.STATUS_DETECTED).strip().lower()
        if present not in self.ALLOWED_TRANSITIONS:
            # Legacy rows may hold a status from before the lifecycle existed.
            present = self.STATUS_DETECTED
        if status != present and status not in self.ALLOWED_TRANSITIONS[present]:
            raise ValueError(
                f"illegal transition {present!r} -> {status!r}; "
                f"allowed from {present!r}: "
                f"{sorted(self.ALLOWED_TRANSITIONS[present]) or 'none (terminal)'}"
            )

        now = datetime.now()
        if status == self.STATUS_ACKNOWLEDGED:
            self.db.execute(
                "UPDATE events SET status = ?, acknowledged_by = ?, "
                "acknowledged_at = ? WHERE id = ?",
                (status, actor, now, event_id),
            )
        elif status == self.STATUS_RESOLVED:
            self.db.execute(
                "UPDATE events SET status = ?, resolved_by = ?, "
                "resolved_at = ? WHERE id = ?",
                (status, actor, now, event_id),
            )
        else:
            self.db.execute(
                "UPDATE events SET status = ? WHERE id = ?", (status, event_id)
            )
        logger.info(
            f"Event {event_id}: {present} -> {status}"
            + (f" by {actor}" if actor else "")
        )
        return self.get_event(event_id)

    def delete_old_events(self, retention_days: int = 30) -> int:
        """Delete events older than retention period"""
        cutoff_date = datetime.now() - timedelta(days=retention_days)
        cursor = self.db.execute(
            "DELETE FROM events WHERE created_at < ?",
            (cutoff_date,)
        )
        deleted_count = cursor.rowcount
        logger.info(f"Deleted {deleted_count} events older than {retention_days} days")
        return deleted_count

    def get_event_stats(self, camera_id: int = None, hours: int = 24) -> Dict:
        """Get event statistics"""
        from_time = datetime.now() - timedelta(hours=hours)
        
        query = """
            SELECT 
                rule_type,
                priority,
                COUNT(*) as count
            FROM events
            WHERE timestamp >= ?
        """
        params = [from_time]
        
        if camera_id:
            query += " AND camera_id = ?"
            params.append(camera_id)
        
        query += " GROUP BY rule_type, priority"
        
        rows = self.db.fetchall(query, tuple(params))
        
        stats = {
            'total': 0,
            'by_rule': {},
            'by_priority': {}
        }
        
        for row in rows:
            count = row['count']
            stats['total'] += count
            
            rule = row['rule_type']
            if rule not in stats['by_rule']:
                stats['by_rule'][rule] = 0
            stats['by_rule'][rule] += count
            
            priority = row['priority']
            if priority not in stats['by_priority']:
                stats['by_priority'][priority] = 0
            stats['by_priority'][priority] += count
        
        return stats


# Global event store instance
_event_store = None


def get_event_store() -> EventStore:
    """Get global event store instance"""
    global _event_store
    if _event_store is None:
        _event_store = EventStore()
    return _event_store
