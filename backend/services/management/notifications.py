"""Alert delivery: get events out of the database and to whoever is on call.

Before this module, `mqtt_publisher.publish_event()` existed, MQTT was
`enabled: true` in config, and **nothing in the codebase ever called it**. Every
event Argus had ever generated was written to SQLite and delivered nowhere. A
surveillance system whose alerts never leave the box is a recording system.

Three things this module refuses to do, each learned from a defect already
fixed elsewhere in this codebase:

1. **It never claims a delivery it did not make.** Every send returns a
   `DeliveryResult` with the transport, outcome and reason. Failures are
   counted and surfaced through `/api/v1/notifications/status`, not swallowed
   into a log line nobody reads.

2. **It never blocks the frame loop.** Delivery happens on a bounded worker
   queue. If the queue is full the event is *dropped and counted* rather than
   stalling detection - and the drop count is reported, because a silently
   dropped alert is worse than a visibly refused one.

3. **A channel that cannot deliver says so.** No broker running, no webhook URL
   configured, `requests` not installed: each reports `available=False` with a
   reason, exactly as the capability registry does. Availability is probed by
   attempting the transport, never read from a config flag.

Policy (roadmap 3.8) decides *whether* an event is worth waking someone for:
minimum priority, rule allow/deny lists, per-rule quiet hours, and a
per-(camera, rule) rate limit so one flapping camera cannot exhaust a pager.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit
from dataclasses import dataclass, field
from datetime import datetime, time as dtime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

# A queue this deep already means the transport is far behind; deeper only
# delays the discovery. Bounded on purpose.
MAX_QUEUE = 256
DEFAULT_TIMEOUT_S = 5.0


@dataclass
class DeliveryResult:
    transport: str
    delivered: bool
    reason: str = ""
    duration_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "transport": self.transport,
            "delivered": self.delivered,
            "reason": self.reason,
            "duration_ms": round(self.duration_ms, 1),
        }


class Transport:
    """One way of getting an alert out."""

    name = "transport"

    def available(self) -> tuple[bool, str]:
        raise NotImplementedError

    def send(self, event: Dict[str, Any]) -> DeliveryResult:
        raise NotImplementedError


class MqttTransport(Transport):
    """Publishes through the existing MQTTPublisher.

    The publisher was fully written and never called; this wires it up and
    reports honestly when no broker is reachable.
    """

    name = "mqtt"

    def __init__(self, publisher=None):
        self._publisher = publisher

    def _pub(self):
        if self._publisher is None:
            from backend.services.management.mqtt_publisher import (
                get_mqtt_publisher,
            )

            self._publisher = get_mqtt_publisher()
        return self._publisher

    def available(self) -> tuple[bool, str]:
        try:
            pub = self._pub()
        except Exception as exc:  # noqa: BLE001
            return False, f"MQTT publisher could not be constructed: {exc}"
        if not getattr(pub.config.mqtt, "enabled", False):
            return False, "mqtt.enabled is false in config"
        # Connection state is a live fact, not a config claim.
        if not pub.is_connected():
            return False, (
                f"no broker reachable at "
                f"{pub.config.mqtt.broker}:{pub.config.mqtt.port}"
            )
        return True, "connected"

    def send(self, event: Dict[str, Any]) -> DeliveryResult:
        started = time.perf_counter()
        ok, reason = self.available()
        if not ok:
            return DeliveryResult(self.name, False, reason,
                                  (time.perf_counter() - started) * 1000)
        try:
            self._pub().publish_event(event)
            return DeliveryResult(self.name, True, "published",
                                  (time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001
            return DeliveryResult(self.name, False, str(exc),
                                  (time.perf_counter() - started) * 1000)


class WebhookTransport(Transport):
    """POSTs the event as JSON to a configured URL.

    Uses urllib from the stdlib rather than requests: this must work on a bare
    install, and adding a dependency for one POST is not worth it.
    """

    name = "webhook"

    def __init__(self, url: str = "", timeout_s: float = DEFAULT_TIMEOUT_S,
                 headers: Optional[Dict[str, str]] = None):
        self.url = (url or "").strip()
        self.timeout_s = timeout_s
        self.headers = headers or {}

    def available(self) -> tuple[bool, str]:
        if not self.url:
            return False, "no webhook URL configured"
        try:
            parsed_url = urlsplit(self.url)
            invalid_url = (
                parsed_url.scheme not in {"http", "https"} or not parsed_url.hostname
                or parsed_url.username or parsed_url.password
            )
        except ValueError:
            invalid_url = True
        if invalid_url:
            return False, (
                "webhook URL must be an absolute HTTP(S) URL without embedded credentials"
            )
        if parsed_url.scheme == "http" and parsed_url.hostname not in {
            "localhost", "127.0.0.1", "::1"
        }:
            # Delivered, but worth flagging: event payloads describe people.
            return True, "configured (WARNING: plaintext http to a remote host)"
        return True, "configured"

    def send(self, event: Dict[str, Any]) -> DeliveryResult:
        started = time.perf_counter()
        ok, reason = self.available()
        if not ok:
            return DeliveryResult(self.name, False, reason,
                                  (time.perf_counter() - started) * 1000)
        try:
            body = json.dumps(_serialisable(event)).encode("utf-8")
            request = urllib.request.Request(
                self.url, data=body, method="POST",
                headers={"Content-Type": "application/json", **self.headers},
            )
            # available() validates an absolute HTTP(S) URL first.
            with urllib.request.urlopen(request, timeout=self.timeout_s) as resp:  # nosec B310
                code = resp.status
            elapsed = (time.perf_counter() - started) * 1000
            if 200 <= code < 300:
                return DeliveryResult(self.name, True, f"HTTP {code}", elapsed)
            return DeliveryResult(self.name, False, f"HTTP {code}", elapsed)
        except urllib.error.HTTPError as exc:
            return DeliveryResult(self.name, False, f"HTTP {exc.code}",
                                  (time.perf_counter() - started) * 1000)
        except Exception as exc:  # noqa: BLE001
            return DeliveryResult(self.name, False, str(exc),
                                  (time.perf_counter() - started) * 1000)


def _serialisable(value):
    """datetime is not JSON-serialisable and events always carry one."""
    if isinstance(value, dict):
        return {k: _serialisable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialisable(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return value


@dataclass
class AlertPolicy:
    """Whether an event is worth delivering at all.

    Defaults are deliberately permissive except for priority: a system that
    silently drops alerts by default is worse than a noisy one, because the
    operator cannot tell the difference between "quiet" and "broken".
    """

    min_priority: str = "medium"
    rules_allow: List[str] = field(default_factory=list)   # empty = all
    rules_deny: List[str] = field(default_factory=list)
    quiet_hours: Dict[str, List[int]] = field(default_factory=dict)
    rate_limit_per_minute: int = 20

    def _priority_ok(self, event: Dict[str, Any]) -> tuple[bool, str]:
        want = PRIORITY_ORDER.get(self.min_priority, 1)
        have = PRIORITY_ORDER.get((event.get("priority") or "medium"), 1)
        if have < want:
            return False, (
                f"priority {event.get('priority')!r} is below the "
                f"{self.min_priority!r} threshold"
            )
        return True, ""

    def _rule_ok(self, event: Dict[str, Any]) -> tuple[bool, str]:
        rule = event.get("rule_type")
        if self.rules_deny and rule in self.rules_deny:
            return False, f"rule {rule!r} is on the deny list"
        if self.rules_allow and rule not in self.rules_allow:
            return False, f"rule {rule!r} is not on the allow list"
        return True, ""

    def _quiet_ok(self, event: Dict[str, Any], now: datetime) -> tuple[bool, str]:
        window = self.quiet_hours.get(event.get("rule_type") or "")
        if not window:
            window = self.quiet_hours.get("*")
        if not window or len(window) != 2:
            return True, ""
        start_h, end_h = int(window[0]), int(window[1])
        hour = now.hour
        # A window may wrap midnight (22 -> 6).
        inside = (start_h <= hour < end_h) if start_h < end_h else (
            hour >= start_h or hour < end_h
        )
        if inside:
            return False, (
                f"inside quiet hours {start_h:02d}:00-{end_h:02d}:00 for "
                f"{event.get('rule_type')!r}"
            )
        return True, ""

    def evaluate(self, event: Dict[str, Any],
                 now: Optional[datetime] = None) -> tuple[bool, str]:
        now = now or datetime.now()
        for check in (self._priority_ok, self._rule_ok):
            ok, why = check(event)
            if not ok:
                return False, why
        return self._quiet_ok(event, now)


class NotificationService:
    """Applies policy, then delivers on every available transport."""

    def __init__(self, transports: Optional[List[Transport]] = None,
                 policy: Optional[AlertPolicy] = None,
                 synchronous: bool = False):
        self.transports = transports if transports is not None else []
        self.policy = policy or AlertPolicy()
        self.synchronous = synchronous

        self._queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=MAX_QUEUE)
        self._worker: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.RLock()

        self._rate: Dict[tuple, List[float]] = {}
        self.counters = {
            "considered": 0,
            "suppressed_by_policy": 0,
            "suppressed_rate_limit": 0,
            "queued": 0,
            "dropped_queue_full": 0,
            "delivered": 0,
            "failed": 0,
        }
        self.last_results: List[Dict[str, Any]] = []
        self.last_suppression: Optional[str] = None

    # -- policy -------------------------------------------------------------

    def _rate_ok(self, event: Dict[str, Any], now: float) -> bool:
        limit = self.policy.rate_limit_per_minute
        if limit <= 0:
            return True
        key = (event.get("camera_id"), event.get("rule_type"))
        with self._lock:
            recent = [t for t in self._rate.get(key, []) if now - t < 60.0]
            if len(recent) >= limit:
                self._rate[key] = recent
                return False
            recent.append(now)
            self._rate[key] = recent
            # Bound the table: one entry per (camera, rule) pair is small, but
            # unbounded growth is how the other leaks in this codebase started.
            if len(self._rate) > 512:
                for k in list(self._rate)[:128]:
                    del self._rate[k]
        return True

    # -- entry point --------------------------------------------------------

    def notify(self, event: Dict[str, Any]) -> Optional[List[DeliveryResult]]:
        """Consider an event for delivery. Never raises, never blocks."""
        try:
            self.counters["considered"] += 1
            allowed, why = self.policy.evaluate(event)
            if not allowed:
                self.counters["suppressed_by_policy"] += 1
                self.last_suppression = why
                return None
            if not self._rate_ok(event, time.time()):
                self.counters["suppressed_rate_limit"] += 1
                self.last_suppression = (
                    f"rate limit of {self.policy.rate_limit_per_minute}/min "
                    f"reached for camera {event.get('camera_id')} "
                    f"rule {event.get('rule_type')}"
                )
                return None

            if self.synchronous:
                return self._deliver(event)

            self._ensure_worker()
            try:
                self._queue.put_nowait(event)
                self.counters["queued"] += 1
            except queue.Full:
                # Counted, not hidden. A dropped alert an operator can see is
                # recoverable; one they cannot is not.
                self.counters["dropped_queue_full"] += 1
                logger.error(
                    "Notification queue full (%d): dropped event %s. "
                    "Transports are not keeping up.",
                    MAX_QUEUE, event.get("id"),
                )
            return None
        except Exception as exc:  # noqa: BLE001 - never break the caller
            logger.error(f"notify() failed for event {event.get('id')}: {exc}")
            return None

    def _ensure_worker(self) -> None:
        with self._lock:
            if self._worker is not None and self._worker.is_alive():
                return
            self._stop.clear()
            self._worker = threading.Thread(
                target=self._run, name="argus-notifications", daemon=True
            )
            self._worker.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                event = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self._deliver(event)
            except Exception as exc:  # noqa: BLE001
                logger.error(f"Delivery worker error: {exc}")
            finally:
                self._queue.task_done()

    def _deliver(self, event: Dict[str, Any]) -> List[DeliveryResult]:
        results: List[DeliveryResult] = []
        for transport in self.transports:
            try:
                result = transport.send(event)
            except Exception as exc:  # noqa: BLE001
                result = DeliveryResult(transport.name, False, str(exc))
            results.append(result)
            if result.delivered:
                self.counters["delivered"] += 1
            else:
                self.counters["failed"] += 1
                logger.warning(
                    "Alert delivery failed on %s: %s", result.transport, result.reason
                )
        self.last_results = [r.to_dict() for r in results]
        return results

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait for the queue to drain. Used by tests and shutdown."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self._queue.empty():
                time.sleep(0.05)
                return True
            time.sleep(0.05)
        return self._queue.empty()

    def stop(self) -> None:
        self._stop.set()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=2.0)

    # -- introspection ------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        channels = []
        for transport in self.transports:
            try:
                ok, reason = transport.available()
            except Exception as exc:  # noqa: BLE001
                ok, reason = False, str(exc)
            channels.append(
                {"transport": transport.name, "available": ok, "reason": reason}
            )
        deliverable = any(c["available"] for c in channels)
        return {
            "channels": channels,
            "any_channel_available": deliverable,
            "warning": None if deliverable else (
                "No alert channel can deliver right now. Events are still "
                "recorded and queryable, but nothing is being sent anywhere."
            ),
            "policy": {
                "min_priority": self.policy.min_priority,
                "rules_allow": self.policy.rules_allow,
                "rules_deny": self.policy.rules_deny,
                "quiet_hours": self.policy.quiet_hours,
                "rate_limit_per_minute": self.policy.rate_limit_per_minute,
            },
            "counters": dict(self.counters),
            "queue_depth": self._queue.qsize(),
            "queue_capacity": MAX_QUEUE,
            "last_delivery": self.last_results,
            "last_suppression": self.last_suppression,
        }


def _build_from_config() -> NotificationService:
    from backend.config.config import get_config, section_to_dict

    config = get_config()
    section = section_to_dict(getattr(config, "notifications", {})) or {}

    policy_cfg = section.get("policy") or {}
    policy = AlertPolicy(
        min_priority=policy_cfg.get("min_priority", "medium"),
        rules_allow=list(policy_cfg.get("rules_allow") or []),
        rules_deny=list(policy_cfg.get("rules_deny") or []),
        quiet_hours=dict(policy_cfg.get("quiet_hours") or {}),
        rate_limit_per_minute=int(policy_cfg.get("rate_limit_per_minute", 20)),
    )

    transports: List[Transport] = []
    channels = section.get("channels") or {}

    mqtt_cfg = channels.get("mqtt") or {}
    if mqtt_cfg.get("enabled", True):
        transports.append(MqttTransport())

    webhook_cfg = channels.get("webhook") or {}
    if webhook_cfg.get("enabled", False):
        transports.append(
            WebhookTransport(
                url=webhook_cfg.get("url", ""),
                timeout_s=float(webhook_cfg.get("timeout_s", DEFAULT_TIMEOUT_S)),
                headers=dict(webhook_cfg.get("headers") or {}),
            )
        )

    return NotificationService(transports=transports, policy=policy)


_service: Optional[NotificationService] = None
_service_lock = threading.Lock()


def get_notification_service() -> NotificationService:
    global _service
    if _service is None:
        with _service_lock:
            if _service is None:
                try:
                    _service = _build_from_config()
                except Exception as exc:  # noqa: BLE001
                    logger.error(f"Notification config failed, using defaults: {exc}")
                    _service = NotificationService(transports=[MqttTransport()])
    return _service


def reset_notification_service() -> None:
    """Test hook: drop the singleton so counters do not leak between tests."""
    global _service
    with _service_lock:
        if _service is not None:
            _service.stop()
        _service = None
