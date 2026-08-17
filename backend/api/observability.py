"""
Observability: Prometheus metrics and structured logging.

Argus previously exposed only a JSON /health and /metrics pair intended for the
dashboard. Neither is scrapeable by standard monitoring, and plain-text logs
cannot be filtered by camera or correlated across subsystems.

This module adds:
  * GET /metrics  - Prometheus text exposition (no extra dependency)
  * JSON logging  - opt-in via ARGUS_LOG_FORMAT=json

The Prometheus client library is deliberately not a dependency: the exposition
format is simple, and Argus already has every number in memory. Adding
prometheus_client can come later without changing the endpoint contract.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

_START_TIME = time.time()


# ── Prometheus exposition ────────────────────────────────────────────────────

def _escape_label(value: str) -> str:
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _fmt(
    name: str,
    value: float,
    labels: Optional[Dict[str, object]] = None,
) -> str:
    if labels:
        rendered = ",".join(f'{k}="{_escape_label(v)}"' for k, v in labels.items())
        return f"{name}{{{rendered}}} {value}"
    return f"{name} {value}"


class MetricsRegistry:
    """Builds the Prometheus text exposition from live subsystem state."""

    def collect(self) -> str:
        lines: List[str] = []

        def add(name: str, help_text: str, metric_type: str, samples: Iterable[str]):
            samples = list(samples)
            if not samples:
                return
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {metric_type}")
            lines.extend(samples)

        # ── Process ──
        add("argus_uptime_seconds", "Seconds since API start", "gauge",
            [_fmt("argus_uptime_seconds", round(time.time() - _START_TIME, 1))])

        try:
            import psutil
            proc = psutil.Process()
            add("argus_process_cpu_percent", "Process CPU usage", "gauge",
                [_fmt("argus_process_cpu_percent", proc.cpu_percent(None))])
            add("argus_process_memory_bytes", "Process resident memory", "gauge",
                [_fmt("argus_process_memory_bytes", proc.memory_info().rss)])
        except Exception:  # noqa: BLE001 - metrics must never break the endpoint
            pass

        # ── Cameras ──
        try:
            from backend.services.management.camera_manager import get_camera_manager
            from backend.services.management.stream_ingestion import get_stream_ingestion

            cameras = get_camera_manager().get_all_cameras()
            ingestion = get_stream_ingestion()

            fps_samples, up_samples, queue_samples = [], [], []
            status_counts: Dict[str, int] = {}

            for cam in cameras:
                labels = {"camera_id": cam["id"], "name": cam.get("name", "")}
                fps_samples.append(_fmt("argus_camera_fps", cam.get("fps") or 0.0, labels))
                up_samples.append(
                    _fmt("argus_camera_up", 1 if cam.get("status") == "online" else 0, labels)
                )
                status = cam.get("status", "unknown")
                status_counts[status] = status_counts.get(status, 0) + 1
                try:
                    depth = ingestion.get_queue_depth(cam["id"])
                    queue_samples.append(_fmt("argus_camera_queue_depth", depth, labels))
                except Exception:  # noqa: BLE001
                    pass

            add("argus_camera_fps", "Frames per second per camera", "gauge", fps_samples)
            add("argus_camera_up", "1 when the camera is online", "gauge", up_samples)
            add("argus_camera_queue_depth", "Pending frames in the capture queue",
                "gauge", queue_samples)
            add("argus_cameras_total", "Cameras by status", "gauge",
                [_fmt("argus_cameras_total", n, {"status": s}) for s, n in status_counts.items()])
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Camera metrics unavailable: {exc}")

        # ── Inference ──
        try:
            from backend.services.core_engine.inference_engine import get_inference_engine
            engine = get_inference_engine()
            add("argus_inference_latency_ms", "Mean inference latency", "gauge",
                [_fmt("argus_inference_latency_ms", round(engine.get_avg_inference_time(), 2))])
            add("argus_model_loaded", "1 when the detection model is loaded", "gauge",
                [_fmt("argus_model_loaded", 1 if engine.is_model_loaded() else 0)])
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Inference metrics unavailable: {exc}")

        # ── Detections / tracking ──
        try:
            from backend.services.core_engine.processing_coordinator import (
                get_processing_coordinator,
            )
            coordinator = get_processing_coordinator()
            det_samples = []
            for camera_id, status in coordinator.get_processing_status().items():
                analysis = status.get("latest_analysis", {})
                det_samples.append(_fmt(
                    "argus_detections_current", analysis.get("detection_count", 0),
                    {"camera_id": camera_id},
                ))
            add("argus_detections_current", "Detections in the latest analysed frame",
                "gauge", det_samples)
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Detection metrics unavailable: {exc}")

        try:
            from backend.services.core_engine.deep_tracker import get_deep_tracker
            tracker = get_deep_tracker()
            add("argus_active_tracks", "Currently tracked objects", "gauge",
                [_fmt("argus_active_tracks", len(getattr(tracker, "tracks", {})))])
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Tracker metrics unavailable: {exc}")

        # ── Events ──
        try:
            from backend.services.management.event_store import get_event_store
            stats = get_event_store().get_event_stats(hours=24)
            add("argus_events_24h_total", "Events in the last 24h by rule", "gauge",
                [_fmt("argus_events_24h_total", n, {"rule": r})
                 for r, n in (stats.get("by_rule") or {}).items()])
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"Event metrics unavailable: {exc}")

        # ── Security posture ──
        try:
            from backend.api.auth import auth_enabled, EPHEMERAL_SECRET_IN_USE
            add("argus_auth_enabled", "1 when API authentication is enforced", "gauge",
                [_fmt("argus_auth_enabled", 1 if auth_enabled() else 0)])
            add("argus_ephemeral_jwt_secret",
                "1 when signing with a throwaway key (tokens die on restart)", "gauge",
                [_fmt("argus_ephemeral_jwt_secret", 1 if EPHEMERAL_SECRET_IN_USE else 0)])
        except Exception:  # noqa: BLE001
            pass

        return "\n".join(lines) + "\n"


_registry: Optional[MetricsRegistry] = None


def get_metrics_registry() -> MetricsRegistry:
    global _registry
    if _registry is None:
        _registry = MetricsRegistry()
    return _registry


# ── Structured logging ───────────────────────────────────────────────────────

class JsonLogFormatter(logging.Formatter):
    """
    Emit one JSON object per log record.

    Plain-text logs cannot be filtered by camera or joined across subsystems.
    Any extra=... fields supplied by the caller are merged into the object.
    """

    _RESERVED = {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "getMessage", "message", "asctime",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "service": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in self._RESERVED and not key.startswith("_"):
                try:
                    json.dumps(value)
                    payload[key] = value
                except (TypeError, ValueError):
                    payload[key] = str(value)

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload)


def configure_logging() -> None:
    """
    Apply the log format selected by ARGUS_LOG_FORMAT (``json`` or ``text``).

    Defaults to text so local development stays readable; production
    deployments set json for ingestion by Loki/Elasticsearch/CloudWatch.
    """
    log_format = os.environ.get("ARGUS_LOG_FORMAT", "text").strip().lower()
    level_name = os.environ.get("ARGUS_LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    root.setLevel(level)

    if log_format == "json":
        handler = logging.StreamHandler()
        handler.setFormatter(JsonLogFormatter())
        root.handlers = [handler]
        logger.info("Structured JSON logging enabled")
