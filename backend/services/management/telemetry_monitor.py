"""
Real-Time Hardware Telemetry Monitor

Provides live CPU, GPU, and RAM metrics to the Consortium Broker for
Dynamic Stress Multiplier computation. Transforms bidding from predictive
into a closed-loop reactive system.

Features:
  - CPU/RAM: psutil (always available)
  - GPU: pynvml or nvidia-smi subprocess (safe fallback)
  - Zero-cost fallback: synthetic simulation when hardware APIs unavailable
  - Thread-safe snapshot cache with configurable TTL
  - Quadratic Dynamic Stress Multiplier curve

Lifecycle:
    __init__(): Load config, init psutil, attempt GPU binding
    start():    Start background polling thread
    stop():     Stop polling thread
    get_snapshot(): Get latest cached HardwareSnapshot
"""

import logging
import math
import threading
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Data Structures
# =============================================================================


@dataclass
class HardwareSnapshot:
    """Immutable, timestamped snapshot of all hardware metrics."""

    timestamp: float
    cpu_utilization_percent: float
    cpu_count: int
    system_ram_percent: float
    system_ram_used_mb: float
    system_ram_total_mb: float
    gpu_available: bool
    gpu_utilization_percent: float
    vram_used_mb: float
    vram_total_mb: float
    gpu_temperature_c: float
    stress_multiplier: float

    def to_dict(self) -> Dict:
        """Serialize to dictionary for API responses."""
        return {
            "timestamp": self.timestamp,
            "cpu_utilization_percent": round(self.cpu_utilization_percent, 1),
            "cpu_count": self.cpu_count,
            "system_ram_percent": round(self.system_ram_percent, 1),
            "system_ram_used_mb": round(self.system_ram_used_mb, 1),
            "system_ram_total_mb": round(self.system_ram_total_mb, 1),
            "gpu_available": self.gpu_available,
            "gpu_utilization_percent": round(self.gpu_utilization_percent, 1),
            "vram_used_mb": round(self.vram_used_mb, 1),
            "vram_total_mb": round(self.vram_total_mb, 1),
            "gpu_temperature_c": round(self.gpu_temperature_c, 1),
            "stress_multiplier": round(self.stress_multiplier, 3),
        }


# =============================================================================
# TelemetryMonitor
# =============================================================================


class TelemetryMonitor:
    """
    Real-time hardware telemetry with zero-cost fallback.

    - CPU/RAM: psutil (always available, no extra deps)
    - GPU: pynvml (NVIDIA) or nvidia-smi subprocess (safe fallback)
    - If all GPU detection fails: gpu_available=False, no GPU throttling
    - Thread-safe RLock-protected snapshot cache
    - Quadratic Dynamic Stress Multiplier curve computed from thresholds
    """

    def __init__(self):
        # Load config lazily (avoids circular imports)
        self._config = None
        self._telemetry_config = None
        self._load_config()

        self.enabled = self._telemetry_config.enabled if self._telemetry_config else False
        self.sampling_interval = (
            (self._telemetry_config.sampling_interval_ms / 1000.0)
            if self._telemetry_config
            else 0.5
        )
        self.cache_ttl = self._telemetry_config.cache_ttl_seconds if self._telemetry_config else 1.0

        # Thread-safe cache
        self._lock = threading.RLock()
        self._latest_snapshot: Optional[HardwareSnapshot] = None
        self._polling_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # psutil reference (loaded locally for zero-leak safety)
        self._psutil = None
        self._psutil_available = False
        self._init_psutil()

        # GPU state
        self._pynvml = None
        self._pynvml_available = False
        self._nvidia_smi_available = False
        self._gpu_available = False
        self._init_gpu()

        # Create initial zero-stress snapshot
        self._latest_snapshot = self._create_synthetic_snapshot(
            cpu=0.0, gpu=0.0, ram=0.0, vram=0.0
        )

        logger.info(
            f"TelemetryMonitor initialized "
            f"(enabled={self.enabled}, psutil={self._psutil_available}, "
            f"gpu={self._gpu_available})"
        )

    def _load_config(self):
        """Load configuration — wrapped in try/except for import safety."""
        try:
            from ...config.config import get_config

            self._config = get_config()
            self._telemetry_config = getattr(self._config, "telemetry", None)
        except Exception as e:
            # Fallback to absolute import
            try:
                from backend.config.config import get_config
                self._config = get_config()
                self._telemetry_config = getattr(self._config, "telemetry", None)
            except Exception as e2:
                logger.warning(f"Could not load telemetry config: {e2}")
                self._telemetry_config = None

    def _init_psutil(self):
        """Initialize psutil — local import for zero-leak safety."""
        try:
            import psutil as _psutil

            self._psutil = _psutil
            self._psutil_available = True
            logger.debug("psutil loaded successfully for CPU/RAM telemetry")
        except ImportError:
            logger.warning("psutil not available — CPU/RAM telemetry disabled (synthetic fallback)")
            self._psutil_available = False

    def _init_gpu(self):
        """Initialize GPU monitoring — local imports, never at module top level."""
        # Attempt 1: pynvml
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._pynvml_available = True
            self._gpu_available = True
            logger.debug("pynvml loaded successfully for GPU telemetry")
            return
        except ImportError:
            logger.debug("pynvml not available — trying nvidia-smi")
        except Exception as e:
            logger.debug(f"pynvml init failed: {e}")

        # Attempt 2: nvidia-smi subprocess
        try:
            import subprocess
            import shutil

            if shutil.which("nvidia-smi"):
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5.0
                )
                if result.returncode == 0 and result.stdout.strip():
                    self._nvidia_smi_available = True
                    self._gpu_available = True
                    logger.debug("nvidia-smi subprocess available for GPU telemetry")
                    return
        except Exception as e:
            logger.debug(f"nvidia-smi subprocess failed: {e}")

        # Fallback: no GPU available
        self._gpu_available = False
        logger.info("No GPU monitoring available — synthetic GPU metrics will be used")

    # ── Lifecycle ───────────────────────────────────────────────────────

    async def start(self):
        """Start background hardware polling thread."""
        if not self.enabled:
            logger.info("TelemetryMonitor is disabled (zero-cost fallback active)")
            return

        if self._polling_thread and self._polling_thread.is_alive():
            logger.warning("TelemetryMonitor already running")
            return

        self._stop_event.clear()
        self._polling_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name="telemetry-poll",
        )
        self._polling_thread.start()
        logger.info(f"TelemetryMonitor started (interval={self.sampling_interval:.2f}s)")

    async def stop(self):
        """Stop background polling thread."""
        self._stop_event.set()
        if self._polling_thread:
            self._polling_thread.join(timeout=3.0)
            self._polling_thread = None
        logger.info("TelemetryMonitor stopped")

    # ── Public API ──────────────────────────────────────────────────────

    def get_snapshot(self) -> HardwareSnapshot:
        """
        Get the latest cached hardware snapshot.

        If cache is missing or expired, returns a synthetic zero-stress snapshot
        (safe fallback — no crash, no blocking).
        """
        with self._lock:
            snapshot = self._latest_snapshot
            if snapshot is None:
                return self._create_synthetic_snapshot(cpu=0.0, gpu=0.0, ram=0.0, vram=0.0)
            # Check cache TTL
            if time.time() - snapshot.timestamp > self.cache_ttl:
                return self._create_synthetic_snapshot(cpu=0.0, gpu=0.0, ram=0.0, vram=0.0)
            return snapshot

    def get_stress_multiplier(self) -> float:
        """Get current Dynamic Stress Multiplier (0.3-1.0)."""
        return self.get_snapshot().stress_multiplier

    def get_status(self) -> Dict:
        """Get full telemetry status for API/monitoring."""
        snapshot = self.get_snapshot()
        return {
            "enabled": self.enabled,
            "psutil_available": self._psutil_available,
            "gpu_available": self._gpu_available,
            "polling_active": self._polling_thread is not None and self._polling_thread.is_alive(),
            "last_snapshot": snapshot.to_dict(),
        }

    # ── Internal Polling ────────────────────────────────────────────────

    def _poll_loop(self):
        """Background polling loop."""
        while not self._stop_event.is_set():
            try:
                cpu_percent, cpu_count = self._read_cpu()
                ram_percent, ram_used_mb, ram_total_mb = self._read_ram()
                gpu_util, vram_used, vram_total, gpu_temp = self._read_gpu()

                stress = self._compute_stress_multiplier(
                    cpu=cpu_percent / 100.0,
                    gpu=gpu_util / 100.0 if self._gpu_available else 0.0,
                    ram=ram_percent / 100.0,
                    vram=vram_used / max(vram_total, 1) if self._gpu_available and vram_total > 0 else 0.0,
                )

                snapshot = HardwareSnapshot(
                    timestamp=time.time(),
                    cpu_utilization_percent=cpu_percent,
                    cpu_count=cpu_count,
                    system_ram_percent=ram_percent,
                    system_ram_used_mb=ram_used_mb,
                    system_ram_total_mb=ram_total_mb,
                    gpu_available=self._gpu_available,
                    gpu_utilization_percent=gpu_util,
                    vram_used_mb=vram_used,
                    vram_total_mb=vram_total,
                    gpu_temperature_c=gpu_temp,
                    stress_multiplier=stress,
                )

                with self._lock:
                    self._latest_snapshot = snapshot

            except Exception as e:
                logger.warning(f"Telemetry poll error: {e}")
                # Create zero-stress fallback snapshot so the broker never stalls
                fallback = self._create_synthetic_snapshot(cpu=0.0, gpu=0.0, ram=0.0, vram=0.0)
                with self._lock:
                    self._latest_snapshot = fallback

            # Sleep for sampling interval (check stop event periodically)
            for _ in range(int(self.sampling_interval * 10)):
                if self._stop_event.is_set():
                    return
                time.sleep(0.1)

    def _read_cpu(self) -> tuple:
        """Read CPU utilization percent and count."""
        if not self._psutil_available or self._psutil is None:
            return 0.0, 1

        try:
            cpu_percent = self._psutil.cpu_percent(interval=None)
            cpu_count = self._psutil.cpu_count()
            return cpu_percent, cpu_count
        except Exception as e:
            logger.warning(f"CPU read error: {e}")
            return 0.0, 1

    def _read_ram(self) -> tuple:
        """Read system RAM percent, used MB, total MB."""
        if not self._psutil_available or self._psutil is None:
            return 0.0, 0.0, 1.0

        try:
            mem = self._psutil.virtual_memory()
            return mem.percent, mem.used / (1024 * 1024), mem.total / (1024 * 1024)
        except Exception as e:
            logger.warning(f"RAM read error: {e}")
            return 0.0, 0.0, 1.0

    def _read_gpu(self) -> tuple:
        """Read GPU utilization %, VRAM used MB, VRAM total MB, temperature C."""
        if not self._gpu_available:
            return 0.0, 0.0, 0.0, 0.0

        # Try pynvml first
        if self._pynvml_available and self._pynvml is not None:
            try:
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(0)
                util = self._pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                temp = self._pynvml.nvmlDeviceGetTemperature(handle, self._pynvml.NVML_TEMPERATURE_GPU)

                return (
                    float(util.gpu),
                    mem_info.used / (1024 * 1024),
                    mem_info.total / (1024 * 1024),
                    float(temp),
                )
            except Exception as e:
                logger.debug(f"pynvml read error: {e}")

        # Fallback to nvidia-smi subprocess
        if self._nvidia_smi_available:
            try:
                import subprocess

                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5.0
                )
                if result.returncode == 0 and result.stdout.strip():
                    parts = result.stdout.strip().split(", ")
                    if len(parts) >= 4:
                        return (
                            float(parts[0]),
                            float(parts[1]),
                            float(parts[2]),
                            float(parts[3]),
                        )
            except Exception as e:
                logger.debug(f"nvidia-smi read error: {e}")

        return 0.0, 0.0, 0.0, 0.0

    def _compute_stress_multiplier(
        self, cpu: float, gpu: float, ram: float, vram: float
    ) -> float:
        """
        Compute Dynamic Stress Multiplier using quadratic curve.

        For each resource:
          if util < warning: factor = 1.0
          if warning <= util <= critical:
            t = (util - warn) / (crit - warn)
            factor = 1.0 - (t ^ exponent) * (1.0 - min_mult)
          if util > critical: factor = min_mult

        Overall multiplier = min(cpu_factor, gpu_factor, ram_factor, vram_factor)
        """
        if self._telemetry_config is None:
            return 1.0

        thresholds = self._telemetry_config.thresholds
        sm_config = self._telemetry_config.stress_multiplier
        min_mult = sm_config.min
        exponent = sm_config.curve_exponent

        def _compute_factor(util: float, warn: float, crit: float) -> float:
            if util <= warn:
                return 1.0
            if util >= crit:
                return min_mult
            t = (util - warn) / (crit - warn)
            return 1.0 - (t ** exponent) * (1.0 - min_mult)

        cpu_factor = _compute_factor(cpu, thresholds.cpu_warning, thresholds.cpu_critical)
        gpu_factor = _compute_factor(gpu, thresholds.gpu_warning, thresholds.gpu_critical)
        ram_factor = _compute_factor(ram, thresholds.ram_warning, thresholds.ram_critical)
        vram_factor = _compute_factor(vram, thresholds.vram_warning, thresholds.vram_critical)

        # If GPU not available, ignore GPU/VRAM factors
        if not self._gpu_available:
            return min(cpu_factor, ram_factor)
        return min(cpu_factor, gpu_factor, ram_factor, vram_factor)

    def _create_synthetic_snapshot(
        self, cpu: float, gpu: float, ram: float, vram: float
    ) -> HardwareSnapshot:
        """Create a synthetic snapshot (used for fallback/initial state)."""
        stress = self._compute_stress_multiplier(cpu, gpu, ram, vram)
        return HardwareSnapshot(
            timestamp=time.time(),
            cpu_utilization_percent=cpu * 100.0,
            cpu_count=1,
            system_ram_percent=ram * 100.0,
            system_ram_used_mb=0.0,
            system_ram_total_mb=1.0,
            gpu_available=self._gpu_available,
            gpu_utilization_percent=gpu * 100.0,
            vram_used_mb=0.0,
            vram_total_mb=1.0,
            gpu_temperature_c=0.0,
            stress_multiplier=stress,
        )


# =============================================================================
# Global Instance
# =============================================================================

_telemetry_monitor: Optional[TelemetryMonitor] = None


def get_telemetry_monitor() -> TelemetryMonitor:
    """Get global TelemetryMonitor instance (singleton)."""
    global _telemetry_monitor
    if _telemetry_monitor is None:
        _telemetry_monitor = TelemetryMonitor()
    return _telemetry_monitor