"""
Frame Buffer Monitor for City OS
Monitors FastAPI frame ingestion queue performance
"""
import time
import asyncio
from collections import deque
from typing import Dict, Any, List

class IngestionBufferMonitor:
    """Monitor frame ingestion performance metrics."""
    
    def __init__(self, max_samples: int = 100):
        self.frame_timestamps: deque = deque(maxlen=max_samples)
        self.dropped_frames: int = 0
        self.total_frames_received: int = 0
        self.processing_times: List[float] = []

    def record_frame_arrival(self, ingestion_timestamp: float):
        """Record when a frame arrives at the buffer."""
        self.total_frames_received += 1
        self.frame_timestamps.append(ingestion_timestamp)

    def record_frame_drop(self):
        """Record when a frame is dropped due to full queue."""
        self.dropped_frames += 1

    def record_processing_time(self, processing_time_ms: float):
        """Record processing latency."""
        self.processing_times.append(processing_time_ms)

    def calculate_metrics(self, pipeline_processing_start: float) -> Dict[str, Any]:
        """Calculate buffer performance metrics."""
        now = time.time()
        
        # Calculate Ingestion FPS
        if len(self.frame_timestamps) > 1:
            duration = self.frame_timestamps[-1] - self.frame_timestamps[0]
            fps = (len(self.frame_timestamps) - 1) / duration if duration > 0 else 0.0
        else:
            fps = 0.0

        # Calculate Frame Drop Rate
        total = self.total_frames_received + self.dropped_frames
        drop_rate = (self.dropped_frames / total) * 100 if total > 0 else 0.0

        # Latent processing delay
        latent_delay_ms = (now - pipeline_processing_start) * 1000

        # Average processing time
        avg_proc_ms = sum(self.processing_times[-30:]) / len(self.processing_times[-30:]) if self.processing_times else 0

        return {
            "ingestion_fps": round(fps, 2),
            "drop_rate_percent": round(drop_rate, 2),
            "latent_delay_ms": round(latent_delay_ms, 2),
            "avg_processing_ms": round(avg_proc_ms, 2),
            "total_received": self.total_frames_received,
            "total_dropped": self.dropped_frames
        }

    def check_health(self) -> Dict[str, bool]:
        """Check if system is operating within healthy bounds."""
        metrics = self.calculate_metrics(time.time() - 0.01)
        
        return {
            "healthy": (
                metrics["ingestion_fps"] >= 15 and  # At least 15 FPS
                metrics["drop_rate_percent"] <= 5 and  # Less than 5% drops
                metrics["latent_delay_ms"] <= 100  # Less than 100ms latency
            ),
            "fps_ok": metrics["ingestion_fps"] >= 15,
            "dropping": metrics["drop_rate_percent"] > 5
        }


# Diagnostic loop
async def diagnostic_loop(monitor: IngestionBufferMonitor):
    """Continuous monitoring loop."""
    while True:
        await asyncio.sleep(1.0)
        
        # Simulate frame arrivals for demo
        monitor.record_frame_arrival(time.time())
        
        metrics = monitor.calculate_metrics(time.time() - 0.012)
        health = monitor.check_health()
        
        status = "✓ PASS" if health["healthy"] else "✗ FAIL"
        
        print(f"[BUFFER DIAGNOSTIC] {status} | "
              f"FPS: {metrics['ingestion_fps']} | "
              f"Drop Rate: {metrics['drop_rate_percent']}% | "
              f"Queue Latency: {metrics['latent_delay_ms']}ms | "
              f"Proc Time: {metrics['avg_processing_ms']}ms")
        
        if not health["healthy"]:
            if health["dropping"]:
                print("  WARNING: Frame dropping detected! Consider reducing FPS or adding more workers.")


if __name__ == "__main__":
    monitor = IngestionBufferMonitor(max_samples=100)
    
    # Demo mode - simulate 30 frames
    for _ in range(30):
        monitor.record_frame_arrival(time.time())
    
    print("Running diagnostic loop... (Ctrl+C to stop)")
    try:
        asyncio.run(diagnostic_loop(monitor))
    except KeyboardInterrupt:
        print("\nStopped.")