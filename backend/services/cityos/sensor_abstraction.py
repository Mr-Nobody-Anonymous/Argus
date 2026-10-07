"""
CityOS sensor abstraction layer.

Sits BELOW the perception engine, closest to the physical hardware:

    Physical sensors (4D FMCW LiDAR, camera, radar)
        -> SensorFrame (timestamped, posed, confident)
        -> clustering / conversion
        -> candidate observations for the fusion layer

A 4D LiDAR frame carries per-point measurements, not just geometry:

    - x, y, z          : sensor-local position (forward/left/up, metres)
    - radial_velocity  : direct Doppler measurement along the line of sight
    - intensity        : return strength
    - return_index     : multiple returns per beam (penetration/order)

Each sensor has a mounting POSE (position + orientation) so sensor-local
points can be transformed into intersection/world coordinates, and a HEALTH
record (frame rate, packet loss, timestamp drift, temperature, calibration
validity) deep enough for the digital twin to report e.g.

    "Intersection degraded - lidar_2 unavailable - coverage reduced 43%"

The layer never assumes the camera detector is the source of truth: LiDAR
observations enter the same fusion pipeline as camera tracks.
"""
import math
import threading
import time
from typing import Dict, List, Optional, Tuple

# ── Geometry helpers ─────────────────────────────────────────────────────────


class SensorPose:
    """Rigid-body mounting pose of a sensor in world (ENU) coordinates.

    Position is metres east/north/up of the intersection centre. Orientation
    is yaw (clockwise from north), pitch and roll in degrees.
    """

    __slots__ = ("east_m", "north_m", "up_m", "yaw_deg", "pitch_deg",
                 "roll_deg")

    def __init__(self, east_m: float = 0.0, north_m: float = 0.0,
                 up_m: float = 5.0, yaw_deg: float = 0.0,
                 pitch_deg: float = -15.0, roll_deg: float = 0.0):
        self.east_m = float(east_m)
        self.north_m = float(north_m)
        self.up_m = float(up_m)
        self.yaw_deg = float(yaw_deg)
        self.pitch_deg = float(pitch_deg)
        self.roll_deg = float(roll_deg)

    def to_world(self, x_fwd: float, y_left: float,
                 z_up: float = 0.0) -> Tuple[float, float, float]:
        """Sensor-local (forward, left, up) -> world (east, north, up)."""
        yr = math.radians(self.yaw_deg)
        pr = math.radians(self.pitch_deg)
        rr = math.radians(self.roll_deg)

        # Roll about forward axis (left/up components).
        y1 = y_left * math.cos(rr) - z_up * math.sin(rr)
        z1 = y_left * math.sin(rr) + z_up * math.cos(rr)
        # Pitch about left axis (forward/up components).
        x2 = x_fwd * math.cos(pr) + z1 * math.sin(pr)
        z2 = -x_fwd * math.sin(pr) + z1 * math.cos(pr)
        # Yaw about up axis: forward -> east/north.
        east = x2 * math.sin(yr) + y1 * math.cos(yr)
        north = x2 * math.cos(yr) - y1 * math.sin(yr)
        return (self.east_m + east, self.north_m + north, self.up_m + z2)

    def to_dict(self) -> Dict:
        return {
            "east_m": self.east_m, "north_m": self.north_m,
            "up_m": self.up_m, "yaw_deg": self.yaw_deg,
            "pitch_deg": self.pitch_deg, "roll_deg": self.roll_deg,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "SensorPose":
        return cls(
            east_m=d.get("east_m", 0.0), north_m=d.get("north_m", 0.0),
            up_m=d.get("up_m", 5.0), yaw_deg=d.get("yaw_deg", 0.0),
            pitch_deg=d.get("pitch_deg", -15.0),
            roll_deg=d.get("roll_deg", 0.0),
        )


# ── Frames ───────────────────────────────────────────────────────────────────


class LidarPoint:
    """One 4D LiDAR return: position + Doppler velocity + intensity."""

    __slots__ = ("x", "y", "z", "radial_velocity", "intensity", "return_index")

    def __init__(self, x: float, y: float, z: float,
                 radial_velocity: float = 0.0, intensity: float = 0.0,
                 return_index: int = 0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)
        self.radial_velocity = float(radial_velocity)
        self.intensity = float(intensity)
        self.return_index = int(return_index)


class SensorFrame:
    """A timestamped, confidence-scored frame from one physical sensor."""

    def __init__(self, sensor_id: str, sensor_type: str, timestamp: float,
                 points: Optional[List[LidarPoint]] = None,
                 detections: Optional[List[Dict]] = None,
                 confidence: float = 1.0, sequence: int = 0,
                 expected_packets: int = 0, received_packets: int = 0):
        self.sensor_id = sensor_id
        self.sensor_type = sensor_type          # lidar | camera | radar
        self.timestamp = float(timestamp)
        self.points = points or []
        self.detections = detections or []
        self.confidence = float(confidence)
        self.sequence = int(sequence)
        self.expected_packets = int(expected_packets)
        self.received_packets = int(received_packets)


# ── Clustering ───────────────────────────────────────────────────────────────

CLUSTER_CELL_M = 1.5         # grid cell for ground-point clustering
CLUSTER_MIN_POINTS = 4       # fewer than this is noise
GROUND_Z_MAX_M = 2.5         # ignore returns above this height (buildings etc.)
OBJECT_Z_MIN_M = 0.2         # ignore ground-plane-only returns


def cluster_point_cloud(points: List[LidarPoint],
                        pose: SensorPose) -> List[Dict]:
    """Grid-cluster ground-level returns into candidate-object observations.

    Returns world-frame observations:
        {east_m, north_m, speed_mps, heading_deg, n_points, intensity_avg,
         radial_velocity_avg, z_max_m}

    Speed comes from the mean per-point radial (Doppler) velocity - a direct
    physics measurement rather than a frame-to-frame tracker estimate. The
    heading is only the bearing FROM THE SENSOR; the fusion layer refines it
    against camera headings.
    """
    grid: Dict[Tuple[int, int], List[LidarPoint]] = {}
    for p in points:
        if p.z > GROUND_Z_MAX_M or p.z < -GROUND_Z_MAX_M:
            continue
        if p.z < OBJECT_Z_MIN_M and abs(p.radial_velocity) < 0.3:
            continue                      # static road surface return
        key = (int(math.floor(p.x / CLUSTER_CELL_M)),
               int(math.floor(p.y / CLUSTER_CELL_M)))
        grid.setdefault(key, []).append(p)

    # Merge adjacent occupied cells into clusters (4-neighbour flood fill).
    seen: Dict[Tuple[int, int], int] = {}
    clusters: List[List[LidarPoint]] = []
    for key in grid:
        if key in seen:
            continue
        stack, members = [key], []
        seen[key] = 1
        while stack:
            cx, cy = stack.pop()
            members.extend(grid[(cx, cy)])
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nk = (cx + dx, cy + dy)
                if nk in grid and nk not in seen:
                    seen[nk] = 1
                    stack.append(nk)
        if len(members) >= CLUSTER_MIN_POINTS:
            clusters.append(members)

    observations: List[Dict] = []
    for members in clusters:
        n = len(members)
        cx = sum(p.x for p in members) / n
        cy = sum(p.y for p in members) / n
        cz = sum(p.z for p in members) / n
        east, north, _ = pose.to_world(cx, cy, cz)
        rv = sum(p.radial_velocity for p in members) / n
        # Radial velocity is along the sensor->object line of sight; signed
        # positive = approaching. Convert to a speed magnitude estimate.
        speed = abs(rv)
        bearing = (math.degrees(math.atan2(east - pose.east_m,
                                           north - pose.north_m)) + 360.0) \
            % 360.0
        observations.append({
            "east_m": round(east, 2),
            "north_m": round(north, 2),
            "speed_mps": round(speed, 2),
            "bearing_from_sensor_deg": round(bearing, 1),
            "heading_deg": round(bearing, 1),
            "n_points": n,
            "intensity_avg": round(sum(p.intensity for p in members) / n, 1),
            "radial_velocity_avg": round(rv, 2),
            "z_mean_m": round(cz, 2),
            "multi_return": any(p.return_index > 0 for p in members),
        })
    return observations


# ── Sensors & registry ───────────────────────────────────────────────────────

STALE_FRAME_S = 5.0
MIN_FPS = 0.3
MAX_TS_DRIFT_MS = 250.0


class Sensor:
    """Health-tracked physical sensor attached to an intersection."""

    def __init__(self, sensor_id: str, sensor_type: str,
                 pose: Optional[SensorPose] = None,
                 label: str = ""):
        self.id = sensor_id
        self.type = sensor_type
        self.label = label or sensor_id
        self.pose = pose or SensorPose()
        self.connected = True
        self.frames_received = 0
        self.dropped_frames = 0
        self.packets_expected = 0
        self.packets_received = 0
        self.last_frame_ts = 0.0
        self.last_wall_ts = 0.0
        self.ts_drift_ms = 0.0
        self.temperature_c: Optional[float] = None
        self.calibration_valid = True
        self.inference_latency_ms = 0.0
        self.notes = ""
        self._frame_times: List[float] = []

    # ── Ingest ──────────────────────────────────────────────────────

    def record_frame(self, frame: SensorFrame):
        now = time.time()
        self.frames_received += 1
        self.packets_expected += max(frame.expected_packets, 0)
        self.packets_received += max(frame.received_packets, 0)
        if self.last_frame_ts:
            if frame.sequence <= 0 or frame.timestamp <= self.last_frame_ts:
                self.dropped_frames += 1
        # Timestamp drift: sensor clock vs edge-node wall clock.
        self.ts_drift_ms = round((now - frame.timestamp) * 1000.0, 1)
        self.last_frame_ts = frame.timestamp
        self.last_wall_ts = now
        self._frame_times.append(now)
        if len(self._frame_times) > 240:
            del self._frame_times[:-120]

    def fps(self) -> float:
        cutoff = time.time() - 60
        recent = [t for t in self._frame_times if t > cutoff]
        return round(len(recent) / 60.0, 2)

    def packet_loss_pct(self) -> float:
        if self.packets_expected <= 0:
            return 0.0
        lost = max(self.packets_expected - self.packets_received, 0)
        return round(lost / self.packets_expected * 100.0, 2)

    # ── Health ──────────────────────────────────────────────────────

    def health(self) -> Dict:
        age = (time.time() - self.last_wall_ts) if self.last_wall_ts else None
        fps = self.fps()
        problems: List[str] = []
        if not self.connected:
            problems.append("disconnected")
        if age is None or age > STALE_FRAME_S:
            problems.append("no recent frames")
        elif fps < MIN_FPS:
            problems.append(f"low frame rate ({fps} fps)")
        if abs(self.ts_drift_ms) > MAX_TS_DRIFT_MS:
            problems.append(f"timestamp drift {self.ts_drift_ms} ms")
        if self.packet_loss_pct() > 5.0:
            problems.append(f"packet loss {self.packet_loss_pct()}%")
        if not self.calibration_valid:
            problems.append("calibration invalid/expired")
        if problems:
            status = "offline" if (not self.connected or age is None
                                   or age > STALE_FRAME_S * 6) else "degraded"
        else:
            status = "ok"
        return {
            "sensor_id": self.id,
            "type": self.type,
            "label": self.label,
            "status": status,
            "problems": problems,
            "connected": self.connected,
            "fps_60s": fps,
            "last_frame_age_s": round(age, 1) if age else None,
            "frames_total": self.frames_received,
            "dropped_frames": self.dropped_frames,
            "packet_loss_pct": self.packet_loss_pct(),
            "timestamp_drift_ms": self.ts_drift_ms,
            "temperature_c": self.temperature_c,
            "calibration_valid": self.calibration_valid,
            "inference_latency_ms": self.inference_latency_ms,
            "pose": self.pose.to_dict(),
            "notes": self.notes,
        }

    def to_dict(self) -> Dict:
        return {
            "sensor_id": self.id, "type": self.type, "label": self.label,
            "pose": self.pose.to_dict(),
        }


class SensorRegistry:
    """Per-intersection registry of physical sensors + aggregate health."""

    def __init__(self):
        self._lock = threading.Lock()
        self.sensors: Dict[str, Sensor] = {}

    def register(self, sensor_id: str, sensor_type: str,
                 pose: Optional[SensorPose] = None,
                 label: str = "") -> Sensor:
        with self._lock:
            s = Sensor(sensor_id, sensor_type, pose=pose, label=label)
            self.sensors[sensor_id] = s
            return s

    def get(self, sensor_id: str) -> Optional[Sensor]:
        with self._lock:
            return self.sensors.get(sensor_id)

    def all(self) -> List[Sensor]:
        with self._lock:
            return list(self.sensors.values())

    def record_frame(self, frame: SensorFrame) -> Optional[Sensor]:
        sensor = self.get(frame.sensor_id)
        if sensor is None:
            return None
        sensor.record_frame(frame)
        return sensor

    def health_summary(self) -> Dict:
        sensors = self.all()
        if not sensors:
            return {"status": "no_sensors_registered", "sensors": [],
                    "coverage_pct": None}
        statuses = [s.health() for s in sensors]
        ok = sum(1 for h in statuses if h["status"] == "ok")
        coverage = round(ok / len(statuses) * 100.0, 1)
        if coverage >= 99.0:
            overall = "ok"
        elif coverage >= 50.0:
            overall = "degraded"
        else:
            overall = "critical"
        return {
            "status": overall,
            "coverage_pct": coverage,
            "note": (f"{len(statuses) - ok}/{len(statuses)} sensors "
                     f"degraded/offline - coverage reduced "
                     f"{round(100 - coverage, 1)}%"
                     if coverage < 99.0 else "all sensors nominal"),
            "sensors": statuses,
        }

    def to_dict(self) -> Dict:
        return {"sensors": [s.to_dict() for s in self.all()]}