"""
Zone-Based Tripwire and Anomaly Detection System
Triggers alerts when objects cross virtual boundaries
"""
import json
import time
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)


@dataclass
class ZoneEvent:
    """Zone-triggered event"""
    camera_id: int
    zone_id: int
    zone_name: str
    track_id: int
    object_type: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    timestamp: float
    image_snapshot: Optional[str] = None


class ZoneAlerts:
    """
    Virtual tripwire and geofence monitoring system.
    
    Features:
    - Polygon and line zone definitions
    - Cross-zone detection
    - Dwell time monitoring (loitering detection)
    - Speed violation detection
    - Automatic snapshot capture
    """

    def __init__(self):
        self.zones: Dict[int, dict] = {}
        self.zone_triggers: Dict[int, dict] = {}  # track_id -> last position
        self.loitering_triggers: Dict[int, dict] = {}  # track_id -> entry time

    def load_zones(self, camera_id: int, zones: List[dict]):
        """Load zone definitions from database"""
        self.zones[camera_id] = {}
        for zone in zones:
            self.zones[camera_id][zone['id']] = {
                'name': zone['name'],
                'type': zone['type'],
                'coordinates': self._parse_coordinates(zone['coordinates'])
            }

    def _parse_coordinates(self, coord_str: str) -> List[Tuple[int, int]]:
        """Parse coordinate string to list of points"""
        try:
            if isinstance(coord_str, str):
                coords = json.loads(coord_str)
            else:
                coords = coord_str
            return [(int(p[0]), int(p[1])) for p in coords]
        except Exception:
            return []

    def check_zone_crossings(self, camera_id: int, detections) -> List[ZoneEvent]:
        """
        Check if any detections cross zone boundaries.
        Returns list of triggered events.
        """
        events = []
        
        if camera_id not in self.zones:
            return events

        for zone in self.zones[camera_id].values():
            zone_events = self._check_zone_zone(zone, detections, camera_id)
            events.extend(zone_events)

        return events

    def _check_zone_zone(self, zone: dict, detections, camera_id: int) -> List[ZoneEvent]:
        """Check crossing for a single zone"""
        events = []
        zone_type = zone['type']
        points = zone['coordinates']

        for det in detections:
            center = det.center
            
            if zone_type == 'line':
                events.extend(self._check_line_crossing(zone, det, camera_id))
            elif zone_type == 'polygon':
                events.extend(self._check_polygon_crossing(zone, det, camera_id, points))
            elif zone_type == 'intrusion':
                events.extend(self._check_intrusion(zone, det, camera_id, points))

        return events

    def _check_line_crossing(self, zone: dict, det, camera_id: int) -> List[ZoneEvent]:
        """Check if object crosses a virtual line (tripwire)"""
        events = []
        track_id = det.track_id
        
        # Get line points
        points = zone['coordinates']
        if len(points) < 2:
            return events

        line_start = points[0]
        line_end = points[1]
        
        # Check if center crosses the line
        is_crossing = self._line_intersection(
            line_start, line_end,
            det.center, self.zone_triggers.get(track_id, {}).get('last_center', det.center)
        )

        if is_crossing:
            events.append(ZoneEvent(
                camera_id=camera_id,
                zone_id=zone.get('id', 0),
                zone_name=zone.get('name', 'tripwire'),
                track_id=track_id,
                object_type=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                timestamp=time.time()
            ))

        # Update trigger state
        if track_id not in self.zone_triggers:
            self.zone_triggers[track_id] = {}
        self.zone_triggers[track_id]['last_center'] = det.center

        return events

    def _check_polygon_crossing(self, zone: dict, det, camera_id: int, points: List) -> List[ZoneEvent]:
        """Check if object enters/exits a polygon zone"""
        events = []
        
        # Check if center is inside polygon
        is_inside = self._point_in_polygon(det.center, points)
        
        track_id = det.track_id
        was_inside = self.zone_triggers.get(track_id, {}).get('inside_polygon', False)
        
        if is_inside and not was_inside:
            # Entry event
            events.append(ZoneEvent(
                camera_id=camera_id,
                zone_id=zone.get('id', 0),
                zone_name=zone.get('name', 'zone'),
                track_id=track_id,
                object_type=det.class_name,
                confidence=det.confidence,
                bbox=det.bbox,
                timestamp=time.time()
            ))
        
        if track_id not in self.zone_triggers:
            self.zone_triggers[track_id] = {}
        self.zone_triggers[track_id]['inside_polygon'] = is_inside

        return events

    def _check_intrusion(self, zone: dict, det, camera_id: int, points: List) -> List[ZoneEvent]:
        """Check for intrusion (object stays in zone too long)"""
        events = []
        track_id = det.track_id
        
        is_inside = self._point_in_polygon(det.center, points)
        
        if is_inside:
            if track_id not in self.loitering_triggers:
                self.loitering_triggers[track_id] = time.time()
            else:
                dwell_time = time.time() - self.loitering_triggers[track_id]
                if dwell_time > 30:  # 30 seconds loitering threshold
                    events.append(ZoneEvent(
                        camera_id=camera_id,
                        zone_id=zone.get('id', 0),
                        zone_name=zone.get('name', 'intrusion'),
                        track_id=track_id,
                        object_type=det.class_name,
                        confidence=det.confidence,
                        bbox=det.bbox,
                        timestamp=time.time()
                    ))
        else:
            self.loitering_triggers.pop(track_id, None)

        return events

    def _line_intersection(self, line1_start, line1_end, point1, point2) -> bool:
        """Check if line segment intersects with motion line"""
        # Simplified line crossing check
        def ccw(A, B, C):
            return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])

        # Check if points are on opposite sides of line
        cross1 = ccw(line1_start, line1_end, point1)
        cross2 = ccw(line1_start, line1_end, point2)
        
        return cross1 != cross2

    def _point_in_polygon(self, point: Tuple[int, int], polygon: List[Tuple[int, int]]) -> bool:
        """Ray casting algorithm for point-in-polygon test"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside

    def capture_snapshot(self, frame: np.ndarray, output_dir: str = "data/snapshots") -> str:
        """Capture and save frame snapshot"""
        from pathlib import Path
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"snapshot_{timestamp}.jpg"
        filepath = output_path / filename
        
        cv2.imwrite(str(filepath), frame)
        return str(filepath)


# Singleton instance
_alerts = None

def get_zone_alerts() -> ZoneAlerts:
    global _alerts
    if _alerts is None:
        _alerts = ZoneAlerts()
    return _alerts