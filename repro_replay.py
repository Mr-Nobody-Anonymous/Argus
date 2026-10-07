import logging
import time

logging.basicConfig(level=logging.DEBUG)

from backend.services.cityos.engine import CityOSEngine

engine = CityOSEngine()
inter = engine.get_intersection("replay_export_t")
inter._last_replay_at = 0.0
engine.ingest(12, [{"track_id": 3, "class_name": "person",
                    "confidence": 0.9,
                    "bbox": {"x1": 0.5, "y1": 0.5, "x2": 0.55, "y2": 0.6}}],
              [], time.time())
print("snapshots:", len(inter.replay))