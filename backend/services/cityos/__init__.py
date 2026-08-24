"""
Argus CityOS - AI-powered intersection intelligence.

Inspired by Aeva CityOS capabilities, implemented on top of Argus's existing
camera-based perception pipeline:

    Sensors (cameras) -> Perception Engine -> Road-user classification ->
    Tracking & trajectory analysis -> Safety analytics (wrong-way,
    near-miss, VRU) -> Traffic-flow analytics -> Signal optimisation ->
    Digital twin -> Operator alerts

Privacy posture: this layer deliberately consumes ONLY detection geometry
(bounding boxes, track ids, classes, speeds). It never touches face
embeddings, plate text or any biometric pipeline, so the intersection model
it builds contains no personally identifiable imagery.
"""
from backend.services.cityos.engine import get_cityos_engine, CityOSEngine

__all__ = ["get_cityos_engine", "CityOSEngine"]