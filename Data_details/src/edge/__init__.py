"""
Edge Engine Package for High-Frequency Industrial IMU / FOG Pipelines:
Supports 10 Hz - 400 Hz streaming with multi-grade noise parameters
(Consumer Smartphone MEMS, Tactical MEMS, Aerospace FOG).
"""

from .sensor_profiles import SensorGrade, SensorProfile, SENSOR_PROFILES
from .edge_fog_runner import EdgeFOGStreamingRunner, EdgeTimingProfile

__all__ = [
    "SensorGrade",
    "SensorProfile",
    "SENSOR_PROFILES",
    "EdgeFOGStreamingRunner",
    "EdgeTimingProfile",
]
