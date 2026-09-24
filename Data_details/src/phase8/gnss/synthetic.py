"""
Phase 8 Synthetic GNSS Degradation and Outage Injection Engine
Strictly adheres to ISRO SIH26168 provenance standard:
REAL_GNSS, SYNTHETIC_GNSS, and REFERENCE_ONLY.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np


class Provenance(str, Enum):
    REAL_GNSS = "REAL_GNSS"              # Smartphone GNSS telemetry from vehicle drive
    SYNTHETIC_GNSS = "SYNTHETIC_GNSS"    # Controlled synthetic noise/blackout injected
    REFERENCE_ONLY = "REFERENCE_ONLY"    # VBOX/RTK high-precision truth for validation


@dataclass
class GNSSMeasurementSample:
    timestamp: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    speed_mps: float
    bearing_deg: float
    acc_m: float
    num_sats: int
    provenance: Provenance
    is_outage: bool = False
    p_enu: Optional[np.ndarray] = None
    v_enu: Optional[np.ndarray] = None


class SyntheticGNSSGenerator:
    """
    Controlled GNSS degradation engine for reproducible scientific benchmarking.
    """

    def __init__(
        self,
        ref_lat_deg: float,
        ref_lon_deg: float,
        ref_alt_m: float = 0.0,
        random_seed: int = 42,
    ):
        self.ref_lat_deg = ref_lat_deg
        self.ref_lon_deg = ref_lon_deg
        self.ref_alt_m = ref_alt_m
        self.rng = np.random.default_rng(random_seed)

    def apply_scenario_degradation(
        self,
        sample: GNSSMeasurementSample,
        outage_intervals: Optional[List[Tuple[float, float]]] = None,
        added_noise_std_m: float = 0.0,
        step_jump_m: Optional[Tuple[float, float, float]] = None,  # (t_jump, jump_east, jump_north)
    ) -> GNSSMeasurementSample:
        """
        Applies controlled outage or degradation to a GNSS sample.
        """
        t = sample.timestamp
        in_outage = False

        # Check outage windows
        if outage_intervals:
            for t_start, t_end in outage_intervals:
                if t_start <= t <= t_end:
                    in_outage = True
                    break

        if in_outage:
            # During blackout, satellites drop to 0 and accuracy blows up
            return GNSSMeasurementSample(
                timestamp=t,
                lat_deg=sample.lat_deg,
                lon_deg=sample.lon_deg,
                alt_m=sample.alt_m,
                speed_mps=0.0,
                bearing_deg=0.0,
                acc_m=999.0,
                num_sats=0,
                provenance=Provenance.SYNTHETIC_GNSS,
                is_outage=True,
                p_enu=None,
                v_enu=None,
            )

        # Apply noise if specified
        lat = sample.lat_deg
        lon = sample.lon_deg
        alt = sample.alt_m
        acc = sample.acc_m
        sats = sample.num_sats
        prov = sample.provenance

        if added_noise_std_m > 0.0:
            prov = Provenance.SYNTHETIC_GNSS
            # Convert added noise in meters to approx degrees
            d_lat = self.rng.normal(0.0, added_noise_std_m) / 111132.954
            d_lon = self.rng.normal(0.0, added_noise_std_m) / (111132.954 * np.cos(np.radians(lat)))
            lat += d_lat
            lon += d_lon
            acc = max(acc, added_noise_std_m)

        # Apply step jump if specified
        if step_jump_m is not None:
            t_jump, d_east, d_north = step_jump_m
            if t >= t_jump:
                prov = Provenance.SYNTHETIC_GNSS
                d_lat = d_north / 111132.954
                d_lon = d_east / (111132.954 * np.cos(np.radians(lat)))
                lat += d_lat
                lon += d_lon

        return GNSSMeasurementSample(
            timestamp=t,
            lat_deg=lat,
            lon_deg=lon,
            alt_m=alt,
            speed_mps=sample.speed_mps,
            bearing_deg=sample.bearing_deg,
            acc_m=acc,
            num_sats=sats,
            provenance=prov,
            is_outage=False,
            p_enu=None,
            v_enu=None,
        )
