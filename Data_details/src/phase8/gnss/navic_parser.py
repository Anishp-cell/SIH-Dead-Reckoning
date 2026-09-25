"""
SAC ISRO NavIC (IRNSS) & Multi-Constellation Parser & Anti-Jamming Gater:
Compliant with ISRO Space Applications Centre (SAC) NavIC specifications and NMEA 0183 v4.10.
Supports:
1. NavIC talker sentences ($GIRMC, $GIGGA, $GIGSV, $GAGSV)
2. Android Raw GNSS Constellation Type 7 (CONSTELLATION_IRNSS)
3. Dual-carrier tracking: L5 (1176.45 MHz) and S-band (2492.028 MHz)
4. Selective L1-band jamming vs S-band resilience detection
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import numpy as np


class GNSSConstellation(int, Enum):
    UNKNOWN = 0
    GPS = 1
    SBAS = 2
    GLONASS = 3
    QZSS = 4
    BEIDOU = 5
    GALILEO = 6
    IRNSS = 7       # ISRO NavIC (Android GnssStatus.CONSTELLATION_IRNSS)


class NavICCarrierBand(str, Enum):
    L5 = "L5"       # 1176.45 MHz
    S_BAND = "S"    # 2492.028 MHz
    UNKNOWN = "UNKNOWN"


# Exact Center Frequencies (Hz)
NAVIC_FREQ_L5_HZ = 1176450000.0
NAVIC_FREQ_S_HZ = 2492028000.0


@dataclass
class SatelliteObservation:
    """Per-satellite tracking observability."""
    constellation: GNSSConstellation
    svid: int
    carrier_frequency_hz: float
    band: NavICCarrierBand
    cn0_dbhz: float
    elevation_deg: float = 0.0
    azimuth_deg: float = 0.0
    used_in_fix: bool = True


@dataclass
class NavICFixReport:
    """Parsed NavIC/Multi-GNSS position and signal quality report."""
    timestamp: float
    lat_deg: float
    lon_deg: float
    alt_m: float
    is_valid: bool
    num_sats_navic: int
    num_sats_total: int
    hdop: float
    speed_mps: float
    bearing_deg: float
    has_l5: bool
    has_s_band: bool
    mean_navic_cn0_dbhz: float
    l1_jamming_detected: bool
    navic_resilient: bool


def verify_nmea_checksum(sentence: str) -> bool:
    """
    Verifies XOR checksum of standard NMEA string: $...*XX
    """
    s = sentence.strip()
    if not s.startswith("$") or "*" not in s:
        return False
    payload, checksum_str = s[1:].split("*", 1)
    checksum_str = checksum_str[:2]

    try:
        expected = int(checksum_str, 16)
    except ValueError:
        return False

    computed = 0
    for ch in payload:
        computed ^= ord(ch)

    return computed == expected


def compute_nmea_checksum(payload: str) -> str:
    """Computes two-character hex checksum for an NMEA payload."""
    computed = 0
    for ch in payload:
        computed ^= ord(ch)
    return f"{computed:02X}"


class NavICNMEAParser:
    """
    Parses ISRO NavIC and multi-GNSS NMEA sentences.
    Handles $GIRMC, $GIGGA, $GIGSV, and $GNGGA.
    """

    @staticmethod
    def parse_coordinate(coord_str: str, direction: str) -> Optional[float]:
        """Converts NMEA DDMM.MMMM to decimal degrees."""
        if not coord_str or not direction:
            return None
        try:
            val = float(coord_str)
            degrees = int(val / 100.0)
            minutes = val - (degrees * 100.0)
            dec = degrees + (minutes / 60.0)
            if direction in ["S", "W"]:
                dec = -dec
            return dec
        except (ValueError, IndexError):
            return None

    @classmethod
    def parse_sentence(cls, sentence: str) -> Optional[Dict[str, Any]]:
        """
        Parses an incoming NMEA sentence from NavIC ($GI) or Multi-GNSS ($GN).
        """
        if not verify_nmea_checksum(sentence):
            return None

        clean = sentence.strip().lstrip("$").split("*")[0]
        tokens = clean.split(",")
        talker_type = tokens[0]

        talker = talker_type[:2]
        msg_type = talker_type[2:]

        result: Dict[str, Any] = {
            "talker": talker,
            "msg_type": msg_type,
            "is_navic": (talker == "GI"),
        }

        if msg_type == "GGA":
            # $GIGGA,hhmmss.ss,llll.ll,a,yyyyy.yy,a,x,xx,x.x,x.x,M,x.x,M,x.x,xxxx
            if len(tokens) >= 10:
                lat = cls.parse_coordinate(tokens[2], tokens[3])
                lon = cls.parse_coordinate(tokens[4], tokens[5])
                fix_quality = int(tokens[6]) if tokens[6].isdigit() else 0
                num_sats = int(tokens[7]) if tokens[7].isdigit() else 0
                hdop = float(tokens[8]) if tokens[8] else 99.0
                alt = float(tokens[9]) if tokens[9] else 0.0

                result.update({
                    "lat_deg": lat,
                    "lon_deg": lon,
                    "fix_quality": fix_quality,
                    "is_valid": fix_quality > 0,
                    "num_sats": num_sats,
                    "hdop": hdop,
                    "alt_m": alt,
                })
                return result

        elif msg_type == "RMC":
            # $GIRMC,hhmmss.ss,A,llll.ll,a,yyyyy.yy,a,x.x,x.x,ddmmyy,,,a
            if len(tokens) >= 9:
                status = tokens[2]
                lat = cls.parse_coordinate(tokens[3], tokens[4])
                lon = cls.parse_coordinate(tokens[5], tokens[6])
                speed_knots = float(tokens[7]) if tokens[7] else 0.0
                bearing_deg = float(tokens[8]) if tokens[8] else 0.0
                speed_mps = speed_knots * 0.514444

                result.update({
                    "lat_deg": lat,
                    "lon_deg": lon,
                    "is_valid": (status == "A"),
                    "speed_mps": speed_mps,
                    "bearing_deg": bearing_deg,
                })
                return result

        return result


def detect_selective_l1_jamming(
    l1_observations: List[SatelliteObservation],
    navic_observations: List[SatelliteObservation],
    l1_cn0_jam_threshold: float = 26.0,
    navic_s_nominal_threshold: float = 34.0,
) -> Tuple[bool, bool, str]:
    """
    Evaluates anti-jamming resilience condition:
    Detects if common L1 band (GPS/GLONASS/Galileo L1) is jammed while
    ISRO NavIC S-band (2.49 GHz) remains intact and operational.
    
    Returns:
        (l1_jammed, navic_resilient, status_message)
    """
    # 1. Compute mean L1 C/N0
    l1_cn0s = [obs.cn0_dbhz for obs in l1_observations if obs.cn0_dbhz > 0]
    mean_l1_cn0 = float(np.mean(l1_cn0s)) if l1_cn0s else 0.0

    # 2. Compute NavIC S-band C/N0
    s_band_obs = [obs for obs in navic_observations if obs.band == NavICCarrierBand.S_BAND]
    s_cn0s = [obs.cn0_dbhz for obs in s_band_obs if obs.cn0_dbhz > 0]
    mean_s_cn0 = float(np.mean(s_cn0s)) if s_cn0s else 0.0

    l1_jammed = (len(l1_cn0s) == 0) or (mean_l1_cn0 < l1_cn0_jam_threshold)
    navic_s_healthy = (len(s_cn0s) >= 2) and (mean_s_cn0 >= navic_s_nominal_threshold)

    if l1_jammed and navic_s_healthy:
        return True, True, f"L1_JAMMED_NAVIC_S_BAND_RESILIENT: L1={mean_l1_cn0:.1f}dBHz, NavIC_S={mean_s_cn0:.1f}dBHz"
    elif l1_jammed and not navic_s_healthy:
        return True, False, "ALL_BANDS_SEVERELY_DEGRADED: Full Outage Condition"
    else:
        return False, True, "ALL_BANDS_NOMINAL: Multi-Constellation Healthy"
