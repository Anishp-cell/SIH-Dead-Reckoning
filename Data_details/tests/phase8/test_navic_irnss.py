"""
Unit Tests for SAC ISRO NavIC (IRNSS) NMEA Parsing, Multi-Band Tracking, and Anti-Jamming Resilience.
"""

import pytest
import numpy as np

from Data_details.src.phase8.gnss.navic_parser import (
    GNSSConstellation,
    NavICCarrierBand,
    SatelliteObservation,
    NavICNMEAParser,
    verify_nmea_checksum,
    compute_nmea_checksum,
    detect_selective_l1_jamming,
    NAVIC_FREQ_L5_HZ,
    NAVIC_FREQ_S_HZ,
)
from Data_details.src.phase8.gnss.quality import GNSSQualityAssessor


class TestNavICParser:
    def test_checksum_computation_and_verification(self):
        payload = "GIRMC,091530.00,A,5224.1031,N,00130.3198,W,19.4,85.2,240926,,,A"
        cs = compute_nmea_checksum(payload)
        sentence = f"${payload}*{cs}"
        assert verify_nmea_checksum(sentence)
        assert not verify_nmea_checksum(f"${payload}*99")  # wrong checksum

    def test_parse_gigga_sentence(self):
        # Format: $GIGGA,091530.00,5224.1031,N,00130.3198,W,1,08,1.2,105.4,M,0.0,M,,*XX
        payload = "GIGGA,091530.00,5224.1031,N,00130.3198,W,1,08,1.2,105.4,M,0.0,M,,"
        cs = compute_nmea_checksum(payload)
        sentence = f"${payload}*{cs}"

        parsed = NavICNMEAParser.parse_sentence(sentence)
        assert parsed is not None
        assert parsed["talker"] == "GI"
        assert parsed["is_navic"] is True
        assert parsed["is_valid"] is True
        assert parsed["num_sats"] == 8
        assert pytest.approx(parsed["lat_deg"], abs=1e-4) == 52.401718
        assert pytest.approx(parsed["lon_deg"], abs=1e-4) == -1.505330
        assert parsed["alt_m"] == 105.4

    def test_parse_girmc_sentence(self):
        payload = "GIRMC,091530.00,A,5224.1031,N,00130.3198,W,19.4,85.2,240926,,,A"
        cs = compute_nmea_checksum(payload)
        sentence = f"${payload}*{cs}"

        parsed = NavICNMEAParser.parse_sentence(sentence)
        assert parsed is not None
        assert parsed["is_valid"] is True
        assert pytest.approx(parsed["speed_mps"], abs=0.1) == 19.4 * 0.514444
        assert pytest.approx(parsed["bearing_deg"], abs=0.1) == 85.2

    def test_navic_carrier_frequencies_and_constellation(self):
        assert GNSSConstellation.IRNSS == 7
        assert pytest.approx(NAVIC_FREQ_L5_HZ, abs=1e3) == 1.17645e9
        assert pytest.approx(NAVIC_FREQ_S_HZ, abs=1e3) == 2.492028e9

    def test_selective_l1_jamming_detection_navic_resilience(self):
        # Scenario: L1 signal jammed (CW jammer on 1575.42 MHz) -> L1 C/N0 plummets to 18 dB-Hz
        l1_obs = [
            SatelliteObservation(GNSSConstellation.GPS, 1, 1575.42e6, NavICCarrierBand.UNKNOWN, 18.2),
            SatelliteObservation(GNSSConstellation.GPS, 3, 1575.42e6, NavICCarrierBand.UNKNOWN, 19.5),
        ]
        # NavIC S-band (2492 MHz) completely unaffected by L1 jammer -> nominal 42 dB-Hz
        navic_obs = [
            SatelliteObservation(GNSSConstellation.IRNSS, 1, NAVIC_FREQ_S_HZ, NavICCarrierBand.S_BAND, 42.1),
            SatelliteObservation(GNSSConstellation.IRNSS, 2, NAVIC_FREQ_S_HZ, NavICCarrierBand.S_BAND, 41.5),
            SatelliteObservation(GNSSConstellation.IRNSS, 4, NAVIC_FREQ_S_HZ, NavICCarrierBand.S_BAND, 39.8),
        ]

        l1_jammed, navic_resilient, msg = detect_selective_l1_jamming(l1_obs, navic_obs)
        assert l1_jammed is True
        assert navic_resilient is True
        assert "L1_JAMMED_NAVIC_S_BAND_RESILIENT" in msg

        # Signal Quality Assessor must accept the fix under NavIC S-band resilience
        assessor = GNSSQualityAssessor(min_sats=4)
        report = assessor.evaluate(
            timestamp=100.0,
            p_enu=np.array([10.0, 20.0, 0.0]),
            stated_acc_m=2.5,
            num_sats=0,  # 0 GPS satellites available due to L1 jamming
            navic_sats=3, # 3 NavIC S-band satellites tracked
            l1_jammed=l1_jammed,
            navic_resilient=navic_resilient,
        )
        assert report.is_valid is True
        assert report.navic_resilient is True
        assert report.jamming_detected is True

    def test_dual_band_full_outage_rejection(self):
        # Scenario: Both L1 and S-band jammed (wideband barrage jamming)
        l1_obs = [
            SatelliteObservation(GNSSConstellation.GPS, 1, 1575.42e6, NavICCarrierBand.UNKNOWN, 15.0),
        ]
        navic_obs = [
            SatelliteObservation(GNSSConstellation.IRNSS, 1, NAVIC_FREQ_S_HZ, NavICCarrierBand.S_BAND, 17.0),
        ]

        l1_jammed, navic_resilient, _ = detect_selective_l1_jamming(l1_obs, navic_obs)
        assert l1_jammed is True
        assert navic_resilient is False

        assessor = GNSSQualityAssessor(min_sats=4)
        report = assessor.evaluate(
            timestamp=100.0,
            p_enu=np.array([10.0, 20.0, 0.0]),
            stated_acc_m=2.5,
            num_sats=0,
            navic_sats=1,
            l1_jammed=l1_jammed,
            navic_resilient=navic_resilient,
        )
        assert report.is_valid is False
        assert any("INSUFFICIENT_SATELLITES" in r for r in report.rejection_reasons)
