"""
Unit Tests for Multi-Grade Sensor Noise Profiles (MEMS, Tactical, FOG).
"""

import pytest
import numpy as np

from Data_details.src.edge.sensor_profiles import (
    SensorGrade,
    SensorProfile,
    SENSOR_PROFILES,
)


class TestSensorProfiles:
    def test_all_profiles_registered(self):
        assert SensorGrade.SMARTPHONE_MEMS in SENSOR_PROFILES
        assert SensorGrade.TACTICAL_MEMS in SENSOR_PROFILES
        assert SensorGrade.AEROSPACE_FOG in SENSOR_PROFILES

    def test_monotonic_noise_hierarchy(self):
        p_phone = SENSOR_PROFILES[SensorGrade.SMARTPHONE_MEMS]
        p_tac = SENSOR_PROFILES[SensorGrade.TACTICAL_MEMS]
        p_fog = SENSOR_PROFILES[SensorGrade.AEROSPACE_FOG]

        # FOG must have lower noise densities than Tactical, and Tactical lower than Smartphone
        assert p_fog.gyro_noise_density < p_tac.gyro_noise_density < p_phone.gyro_noise_density
        assert p_fog.accel_noise_density < p_tac.accel_noise_density < p_phone.accel_noise_density
        assert p_fog.gyro_bias_instability < p_tac.gyro_bias_instability < p_phone.gyro_bias_instability

    def test_discrete_noise_scaling(self):
        p = SENSOR_PROFILES[SensorGrade.TACTICAL_MEMS]
        # At dt = 0.01s (100 Hz) vs dt = 0.005s (200 Hz)
        sig_a_100, sig_w_100, _, _ = p.get_discrete_noises(dt=0.01)
        sig_a_200, sig_w_200, _, _ = p.get_discrete_noises(dt=0.005)

        # Smaller dt means larger discrete standard deviation (since sigma_d = PSD / sqrt(dt))
        assert sig_a_200 > sig_a_100
        assert sig_w_200 > sig_w_100
        assert pytest.approx(sig_a_200 / sig_a_100, rel=1e-3) == np.sqrt(2.0)

    def test_process_noise_covariances_positive_definite(self):
        for grade, profile in SENSOR_PROFILES.items():
            Q_v, Q_th, Q_ba, Q_bg = profile.get_process_noise_covariances(dt=0.005)
            for Q in [Q_v, Q_th, Q_ba, Q_bg]:
                assert Q.shape == (3, 3)
                eigvals = np.linalg.eigvals(Q)
                assert np.all(eigvals > 0.0)

    def test_simulate_step_noise(self):
        p = SENSOR_PROFILES[SensorGrade.SMARTPHONE_MEMS]
        rng = np.random.RandomState(42)
        n_samples = 2000
        noises_w = np.array([p.simulate_step_noise(0.01, rng=rng)[1] for _ in range(n_samples)])

        _, sig_w, _, _ = p.get_discrete_noises(0.01)
        measured_std = np.std(noises_w)
        # Sample std should match theoretical sigma within 10%
        assert pytest.approx(measured_std, rel=0.10) == sig_w
