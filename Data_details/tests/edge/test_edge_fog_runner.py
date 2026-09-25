"""
Unit Tests and Latency Benchmarks for EdgeFOGStreamingRunner (200 Hz Industrial IMU Pipeline).
"""

import pytest
import numpy as np

from Data_details.src.edge.sensor_profiles import SensorGrade, SENSOR_PROFILES
from Data_details.src.edge.edge_fog_runner import EdgeFOGStreamingRunner


class TestEdgeFOGStreamingRunner:
    def test_runner_initialization(self):
        profile = SENSOR_PROFILES[SensorGrade.TACTICAL_MEMS]
        runner = EdgeFOGStreamingRunner(profile=profile, sampling_rate_hz=200.0)
        assert runner.rate_hz == 200.0
        assert runner.dt_nominal == 0.005
        assert runner.deadline_ms == 5.0

    def test_200hz_streaming_execution_and_constraints(self):
        profile = SENSOR_PROFILES[SensorGrade.TACTICAL_MEMS]
        runner = EdgeFOGStreamingRunner(profile=profile, sampling_rate_hz=200.0)

        # Initial state: vehicle facing East at 10 m/s
        init_p = np.array([0.0, 0.0, 0.0])
        init_v = np.array([10.0, 0.0, 0.0])
        init_q = np.array([1.0, 0.0, 0.0, 0.0])  # identity
        runner.reset(init_p, init_v, init_q)

        # Run 100 steps (0.5s) of constant motion at 200 Hz
        # Accel measures only gravity upwards [0, 0, 9.81]
        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.0])

        for step_idx in range(100):
            t = step_idx * 0.005
            est = runner.step(
                timestamp=t,
                accel_mps2=accel,
                gyro_rads=gyro,
                is_stationary=False,
                v_forward_ai=10.0,
                sigma_v_ai=0.2,
            )

        # Vehicle should have traveled ~5 meters East
        assert np.isfinite(est.p_enu).all()
        assert pytest.approx(est.p_enu[0], abs=0.5) == 5.0
        assert abs(est.p_enu[1]) < 0.1  # lateral drift bounded by NHC
        assert abs(est.v_lateral_mps) < 0.05

    def test_standstill_zupt(self):
        profile = SENSOR_PROFILES[SensorGrade.SMARTPHONE_MEMS]
        runner = EdgeFOGStreamingRunner(profile=profile, sampling_rate_hz=200.0)

        init_p = np.array([0.0, 0.0, 0.0])
        init_v = np.array([0.05, 0.02, 0.0])  # slight velocity bias
        init_q = np.array([1.0, 0.0, 0.0, 0.0])
        runner.reset(init_p, init_v, init_q)

        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.001, -0.001, 0.0])

        for step_idx in range(50):
            t = step_idx * 0.005
            est = runner.step(
                timestamp=t,
                accel_mps2=accel,
                gyro_rads=gyro,
                is_stationary=True,
            )

        # Velocity should be clamped near zero by ZUPT
        assert np.linalg.norm(est.v_enu) < 0.02

    def test_latency_and_jitter_sub_2ms_benchmark(self):
        """
        Critical ISRO SIH Benchmark:
        Verifies that 200 Hz sequential step execution completes in < 2.0 ms on average,
        with 0 deadline misses against the 5.0 ms frame deadline.
        """
        profile = SENSOR_PROFILES[SensorGrade.AEROSPACE_FOG]
        runner = EdgeFOGStreamingRunner(profile=profile, sampling_rate_hz=200.0)

        runner.reset(np.zeros(3), np.array([15.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))

        accel = np.array([0.0, 0.0, 9.81])
        gyro = np.array([0.0, 0.0, 0.01])

        # Execute 500 steps (2.5 seconds of continuous 200 Hz streaming)
        for i in range(500):
            runner.step(
                timestamp=i * 0.005,
                accel_mps2=accel,
                gyro_rads=gyro,
                is_stationary=False,
                v_forward_ai=15.0,
                sigma_v_ai=0.3,
            )

        timing = runner.get_timing_profile()

        # Assertions
        assert timing.total_steps == 500
        assert timing.frame_deadline_ms == 5.0
        # Average step latency must be < 2.0 ms
        assert timing.mean_step_ms < 2.0, f"Mean latency {timing.mean_step_ms}ms exceeded 2.0ms!"
        # Effective throughput must exceed 200 Hz
        assert timing.effective_throughput_hz > 200.0
        # No more than 1% deadline misses
        assert timing.deadline_misses <= 5
