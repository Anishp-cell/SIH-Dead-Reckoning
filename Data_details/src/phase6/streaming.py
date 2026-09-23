"""
Phase 6 Streaming Navigation Engine: Sequential Sample-by-Sample Real-Time ESKF.
Fuses Phase 3 IMU, Phase 4 AI motion intelligence, NHC, ZUPT, and ZARU constraints.
"""

from typing import Optional, Union, List
import numpy as np

from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF, Phase6NavigationEstimate
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine, MotionEstimate


class Phase6StreamingEngine:
    """
    Real-time sequential navigation engine executing Phase 6 vehicle-physics ESKF.
    Guarantees strict causality: processes 12-channel IMU samples sequentially at 10 Hz without future buffering.
    """

    def __init__(
        self,
        eskf: Optional[Phase6ESKF] = None,
        ai_engine: Optional[CausalStreamingInferenceEngine] = None,
    ):
        self.eskf = eskf or Phase6ESKF()
        self.ai_engine = ai_engine

        self.last_timestamp: Optional[float] = None
        self.step_count = 0

    def initialize(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        q0: np.ndarray,
        ba0: Optional[np.ndarray] = None,
        bg0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        initial_timestamp: float = 0.0,
    ) -> None:
        """Initializes the navigation filter state and clears detector histories."""
        self.eskf.initialize(p0, v0, q0, ba0, bg0, P0)
        self.last_timestamp = initial_timestamp
        self.step_count = 0
        self.eskf.stationary_detector.reset()
        if self.ai_engine is not None:
            self.ai_engine.reset()

    def step(
        self,
        imu_sample_12ch: Union[np.ndarray, List[float]],
        timestamp: float,
        dt_override: Optional[float] = None,
        enable_ai_speed: bool = True,
        fixed_sigma_override: Optional[float] = None,
    ) -> Phase6NavigationEstimate:
        """
        Executes one full streaming navigation cycle:
        1. Extract specific force (negating inverted mount channel 2) and angular rate.
        2. IMU Kinematic Prediction: propagate nominal INS state and covariance P forward by dt.
        3. Phase 4 AI Speed Inference & Kalman Update.
        4. Disturbance & Stationary Detection.
        5. ZUPT & ZARU Updates (if stationary).
        6. Non-Holonomic Constraint (NHC) Update with adaptive disturbance covariance.
        7. Return structured Phase 6 navigation estimate.
        """
        sample_arr = np.asarray(imu_sample_12ch, dtype=np.float64).ravel()
        if len(sample_arr) < 6:
            raise ValueError(f"Expected at least 6 IMU channels, got {len(sample_arr)}")

        # Calculate time increment
        if dt_override is not None:
            dt = dt_override
        elif self.last_timestamp is not None:
            dt = max(1e-4, timestamp - self.last_timestamp)
        else:
            dt = 0.100  # Default nominal 10 Hz

        self.last_timestamp = timestamp
        self.step_count += 1

        # Reconcile body specific force: negate inverted channel 2 (acc_up_veh) so +g is upwards
        f_meas = np.array([sample_arr[0], sample_arr[1], -sample_arr[2]], dtype=np.float64)
        omega_meas = sample_arr[3:6]
        vibe_energy = float(sample_arr[10]) if len(sample_arr) > 10 else 0.0

        # 1. IMU Prediction step
        self.eskf.predict(f_meas, omega_meas, dt)

        # 2. Phase 4 AI Forward Speed Update
        ai_speed_val = 0.0
        if enable_ai_speed and self.ai_engine is not None:
            motion = self.ai_engine.update(sample_arr, timestamp=timestamp)
            ai_speed_val = motion.forward_speed_mps
            sigma_val = fixed_sigma_override if fixed_sigma_override is not None else motion.speed_uncertainty
            self.eskf.update_speed(ai_speed_val, sigma_val)

        # 3. Causal Disturbance Detection
        dist_report = self.eskf.disturbance_detector.evaluate(f_meas, omega_meas, vibration_energy=vibe_energy)
        self.eskf.last_nhc_conf = dist_report.c_nhc

        # 4. Stationary Detection with Hysteresis
        is_stat, stat_dur, stat_conf = self.eskf.stationary_detector.update(
            f_meas, omega_meas, ai_speed_mps=ai_speed_val if enable_ai_speed else None, dt=dt
        )
        self.eskf.last_is_stationary = is_stat
        self.eskf.last_stat_duration = stat_dur

        # 5. Stationary Updates (ZUPT + ZARU)
        if is_stat:
            self.eskf.update_stationary_constraints(omega_meas)
        else:
            self.eskf.last_zupt_accepted = False
            self.eskf.last_zaru_accepted = False

        # 6. Non-Holonomic Constraints (NHC) Update
        self.eskf.update_nhc(d_lat=dist_report.d_lat, d_up=dist_report.d_up)

        return self.eskf.get_phase6_estimate(timestamp=timestamp)
