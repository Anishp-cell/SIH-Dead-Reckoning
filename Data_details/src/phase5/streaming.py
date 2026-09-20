"""
Phase 5 Streaming Navigation Engine: Sequential Sample-by-Sample Real-Time ESKF.
Couples Phase 3 calibrated IMU telemetry with Phase 4 AI Motion Intelligence.
"""

from typing import Optional, Union, List
import numpy as np

from Data_details.src.phase5.core.eskf import ESKF15State
from Data_details.src.phase5.core.state import NavigationEstimate
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine, MotionEstimate


class StreamingNavigationEngine:
    """
    Real-time streaming inertial navigation engine.
    Processes sequential 12-channel IMU samples at 10 Hz without future buffering.
    """

    def __init__(
        self,
        eskf: Optional[ESKF15State] = None,
        ai_engine: Optional[CausalStreamingInferenceEngine] = None,
    ):
        self.eskf = eskf or ESKF15State()
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
        """Initializes the navigation filter state."""
        self.eskf.initialize(p0, v0, q0, ba0, bg0, P0)
        self.last_timestamp = initial_timestamp
        self.step_count = 0
        if self.ai_engine is not None:
            self.ai_engine.reset()

    def step(
        self,
        imu_sample_12ch: Union[np.ndarray, List[float]],
        timestamp: float,
        dt_override: Optional[float] = None,
        enable_ai_speed: bool = True,
    ) -> NavigationEstimate:
        """
        Executes one full streaming navigation step:
        1. Parse 12-channel features:
           Channel 0: acc_fwd_veh
           Channel 1: acc_lat_veh
           Channel 2: acc_up_veh
           Channel 3: gyro_roll_veh
           Channel 4: gyro_pitch_veh
           Channel 5: gyro_yaw_veh
           ...
        2. Propagate nominal state and covariance by dt.
        3. Infer AI forward speed & uncertainty via Phase 4 engine.
        4. Apply Kalman speed update with NIS gating.
        5. Return updated NavigationEstimate.
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

        # Extract vehicle-frame IMU acceleration and angular velocity
        # Channel 2 (acc_up_veh) was inverted in Phase 2 mount alignment; negate so upward force is +g in ENU
        f_meas = np.array([sample_arr[0], sample_arr[1], -sample_arr[2]], dtype=np.float64)
        omega_meas = sample_arr[3:6]

        # 1. ESKF Prediction step
        self.eskf.predict(f_meas, omega_meas, dt)

        # 2. AI Motion Intelligence Inference & Kalman Correction
        if enable_ai_speed and self.ai_engine is not None:
            motion = self.ai_engine.update(sample_arr, timestamp=timestamp)
            self.eskf.update_speed(
                z_speed_mps=motion.forward_speed_mps,
                sigma_v_mps=motion.speed_uncertainty,
            )

        return self.eskf.get_estimate(timestamp=timestamp)
