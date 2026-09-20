"""
Phase 5 15-State Error-State Kalman Filter Navigation Core Engine.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any

from Data_details.src.phase5.core.state import NominalState, NavigationEstimate
from Data_details.src.phase5.core.propagation import (
    propagate_nominal_ins,
    compute_discrete_F_matrix,
    propagate_covariance,
)
from Data_details.src.phase5.core.measurement import (
    predict_forward_speed,
    kalman_update_speed,
)
from Data_details.src.phase5.models.imu_noise import IMUNoiseParameters


class ESKF15State:
    """
    15-State Error-State Kalman Filter for Inertial Navigation.
    
    Nominal state (16D): [p, v, q, ba, bg]
    Error state (15D): [delta_p, delta_v, delta_theta, delta_ba, delta_bg]
    """

    def __init__(
        self,
        noise_params: Optional[IMUNoiseParameters] = None,
        nis_threshold: float = 9.0,
    ):
        self.noise_params = noise_params or IMUNoiseParameters()
        self.nis_threshold = nis_threshold

        self.state = NominalState()
        self.P = np.eye(15, dtype=np.float64)
        self.initialized = False

        # Diagnostic telemetry
        self.last_nu = 0.0
        self.last_S = 1.0
        self.last_nis = 0.0
        self.last_accepted = True

    def initialize(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        q0: np.ndarray,
        ba0: Optional[np.ndarray] = None,
        bg0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        """Initializes the filter nominal state and prior covariance."""
        ba = np.zeros(3, dtype=np.float64) if ba0 is None else np.asarray(ba0, dtype=np.float64)
        bg = np.zeros(3, dtype=np.float64) if bg0 is None else np.asarray(bg0, dtype=np.float64)

        self.state = NominalState(
            p=np.asarray(p0, dtype=np.float64).copy(),
            v=np.asarray(v0, dtype=np.float64).copy(),
            q=np.asarray(q0, dtype=np.float64).copy(),
            ba=ba.copy(),
            bg=bg.copy(),
        )

        if P0 is not None:
            self.P = np.asarray(P0, dtype=np.float64).copy()
        else:
            # Default physically justified initial covariance
            self.P = np.diag([
                1.0, 1.0, 1.0,           # Position uncertainty: 1 m^2
                0.25, 0.25, 0.25,        # Velocity uncertainty: 0.5 m/s (0.25 m^2/s^2)
                (np.radians(2.0))**2,    # Roll/Pitch tilt uncertainty: 2 deg
                (np.radians(2.0))**2,
                (np.radians(5.0))**2,    # Yaw heading uncertainty: 5 deg
                (0.05)**2, (0.05)**2, (0.05)**2, # Accel bias uncertainty: 0.05 m/s^2
                (1e-3)**2, (1e-3)**2, (1e-3)**2, # Gyro bias uncertainty: 1e-3 rad/s
            ]).astype(np.float64)

        self.initialized = True

    def predict(
        self,
        f_meas: np.ndarray,
        omega_meas: np.ndarray,
        dt: float,
    ) -> None:
        """
        Executes nominal INS propagation and error covariance propagation forward by dt.
        """
        if not self.initialized:
            raise RuntimeError("ESKF must be initialized before predict() is called.")

        # 1. Propagate nominal state
        next_state, f_b_corr, omega_b_corr = propagate_nominal_ins(
            self.state, f_meas, omega_meas, dt
        )

        # 2. Compute state transition Jacobian F_k
        F = compute_discrete_F_matrix(self.state, f_b_corr, omega_b_corr, dt)

        # 3. Compute discrete process noise Q_k
        Q = self.noise_params.build_discrete_Q(dt)

        # 4. Propagate covariance P
        self.P = propagate_covariance(self.P, F, Q)

        # Update nominal state
        self.state = next_state

    def update_speed(
        self,
        z_speed_mps: float,
        sigma_v_mps: float,
    ) -> bool:
        """
        Executes Kalman correction with scalar AI forward speed measurement.
        Returns: True if measurement was accepted by NIS gate, False if rejected.
        """
        if not self.initialized:
            raise RuntimeError("ESKF must be initialized before update_speed() is called.")

        self.state, self.P, self.last_nu, self.last_S, self.last_nis, self.last_accepted = kalman_update_speed(
            self.state,
            self.P,
            z_speed_mps,
            sigma_v_mps,
            nis_threshold=self.nis_threshold,
        )
        return self.last_accepted

    def get_estimate(self, timestamp: float = 0.0) -> NavigationEstimate:
        """Returns structured navigation estimate snapshot."""
        v_fwd, _ = predict_forward_speed(self.state)
        cov_diag = np.diag(self.P).copy()
        filter_healthy = bool(np.all(np.isfinite(cov_diag)) and np.all(cov_diag > 0))

        return NavigationEstimate(
            timestamp=timestamp,
            p=self.state.p.copy(),
            v=self.state.v.copy(),
            q=self.state.q.copy(),
            ba=self.state.ba.copy(),
            bg=self.state.bg.copy(),
            cov_diagonal=cov_diag,
            forward_speed_mps=v_fwd,
            speed_innovation=self.last_nu,
            innovation_variance=self.last_S,
            nis=self.last_nis,
            measurement_accepted=self.last_accepted,
            filter_healthy=filter_healthy,
        )
