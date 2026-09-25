"""
Dynamic In-Vehicle Alignment & Mount Slip Detector:
Continuously monitors smartphone orientation relative to the vehicle chassis.
Detects abrupt phone mount slips, shifts, or falls during aggressive driving
(braking, potholes, collisions) and performs autonomous online recalibration
with attitude covariance inflation to prevent filter divergence.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, List
import numpy as np


class MountSlipState(str, Enum):
    STABLE = "STABLE"
    SLIP_DETECTED = "SLIP_DETECTED"
    RECALIBRATING = "RECALIBRATING"


@dataclass
class DynamicAlignmentConfig:
    """Configuration thresholds for dynamic mount slip detection and recalibration."""
    slip_angle_threshold_deg: float = 4.0        # Minimum gravity deflection to trigger slip
    gyro_spike_threshold_rads: float = 1.2       # High angular rate threshold (~70 deg/s)
    short_window_tau_s: float = 0.3              # Fast gravity EMA time constant
    long_window_tau_s: float = 4.0               # Baseline reference gravity EMA time constant
    stabilization_duration_s: float = 0.8        # Settling time before accepting new alignment
    covariance_inflation_factor: float = 15.0    # Factor to inflate attitude covariance upon slip


@dataclass
class MountSlipEvent:
    """Structured record of a verified mount slip and recovery event."""
    timestamp: float
    slip_angle_deg: float
    gyro_peak_rads: float
    R_before: np.ndarray
    R_after: np.ndarray
    covariance_inflation: float


class DynamicMountSlipDetector:
    """
    Causal streaming detector and online recalibrator for phone-to-vehicle alignment.
    """

    def __init__(
        self,
        config: Optional[DynamicAlignmentConfig] = None,
        initial_R_phone_to_veh: Optional[np.ndarray] = None,
    ):
        self.cfg = config or DynamicAlignmentConfig()
        self.R_phone_to_veh = (
            np.asarray(initial_R_phone_to_veh, dtype=np.float64).copy()
            if initial_R_phone_to_veh is not None
            else np.eye(3, dtype=np.float64)
        )

        self.state = MountSlipState.STABLE
        self.slip_onset_time: Optional[float] = None
        self.slip_events: List[MountSlipEvent] = []

        # Internal EMA low-pass filters for specific force
        self.g_short: Optional[np.ndarray] = None
        self.g_long: Optional[np.ndarray] = None
        self.gyro_peak: float = 0.0

    def reset(self, R_initial: Optional[np.ndarray] = None) -> None:
        """Resets the detector state."""
        if R_initial is not None:
            self.R_phone_to_veh = np.asarray(R_initial, dtype=np.float64).copy()
        self.state = MountSlipState.STABLE
        self.slip_onset_time = None
        self.g_short = None
        self.g_long = None
        self.gyro_peak = 0.0

    def step(
        self,
        timestamp: float,
        accel_phone: np.ndarray,
        gyro_phone: np.ndarray,
        dt: float = 0.10,
        is_stationary: bool = False,
    ) -> Tuple[MountSlipState, np.ndarray, float]:
        """
        Processes one IMU measurement sample:
        
        Returns:
            (mount_state, current_R_phone_to_veh, attitude_inflation_factor)
        """
        a_p = np.asarray(accel_phone, dtype=np.float64)
        w_p = np.asarray(gyro_phone, dtype=np.float64)
        w_norm = float(np.linalg.norm(w_p))

        # Initialize EMAs on first sample
        if self.g_short is None:
            self.g_short = a_p.copy()
            self.g_long = a_p.copy()
            return self.state, self.R_phone_to_veh.copy(), 1.0

        # Update Low-pass filter weights
        alpha_short = float(np.clip(dt / max(1e-4, self.cfg.short_window_tau_s), 0.01, 0.99))
        alpha_long = float(np.clip(dt / max(1e-4, self.cfg.long_window_tau_s), 0.001, 0.50))

        self.g_short = (1.0 - alpha_short) * self.g_short + alpha_short * a_p

        # In STABLE state, baseline gravity slowly adapts; during slip, baseline is frozen
        if self.state == MountSlipState.STABLE:
            self.g_long = (1.0 - alpha_long) * self.g_long + alpha_long * a_p

        # Compute angular deviation between short-term gravity and reference baseline
        norm_s = float(np.linalg.norm(self.g_short))
        norm_l = float(np.linalg.norm(self.g_long))

        inflation = 1.0

        if norm_s > 1e-3 and norm_l > 1e-3:
            cos_theta = float(np.clip(np.dot(self.g_short, self.g_long) / (norm_s * norm_l), -1.0, 1.0))
            angle_deg = float(np.degrees(np.arccos(cos_theta)))
        else:
            angle_deg = 0.0

        # State Machine
        if self.state == MountSlipState.STABLE:
            if (
                angle_deg >= self.cfg.slip_angle_threshold_deg
                and w_norm >= self.cfg.gyro_spike_threshold_rads
            ):
                self.state = MountSlipState.SLIP_DETECTED
                self.slip_onset_time = timestamp
                self.gyro_peak = w_norm
                inflation = self.cfg.covariance_inflation_factor

        elif self.state == MountSlipState.SLIP_DETECTED:
            self.gyro_peak = max(self.gyro_peak, w_norm)
            inflation = self.cfg.covariance_inflation_factor

            # Check if phone has settled into its new orientation
            if self.slip_onset_time is not None:
                elapsed = timestamp - self.slip_onset_time
                if elapsed >= self.cfg.stabilization_duration_s and w_norm < 0.3:
                    self.state = MountSlipState.RECALIBRATING

        elif self.state == MountSlipState.RECALIBRATING:
            # Perform online recalibration
            R_old = self.R_phone_to_veh.copy()
            R_new = self._recalibrate_alignment(self.g_short, self.R_phone_to_veh)
            self.R_phone_to_veh = R_new

            self.slip_events.append(
                MountSlipEvent(
                    timestamp=timestamp,
                    slip_angle_deg=angle_deg,
                    gyro_peak_rads=self.gyro_peak,
                    R_before=R_old,
                    R_after=R_new,
                    covariance_inflation=self.cfg.covariance_inflation_factor,
                )
            )

            # Re-seed baseline filter to new orientation
            self.g_long = self.g_short.copy()
            self.state = MountSlipState.STABLE
            inflation = 1.0

        return self.state, self.R_phone_to_veh.copy(), inflation

    @staticmethod
    def _recalibrate_alignment(
        new_gravity_phone: np.ndarray,
        old_R_phone_to_veh: np.ndarray,
    ) -> np.ndarray:
        """
        Recomputes orthonormal R_{phone->vehicle} matrix:
        1. Vertical axis z_v in phone frame = -g_new / ||g_new||
        2. Preserves heading continuity by projecting old forward axis perpendicular to z_v
        3. Enforces strict SO(3) orthonormality via Gram-Schmidt
        """
        norm_g = np.linalg.norm(new_gravity_phone)
        if norm_g < 1e-4:
            return old_R_phone_to_veh.copy()

        # New vertical axis (Z points UP in vehicle frame, opposite to gravity)
        z_phone = -new_gravity_phone / norm_g

        # Retrieve previous forward axis in phone frame: Col 0 of old R_phone_to_veh.T
        x_phone_old = old_R_phone_to_veh[0, :].copy()

        # Project old forward axis onto the new horizontal plane perpendicular to z_phone
        x_proj = x_phone_old - np.dot(x_phone_old, z_phone) * z_phone
        norm_x = np.linalg.norm(x_proj)

        if norm_x < 1e-4:
            # Degenerate case (phone flipped on side)
            x_phone = np.array([1.0, 0.0, 0.0]) if abs(z_phone[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x_proj = x_phone - np.dot(x_phone, z_phone) * z_phone
            norm_x = np.linalg.norm(x_proj)

        x_phone = x_proj / norm_x

        # Lateral axis (Y = Right in vehicle frame): Y = Z x X
        y_phone = np.cross(z_phone, x_phone)
        y_phone = y_phone / np.linalg.norm(y_phone)

        # Re-ensure exact perpendicularity: X = Y x Z
        x_phone = np.cross(y_phone, z_phone)
        x_phone = x_phone / np.linalg.norm(x_phone)

        # Assemble R_{phone->veh} = [x_phone, y_phone, z_phone]
        R_new = np.vstack([x_phone, y_phone, z_phone])

        # Ensure det(R) = +1
        if np.linalg.det(R_new) < 0:
            R_new[1, :] = -R_new[1, :]

        return R_new
