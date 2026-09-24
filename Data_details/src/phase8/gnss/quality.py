"""
Phase 8 GNSS Signal Quality Assessment and Outlier Pre-Filter
Evaluates constellation geometry, stated horizontal accuracy, physical velocity limits,
and step jump anomalies before Kalman gating.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np


@dataclass
class GNSSQualityReport:
    """
    Structured outcome of the GNSS Quality Assessment.
    """
    is_valid: bool
    rejection_reasons: List[str] = field(default_factory=list)
    num_sats: int = 0
    stated_acc_m: float = 0.0
    scaled_acc_m: float = 0.0
    speed_mps: float = 0.0
    bearing_deg: float = 0.0
    jump_detected: bool = False
    step_jump_m: float = 0.0
    apparent_vel_mps: float = 0.0


class GNSSQualityAssessor:
    """
    Autonomous signal quality assessor for smartphone / receiver GNSS.
    """

    def __init__(
        self,
        min_sats: int = 4,
        max_stated_acc_m: float = 15.0,
        max_physical_speed_mps: float = 45.0,     # ~160 km/h
        max_step_jump_m: float = 20.0,
    ):
        self.min_sats = min_sats
        self.max_stated_acc_m = max_stated_acc_m
        self.max_physical_speed_mps = max_physical_speed_mps
        self.max_step_jump_m = max_step_jump_m

        self.last_timestamp: Optional[float] = None
        self.last_p_enu: Optional[np.ndarray] = None

    def reset(self) -> None:
        """Clears memory of previous fixes."""
        self.last_timestamp = None
        self.last_p_enu = None

    def evaluate(
        self,
        timestamp: float,
        p_enu: np.ndarray,
        stated_acc_m: float,
        num_sats: int,
        speed_mps: Optional[float] = None,
        bearing_deg: Optional[float] = None,
    ) -> GNSSQualityReport:
        """
        Assesses a new GNSS sample. Returns GNSSQualityReport.
        """
        reasons: List[str] = []
        p_arr = np.asarray(p_enu, dtype=np.float64).ravel()[:3]

        # 1. Finite numerical check
        if not np.all(np.isfinite(p_arr)):
            reasons.append("NON_FINITE_POSITION")
        if not np.isfinite(stated_acc_m) or stated_acc_m <= 0.0:
            reasons.append("INVALID_STATED_ACCURACY")

        # 2. Minimum Satellite threshold
        if num_sats < self.min_sats:
            reasons.append(f"INSUFFICIENT_SATELLITES_{num_sats}_LT_{self.min_sats}")

        # 3. Maximum Stated Accuracy threshold
        if stated_acc_m > self.max_stated_acc_m:
            reasons.append(f"HIGH_STATED_UNCERTAINTY_{stated_acc_m:.1f}m_GT_{self.max_stated_acc_m:.1f}m")

        # 4. Physical Jump & Velocity check relative to previous fix
        jump_detected = False
        step_dist = 0.0
        apparent_vel = 0.0

        if self.last_p_enu is not None and self.last_timestamp is not None and np.all(np.isfinite(p_arr)):
            dt = timestamp - self.last_timestamp
            if 0.0 < dt < 3.0:
                delta_pos = p_arr - self.last_p_enu
                step_dist = float(np.linalg.norm(delta_pos[:2]))
                apparent_vel = step_dist / dt

                if step_dist > self.max_step_jump_m:
                    jump_detected = True
                    reasons.append(f"POSITION_STEP_JUMP_{step_dist:.1f}m_GT_{self.max_step_jump_m:.1f}m")

                if apparent_vel > self.max_physical_speed_mps:
                    jump_detected = True
                    reasons.append(f"UNPHYSICAL_VELOCITY_{apparent_vel:.1f}mps_GT_{self.max_physical_speed_mps:.1f}mps")

        # Compute adaptive scaled accuracy
        base_sigma = max(0.5, stated_acc_m)
        scale_factor = 1.0
        if num_sats <= 4:
            scale_factor *= 1.5
        elif num_sats <= 6:
            scale_factor *= 1.2

        if base_sigma > 8.0:
            scale_factor *= 1.3

        scaled_acc = base_sigma * scale_factor

        is_valid = (len(reasons) == 0)

        # Update last known position if valid
        if is_valid:
            self.last_p_enu = p_arr.copy()
            self.last_timestamp = timestamp

        return GNSSQualityReport(
            is_valid=is_valid,
            rejection_reasons=reasons,
            num_sats=num_sats,
            stated_acc_m=float(stated_acc_m),
            scaled_acc_m=float(scaled_acc),
            speed_mps=float(speed_mps if speed_mps is not None else 0.0),
            bearing_deg=float(bearing_deg if bearing_deg is not None else 0.0),
            jump_detected=jump_detected,
            step_jump_m=step_dist,
            apparent_vel_mps=apparent_vel,
        )
