"""
Phase 8 Smooth Recovery Manager & Recovery Continuity Metrics
Enforces zero-teleportation continuity bounds during GNSS recovery transitions.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import numpy as np

from Data_details.src.phase5.core.quaternion import quaternion_to_euler_zyx


@dataclass
class RecoveryMetricsReport:
    """
    Quantifies trajectory continuity across a GNSS recovery transition.
    """
    timestamp: float
    p_step_m: float                     # Position discontinuity (target <= 3.5m)
    v_step_mps: float                   # Velocity discontinuity (target <= 1.0m/s)
    yaw_step_deg: float                 # Heading discontinuity (target <= 3.0 deg)
    pseudo_accel_mps2: float            # Apparent acceleration spike (target <= 2.5 m/s^2)
    zero_teleportation_passed: bool     # True if all 4 metrics are within safe bounds


class SmoothRecoveryManager:
    """
    Supervises and validates state transitions exiting GNSS outages.
    """

    def __init__(
        self,
        max_p_step_m: float = 3.5,
        max_v_step_mps: float = 1.0,
        max_yaw_step_deg: float = 3.0,
        max_pseudo_accel_mps2: float = 2.5,
    ):
        self.max_p_step_m = max_p_step_m
        self.max_v_step_mps = max_v_step_mps
        self.max_yaw_step_deg = max_yaw_step_deg
        self.max_pseudo_accel_mps2 = max_pseudo_accel_mps2

        self.reports: List[RecoveryMetricsReport] = []

    def evaluate_transition(
        self,
        timestamp: float,
        p_pre: np.ndarray,
        p_post: np.ndarray,
        v_pre: np.ndarray,
        v_post: np.ndarray,
        q_pre: np.ndarray,
        q_post: np.ndarray,
        dt: float = 0.1,
    ) -> RecoveryMetricsReport:
        """
        Evaluates pre-update vs post-update state discontinuity across an update step.
        """
        p_step = float(np.linalg.norm(p_post[:2] - p_pre[:2]))
        v_step = float(np.linalg.norm(v_post[:2] - v_pre[:2]))

        _, _, yaw_pre = quaternion_to_euler_zyx(q_pre)
        _, _, yaw_post = quaternion_to_euler_zyx(q_post)
        yaw_diff_rad = np.arctan2(np.sin(yaw_post - yaw_pre), np.cos(yaw_post - yaw_pre))
        yaw_step = float(np.abs(np.degrees(yaw_diff_rad)))

        dt_safe = max(1e-4, dt)
        pseudo_a = v_step / dt_safe

        passed = (
            p_step <= self.max_p_step_m and
            v_step <= self.max_v_step_mps and
            yaw_step <= self.max_yaw_step_deg and
            pseudo_a <= self.max_pseudo_accel_mps2
        )

        report = RecoveryMetricsReport(
            timestamp=timestamp,
            p_step_m=p_step,
            v_step_mps=v_step,
            yaw_step_deg=yaw_step,
            pseudo_accel_mps2=pseudo_a,
            zero_teleportation_passed=passed,
        )
        self.reports.append(report)
        return report


# Alias matching terminology in audit test suite
SoftRecoveryManager = SmoothRecoveryManager
