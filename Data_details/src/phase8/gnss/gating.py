"""
Phase 8 3-DOF Chi-Square Normalized Innovation Squared (NIS) Gating
Applies strict statistical outlier rejection and adaptive Huber downweighting
on raw, unclipped innovation vectors.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Tuple
import numpy as np


class GatingDecision(str, Enum):
    ACCEPT = "ACCEPT"
    DOWNWEIGHT = "DOWNWEIGHT"
    REJECT = "REJECT"


@dataclass
class GatingResult:
    decision: GatingDecision
    nis: float
    inflation_factor: float
    threshold_99: float = 11.345


class ChiSquareGating3DOF:
    """
    3-DOF Chi-Square Gating for GNSS 3D position and velocity innovations.
    DOF = 3:
      - 95% confidence threshold: 7.815
      - 99% confidence threshold: 11.345
      - Severe outlier threshold: 25.0
    """

    def __init__(
        self,
        threshold_accept: float = 11.345,   # 99% Chi2 threshold for 3 DOF
        threshold_reject: float = 25.000,   # Severe outlier cutoff
    ):
        self.threshold_accept = threshold_accept
        self.threshold_reject = threshold_reject

    def evaluate(
        self,
        nu: np.ndarray,
        S: np.ndarray,
    ) -> GatingResult:
        """
        Computes NIS = nu^T S^{-1} nu and determines gating decision.
        """
        nu_vec = np.asarray(nu, dtype=np.float64).ravel()
        if len(nu_vec) != 3 or S.shape != (3, 3):
            raise ValueError(f"Expected 3-DOF innovation and 3x3 S matrix, got {len(nu_vec)} and {S.shape}")

        if not (np.all(np.isfinite(nu_vec)) and np.all(np.isfinite(S))):
            return GatingResult(
                decision=GatingDecision.REJECT,
                nis=float("inf"),
                inflation_factor=1.0,
                threshold_99=self.threshold_accept,
            )

        try:
            # Solve S * x = nu -> x = S^{-1} nu
            sol = np.linalg.solve(S, nu_vec)
            nis = float(nu_vec @ sol)
        except np.linalg.LinAlgError:
            # Singular S matrix
            return GatingResult(
                decision=GatingDecision.REJECT,
                nis=float("inf"),
                inflation_factor=1.0,
                threshold_99=self.threshold_accept,
            )

        if nis <= self.threshold_accept:
            return GatingResult(
                decision=GatingDecision.ACCEPT,
                nis=nis,
                inflation_factor=1.0,
                threshold_99=self.threshold_accept,
            )
        elif nis <= self.threshold_reject:
            # Downweight by inflating covariance proportionally
            inflation = max(1.0, nis / self.threshold_accept)
            return GatingResult(
                decision=GatingDecision.DOWNWEIGHT,
                nis=nis,
                inflation_factor=inflation,
                threshold_99=self.threshold_accept,
            )
        else:
            return GatingResult(
                decision=GatingDecision.REJECT,
                nis=nis,
                inflation_factor=1.0,
                threshold_99=self.threshold_accept,
            )


class ChiSquareGating2DOF:
    """
    2-DOF Chi-Square Gating for GNSS 2D horizontal velocity innovations [v_E, v_N].
    DOF = 2:
      - 95% confidence threshold: 5.991
      - 99% confidence threshold: 9.210
      - Severe outlier threshold: 20.000
    """

    def __init__(
        self,
        threshold_accept: float = 9.210,    # 99% Chi2 threshold for 2 DOF
        threshold_reject: float = 20.000,   # Severe outlier cutoff
    ):
        self.threshold_accept = threshold_accept
        self.threshold_reject = threshold_reject

    def evaluate(
        self,
        nu: np.ndarray,
        S: np.ndarray,
    ) -> GatingResult:
        """
        Computes NIS = nu^T S^{-1} nu and determines gating decision for 2D vector.
        """
        nu_vec = np.asarray(nu, dtype=np.float64).ravel()
        if len(nu_vec) != 2 or S.shape != (2, 2):
            raise ValueError(f"Expected 2-DOF innovation and 2x2 S matrix, got {len(nu_vec)} and {S.shape}")

        if not (np.all(np.isfinite(nu_vec)) and np.all(np.isfinite(S))):
            return GatingResult(
                decision=GatingDecision.REJECT,
                nis=float("inf"),
                inflation_factor=1.0,
                threshold_99=self.threshold_accept,
            )

        try:
            sol = np.linalg.solve(S, nu_vec)
            nis = float(nu_vec @ sol)
        except np.linalg.LinAlgError:
            return GatingResult(
                decision=GatingDecision.REJECT,
                nis=float("inf"),
                inflation_factor=1.0,
                threshold_99=self.threshold_accept,
            )

        if nis <= self.threshold_accept:
            return GatingResult(
                decision=GatingDecision.ACCEPT,
                nis=nis,
                inflation_factor=1.0,
                threshold_99=self.threshold_accept,
            )
        elif nis <= self.threshold_reject:
            inflation = max(1.0, nis / self.threshold_accept)
            return GatingResult(
                decision=GatingDecision.DOWNWEIGHT,
                nis=nis,
                inflation_factor=inflation,
                threshold_99=self.threshold_accept,
            )
        else:
            return GatingResult(
                decision=GatingDecision.REJECT,
                nis=nis,
                inflation_factor=1.0,
                threshold_99=self.threshold_accept,
            )

