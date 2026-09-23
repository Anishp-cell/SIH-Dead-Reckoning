"""
Phase 6 Robust Measurement Updater: Generalized Joseph-Form Vector Kalman Update Engine.
Supports arbitrary measurement dimension m with guaranteed positive semi-definiteness.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np

from Data_details.src.phase5.core.state import NominalState
from Data_details.src.phase5.core.quaternion import inject_small_angle_error
from Data_details.src.phase6.measurement.gating import evaluate_nis_gate, get_chi2_threshold


@dataclass
class UpdateResult:
    """Standardized output of a Kalman measurement update."""
    state_plus: NominalState
    P_plus: np.ndarray
    nu: np.ndarray
    S: np.ndarray
    nis: float
    accepted: bool


class RobustMeasurementUpdater:
    """
    Executes indirect error-state Kalman update for arbitrary m-dimensional measurement:
    1. Innovation residual: nu = z - h(x)
    2. Innovation covariance: S = H * P * H^T + R
    3. Multi-dimensional Chi-square NIS gating check
    4. Numerically stable Kalman Gain: K = P * H^T * S^{-1} via linear solve
    5. Joseph-form covariance stabilization: P^+ = (I - KH) P (I - KH)^T + K R K^T
    6. Non-linear error injection & error reset
    """

    @staticmethod
    def update(
        nominal_state: NominalState,
        P: np.ndarray,
        z: np.ndarray,
        h_val: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        nis_threshold: Optional[float] = None,
        min_variance: float = 1e-6,
    ) -> UpdateResult:
        z_vec = np.asarray(z, dtype=np.float64).ravel()
        h_vec = np.asarray(h_val, dtype=np.float64).ravel()
        m = len(z_vec)

        if len(h_vec) != m:
            raise ValueError(f"Measurement z dim ({m}) does not match prediction h dim ({len(h_vec)})")

        H_mat = np.asarray(H, dtype=np.float64)
        if H_mat.shape != (m, 15):
            raise ValueError(f"Jacobian H shape {H_mat.shape} must be ({m}, 15)")

        R_mat = np.asarray(R, dtype=np.float64)
        if R_mat.ndim == 1:
            R_mat = np.diag(R_mat)
        elif R_mat.shape != (m, m):
            raise ValueError(f"R shape {R_mat.shape} must be ({m}, {m})")

        # 1. Innovation
        nu = z_vec - h_vec

        # 2. Innovation covariance S = H * P * H^T + R
        HP = H_mat @ P  # (m, 15)
        S = HP @ H_mat.T + R_mat  # (m, m)

        # Enforce minimum diagonal variance
        for i in range(m):
            if S[i, i] < min_variance:
                S[i, i] = min_variance

        # 3. NIS gating check
        thresh = nis_threshold if nis_threshold is not None else get_chi2_threshold(m)
        accepted, nis = evaluate_nis_gate(nu, S, threshold=thresh)

        if not accepted:
            # Outlier rejected: return unperturbed prior state and covariance
            return UpdateResult(
                state_plus=nominal_state.copy(),
                P_plus=P.copy(),
                nu=nu,
                S=S,
                nis=nis,
                accepted=False,
            )

        # 4. Kalman Gain K = P * H^T * S^{-1}
        # Solve S * (K^T) = H * P  ->  K = ( (H * P)^T * S^{-1} )
        try:
            # sol: (m, 15) such that S @ sol = HP
            sol = np.linalg.solve(S, HP)
            K = sol.T  # (15, m)
        except np.linalg.LinAlgError:
            # Ill-conditioned S: fall back to pseudo-inverse
            K = (P @ H_mat.T) @ np.linalg.pinv(S)

        # 5. Error state vector delta_x = K * nu
        delta_x = K @ nu  # (15,)

        # 6. Joseph-form covariance update
        I15 = np.eye(15, dtype=np.float64)
        I_KH = I15 - K @ H_mat  # (15, 15)
        P_plus = I_KH @ P @ I_KH.T + K @ R_mat @ K.T

        # Symmetry enforcement
        P_plus = 0.5 * (P_plus + P_plus.T)

        # 7. Non-linear error injection into nominal state
        dp = delta_x[0:3]
        dv = delta_x[3:6]
        dtheta = delta_x[6:9]
        dba = delta_x[9:12]
        dbg = delta_x[12:15]

        p_new = nominal_state.p + dp
        v_new = nominal_state.v + dv
        q_new = inject_small_angle_error(nominal_state.q, dtheta)
        ba_new = nominal_state.ba + dba
        bg_new = nominal_state.bg + dbg

        state_plus = NominalState(
            p=p_new,
            v=v_new,
            q=q_new,
            ba=ba_new,
            bg=bg_new,
        )

        return UpdateResult(
            state_plus=state_plus,
            P_plus=P_plus,
            nu=nu,
            S=S,
            nis=nis,
            accepted=True,
        )
