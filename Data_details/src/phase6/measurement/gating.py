"""
Phase 6 Statistical NIS Innovation Gating for Multi-Dimensional Measurements.
Supports 1D (Speed), 2D (NHC), and 3D (ZUPT / ZARU) updates.
"""

from typing import Tuple, Optional
import numpy as np


CHI2_TABLE_99_PCT = {
    1: 6.635,   # DOF=1 (99.0%) - or 9.0 for 3-sigma
    2: 9.210,   # DOF=2 (99.0%)
    3: 11.345,  # DOF=3 (99.0%)
}

CHI2_TABLE_99_73_PCT = {
    1: 9.000,   # 3-sigma
    2: 11.829,  # 3-sigma
    3: 14.156,  # 3-sigma
}


def get_chi2_threshold(dof: int, use_three_sigma: bool = True) -> float:
    """Returns Chi-square threshold for given degrees of freedom."""
    table = CHI2_TABLE_99_73_PCT if use_three_sigma else CHI2_TABLE_99_PCT
    if dof in table:
        return table[dof]
    # Fallback approximation for higher dof: dof + 3.0 * sqrt(2 * dof)
    return float(dof + 3.0 * np.sqrt(2.0 * dof))


def evaluate_nis_gate(
    nu: np.ndarray,
    S: np.ndarray,
    threshold: Optional[float] = None,
) -> Tuple[bool, float]:
    """
    Computes Normalized Innovation Squared (NIS) and evaluates acceptance gate.
    
    NIS = nu^T * S^{-1} * nu
    
    Parameters:
        nu: Innovation residual vector (m,)
        S: Innovation covariance matrix (m, m) or scalar float
        threshold: Optional custom gate threshold. If None, uses Chi-square 99.73% bound.
        
    Returns:
        accepted: True if NIS <= threshold, False if rejected
        nis: Scalar NIS value
    """
    nu_vec = np.asarray(nu, dtype=np.float64).ravel()
    dof = len(nu_vec)
    thresh = threshold if threshold is not None else get_chi2_threshold(dof)

    if dof == 1:
        s_val = float(S.item() if isinstance(S, np.ndarray) else S)
        if s_val <= 0.0 or not np.isfinite(s_val):
            return False, float("inf")
        nis = float((nu_vec[0] ** 2) / s_val)
    else:
        S_mat = np.asarray(S, dtype=np.float64)
        if S_mat.shape != (dof, dof):
            raise ValueError(f"S matrix shape {S_mat.shape} incompatible with nu length {dof}")
        
        try:
            # Solve S * x = nu for numerical stability instead of inv(S)
            sol = np.linalg.solve(S_mat, nu_vec)
            nis = float(nu_vec @ sol)
        except np.linalg.LinAlgError:
            return False, float("inf")

    if not np.isfinite(nis) or nis < 0.0:
        return False, float("inf")

    accepted = bool(nis <= thresh)
    return accepted, nis
