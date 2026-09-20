"""
Phase 5 Filter Consistency & Credibility Analysis: NIS & NEES Statistics.
"""

from typing import Dict, Any, List
import numpy as np


def compute_nis_statistics(
    nu_list: List[float],
    S_list: List[float],
    threshold: float = 9.0,
) -> Dict[str, Any]:
    """
    Computes Normalized Innovation Squared (NIS) consistency metrics.
    For a 1-DOF scalar measurement: NIS = nu^2 / S ~ chi^2(1).
    Theoretical mean = 1.0, theoretical 95% bound = 3.84, 99.7% bound = 9.0.
    """
    nu = np.asarray(nu_list, dtype=np.float64)
    S = np.asarray(S_list, dtype=np.float64)

    valid_mask = (S > 0) & np.isfinite(nu) & np.isfinite(S)
    if not np.any(valid_mask):
        return {"mean_nis": 0.0, "acceptance_rate_pct": 0.0, "total_updates": 0}

    nis_vals = (nu[valid_mask] ** 2) / S[valid_mask]
    mean_nis = float(np.mean(nis_vals))
    accepted_mask = nis_vals <= threshold
    acceptance_rate = float(np.mean(accepted_mask) * 100.0)

    return {
        "mean_nis": round(mean_nis, 4),
        "median_nis": round(float(np.median(nis_vals)), 4),
        "p95_nis": round(float(np.percentile(nis_vals, 95)), 4),
        "max_nis": round(float(np.max(nis_vals)), 4),
        "acceptance_rate_pct": round(acceptance_rate, 2),
        "total_updates": int(len(nis_vals)),
        "accepted_updates": int(np.sum(accepted_mask)),
        "rejected_updates": int(np.sum(~accepted_mask)),
    }


def compute_nees_statistics(
    p_est: np.ndarray,
    p_ref: np.ndarray,
    p_cov_diags: np.ndarray,
) -> Dict[str, Any]:
    """
    Computes Normalized Estimation Error Squared (NEES) for 2D position.
    NEES_2d = (dx^2 / sigma_x^2) + (dy^2 / sigma_y^2).
    Theoretical expected value for 2-DOF = 2.0.
    """
    dx = p_est[:, 0] - p_ref[:, 0]
    dy = p_est[:, 1] - p_ref[:, 1]

    var_x = p_cov_diags[:, 0]
    var_y = p_cov_diags[:, 1]

    # Guard against near-zero variances
    var_x = np.maximum(var_x, 1e-6)
    var_y = np.maximum(var_y, 1e-6)

    nees_2d = (dx ** 2) / var_x + (dy ** 2) / var_y
    mean_nees = float(np.mean(nees_2d))

    return {
        "mean_nees_2d": round(mean_nees, 4),
        "median_nees_2d": round(float(np.median(nees_2d)), 4),
        "p95_nees_2d": round(float(np.percentile(nees_2d, 95)), 4),
        "nees_in_bounds_pct": round(float(np.mean(nees_2d <= 5.99) * 100.0), 2),  # 95% bound for chi^2(2) = 5.99
    }
