"""
Task 14: Trajectory Error Metrics for IO-VNBD baseline evaluation.
Computes position error, RMSE, endpoint drift, drift percentage, and error over time.
"""

import logging
from typing import Dict, Any
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_trajectory_metrics(
    ref_east: np.ndarray,
    ref_north: np.ndarray,
    est_east: np.ndarray,
    est_north: np.ndarray,
    time_s: np.ndarray,
    distance_travelled_m: float,
) -> Dict[str, Any]:
    """
    Computes trajectory comparison metrics between reference and estimated positions.
    
    Parameters:
        ref_east, ref_north: Reference trajectory in ENU (meters)
        est_east, est_north: Estimated trajectory in ENU (meters)
        time_s: Timestamps (seconds)
        distance_travelled_m: Total geodesic distance during evaluation window
    
    Returns:
        Dictionary of error metrics
    """
    # Position error at each timestep
    error = np.sqrt((ref_east - est_east)**2 + (ref_north - est_north)**2)
    
    # Endpoint error
    endpoint_error = error[-1] if len(error) > 0 else 0.0
    
    # RMSE
    rmse = np.sqrt(np.mean(error**2))
    
    # Mean absolute error
    mae = np.mean(error)
    
    # Max error
    max_error = np.max(error)
    
    # Drift percentage
    drift_pct = (endpoint_error / distance_travelled_m * 100) if distance_travelled_m > 0 else float("inf")
    
    metrics = {
        "endpoint_error_m": round(float(endpoint_error), 2),
        "rmse_m": round(float(rmse), 2),
        "mae_m": round(float(mae), 2),
        "max_error_m": round(float(max_error), 2),
        "distance_travelled_m": round(float(distance_travelled_m), 2),
        "drift_pct": round(float(drift_pct), 2),
        "duration_s": round(float(time_s[-1] - time_s[0]), 1) if len(time_s) > 0 else 0,
        "n_samples": len(error),
    }
    
    logger.info(f"Trajectory metrics: endpoint={endpoint_error:.1f}m, RMSE={rmse:.1f}m, "
                f"drift={drift_pct:.1f}%, distance={distance_travelled_m:.0f}m")
    
    return metrics


def compute_error_timeseries(
    ref_east: np.ndarray,
    ref_north: np.ndarray,
    est_east: np.ndarray,
    est_north: np.ndarray,
    time_s: np.ndarray,
) -> pd.DataFrame:
    """
    Computes position error at each timestep.
    
    Returns:
        DataFrame with columns: time_s, error_m, ref_east, ref_north, est_east, est_north
    """
    error = np.sqrt((ref_east - est_east)**2 + (ref_north - est_north)**2)
    
    return pd.DataFrame({
        "time_s": time_s,
        "error_m": error,
        "ref_east": ref_east,
        "ref_north": ref_north,
        "est_east": est_east,
        "est_north": est_north,
    })
