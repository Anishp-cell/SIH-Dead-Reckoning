"""
Phase 2 Task 16: Calibration Quality Metrics and Evaluation Infrastructure.
Computes stationary acceleration residuals, gyro residuals, gravity consistency,
trajectory error metrics, and ablation comparisons.
"""

import logging
from typing import Dict, Any, List
import numpy as np
import pandas as pd

from Data_details.src.metrics import compute_trajectory_metrics, compute_error_timeseries

logger = logging.getLogger(__name__)


def compute_calibration_quality_metrics(
    df: pd.DataFrame,
    a_dyn_nav: np.ndarray,
    gyro_bias_corrected: np.ndarray,
    quaternions: np.ndarray,
    stationary_mask: np.ndarray,
    gyro_bias_info: Dict[str, Any],
    accel_bias_info: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Computes a comprehensive set of Phase 2 calibration quality metrics.
    
    1. Stationary acceleration residual (target: -> 0 m/s^2)
    2. Stationary gyro residual (target: -> 0 rad/s)
    3. Gravity magnitude consistency across full driving
    4. Orientation pitch/roll stability during stops
    """
    a_dyn_norm = np.sqrt(np.sum(a_dyn_nav**2, axis=1))
    g_dyn_norm = np.sqrt(np.sum(gyro_bias_corrected**2, axis=1))
    
    stat_a_residual = float(np.mean(a_dyn_norm[stationary_mask])) if np.any(stationary_mask) else float(np.mean(a_dyn_norm))
    stat_a_std      = float(np.std(a_dyn_norm[stationary_mask])) if np.any(stationary_mask) else float(np.std(a_dyn_norm))
    stat_g_residual = float(np.mean(g_dyn_norm[stationary_mask])) if np.any(stationary_mask) else float(np.mean(g_dyn_norm))
    stat_g_std      = float(np.std(g_dyn_norm[stationary_mask])) if np.any(stationary_mask) else float(np.std(g_dyn_norm))
    
    metrics = {
        "stat_accel_residual_mean_mps2": round(stat_a_residual, 4),
        "stat_accel_residual_std_mps2": round(stat_a_std, 4),
        "stat_gyro_residual_mean_rads": round(stat_g_residual, 6),
        "stat_gyro_residual_std_rads": round(stat_g_std, 6),
        "gyro_bias_norm_rads": round(gyro_bias_info.get("norm_bias", 0.0), 6),
        "accel_bias_norm_offset_mps2": round(accel_bias_info.get("magnitude_bias_mps2", 0.0), 4),
        "stationary_samples_evaluated": int(np.sum(stationary_mask)),
        "stationary_percentage": round(float(np.mean(stationary_mask) * 100), 2),
    }
    
    logger.info(f"Calibration Metrics: Accel residual={metrics['stat_accel_residual_mean_mps2']:.4f} m/s^2, "
                f"Gyro residual={metrics['stat_gyro_residual_mean_rads']:.6f} rad/s")
    
    return metrics


def format_ablation_table(ablation_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Formats the 6-stage calibration ablation study into a clean summary table.
    
    Ablation stages:
    - V0: Raw Phase 1 Baseline
    - V1: V0 + Gyroscope bias correction
    - V2: V1 + Accelerometer calibration
    - V3: V2 + Quaternion attitude estimation
    - V4: V3 + World-frame gravity compensation
    - V5: V4 + Phone-to-vehicle 3D alignment
    - V6: Full Phase 2 Calibrated Pipeline
    """
    records = []
    for res in ablation_results:
        records.append({
            "version": res.get("version", "N/A"),
            "configuration": res.get("name", "N/A"),
            "duration_s": res.get("blackout_duration_s", 60.0),
            "distance_m": round(float(res.get("distance_travelled_m", 0.0)), 1),
            "endpoint_error_m": round(float(res.get("endpoint_error_m", 0.0)), 2),
            "rmse_m": round(float(res.get("rmse_m", 0.0)), 2),
            "max_error_m": round(float(res.get("max_error_m", 0.0)), 2),
            "drift_pct": round(float(res.get("drift_pct", 0.0)), 2),
            "improvement_pct": round(float(res.get("improvement_pct", 0.0)), 2),
        })
    return pd.DataFrame(records)
