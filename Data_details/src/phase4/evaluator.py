"""
Phase 4 Evaluation Engine: Scientific Metrics, Speed-Regime Analysis, and Latency Profiling.

Computes MAE, RMSE, R^2, Max Error, 95th Percentile Error, Bias,
speed-regime breakdowns, dynamic motion condition breakdowns, and CPU/GPU execution benchmarks.
"""

import time
import logging
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Computes comprehensive regression metrics."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.asarray(y_pred, dtype=np.float64).ravel()
    
    residuals = y_pred - y_true
    abs_errors = np.abs(residuals)
    
    mae = float(np.mean(abs_errors))
    rmse = float(np.sqrt(np.mean(residuals ** 2)))
    
    # R^2 calculation
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    ss_res = np.sum(residuals ** 2)
    r2 = float(1.0 - (ss_res / max(ss_tot, 1e-8)))
    
    max_err = float(np.max(abs_errors))
    p95_err = float(np.percentile(abs_errors, 95))
    bias = float(np.mean(residuals))
    
    return {
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2": round(r2, 4),
        "max_error": round(max_err, 4),
        "p95_error": round(p95_err, 4),
        "bias": round(bias, 4),
        "count": int(len(y_true)),
    }


def analyze_speed_regimes(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    bins: Optional[List[Tuple[float, float, str]]] = None,
) -> Dict[str, Any]:
    """
    Evaluates speed estimation performance across discrete velocity regimes.
    Default regimes:
    - Stationary: [0.0, 0.5) m/s
    - Low speed (urban): [0.5, 5.0) m/s
    - Medium speed (suburban): [5.0, 10.0) m/s
    - High speed (arterial): [10.0, 15.0) m/s
    - Highway: [15.0, 30.0) m/s
    """
    if bins is None:
        bins = [
            (0.0, 0.5, "Stationary (<0.5 m/s)"),
            (0.5, 5.0, "Low Speed (0.5-5.0 m/s)"),
            (5.0, 10.0, "Medium Speed (5.0-10.0 m/s)"),
            (10.0, 15.0, "High Speed (10.0-15.0 m/s)"),
            (15.0, 30.0, "Highway (15.0+ m/s)"),
        ]
        
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    
    regime_results = {}
    for low, high, label in bins:
        mask = (y_true >= low) & (y_true < high)
        n_samples = np.sum(mask)
        if n_samples > 0:
            regime_results[label] = compute_metrics(y_true[mask], y_pred[mask])
        else:
            regime_results[label] = {"count": 0, "mae": None, "rmse": None}
            
    return regime_results


def analyze_motion_conditions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    motion_states: np.ndarray,
    state_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluates error breakdown conditioned on vehicle dynamic motion states."""
    if state_names is None:
        state_names = ["STANDSTILL", "CRUISING", "ACCELERATING", "BRAKING", "TURNING"]
        
    condition_results = {}
    for state_idx, name in enumerate(state_names):
        mask = (motion_states == state_idx)
        n_samples = np.sum(mask)
        if n_samples > 0:
            condition_results[name] = compute_metrics(y_true[mask], y_pred[mask])
        else:
            condition_results[name] = {"count": 0, "mae": None, "rmse": None}
            
    return condition_results


def benchmark_model_latency(
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (1, 30, 12),
    device: str = "cpu",
    warmup_runs: int = 50,
    benchmark_runs: int = 200,
) -> Dict[str, float]:
    """
    Profiles inference latency (in ms), throughput (samples/sec),
    and model parameter size.
    """
    model.eval()
    dev = torch.device(device)
    model.to(dev)
    
    dummy_input = torch.randn(*input_shape, device=dev)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup_runs):
            _ = model(dummy_input)
            
    if device == "cuda":
        torch.cuda.synchronize()
        
    # Timed runs
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(benchmark_runs):
            _ = model(dummy_input)
            
    if device == "cuda":
        torch.cuda.synchronize()
        
    elapsed = time.perf_counter() - t0
    avg_latency_ms = (elapsed / benchmark_runs) * 1000.0
    throughput = benchmark_runs / elapsed
    
    n_params = sum(p.numel() for p in model.parameters())
    size_kb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024.0
    
    return {
        "device": device,
        "avg_latency_ms": round(float(avg_latency_ms), 4),
        "throughput_hz": round(float(throughput), 1),
        "parameters": int(n_params),
        "model_size_kb": round(float(size_kb), 2),
    }
