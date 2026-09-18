"""
Phase 4 Robustness Evaluation Engine: Synthetic Perturbation Stress-Testing.

Evaluates model resilience under controlled synthetic physical perturbations:
1. Additive Gaussian noise (high-frequency road chatter amplification)
2. Accelerometer bias shift (uncompensated thermal/sensor drift)
3. Pothole / curb transient impulses (sudden sharp spikes)
4. Sensor dropout / missing packet simulation
"""

import copy
import logging
from typing import Dict, Any, List
import numpy as np
import torch
import torch.nn as nn

from Data_details.src.phase4.evaluator import compute_metrics

logger = logging.getLogger(__name__)


def evaluate_robustness_perturbations(
    model: nn.Module,
    X_test_norm: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Applies controlled physical perturbations to test windows and records performance degradation.
    """
    model.eval()
    results = {}
    
    # Baseline unperturbed
    with torch.no_grad():
        x_tensor = torch.from_numpy(X_test_norm.astype(np.float32)).to(device)
        y_pred_base = model(x_tensor)["speed"].cpu().numpy().ravel()
    results["baseline"] = compute_metrics(y_test, y_pred_base)
    
    # 1. Additive Gaussian Noise (simulating severe road surface roughness)
    for noise_std in [0.2, 0.5, 1.0]:
        noise = np.random.normal(0, noise_std, size=X_test_norm.shape).astype(np.float32)
        X_perturbed = X_test_norm + noise
        with torch.no_grad():
            x_t = torch.from_numpy(X_perturbed).to(device)
            y_pred = model(x_t)["speed"].cpu().numpy().ravel()
        results[f"gaussian_noise_std_{noise_std}"] = compute_metrics(y_test, y_pred)
        
    # 2. Accelerometer Longitudinal Bias Shift (thermal drift)
    # Channel 0 is acc_fwd_veh
    for bias_val in [0.2, 0.5, 1.0]:
        X_perturbed = copy.deepcopy(X_test_norm)
        X_perturbed[:, :, 0] += bias_val  # Add bias to normalized forward acc
        with torch.no_grad():
            x_t = torch.from_numpy(X_perturbed.astype(np.float32)).to(device)
            y_pred = model(x_t)["speed"].cpu().numpy().ravel()
        results[f"fwd_acc_bias_{bias_val}"] = compute_metrics(y_test, y_pred)
        
    # 3. Pothole / Shock Impulses (sudden transient spikes in vertical acceleration)
    # Channel 2 is acc_up_veh
    X_pothole = copy.deepcopy(X_test_norm)
    # In 20% of windows, inject a 3-step impulse spike
    spike_indices = np.random.choice(len(X_pothole), size=int(0.20 * len(X_pothole)), replace=False)
    for idx in spike_indices:
        t_loc = np.random.randint(10, 25)
        X_pothole[idx, t_loc:t_loc+3, 2] += 3.0  # 3-sigma vertical shock
    with torch.no_grad():
        x_t = torch.from_numpy(X_pothole.astype(np.float32)).to(device)
        y_pred = model(x_t)["speed"].cpu().numpy().ravel()
    results["pothole_transient_shock"] = compute_metrics(y_test, y_pred)
    
    # 4. Sensor Dropout (temporary zeroing of gyroscope channels)
    # Channels 3, 4, 5 are gyroscopes
    X_dropout = copy.deepcopy(X_test_norm)
    drop_indices = np.random.choice(len(X_dropout), size=int(0.15 * len(X_dropout)), replace=False)
    for idx in drop_indices:
        X_dropout[idx, :, 3:6] = 0.0  # Zero out gyro
    with torch.no_grad():
        x_t = torch.from_numpy(X_dropout.astype(np.float32)).to(device)
        y_pred = model(x_t)["speed"].cpu().numpy().ravel()
    results["gyro_dropout_15pct"] = compute_metrics(y_test, y_pred)
    
    return results
