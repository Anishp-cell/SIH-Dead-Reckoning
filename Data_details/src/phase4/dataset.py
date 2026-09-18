"""
Phase 4 Dataset Module: Causal Temporal Windowing, Target Alignment & Purge-Gapped Splits.

Loads Phase 3 conditioned IMU signals and synchronizes with vehicle CAN/VBOX ground-truth speed.
Enforces strict time-block splitting, temporal purge gaps, and train-only normalization.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from Data_details.src.dataset_loader import DatasetLoader

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "acc_fwd_veh",
    "acc_lat_veh",
    "acc_up_veh",
    "gyro_roll_veh",
    "gyro_pitch_veh",
    "gyro_yaw_veh",
    "jerk_fwd",
    "acc_horiz_norm",
    "gyro_norm",
    "ori_pitch_deg",
    "vibration_energy",
    "is_stationary",
]

MOTION_STATE_NAMES = ["STANDSTILL", "CRUISING", "ACCELERATING", "BRAKING", "TURNING"]


def compute_motion_states(v_speed: np.ndarray, a_fwd: np.ndarray, gyro_yaw: np.ndarray) -> np.ndarray:
    """
    Classifies vehicle dynamic state into 5 discrete physical regimes:
    0: Standstill (v < 0.2 m/s)
    1: Cruising (|a_fwd| <= 0.4 m/s^2, |gyro_yaw| <= 0.08 rad/s, v >= 0.2 m/s)
    2: Accelerating (a_fwd > 0.4 m/s^2, v >= 0.2 m/s)
    3: Braking (a_fwd < -0.4 m/s^2, v >= 0.2 m/s)
    4: Turning (|gyro_yaw| > 0.08 rad/s, v >= 0.2 m/s)
    """
    states = np.ones(len(v_speed), dtype=np.int64)  # Default: Cruising (1)
    
    # Standstill
    standstill_mask = (v_speed < 0.2)
    states[standstill_mask] = 0
    
    moving_mask = ~standstill_mask
    
    # Turning takes priority when rotating
    turning_mask = moving_mask & (np.abs(gyro_yaw) > 0.08)
    states[turning_mask] = 4
    
    # Acceleration & Braking for non-turning
    non_turning_moving = moving_mask & ~turning_mask
    accel_mask = non_turning_moving & (a_fwd > 0.4)
    states[accel_mask] = 2
    
    brake_mask = non_turning_moving & (a_fwd < -0.4)
    states[brake_mask] = 3
    
    return states


class SpeedDataset(Dataset):
    """PyTorch Dataset for Phase 4 Causal Temporal Windows."""

    def __init__(
        self,
        X: np.ndarray,
        y_speed: np.ndarray,
        y_state: np.ndarray,
        timestamps: np.ndarray,
    ):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y_speed = torch.from_numpy(y_speed.astype(np.float32)).unsqueeze(-1)
        self.y_state = torch.from_numpy(y_state.astype(np.int64))
        self.timestamps = timestamps

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x": self.X[idx],
            "speed": self.y_speed[idx],
            "state": self.y_state[idx],
            "timestamp": torch.tensor(self.timestamps[idx], dtype=torch.float64),
        }


def prepare_phase4_data(
    processed_imu_path: str = "Data_details/outputs/phase3/processed/s1_filtered_causal_imu.csv",
    vehicle_raw_path: str = "V-Dataset/V-S1.csv",
    window_length: int = 30,
    stride: int = 5,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    purge_gap_samples: int = 50,
    normalization_save_path: Optional[str] = "Data_details/outputs/phase4/models/normalization.json",
    manifest_save_path: Optional[str] = "Data_details/outputs/phase4/models/data_manifest.json",
) -> Dict[str, Any]:
    """
    Extracts causal sliding windows from Phase 3 IMU signals, synchronizes vehicle CAN speed,
    executes temporal block partitioning with purge gaps, and normalizes using train statistics.
    """
    logger.info(f"Loading Phase 3 conditioned IMU signals from {processed_imu_path}...")
    df_imu = pd.read_csv(processed_imu_path)
    
    logger.info("Loading ground truth vehicle reference from vehicle dataset...")
    loader = DatasetLoader()
    _, v_df = loader.load_sequence("S-Dataset/S-S1.csv", vehicle_raw_path)
    
    if v_df is None or "v_speed_mps" not in v_df.columns:
        raise RuntimeError("Failed to load vehicle reference speed (v_speed_mps) from V-Dataset.")
        
    n_samples = len(df_imu)
    assert len(v_df) == n_samples, f"Sample length mismatch: IMU={n_samples}, Vehicle={len(v_df)}"
    
    # Ground-truth forward speed from CAN / ECU
    target_speed = v_df["v_speed_mps"].values.astype(np.float32)
    time_s = df_imu["time_s"].values.astype(np.float64)
    
    # Verify all 12 feature channels exist
    for col in FEATURE_COLUMNS:
        if col not in df_imu.columns:
            raise KeyError(f"Required Phase 3 feature column '{col}' missing from {processed_imu_path}")
            
    feature_matrix = df_imu[FEATURE_COLUMNS].values.astype(np.float32)
    a_fwd = df_imu["acc_fwd_veh"].values
    gyro_yaw = df_imu["gyro_yaw_veh"].values
    motion_states = compute_motion_states(target_speed, a_fwd, gyro_yaw)
    
    # Window extraction
    n_windows = (n_samples - window_length) // stride + 1
    logger.info(f"Extracting {n_windows} windows (W={window_length}, S={stride})...")
    
    X_all = np.zeros((n_windows, window_length, len(FEATURE_COLUMNS)), dtype=np.float32)
    y_speed_all = np.zeros(n_windows, dtype=np.float32)
    y_state_all = np.zeros(n_windows, dtype=np.int64)
    timestamps_all = np.zeros(n_windows, dtype=np.float64)
    window_end_indices = np.zeros(n_windows, dtype=np.int64)
    
    for i in range(n_windows):
        start_idx = i * stride
        end_idx = start_idx + window_length
        X_all[i] = feature_matrix[start_idx:end_idx]
        y_speed_all[i] = target_speed[end_idx - 1]
        y_state_all[i] = motion_states[end_idx - 1]
        timestamps_all[i] = time_s[end_idx - 1]
        window_end_indices[i] = end_idx - 1
        
    # Temporal block splitting with purge gap
    # Step 1: Raw index split points
    train_sample_cutoff = int(n_samples * train_ratio)
    val_sample_start = train_sample_cutoff + purge_gap_samples
    val_sample_cutoff = val_sample_start + int(n_samples * val_ratio)
    test_sample_start = val_sample_cutoff + purge_gap_samples
    
    # Step 2: Assign windows based on their endpoint timestamp
    train_mask = window_end_indices < train_sample_cutoff
    val_mask = (window_end_indices >= val_sample_start) & (window_end_indices < val_sample_cutoff)
    test_mask = window_end_indices >= test_sample_start
    
    train_indices = np.where(train_mask)[0]
    val_indices = np.where(val_mask)[0]
    test_indices = np.where(test_mask)[0]
    
    logger.info(
        f"Split partition windows: Train={len(train_indices)}, "
        f"PurgeGap1={np.sum((window_end_indices >= train_sample_cutoff) & (window_end_indices < val_sample_start))}, "
        f"Val={len(val_indices)}, "
        f"PurgeGap2={np.sum((window_end_indices >= val_sample_cutoff) & (window_end_indices < test_sample_start))}, "
        f"Test={len(test_indices)}"
    )
    
    # Train-Only Normalization
    X_train_raw = X_all[train_indices]
    means = np.mean(X_train_raw, axis=(0, 1)).astype(np.float32)
    stds = np.std(X_train_raw, axis=(0, 1)).astype(np.float32)
    stds[stds < 1e-6] = 1.0  # Numerical stability for constant features
    
    # Apply normalization
    X_all_norm = (X_all - means) / stds
    
    norm_dict = {
        "features": FEATURE_COLUMNS,
        "mean": means.tolist(),
        "std": stds.tolist(),
        "train_samples_count": int(len(train_indices)),
        "sampling_rate_hz": 10.0,
        "window_length": window_length,
    }
    
    if normalization_save_path:
        norm_path = Path(normalization_save_path)
        norm_path.parent.mkdir(parents=True, exist_ok=True)
        with open(norm_path, "w") as f:
            json.dump(norm_dict, f, indent=2)
        logger.info(f"Saved feature normalization parameters to {norm_path}")
        
    manifest = {
        "sequence": "S1 (Coventry, UK)",
        "total_samples": int(n_samples),
        "window_length": int(window_length),
        "stride": int(stride),
        "purge_gap_samples": int(purge_gap_samples),
        "train_windows": int(len(train_indices)),
        "val_windows": int(len(val_indices)),
        "test_windows": int(len(test_indices)),
        "train_time_range_s": [float(timestamps_all[train_indices[0]]), float(timestamps_all[train_indices[-1]])],
        "val_time_range_s": [float(timestamps_all[val_indices[0]]), float(timestamps_all[val_indices[-1]])],
        "test_time_range_s": [float(timestamps_all[test_indices[0]]), float(timestamps_all[test_indices[-1]])],
        "target_source": "Ford Fiesta CAN indicated vehicle speed (v_speed_mps)",
        "target_stats": {
            "train_mean_mps": float(np.mean(y_speed_all[train_indices])),
            "val_mean_mps": float(np.mean(y_speed_all[val_indices])),
            "test_mean_mps": float(np.mean(y_speed_all[test_indices])),
            "train_max_mps": float(np.max(y_speed_all[train_indices])),
            "test_max_mps": float(np.max(y_speed_all[test_indices])),
        },
    }
    
    if manifest_save_path:
        man_path = Path(manifest_save_path)
        man_path.parent.mkdir(parents=True, exist_ok=True)
        with open(man_path, "w") as f:
            json.dump(manifest, f, indent=2)
        logger.info(f"Saved dataset manifest to {man_path}")
        
    datasets = {
        "train": SpeedDataset(X_all_norm[train_indices], y_speed_all[train_indices], y_state_all[train_indices], timestamps_all[train_indices]),
        "val": SpeedDataset(X_all_norm[val_indices], y_speed_all[val_indices], y_state_all[val_indices], timestamps_all[val_indices]),
        "test": SpeedDataset(X_all_norm[test_indices], y_speed_all[test_indices], y_state_all[test_indices], timestamps_all[test_indices]),
        "full": SpeedDataset(X_all_norm, y_speed_all, y_state_all, timestamps_all),
        "norm_params": norm_dict,
        "manifest": manifest,
        "raw_arrays": {
            "X_all_norm": X_all_norm,
            "y_speed_all": y_speed_all,
            "y_state_all": y_state_all,
            "timestamps_all": timestamps_all,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "test_indices": test_indices,
        }
    }
    
    return datasets
