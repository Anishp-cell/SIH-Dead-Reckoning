"""
Task 12: Synthetic GNSS Blackout Simulator for IO-VNBD.
Creates non-destructive derived datasets where GPS fields are masked during configurable windows.
"""

import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# GPS-related columns to mask during blackout
SMARTPHONE_GPS_COLS = [
    "gps_lat", "gps_lon", "gps_alt", "gps_speed_kmh", "gps_speed_mps",
    "gps_acc_m", "gps_heading_deg", "gps_sats",
]

VEHICLE_GPS_COLS = [
    "v_gps_lat", "v_gps_lon", "v_gps_speed_kmh", "v_gps_heading_deg",
    "v_gps_height_km", "v_vert_vel_kmh", "v_gps_sats",
]


def create_blackout_mask(
    time_s: np.ndarray,
    blackout_start_s: float,
    blackout_duration_s: float,
) -> np.ndarray:
    """
    Creates a boolean mask indicating which samples fall within the blackout window.
    True = sample is within blackout (GPS unavailable).
    """
    blackout_end_s = blackout_start_s + blackout_duration_s
    mask = (time_s >= blackout_start_s) & (time_s < blackout_end_s)
    
    n_masked = mask.sum()
    logger.info(f"Blackout mask: {blackout_start_s:.1f}s to {blackout_end_s:.1f}s "
                f"({blackout_duration_s:.0f}s), {n_masked} samples masked")
    
    return mask


def apply_blackout(
    df: pd.DataFrame,
    blackout_mask: np.ndarray,
    domain: str = "smartphone",
) -> pd.DataFrame:
    """
    Creates a copy of the DataFrame with GPS columns set to NaN during the blackout.
    Does NOT modify the original DataFrame.
    """
    df_out = df.copy()
    
    gps_cols = SMARTPHONE_GPS_COLS if domain == "smartphone" else VEHICLE_GPS_COLS
    
    cols_masked = []
    for col in gps_cols:
        if col in df_out.columns:
            df_out.loc[blackout_mask, col] = np.nan
            cols_masked.append(col)
    
    # Add blackout indicator column
    df_out["gnss_available"] = ~blackout_mask
    
    logger.info(f"Blackout applied: masked {len(cols_masked)} GPS columns for "
                f"{blackout_mask.sum()} samples")
    
    return df_out


def simulate_blackout(
    df: pd.DataFrame,
    blackout_start_s: float,
    blackout_duration_s: float,
    domain: str = "smartphone",
) -> pd.DataFrame:
    """
    Convenience function: creates and applies a single blackout.
    Returns a new DataFrame with GPS masked.
    """
    mask = create_blackout_mask(df["time_s"].values, blackout_start_s, blackout_duration_s)
    return apply_blackout(df, mask, domain)


def simulate_multiple_blackouts(
    df: pd.DataFrame,
    blackout_start_s: float,
    durations_s: List[float],
    domain: str = "smartphone",
) -> Dict[float, pd.DataFrame]:
    """
    Creates multiple blackout scenarios with different durations.
    Returns dict mapping duration -> blacked-out DataFrame.
    """
    results = {}
    for dur in durations_s:
        results[dur] = simulate_blackout(df, blackout_start_s, dur, domain)
    return results


def get_blackout_window_info(
    df: pd.DataFrame,
    blackout_start_s: float,
    blackout_duration_s: float,
) -> Dict[str, Any]:
    """Returns metadata about a blackout window."""
    blackout_end_s = blackout_start_s + blackout_duration_s
    mask = create_blackout_mask(df["time_s"].values, blackout_start_s, blackout_duration_s)
    
    # Reference GPS positions at start and end
    start_idx = np.argmax(mask)
    end_idx = len(mask) - 1 - np.argmax(mask[::-1])
    
    info = {
        "blackout_start_s": blackout_start_s,
        "blackout_end_s": blackout_end_s,
        "blackout_duration_s": blackout_duration_s,
        "n_samples_masked": int(mask.sum()),
        "start_idx": int(start_idx),
        "end_idx": int(end_idx),
    }
    
    # Add GPS reference at blackout boundaries if available
    for col in ["gps_lat", "gps_lon", "gps_speed_kmh"]:
        if col in df.columns:
            info[f"start_{col}"] = float(df[col].iloc[start_idx])
            info[f"end_{col}"] = float(df[col].iloc[end_idx])
    
    return info


if __name__ == "__main__":
    import yaml
    import sys
    sys.path.insert(0, ".")
    from Data_details.src.dataset_loader import DatasetLoader
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    with open("Data_details/config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    loader = DatasetLoader(cfg["paths"]["repo_root"], cfg["paths"]["data_cache_dir"])
    seq = cfg["selected_sequence"]
    s_df, v_df = loader.load_sequence(seq["smartphone_file"], seq.get("vehicle_file"))
    
    bl_cfg = cfg["blackout_simulation"]
    start_s = bl_cfg["default_start_time_s"]
    durations = bl_cfg["durations_s"]
    
    results = simulate_multiple_blackouts(s_df, start_s, durations, "smartphone")
    
    print(f"\n{'='*60}")
    print("GNSS BLACKOUT SIMULATION COMPLETE")
    print(f"{'='*60}")
    for dur, bo_df in results.items():
        n_masked = (~bo_df["gnss_available"]).sum()
        print(f"  Duration {dur:.0f}s: {n_masked} samples masked")
    print(f"{'='*60}\n")
