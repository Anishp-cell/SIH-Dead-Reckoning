"""
Task 7: Data Quality Analysis for IO-VNBD sequences.
Checks for missing values, infinities, duplicate timestamps, impossible values,
constant columns, extreme outliers, and distributions.
"""

import logging
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# Physically reasonable ranges for validation
SMARTPHONE_VALID_RANGES = {
    "gps_lat": (-90, 90),
    "gps_lon": (-180, 180),
    "gps_alt": (-500, 100000),
    "gps_speed_kmh": (0, 300),
    "gps_acc_m": (0, 1000),
    "acc_x": (-200, 200),
    "acc_y": (-200, 200),
    "acc_z": (-200, 200),
    "grav_x": (-15, 15),
    "grav_y": (-15, 15),
    "grav_z": (-15, 15),
    "gyro_x": (-50, 50),
    "gyro_y": (-50, 50),
    "gyro_z": (-50, 50),
    "mag_x": (-500, 500),
    "mag_y": (-500, 500),
    "mag_z": (-500, 500),
}

VEHICLE_VALID_RANGES = {
    "v_gps_lat": (-90, 90),
    "v_gps_lon": (-180, 180),
    "v_gps_speed_kmh": (0, 300),
    "v_speed_kmh": (0, 300),
    "v_engine_rpm": (0, 10000),
    "v_long_acc_g": (-5, 5),
    "v_lat_acc_g": (-5, 5),
    "v_yaw_rate_degs": (-500, 500),
    "v_battery_v": (0, 30),
    "v_coolant_temp_c": (-40, 150),
    "v_air_temp_c": (-40, 60),
}


def analyze_data_quality(df: pd.DataFrame, domain: str = "smartphone") -> pd.DataFrame:
    """
    Performs comprehensive data quality analysis on a standardized DataFrame.
    Returns a DataFrame with quality metrics per column.
    """
    valid_ranges = SMARTPHONE_VALID_RANGES if domain == "smartphone" else VEHICLE_VALID_RANGES
    
    records = []
    for col in df.columns:
        series = df[col]
        rec = {
            "column": col,
            "domain": domain,
            "dtype": str(series.dtype),
            "total_count": len(series),
            "non_null_count": int(series.count()),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
        }
        
        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            rec["inf_count"] = int(np.isinf(valid).sum()) if len(valid) > 0 else 0
            rec["min"] = float(valid.min()) if len(valid) > 0 else None
            rec["max"] = float(valid.max()) if len(valid) > 0 else None
            rec["mean"] = round(float(valid.mean()), 6) if len(valid) > 0 else None
            rec["std"] = round(float(valid.std()), 6) if len(valid) > 0 else None
            rec["median"] = round(float(valid.median()), 6) if len(valid) > 0 else None
            rec["is_constant"] = bool(valid.nunique() <= 1) if len(valid) > 0 else True
            
            # Check for physically impossible values
            if col in valid_ranges:
                lo, hi = valid_ranges[col]
                n_impossible = int(((valid < lo) | (valid > hi)).sum())
                rec["impossible_count"] = n_impossible
                rec["impossible_pct"] = round(n_impossible / len(valid) * 100, 2) if len(valid) > 0 else 0
            else:
                rec["impossible_count"] = 0
                rec["impossible_pct"] = 0.0
            
            # Outlier detection (beyond 5 std from mean)
            if len(valid) > 10 and valid.std() > 0:
                z_scores = np.abs((valid - valid.mean()) / valid.std())
                rec["outlier_5std_count"] = int((z_scores > 5).sum())
            else:
                rec["outlier_5std_count"] = 0
        else:
            rec["inf_count"] = 0
            rec["min"] = None
            rec["max"] = None
            rec["mean"] = None
            rec["std"] = None
            rec["median"] = None
            rec["is_constant"] = bool(series.nunique() <= 1)
            rec["impossible_count"] = 0
            rec["impossible_pct"] = 0.0
            rec["outlier_5std_count"] = 0
        
        records.append(rec)
    
    # Check for duplicate timestamps
    if "time_s" in df.columns:
        n_dup_ts = int(df["time_s"].duplicated().sum())
    else:
        n_dup_ts = 0
    
    quality_df = pd.DataFrame(records)
    logger.info(f"Quality analysis ({domain}): {len(quality_df)} columns, "
                f"{n_dup_ts} duplicate timestamps, "
                f"{quality_df['null_count'].sum()} total nulls")
    
    return quality_df


def generate_quality_report_md(
    s_quality: pd.DataFrame,
    v_quality: pd.DataFrame = None,
    output_dir: str = "Data_details/outputs/reports",
) -> None:
    """Generates a human-readable markdown data quality report."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    lines = [
        "# Data Quality Report — IO-VNBD Phase 1\n",
        "## Smartphone Data Quality\n",
    ]
    
    _add_quality_section(lines, s_quality, "Smartphone")
    
    if v_quality is not None:
        lines.append("\n## Vehicle Data Quality\n")
        _add_quality_section(lines, v_quality, "Vehicle")
    
    report_path = out_path / "data_quality_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    logger.info(f"Quality report saved to {report_path}")


def _add_quality_section(lines: list, quality_df: pd.DataFrame, label: str) -> None:
    """Helper to add quality section to report."""
    total_nulls = quality_df["null_count"].sum()
    total_cells = quality_df["total_count"].sum()
    overall_null_pct = total_nulls / total_cells * 100 if total_cells > 0 else 0
    
    lines.append(f"- **Total columns**: {len(quality_df)}")
    lines.append(f"- **Total null values**: {total_nulls} ({overall_null_pct:.2f}%)")
    
    constant_cols = quality_df[quality_df["is_constant"] == True]
    if len(constant_cols) > 0:
        lines.append(f"- **Constant columns** (no variation): {', '.join(constant_cols['column'].tolist())}")
    
    impossible = quality_df[quality_df["impossible_count"] > 0]
    if len(impossible) > 0:
        lines.append(f"\n### Columns with Physically Impossible Values\n")
        lines.append("| Column | Impossible Count | Impossible % |")
        lines.append("|--------|-----------------|-------------|")
        for _, row in impossible.iterrows():
            lines.append(f"| {row['column']} | {row['impossible_count']} | {row['impossible_pct']}% |")
    
    high_null = quality_df[quality_df["null_pct"] > 1.0]
    if len(high_null) > 0:
        lines.append(f"\n### Columns with >1% Missing Values\n")
        lines.append("| Column | Null Count | Null % |")
        lines.append("|--------|-----------|--------|")
        for _, row in high_null.iterrows():
            lines.append(f"| {row['column']} | {row['null_count']} | {row['null_pct']}% |")
    
    outlier_cols = quality_df[quality_df["outlier_5std_count"] > 0]
    if len(outlier_cols) > 0:
        lines.append(f"\n### Columns with Extreme Outliers (>5σ from mean)\n")
        lines.append("| Column | Outlier Count |")
        lines.append("|--------|--------------|")
        for _, row in outlier_cols.iterrows():
            lines.append(f"| {row['column']} | {row['outlier_5std_count']} |")


def save_quality_csv(
    quality_df: pd.DataFrame,
    output_dir: str = "Data_details/outputs/tables",
) -> None:
    """Saves quality report as CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "data_quality_report.csv"
    quality_df.to_csv(csv_path, index=False)
    logger.info(f"Quality CSV saved to {csv_path}")


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

    s_quality = analyze_data_quality(s_df, "smartphone")
    v_quality = analyze_data_quality(v_df, "vehicle") if v_df is not None else None

    combined = pd.concat([s_quality, v_quality], ignore_index=True) if v_quality is not None else s_quality
    save_quality_csv(combined, cfg["paths"]["tables_dir"])
    generate_quality_report_md(s_quality, v_quality, cfg["paths"]["reports_dir"])

    print(f"\n{'='*60}")
    print("DATA QUALITY ANALYSIS COMPLETE")
    print(f"{'='*60}")
    print(f"Smartphone: {s_quality['null_count'].sum()} total nulls across {len(s_quality)} columns")
    if v_quality is not None:
        print(f"Vehicle:    {v_quality['null_count'].sum()} total nulls across {len(v_quality)} columns")
    print(f"{'='*60}\n")
