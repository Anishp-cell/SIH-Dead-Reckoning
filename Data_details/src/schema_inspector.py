"""
Task 3 & 4: Schema Discovery and Data Dictionary Builder for IO-VNBD.
Inspects CSV headers, dtypes, column statistics, and builds a comprehensive data dictionary.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# === Semantic mapping tables from the IO-VNBD paper ===

SMARTPHONE_COLUMN_SEMANTICS = {
    "gps_lat": {"meaning": "GPS latitude from smartphone receiver", "sensor": "Smartphone GPS", "unit": "degrees", "source": "paper Table 4"},
    "gps_lon": {"meaning": "GPS longitude from smartphone receiver", "sensor": "Smartphone GPS", "unit": "degrees", "source": "paper Table 4"},
    "gps_alt": {"meaning": "GPS altitude above sea level", "sensor": "Smartphone GPS", "unit": "m", "source": "paper Table 4"},
    "gps_speed_kmh": {"meaning": "Speed from smartphone GPS receiver", "sensor": "Smartphone GPS", "unit": "km/h", "source": "paper Table 4"},
    "gps_acc_m": {"meaning": "Estimated horizontal accuracy of GPS fix", "sensor": "Smartphone GPS", "unit": "m", "source": "paper Table 4"},
    "gps_heading_deg": {"meaning": "GPS-derived heading/bearing of travel", "sensor": "Smartphone GPS", "unit": "degrees", "source": "paper Table 4"},
    "gps_sats": {"meaning": "Number of GPS satellites in range (may be 'used / total')", "sensor": "Smartphone GPS", "unit": "count", "source": "paper Table 4"},
    "time_ms": {"meaning": "Time elapsed since recording start", "sensor": "AndroSensor app", "unit": "ms", "source": "paper Table 4"},
    "datetime_raw": {"meaning": "Wall-clock date and time of each sample", "sensor": "System clock", "unit": "YYYY-MO-DD HH-MI-SS_SSS", "source": "paper Table 4"},
    "acc_x": {"meaning": "Accelerometer X-axis (phone body frame, includes gravity)", "sensor": "3-axis accelerometer", "unit": "m/s²", "source": "paper Table 4"},
    "acc_y": {"meaning": "Accelerometer Y-axis (phone body frame, includes gravity)", "sensor": "3-axis accelerometer", "unit": "m/s²", "source": "paper Table 4"},
    "acc_z": {"meaning": "Accelerometer Z-axis (phone body frame, includes gravity)", "sensor": "3-axis accelerometer", "unit": "m/s²", "source": "paper Table 4"},
    "grav_x": {"meaning": "Gravity component along X-axis (from Android sensor fusion)", "sensor": "Android gravity sensor", "unit": "m/s²", "source": "paper Table 4"},
    "grav_y": {"meaning": "Gravity component along Y-axis (from Android sensor fusion)", "sensor": "Android gravity sensor", "unit": "m/s²", "source": "paper Table 4"},
    "grav_z": {"meaning": "Gravity component along Z-axis (from Android sensor fusion)", "sensor": "Android gravity sensor", "unit": "m/s²", "source": "paper Table 4"},
    "gyro_x": {"meaning": "Gyroscope yaw angular velocity around X-axis", "sensor": "3-axis gyroscope", "unit": "rad/s", "source": "paper Table 4"},
    "gyro_y": {"meaning": "Gyroscope pitch angular velocity around Y-axis", "sensor": "3-axis gyroscope", "unit": "rad/s", "source": "paper Table 4"},
    "gyro_z": {"meaning": "Gyroscope roll angular velocity around Z-axis", "sensor": "3-axis gyroscope", "unit": "rad/s", "source": "paper Table 4"},
    "mag_x": {"meaning": "Magnetic field strength along X-axis", "sensor": "3-axis magnetometer", "unit": "μT", "source": "paper Table 4"},
    "mag_y": {"meaning": "Magnetic field strength along Y-axis", "sensor": "3-axis magnetometer", "unit": "μT", "source": "paper Table 4"},
    "mag_z": {"meaning": "Magnetic field strength along Z-axis", "sensor": "3-axis magnetometer", "unit": "μT", "source": "paper Table 4"},
    "ori_yaw_deg": {"meaning": "Orientation azimuth/yaw from Android sensor fusion", "sensor": "Android orientation sensor", "unit": "degrees", "source": "paper Table 4"},
    "ori_pitch_deg": {"meaning": "Orientation pitch from Android sensor fusion", "sensor": "Android orientation sensor", "unit": "degrees", "source": "paper Table 4"},
    "ori_roll_deg": {"meaning": "Orientation roll from Android sensor fusion", "sensor": "Android orientation sensor", "unit": "degrees", "source": "paper Table 4"},
}

VEHICLE_COLUMN_SEMANTICS = {
    "v_gps_sats": {"meaning": "Number of GPS satellites available to VBOX", "sensor": "Racelogic VBOX GPS", "unit": "count", "source": "paper Table 3"},
    "v_time_day_s": {"meaning": "Seconds elapsed since start of day (GPS time-of-day)", "sensor": "Racelogic VBOX", "unit": "seconds", "source": "paper Table 3"},
    "v_gps_lat": {"meaning": "GPS latitude from VBOX antenna", "sensor": "Racelogic VBOX GPS", "unit": "degrees", "source": "paper Table 3"},
    "v_gps_lon": {"meaning": "GPS longitude from VBOX antenna", "sensor": "Racelogic VBOX GPS", "unit": "degrees", "source": "paper Table 3"},
    "v_gps_speed_kmh": {"meaning": "GPS-derived velocity from VBOX", "sensor": "Racelogic VBOX GPS", "unit": "km/h", "source": "paper Table 3"},
    "v_gps_heading_deg": {"meaning": "GPS-derived heading from VBOX", "sensor": "Racelogic VBOX GPS", "unit": "degrees", "source": "paper Table 3"},
    "v_gps_height_km": {"meaning": "GPS-derived height above sea level", "sensor": "Racelogic VBOX GPS", "unit": "km", "source": "paper Table 3"},
    "v_vert_vel_kmh": {"meaning": "Vertical velocity from GPS", "sensor": "Racelogic VBOX GPS", "unit": "km/h", "source": "paper Table 3"},
    "v_sample_period_s": {"meaning": "Actual sample period between consecutive readings", "sensor": "Racelogic VBOX", "unit": "seconds", "source": "paper Table 3"},
    "v_steering_deg": {"meaning": "Steering wheel angle from CAN bus", "sensor": "Ford Fiesta CAN bus", "unit": "degrees", "source": "paper Table 3"},
    "v_wheel_fl_rads": {"meaning": "Front-left wheel speed", "sensor": "Ford Fiesta wheel encoder", "unit": "rad/s", "source": "paper Table 3"},
    "v_wheel_fr_rads": {"meaning": "Front-right wheel speed", "sensor": "Ford Fiesta wheel encoder", "unit": "rad/s", "source": "paper Table 3"},
    "v_wheel_rl_rads": {"meaning": "Rear-left wheel speed", "sensor": "Ford Fiesta wheel encoder", "unit": "rad/s", "source": "paper Table 3"},
    "v_wheel_rr_rads": {"meaning": "Rear-right wheel speed", "sensor": "Ford Fiesta wheel encoder", "unit": "rad/s", "source": "paper Table 3"},
    "v_yaw_rate_degs": {"meaning": "Vehicle yaw rate from CAN bus IMU", "sensor": "Ford Fiesta CAN bus IMU", "unit": "deg/s", "source": "paper Table 3"},
    "v_speed_kmh": {"meaning": "Indicated vehicle speed from CAN bus (speedometer)", "sensor": "Ford Fiesta CAN bus", "unit": "km/h", "source": "paper Table 3"},
    "v_long_acc_g": {"meaning": "Longitudinal acceleration from CAN bus", "sensor": "Ford Fiesta CAN bus", "unit": "g", "source": "paper Table 3"},
    "v_lat_acc_g": {"meaning": "Lateral acceleration from CAN bus", "sensor": "Ford Fiesta CAN bus", "unit": "g", "source": "paper Table 3"},
    "v_handbrake": {"meaning": "Handbrake status", "sensor": "Ford Fiesta CAN bus", "unit": "0 or 1", "source": "paper Table 3"},
    "v_gear_req": {"meaning": "Gear requested by driver", "sensor": "Ford Fiesta CAN bus", "unit": "gear number 1-5", "source": "paper Table 3"},
    "v_gear": {"meaning": "Actual gear engaged", "sensor": "Ford Fiesta CAN bus", "unit": "gear number 1-5", "source": "paper Table 3"},
    "v_engine_rpm": {"meaning": "Engine rotational speed", "sensor": "Ford Fiesta CAN bus", "unit": "rev/min", "source": "paper Table 3"},
    "v_coolant_temp_c": {"meaning": "Engine coolant temperature", "sensor": "Ford Fiesta CAN bus", "unit": "°C", "source": "paper Table 3"},
    "v_clutch": {"meaning": "Clutch pedal position", "sensor": "Ford Fiesta CAN bus", "unit": "0 or 1", "source": "paper Table 3"},
    "v_brake_psi": {"meaning": "Brake line pressure", "sensor": "Ford Fiesta CAN bus", "unit": "PSI", "source": "paper Table 3"},
    "v_brake_pos": {"meaning": "Brake pedal position", "sensor": "Ford Fiesta CAN bus", "unit": "0 or 1", "source": "paper Table 3"},
    "v_battery_v": {"meaning": "Vehicle battery voltage", "sensor": "Ford Fiesta CAN bus", "unit": "volts", "source": "paper Table 3"},
    "v_air_temp_c": {"meaning": "Ambient air temperature", "sensor": "Ford Fiesta CAN bus", "unit": "°C", "source": "paper Table 3"},
    "v_accel_pedal": {"meaning": "Accelerator pedal position", "sensor": "Ford Fiesta CAN bus", "unit": "% activation", "source": "paper Table 3"},
}


def inspect_raw_schema(file_path: Path, encoding: str = "latin1") -> Dict[str, Any]:
    """Reads a CSV file header and first/last rows to determine schema."""
    logger.info(f"Inspecting schema for: {file_path.name}")
    
    df = pd.read_csv(file_path, encoding=encoding, nrows=0)
    raw_columns = [c.strip() for c in df.columns.tolist()]
    
    # Read full file for statistics
    df_full = pd.read_csv(file_path, encoding=encoding)
    df_full.columns = df_full.columns.str.strip()
    
    n_rows = len(df_full)
    n_cols = len(df_full.columns)
    
    col_info = []
    for col in df_full.columns:
        series = df_full[col]
        info = {
            "raw_column_name": col,
            "dtype": str(series.dtype),
            "non_null_count": int(series.count()),
            "null_count": int(series.isna().sum()),
            "null_pct": round(series.isna().mean() * 100, 2),
        }
        
        if pd.api.types.is_numeric_dtype(series):
            info["is_numeric"] = True
            info["min"] = float(series.min()) if series.count() > 0 else None
            info["max"] = float(series.max()) if series.count() > 0 else None
            info["mean"] = float(series.mean()) if series.count() > 0 else None
            info["std"] = float(series.std()) if series.count() > 0 else None
            info["is_timestamp_like"] = ("time" in col.lower() or "date" in col.lower())
        else:
            info["is_numeric"] = False
            info["unique_count"] = int(series.nunique())
            info["sample_values"] = series.dropna().head(3).tolist()
            info["is_timestamp_like"] = ("time" in col.lower() or "date" in col.lower())
        
        col_info.append(info)
    
    return {
        "file_name": file_path.name,
        "file_size_bytes": file_path.stat().st_size,
        "encoding": encoding,
        "delimiter": ",",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "raw_columns": raw_columns,
        "columns": col_info,
    }


def build_data_dictionary(
    schema_info: Dict[str, Any],
    standardized_df: pd.DataFrame,
    domain: str,
    source_file: str,
) -> pd.DataFrame:
    """
    Builds a comprehensive data dictionary for a standardized DataFrame,
    mapping each column to its semantic meaning, unit, sensor, and stats.
    """
    semantics = SMARTPHONE_COLUMN_SEMANTICS if domain == "smartphone" else VEHICLE_COLUMN_SEMANTICS
    
    records = []
    for i, col in enumerate(standardized_df.columns):
        series = standardized_df.iloc[:, i]
        sem = semantics.get(col, {})
        
        rec = {
            "column_name": col,
            "source_file": source_file,
            "domain": domain,
            "semantic_meaning": sem.get("meaning", "Derived/computed field" if col in ["time_s", "dt", "gps_speed_mps", "v_speed_mps"] else "UNKNOWN"),
            "sensor": sem.get("sensor", "Computed" if col in ["time_s", "dt", "gps_speed_mps", "v_speed_mps"] else "UNKNOWN"),
            "unit": sem.get("unit", "s" if col == "time_s" else ("s" if col == "dt" else ("m/s" if "mps" in col else "UNKNOWN"))),
            "dtype": str(series.dtype),
            "interpretation_source": sem.get("source", "inferred from data"),
        }
        
        if pd.api.types.is_numeric_dtype(series):
            valid = series.dropna()
            rec["observed_min"] = float(valid.min()) if len(valid) > 0 else None
            rec["observed_max"] = float(valid.max()) if len(valid) > 0 else None
            rec["observed_mean"] = round(float(valid.mean()), 6) if len(valid) > 0 else None
            rec["observed_std"] = round(float(valid.std()), 6) if len(valid) > 0 else None
            rec["missing_count"] = int(series.isna().sum())
            rec["missing_pct"] = round(series.isna().mean() * 100, 2)
        else:
            rec["observed_min"] = None
            rec["observed_max"] = None
            rec["observed_mean"] = None
            rec["observed_std"] = None
            rec["missing_count"] = int(series.isna().sum())
            rec["missing_pct"] = round(series.isna().mean() * 100, 2)
        
        records.append(rec)
    
    return pd.DataFrame(records)


def generate_schema_report(
    s_schema: Dict[str, Any],
    v_schema: Optional[Dict[str, Any]],
    output_dir: str = "Data_details/outputs/tables",
) -> None:
    """Saves schema report as CSV and JSON."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    all_schemas = [s_schema]
    if v_schema:
        all_schemas.append(v_schema)
    
    # Flatten column info for CSV
    rows = []
    for schema in all_schemas:
        for col_info in schema["columns"]:
            row = {"file_name": schema["file_name"], "n_rows": schema["n_rows"]}
            row.update(col_info)
            rows.append(row)
    
    df = pd.DataFrame(rows)
    csv_path = out_path / "dataset_schema.csv"
    json_path = out_path / "dataset_schema.json"
    
    df.to_csv(csv_path, index=False)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_schemas, f, indent=2, default=str)
    
    logger.info(f"Schema report saved to {csv_path} and {json_path}")


def save_data_dictionary(
    dict_df: pd.DataFrame,
    output_dir: str = "Data_details/outputs/tables",
) -> None:
    """Saves data dictionary as CSV."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "data_dictionary.csv"
    dict_df.to_csv(csv_path, index=False)
    logger.info(f"Data dictionary saved to {csv_path}")


if __name__ == "__main__":
    import yaml
    import sys
    sys.path.insert(0, ".")
    from Data_details.src.dataset_loader import DatasetLoader
    
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    
    with open("Data_details/config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    
    loader = DatasetLoader(
        repo_root=cfg["paths"]["repo_root"],
        cache_dir=cfg["paths"]["data_cache_dir"],
    )
    
    seq = cfg["selected_sequence"]
    s_df, v_df = loader.load_sequence(seq["smartphone_file"], seq.get("vehicle_file"))
    
    # Schema inspection on raw files
    s_path = loader.get_real_file_path(seq["smartphone_file"])
    v_path = loader.get_real_file_path(seq["vehicle_file"]) if seq.get("vehicle_file") else None
    
    s_schema = inspect_raw_schema(s_path)
    v_schema = inspect_raw_schema(v_path) if v_path else None
    
    generate_schema_report(s_schema, v_schema, cfg["paths"]["tables_dir"])
    
    # Data dictionary
    s_dict = build_data_dictionary(s_schema, s_df, "smartphone", seq["smartphone_file"])
    if v_df is not None:
        v_dict = build_data_dictionary(v_schema, v_df, "vehicle", seq["vehicle_file"])
        combined = pd.concat([s_dict, v_dict], ignore_index=True)
    else:
        combined = s_dict
    
    save_data_dictionary(combined, cfg["paths"]["tables_dir"])
    
    print(f"\n{'='*60}")
    print("SCHEMA INSPECTION COMPLETE")
    print(f"{'='*60}")
    print(f"Smartphone: {s_schema['n_rows']} rows x {s_schema['n_cols']} cols")
    if v_schema:
        print(f"Vehicle:    {v_schema['n_rows']} rows x {v_schema['n_cols']} cols")
    print(f"Data dictionary: {len(combined)} columns documented")
    print(f"{'='*60}\n")
