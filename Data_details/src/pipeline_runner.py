"""
Phase 1 Pipeline Runner — Orchestrates all Phase 1 tasks end-to-end.
Run from project root: python -m Data_details.src.pipeline_runner
"""

import sys
import time
import logging
from pathlib import Path
import yaml

# Ensure project root is on path
sys.path.insert(0, ".")

logger = logging.getLogger("phase1_pipeline")


def setup_logging(cfg: dict) -> None:
    """Configures logging for the pipeline run."""
    log_dir = Path(cfg["paths"]["logs_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / "pipeline_run.log"
    log_cfg = cfg.get("logging", {})

    logging.basicConfig(
        level=getattr(logging, log_cfg.get("level", "INFO")),
        format=log_cfg.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
        handlers=[
            logging.FileHandler(log_file, mode="w", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


def run_pipeline():
    """Runs the full Phase 1 pipeline."""
    t0 = time.time()

    # Load config
    with open("Data_details/config/config.yaml", "r") as f:
        cfg = yaml.safe_load(f)

    setup_logging(cfg)
    logger.info("=" * 70)
    logger.info("PHASE 1 PIPELINE — IO-VNBD Dataset Exploration & Baseline")
    logger.info("=" * 70)

    # Ensure output directories exist
    for key in ["outputs_dir", "logs_dir", "inventory_dir", "tables_dir", "plots_dir", "reports_dir", "processed_dir"]:
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    seq = cfg["selected_sequence"]

    # ====== TASK 1: Repository Inventory ======
    logger.info("\n--- TASK 1: Repository Inventory ---")
    from Data_details.src.dataset_inventory import generate_inventory
    inv_df = generate_inventory(cfg["paths"]["repo_root"], cfg["paths"]["inventory_dir"])

    # ====== TASK 2: Documentation Summary ======
    logger.info("\n--- TASK 2: Documentation Summary ---")
    from Data_details.src.report_generator import generate_documentation_summary
    generate_documentation_summary(cfg["paths"]["reports_dir"])

    # ====== Load Selected Sequence ======
    logger.info("\n--- Loading Selected Sequence ---")
    from Data_details.src.dataset_loader import DatasetLoader
    loader = DatasetLoader(cfg["paths"]["repo_root"], cfg["paths"]["data_cache_dir"])
    s_df, v_df = loader.load_sequence(seq["smartphone_file"], seq.get("vehicle_file"))
    logger.info(f"Smartphone: {len(s_df)} rows, {len(s_df.columns)} cols")
    if v_df is not None:
        logger.info(f"Vehicle:    {len(v_df)} rows, {len(v_df.columns)} cols")

    # ====== TASK 3 & 4: Schema & Data Dictionary ======
    logger.info("\n--- TASK 3 & 4: Schema Discovery & Data Dictionary ---")
    from Data_details.src.schema_inspector import (
        inspect_raw_schema, generate_schema_report, build_data_dictionary, save_data_dictionary,
    )
    import pandas as pd

    s_path = loader.get_real_file_path(seq["smartphone_file"])
    v_path = loader.get_real_file_path(seq["vehicle_file"]) if seq.get("vehicle_file") else None

    s_schema = inspect_raw_schema(s_path)
    v_schema = inspect_raw_schema(v_path) if v_path else None
    generate_schema_report(s_schema, v_schema, cfg["paths"]["tables_dir"])

    s_dict = build_data_dictionary(s_schema, s_df, "smartphone", seq["smartphone_file"])
    if v_df is not None and v_schema:
        v_dict = build_data_dictionary(v_schema, v_df, "vehicle", seq["vehicle_file"])
        combined_dict = pd.concat([s_dict, v_dict], ignore_index=True)
    else:
        combined_dict = s_dict
    save_data_dictionary(combined_dict, cfg["paths"]["tables_dir"])

    # ====== TASK 5: Selected Sequence Info ======
    logger.info("\n--- TASK 5: Selected Sequence ---")
    seq_info = {
        "name": seq["name"],
        "driver": seq["driver"],
        "location": seq["location"],
        "smartphone_file": seq["smartphone_file"],
        "vehicle_file": seq.get("vehicle_file"),
        "description": seq["description"],
        "smartphone_rows": len(s_df),
        "smartphone_cols": len(s_df.columns),
        "vehicle_rows": len(v_df) if v_df is not None else 0,
        "vehicle_cols": len(v_df.columns) if v_df is not None else 0,
        "duration_min": round(float(s_df["time_s"].iloc[-1] / 60.0), 1),
    }
    seq_path = Path(cfg["paths"]["inventory_dir"]) / "selected_sequence.yaml"
    with open(seq_path, "w") as f:
        yaml.dump(seq_info, f, default_flow_style=False)
    logger.info(f"Selected sequence saved to {seq_path}")

    # ====== TASK 6: Timestamp Analysis ======
    logger.info("\n--- TASK 6: Timestamp Analysis ---")
    from Data_details.src.timestamp_analysis import (
        analyze_timestamps, plot_timestamp_analysis, save_timestamp_report,
    )
    s_ts = analyze_timestamps(s_df, "smartphone")
    plot_timestamp_analysis(s_df, s_ts,
                           str(Path(cfg["paths"]["plots_dir"]) / "sampling_interval_dt_smartphone.png"),
                           "smartphone")
    v_ts = None
    if v_df is not None:
        v_ts = analyze_timestamps(v_df, "vehicle")
        plot_timestamp_analysis(v_df, v_ts,
                               str(Path(cfg["paths"]["plots_dir"]) / "sampling_interval_dt_vehicle.png"),
                               "vehicle")
    save_timestamp_report(s_ts, v_ts, cfg["paths"]["tables_dir"])

    # ====== TASK 7: Data Quality ======
    logger.info("\n--- TASK 7: Data Quality Analysis ---")
    from Data_details.src.data_quality import (
        analyze_data_quality, save_quality_csv, generate_quality_report_md,
    )
    s_quality = analyze_data_quality(s_df, "smartphone")
    v_quality = analyze_data_quality(v_df, "vehicle") if v_df is not None else None
    combined_q = pd.concat([s_quality, v_quality], ignore_index=True) if v_quality is not None else s_quality
    save_quality_csv(combined_q, cfg["paths"]["tables_dir"])
    generate_quality_report_md(s_quality, v_quality, cfg["paths"]["reports_dir"])

    # ====== TASK 8: Sensor Visualization ======
    logger.info("\n--- TASK 8: Sensor Visualization ---")
    from Data_details.src.sensor_analysis import plot_smartphone_sensors, plot_vehicle_sensors
    plot_smartphone_sensors(s_df, cfg["paths"]["plots_dir"], seq["name"])
    if v_df is not None:
        plot_vehicle_sensors(v_df, cfg["paths"]["plots_dir"], seq["name"])

    # ====== TASK 9: GPS Trajectory ======
    logger.info("\n--- TASK 9: GPS Trajectory ---")
    from Data_details.src.gps_analysis import analyze_gps_trajectory, plot_gps_trajectory, plot_enu_trajectory
    gps_stats = analyze_gps_trajectory(s_df)
    plot_gps_trajectory(s_df, cfg["paths"]["plots_dir"], seq["name"])
    origin_info = plot_enu_trajectory(s_df, cfg["paths"]["plots_dir"], seq["name"])
    logger.info(f"GPS: {gps_stats['total_distance_km']} km, {gps_stats['duration_min']} min")

    # ====== TASK 11: Stationary Analysis ======
    logger.info("\n--- TASK 11: Stationary Period Analysis ---")
    from Data_details.src.stationary_analysis import (
        detect_stationary_periods, analyze_stationary_bias, plot_stationary_analysis,
    )
    stat_cfg = cfg.get("stationary_detection", {})
    periods = detect_stationary_periods(
        s_df,
        speed_threshold=stat_cfg.get("speed_threshold_mps", 0.5),
        min_duration_s=stat_cfg.get("min_duration_s", 5.0),
    )
    bias_df = analyze_stationary_bias(s_df, periods)
    bias_df.to_csv(Path(cfg["paths"]["tables_dir"]) / "stationary_bias_stats.csv", index=False)
    plot_stationary_analysis(s_df, periods, cfg["paths"]["plots_dir"], seq["name"])

    # ====== TASK 12 & 13 & 14: Blackout + Inertial Baseline ======
    logger.info("\n--- TASKS 12-14: GNSS Blackout & Inertial Baseline ---")
    from Data_details.src.inertial_baseline import run_inertial_baseline, plot_baseline_comparison, save_baseline_metrics

    bl_cfg = cfg["blackout_simulation"]
    start_s = bl_cfg["default_start_time_s"]
    durations = bl_cfg["durations_s"]

    all_metrics = []
    primary_result = None
    for dur in durations:
        logger.info(f"Running baseline for {dur:.0f}s blackout...")
        result = run_inertial_baseline(s_df, start_s, dur)
        if result:
            all_metrics.append(result["metrics"])
            if dur == bl_cfg["primary_test_duration_s"]:
                primary_result = result
                plot_baseline_comparison(result, cfg["paths"]["plots_dir"], seq["name"])

    save_baseline_metrics(all_metrics, cfg["paths"]["tables_dir"])

    # ====== TASK 15: Failure Analysis ======
    logger.info("\n--- TASK 15: Failure Analysis ---")
    if primary_result:
        from Data_details.src.failure_analysis import generate_failure_report
        generate_failure_report(primary_result, s_df, cfg["paths"]["reports_dir"], seq["name"])

    # ====== TASK 16: Final Reports ======
    logger.info("\n--- TASK 16: Generating Reports ---")
    from Data_details.src.report_generator import generate_phase1_report, generate_simple_explanation
    generate_phase1_report(cfg, cfg["paths"]["reports_dir"])
    generate_simple_explanation(cfg["paths"]["reports_dir"])

    # ====== SUMMARY ======
    elapsed = time.time() - t0
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1 PIPELINE COMPLETE")
    logger.info(f"Total time: {elapsed:.1f}s")
    logger.info("=" * 70)

    print(f"\n{'='*70}")
    print("PHASE 1 PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Time: {elapsed:.1f}s")
    print(f"\nOutputs:")
    print(f"  Inventory:  {cfg['paths']['inventory_dir']}/")
    print(f"  Tables:     {cfg['paths']['tables_dir']}/")
    print(f"  Plots:      {cfg['paths']['plots_dir']}/")
    print(f"  Reports:    {cfg['paths']['reports_dir']}/")
    print(f"  Logs:       {cfg['paths']['logs_dir']}/")
    if all_metrics:
        print(f"\nBaseline Results:")
        for m in all_metrics:
            print(f"  {m['blackout_duration_s']:.0f}s blackout: "
                  f"endpoint={m['endpoint_error_m']:.1f}m, "
                  f"drift={m['drift_pct']:.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_pipeline()
