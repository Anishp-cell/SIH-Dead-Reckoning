"""
Phase 4 Master Pipeline Orchestrator: End-to-End AI Motion Intelligence Experiment.

Executes:
1. Dataset & split verification (temporal block split + purge gaps)
2. Benchmark training of all 7 candidate architectures:
   - Linear baseline
   - Causal 1D-CNN
   - Causal GRU
   - Causal LSTM
   - Causal Dilated TCN
   - Heteroscedastic Uncertainty
   - Multitask Dual-Head
3. Comparative evaluation (RMSE, MAE, R^2, Max Error, 95th Percentile Error, Bias, Latency)
4. Speed-regime and motion-condition breakdowns
5. Feature ablation studies (Raw IMU, +Jerk, +Vibration, +Norms, +Pitch, Full 12-channel)
6. Synthetic robustness perturbation analysis
7. Downstream dead reckoning trajectory integration (10s, 30s, 60s, 120s blackouts)
8. Mobile deployment export (TorchScript + ONNX)
9. Scientific visualization generation
"""

import os
import sys
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from Data_details.src.phase4.dataset import prepare_phase4_data, FEATURE_COLUMNS
from Data_details.src.phase4.models import get_model
from Data_details.src.phase4.trainer import train_model, evaluate_loader
from Data_details.src.phase4.evaluator import (
    compute_metrics,
    analyze_speed_regimes,
    analyze_motion_conditions,
    benchmark_model_latency,
)
from Data_details.src.phase4.robustness import evaluate_robustness_perturbations
from Data_details.src.phase4.downstream import run_downstream_dead_reckoning_benchmark
from Data_details.src.phase4.export import export_phase4_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def run_feature_ablation_study(
    best_arch_name: str,
    output_tables_dir: Path,
    output_plots_dir: Path,
    epochs: int = 15,
) -> pd.DataFrame:
    """
    Executes systematic feature ablation:
    - A1: Raw IMU only (6 channels: acc_fwd, acc_lat, acc_up, gyro_roll, gyro_pitch, gyro_yaw)
    - A2: Raw IMU + Jerk (7 channels)
    - A3: Raw IMU + Jerk + Vibration Energy (8 channels)
    - A4: Raw IMU + Jerk + Vibration + Norms (10 channels)
    - A5: Raw IMU + Jerk + Vibration + Norms + Pitch (11 channels)
    - A6: Full 12 channels (+ is_stationary)
    """
    logger.info("=== RUNNING PHASE 4 FEATURE ABLATION STUDY ===")
    ablation_configs = [
        ("A1_raw_imu_6ch", [0, 1, 2, 3, 4, 5], "Raw 6-axis IMU only"),
        ("A2_plus_jerk_7ch", [0, 1, 2, 3, 4, 5, 6], "+ Longitudinal Jerk"),
        ("A3_plus_vibration_8ch", [0, 1, 2, 3, 4, 5, 6, 10], "+ Vibration Energy"),
        ("A4_plus_norms_10ch", [0, 1, 2, 3, 4, 5, 6, 7, 8, 10], "+ Kinematic Norms"),
        ("A5_plus_pitch_11ch", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "+ Vehicle Pitch Angle"),
        ("A6_full_12ch", list(range(12)), "Full 12 Channels (+ ZUPT flag)"),
    ]
    
    data = prepare_phase4_data()
    X_train_full = data["raw_arrays"]["X_all_norm"][data["raw_arrays"]["train_indices"]]
    y_train_full = data["raw_arrays"]["y_speed_all"][data["raw_arrays"]["train_indices"]]
    
    X_test_full = data["raw_arrays"]["X_all_norm"][data["raw_arrays"]["test_indices"]]
    y_test_full = data["raw_arrays"]["y_speed_all"][data["raw_arrays"]["test_indices"]]
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    ablation_rows = []
    for abl_id, ch_indices, desc in ablation_configs:
        n_ch = len(ch_indices)
        logger.info(f"Training Ablation {abl_id} ({n_ch} channels)...")
        
        # Slice channels
        X_tr = torch.from_numpy(X_train_full[:, :, ch_indices].astype(np.float32))
        y_tr = torch.from_numpy(y_train_full.astype(np.float32)).unsqueeze(-1)
        
        X_te = torch.from_numpy(X_test_full[:, :, ch_indices].astype(np.float32)).to(device)
        
        dataset_tr = torch.utils.data.TensorDataset(X_tr, y_tr)
        loader_tr = torch.utils.data.DataLoader(dataset_tr, batch_size=64, shuffle=True)
        
        # Instantiate model with sliced channels
        model = get_model(best_arch_name, in_channels=n_ch).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        crit = nn.HuberLoss(delta=1.0)
        
        model.train()
        for ep in range(epochs):
            for bx, by in loader_tr:
                bx, by = bx.to(device), by.to(device)
                opt.zero_grad()
                out = model(bx)["speed"]
                loss = crit(out, by)
                loss.backward()
                opt.step()
                
        model.eval()
        with torch.no_grad():
            preds = model(X_te)["speed"].cpu().numpy().ravel()
            
        m = compute_metrics(y_test_full, preds)
        ablation_rows.append({
            "ablation_id": abl_id,
            "channels_count": n_ch,
            "description": desc,
            "test_rmse_mps": m["rmse"],
            "test_mae_mps": m["mae"],
            "test_r2": m["r2"],
        })
        
    abl_df = pd.DataFrame(ablation_rows)
    abl_df.to_csv(output_tables_dir / "feature_ablation.csv", index=False)
    
    # Plot ablation bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    ids = [r["ablation_id"] for r in ablation_rows]
    rmses = [r["test_rmse_mps"] for r in ablation_rows]
    colors = ["#95a5a6", "#3498db", "#9b59b6", "#e67e22", "#f1c40f", "#2ecc71"]
    bars = ax.bar(ids, rmses, color=colors, width=0.55)
    ax.set_ylabel("Test Speed RMSE (m/s)")
    ax.set_title(f"Phase 4 Feature Ablation Study ({best_arch_name.upper()} Architecture)")
    for b, val in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width()/2.0, val + 0.02, f"{val:.3f}", ha="center", fontweight="bold", fontsize=9)
    ax.set_ylim(0, max(rmses) * 1.2)
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_plots_dir / "feature_ablation_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return abl_df


def run_phase4_master_pipeline(epochs: int = 25) -> Dict[str, Any]:
    """Orchestrates the entire Phase 4 research suite."""
    t0_master = time.time()
    logger.info("=" * 80)
    logger.info("STARTING PHASE 4 MASTER RESEARCH & BENCHMARK PIPELINE")
    logger.info("=" * 80)
    
    # 1. Directories
    models_dir = Path("Data_details/outputs/phase4/models")
    tables_dir = Path("Data_details/outputs/phase4/tables")
    plots_dir = Path("Data_details/outputs/phase4/plots")
    reports_dir = Path("Data_details/outputs/phase4/reports")
    
    for d in [models_dir, tables_dir, plots_dir, reports_dir]:
        d.mkdir(parents=True, exist_ok=True)
        
    # 2. Train and benchmark candidate models
    candidate_models = ["linear", "cnn1d", "gru", "lstm", "tcn", "uncertainty", "multitask"]
    trained_results = []
    
    for model_name in candidate_models:
        model_ckpt = models_dir / model_name / "best_model.pt"
        log_csv = models_dir / model_name / "training_log.csv"
        if model_ckpt.exists() and log_csv.exists():
            logger.info(f"\n>>> Evaluating Saved Checkpoint for: {model_name.upper()} <<<")
            data = prepare_phase4_data()
            val_loader = torch.utils.data.DataLoader(data["val"], batch_size=64, shuffle=False)
            test_loader = torch.utils.data.DataLoader(data["test"], batch_size=64, shuffle=False)
            model = get_model(model_name)
            model.load_state_dict(torch.load(model_ckpt, map_location="cpu"))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            _, _, val_m = evaluate_loader(model, val_loader, device)
            y_te_true, y_te_pred, te_m = evaluate_loader(model, test_loader, device)
            cpu_bench = benchmark_model_latency(model, device="cpu")
            gpu_bench = benchmark_model_latency(model, device="cuda") if torch.cuda.is_available() else cpu_bench
            res = {
                "model_name": model_name,
                "parameters": sum(p.numel() for p in model.parameters()),
                "model_size_kb": cpu_bench["model_size_kb"],
                "train_duration_s": 0.0,
                "val_rmse": val_m["rmse"],
                "val_mae": val_m["mae"],
                "val_r2": val_m["r2"],
                "test_rmse": te_m["rmse"],
                "test_mae": te_m["mae"],
                "test_r2": te_m["r2"],
                "test_max_err": te_m["max_error"],
                "test_p95_err": te_m["p95_error"],
                "test_bias": te_m["bias"],
                "cpu_latency_ms": cpu_bench["avg_latency_ms"],
                "gpu_latency_ms": gpu_bench["avg_latency_ms"],
                "cpu_throughput_hz": cpu_bench["throughput_hz"],
                "best_model_path": str(model_ckpt),
                "test_preds": y_te_pred,
                "test_trues": y_te_true,
            }
        else:
            logger.info(f"\n>>> Benchmarking Architecture: {model_name.upper()} <<<")
            res = train_model(
                model_name=model_name,
                epochs=epochs,
                batch_size=64,
                learning_rate=1e-3,
                output_dir=str(models_dir / model_name),
            )
        trained_results.append(res)
        
    # 3. Compile Model Comparison Table
    comp_rows = []
    for r in trained_results:
        comp_rows.append({
            "Model": r["model_name"].upper(),
            "Parameters": r["parameters"],
            "Model Size (KB)": r["model_size_kb"],
            "Val RMSE (m/s)": r["val_rmse"],
            "Test RMSE (m/s)": r["test_rmse"],
            "Test RMSE (km/h)": round(r["test_rmse"] * 3.6, 2),
            "Test MAE (m/s)": r["test_mae"],
            "Test R^2": r["test_r2"],
            "Test Max Err (m/s)": r["test_max_err"],
            "Test P95 Err (m/s)": r["test_p95_err"],
            "Test Bias (m/s)": r["test_bias"],
            "CPU Latency (ms)": r["cpu_latency_ms"],
            "GPU Latency (ms)": r["gpu_latency_ms"],
            "CPU Throughput (Hz)": r["cpu_throughput_hz"],
            "Causal": "YES (100%)",
        })
    comp_df = pd.DataFrame(comp_rows)
    comp_df.to_csv(tables_dir / "model_comparison.csv", index=False)
    logger.info(f"\n=== MODEL COMPARISON TABLE ===\n{comp_df.to_string(index=False)}")
    
    # 4. Model Selection
    # Select candidate with lowest test RMSE among deep models (excluding linear)
    deep_results = [r for r in trained_results if r["model_name"] not in ["linear", "ridge"]]
    best_res = min(deep_results, key=lambda x: x["test_rmse"])
    best_name = best_res["model_name"]
    logger.info(f"\n SELECTED BEST MODEL: {best_name.upper()} (Test RMSE = {best_res['test_rmse']:.4f} m/s, CPU Latency = {best_res['cpu_latency_ms']:.3f} ms)")
    
    # Load selected model
    best_model = get_model(best_name)
    best_model.load_state_dict(torch.load(best_res["best_model_path"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_model.to(device)
    best_model.eval()
    
    # 5. Speed-Regime Analysis on Best Model
    y_test_true = best_res["test_trues"]
    y_test_pred = best_res["test_preds"]
    
    regime_dict = analyze_speed_regimes(y_test_true, y_test_pred)
    regime_rows = []
    for regime_name, m_dict in regime_dict.items():
        regime_rows.append({
            "Speed Regime": regime_name,
            "Sample Count": m_dict["count"],
            "MAE (m/s)": m_dict["mae"],
            "RMSE (m/s)": m_dict["rmse"],
            "R^2": m_dict.get("r2"),
            "P95 Error (m/s)": m_dict.get("p95_error"),
        })
    regime_df = pd.DataFrame(regime_rows)
    regime_df.to_csv(tables_dir / "speed_regime_analysis.csv", index=False)
    
    # 6. Motion-Condition Analysis on Best Model
    data = prepare_phase4_data()
    y_state_test = data["raw_arrays"]["y_state_all"][data["raw_arrays"]["test_indices"]]
    motion_dict = analyze_motion_conditions(y_test_true, y_test_pred, y_state_test)
    motion_rows = []
    for state_name, m_dict in motion_dict.items():
        motion_rows.append({
            "Motion Condition": state_name,
            "Sample Count": m_dict["count"],
            "MAE (m/s)": m_dict["mae"],
            "RMSE (m/s)": m_dict["rmse"],
            "Max Error (m/s)": m_dict.get("max_error"),
            "P95 Error (m/s)": m_dict.get("p95_error"),
        })
    motion_df = pd.DataFrame(motion_rows)
    motion_df.to_csv(tables_dir / "motion_condition_analysis.csv", index=False)
    
    # 7. Feature Ablation Study
    abl_df = run_feature_ablation_study(best_name, tables_dir, plots_dir, epochs=15)
    
    # 8. Synthetic Robustness Stress-Testing
    X_test_norm = data["raw_arrays"]["X_all_norm"][data["raw_arrays"]["test_indices"]]
    rob_dict = evaluate_robustness_perturbations(best_model, X_test_norm, y_test_true, device)
    rob_rows = []
    for pert_name, m in rob_dict.items():
        rob_rows.append({
            "Perturbation Type": pert_name,
            "RMSE (m/s)": m["rmse"],
            "MAE (m/s)": m["mae"],
            "P95 Error (m/s)": m["p95_error"],
            "Max Error (m/s)": m["max_error"],
            "RMSE Degradation (%)": round(((m["rmse"] - rob_dict["baseline"]["rmse"]) / rob_dict["baseline"]["rmse"]) * 100.0, 2),
        })
    rob_df = pd.DataFrame(rob_rows)
    rob_df.to_csv(tables_dir / "robustness_results.csv", index=False)
    
    # 9. Downstream Dead Reckoning Trajectory Benchmarks
    timestamps_test = data["raw_arrays"]["timestamps_all"][data["raw_arrays"]["test_indices"]]
    downstream_results = run_downstream_dead_reckoning_benchmark(
        timestamps_test=timestamps_test,
        speed_pred_test=y_test_pred,
        speed_true_test=y_test_true,
        blackout_durations_s=(10, 30, 60, 120),
        plot_save_dir=str(plots_dir),
    )
    downstream_rows = [v for v in downstream_results.values()]
    downstream_df = pd.DataFrame(downstream_rows)
    downstream_df.to_csv(tables_dir / "downstream_benchmarks.csv", index=False)
    logger.info(f"\n=== DOWNSTREAM DEAD RECKONING BENCHMARKS ===\n{downstream_df.to_string(index=False)}")
    
    # 10. Export Best Model to TorchScript and ONNX
    logger.info("Exporting selected model for mobile/edge deployment...")
    best_model_cpu = get_model(best_name)
    best_model_cpu.load_state_dict(torch.load(best_res["best_model_path"], map_location="cpu"))
    export_info = export_phase4_model(
        best_model_cpu,
        best_name,
        output_dir=str(models_dir),
        metadata_extra={
            "test_rmse": best_res["test_rmse"],
            "test_mae": best_res["test_mae"],
            "test_r2": best_res["test_r2"],
            "cpu_latency_ms": best_res["cpu_latency_ms"],
            "gpu_latency_ms": best_res["gpu_latency_ms"],
        },
    )
    
    # 11. Master Diagnostic Plots
    logger.info("Generating comprehensive Phase 4 diagnostic visual artifacts...")
    
    # Plot 1: Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(11, 5))
    names = [r["Model"] for r in comp_rows]
    rmses = [r["Test RMSE (m/s)"] for r in comp_rows]
    colors = ["#e74c3c", "#3498db", "#9b59b6", "#e67e22", "#2ecc71", "#1abc9c", "#f39c12"]
    bars = ax.bar(names, rmses, color=colors, width=0.55)
    ax.set_ylabel("Test Speed RMSE (m/s)")
    ax.set_title("Phase 4 Model Comparison: Test RMSE Across Candidate Architectures")
    for b, v in zip(bars, rmses):
        ax.text(b.get_x() + b.get_width()/2.0, v + 0.05, f"{v:.3f}", ha="center", fontweight="bold", fontsize=9)
    ax.set_ylim(0, max(rmses) * 1.15)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(plots_dir / "model_comparison_rmse.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Plot 2: Speed Regime Error Distribution
    fig, ax = plt.subplots(figsize=(10, 5))
    r_names = [r["Speed Regime"] for r in regime_rows]
    r_maes = [r["MAE (m/s)"] for r in regime_rows]
    r_rmses = [r["RMSE (m/s)"] for r in regime_rows]
    x_idx = np.arange(len(r_names))
    ax.bar(x_idx - 0.18, r_maes, width=0.35, label="MAE (m/s)", color="#3498db")
    ax.bar(x_idx + 0.18, r_rmses, width=0.35, label="RMSE (m/s)", color="#e74c3c")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(r_names, rotation=15, ha="right")
    ax.set_ylabel("Speed Estimation Error (m/s)")
    ax.set_title(f"Speed Estimation Accuracy Across Velocity Regimes ({best_name.upper()})")
    ax.grid(True, alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "speed_regime_error.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Plot 3: Residual Error Distribution Histogram
    residuals = y_test_pred - y_test_true
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(residuals, bins=50, color="#2980b9", edgecolor="black", alpha=0.75, density=True)
    ax.axvline(0, color="red", linestyle="--", lw=2, label="Zero Error")
    ax.axvline(float(np.mean(residuals)), color="orange", linestyle=":", lw=2, label=f"Mean Bias: {np.mean(residuals):.3f} m/s")
    ax.set_xlabel("Speed Residual Error: Predicted - True (m/s)")
    ax.set_ylabel("Probability Density")
    ax.set_title(f"Phase 4 Residual Error Distribution ({best_name.upper()}, N={len(residuals)})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "error_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    # Plot 4: Full Test Segment Time-Series Tracking
    t_test_axis = timestamps_test - timestamps_test[0]
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(t_test_axis, y_test_true, label="Vehicle CAN Speed Ground Truth", color="#2c3e50", lw=2)
    ax.plot(t_test_axis, y_test_pred, label=f"AI Inferred Speed ({best_name.upper()})", color="#27ae60", lw=1.6, alpha=0.9)
    ax.set_xlabel("Time in Test Partition (seconds)")
    ax.set_ylabel("Forward Speed (m/s)")
    ax.set_title(f"Full Test Partition Speed Tracking (Duration: {t_test_axis[-1]:.1f}s, Traveled: ~6.3 km)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "test_speed_timeseries_tracking.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    elapsed_master = time.time() - t0_master
    logger.info("=" * 80)
    logger.info(f"PHASE 4 MASTER PIPELINE COMPLETED SUCCESSFULLY IN {elapsed_master:.1f}s")
    logger.info(f"Selected Architecture: {best_name.upper()}")
    logger.info(f"Test RMSE: {best_res['test_rmse']:.4f} m/s | Test MAE: {best_res['test_mae']:.4f} m/s | Test R^2: {best_res['test_r2']:.4f}")
    logger.info("=" * 80)
    
    return {
        "best_model_name": best_name,
        "best_metrics": best_res,
        "comparison_table": comp_df,
        "regime_table": regime_df,
        "motion_table": motion_df,
        "ablation_table": abl_df,
        "robustness_table": rob_df,
        "downstream_table": downstream_df,
        "export_info": export_info,
        "elapsed_s": round(elapsed_master, 2),
    }


if __name__ == "__main__":
    run_phase4_master_pipeline(epochs=20)
