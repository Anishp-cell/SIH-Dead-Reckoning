"""
Phase 4 Training Engine: Multi-Architecture Model Trainer with Early Stopping & Robust Loss.

Trains any candidate architecture under the standardized temporal block split.
Logs epoch metrics, tracks best validation checkpoint, and records detailed training logs.
"""

import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Data_details.src.phase4.dataset import prepare_phase4_data
from Data_details.src.phase4.models import get_model
from Data_details.src.phase4.evaluator import compute_metrics, benchmark_model_latency

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def train_single_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    model_type: str = "standard",
) -> float:
    """Executes one training epoch."""
    model.train()
    running_loss = 0.0
    total_samples = 0
    
    for batch in loader:
        x = batch["x"].to(device)
        y_speed = batch["speed"].to(device)
        y_state = batch["state"].to(device)
        b_size = x.size(0)
        
        optimizer.zero_grad()
        out = model(x)
        
        if model_type == "uncertainty":
            # Gaussian NLL loss: 0.5 * (log_var + (y - mu)^2 / exp(log_var))
            mu = out["speed"]
            log_var = out["log_var"]
            loss = 0.5 * torch.mean(log_var + torch.exp(-log_var) * ((y_speed - mu) ** 2))
        elif model_type == "multitask":
            loss_reg = criterion(out["speed"], y_speed)
            loss_cls = F_cross_entropy = nn.CrossEntropyLoss()(out["state_logits"], y_state)
            loss = loss_reg + 0.2 * loss_cls
        else:
            loss = criterion(out["speed"], y_speed)
            
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()
        
        running_loss += loss.item() * b_size
        total_samples += b_size
        
    return running_loss / max(total_samples, 1)


def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Evaluates full dataset loader and returns predictions and metrics."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["speed"]
            out = model(x)
            pred = out["speed"].cpu().numpy()
            all_preds.append(pred)
            all_targets.append(y.numpy())
            
    y_pred = np.vstack(all_preds).ravel()
    y_true = np.vstack(all_targets).ravel()
    metrics = compute_metrics(y_true, y_pred)
    return y_true, y_pred, metrics


def train_model(
    model_name: str,
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    device_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    save_plots: bool = True,
) -> Dict[str, Any]:
    """Complete training, validation, testing, and benchmarking pipeline for a model."""
    if device_name is None:
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    
    logger.info(f"--- Training Phase 4 Model: '{model_name.upper()}' on {device} ---")
    
    # 1. Prepare data
    data_dict = prepare_phase4_data()
    train_loader = DataLoader(data_dict["train"], batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(data_dict["val"], batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(data_dict["test"], batch_size=batch_size, shuffle=False)
    
    # 2. Instantiate model
    model = get_model(model_name).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {model_name.upper()} | Parameter Count: {n_params:,}")
    
    # 3. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)
    criterion = nn.HuberLoss(delta=1.0)
    
    model_type = "uncertainty" if model_name == "uncertainty" else ("multitask" if model_name == "multitask" else "standard")
    
    # Paths
    run_dir = Path(output_dir or f"Data_details/outputs/phase4/models/{model_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = run_dir / "best_model.pt"
    final_model_path = run_dir / "final_model.pt"
    
    # 4. Training loop
    best_val_rmse = float("inf")
    patience = 8
    patience_counter = 0
    logs = []
    
    t_start = time.time()
    for epoch in range(1, epochs + 1):
        t0_ep = time.time()
        train_loss = train_single_epoch(model, train_loader, optimizer, criterion, device, model_type=model_type)
        
        _, _, val_metrics = evaluate_loader(model, val_loader, device)
        val_rmse = val_metrics["rmse"]
        val_mae = val_metrics["mae"]
        val_r2 = val_metrics["r2"]
        
        scheduler.step(val_rmse)
        lr_current = optimizer.param_groups[0]["lr"]
        ep_duration = time.time() - t0_ep
        
        is_best = val_rmse < best_val_rmse
        if is_best:
            best_val_rmse = val_rmse
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            patience_counter += 1
            
        logs.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "val_r2": val_r2,
            "lr": lr_current,
            "duration_s": round(ep_duration, 2),
            "is_best": is_best,
        })
        
        if epoch % 5 == 0 or is_best or epoch == 1:
            logger.info(
                f"Epoch {epoch:02d}/{epochs} | Train Loss: {train_loss:.4f} | "
                f"Val RMSE: {val_rmse:.4f} m/s | Val MAE: {val_mae:.4f} m/s | Val R2: {val_r2:.4f} | "
                f"{'*BEST*' if is_best else ''}"
            )
            
        if patience_counter >= patience:
            logger.info(f"Early stopping triggered at epoch {epoch} (no val improvement for {patience} epochs).")
            break
            
    train_duration = time.time() - t_start
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Training completed in {train_duration:.1f}s.")
    
    # Save training log
    log_df = pd.DataFrame(logs)
    log_df.to_csv(run_dir / "training_log.csv", index=False)
    
    # 5. Load best model for test evaluation
    model.load_state_dict(torch.load(best_model_path))
    y_test_true, y_test_pred, test_metrics = evaluate_loader(model, test_loader, device)
    _, _, val_best_metrics = evaluate_loader(model, val_loader, device)
    
    logger.info(f"=== {model_name.upper()} FINAL TEST RESULTS ===")
    logger.info(f"Test RMSE: {test_metrics['rmse']:.4f} m/s ({test_metrics['rmse']*3.6:.2f} km/h)")
    logger.info(f"Test MAE:  {test_metrics['mae']:.4f} m/s ({test_metrics['mae']*3.6:.2f} km/h)")
    logger.info(f"Test R^2:  {test_metrics['r2']:.4f}")
    logger.info(f"Test P95:  {test_metrics['p95_error']:.4f} m/s")
    logger.info(f"Test Max:  {test_metrics['max_error']:.4f} m/s")
    
    # 6. Latency & Resource Benchmarks
    cpu_bench = benchmark_model_latency(model, device="cpu")
    gpu_bench = benchmark_model_latency(model, device="cuda") if torch.cuda.is_available() else cpu_bench
    
    logger.info(f"CPU Latency: {cpu_bench['avg_latency_ms']:.3f} ms | GPU Latency: {gpu_bench['avg_latency_ms']:.3f} ms")
    
    # 7. Diagnostic plot
    if save_plots:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        # Loss curves
        axes[0].plot(log_df["epoch"], log_df["train_loss"], label="Train Loss (Huber)", color="#2980b9", lw=2)
        axes[0].plot(log_df["epoch"], log_df["val_rmse"], label="Val RMSE (m/s)", color="#e67e22", lw=2, linestyle="--")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss / Metric")
        axes[0].set_title(f"{model_name.upper()}: Training Convergence")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Test true vs predicted (first 300 test windows)
        n_show = min(300, len(y_test_true))
        t_axis = np.arange(n_show) * 0.5  # Stride = 0.5s
        axes[1].plot(t_axis, y_test_true[:n_show], label="True Speed (CAN)", color="#2c3e50", lw=2)
        axes[1].plot(t_axis, y_test_pred[:n_show], label=f"Predicted ({model_name})", color="#27ae60", lw=1.8, linestyle="--")
        axes[1].set_xlabel("Time in Test Segment (s)")
        axes[1].set_ylabel("Forward Speed (m/s)")
        axes[1].set_title(f"Test Prediction Tracking (RMSE: {test_metrics['rmse']:.3f} m/s)")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(run_dir / "training_diagnostic.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        
    return {
        "model_name": model_name,
        "parameters": n_params,
        "model_size_kb": cpu_bench["model_size_kb"],
        "train_duration_s": round(train_duration, 2),
        "val_rmse": val_best_metrics["rmse"],
        "val_mae": val_best_metrics["mae"],
        "val_r2": val_best_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_max_err": test_metrics["max_error"],
        "test_p95_err": test_metrics["p95_error"],
        "test_bias": test_metrics["bias"],
        "cpu_latency_ms": cpu_bench["avg_latency_ms"],
        "gpu_latency_ms": gpu_bench["avg_latency_ms"],
        "cpu_throughput_hz": cpu_bench["throughput_hz"],
        "best_model_path": str(best_model_path),
        "test_preds": y_test_pred,
        "test_trues": y_test_true,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Phase 4 Motion Intelligence Model")
    parser.add_argument("--model", type=str, default="tcn", help="Model: linear, cnn1d, gru, lstm, tcn, uncertainty, multitask")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda or cpu")
    args = parser.parse_args()
    
    train_model(args.model, epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.lr, device_name=args.device)
