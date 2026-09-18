"""
Phase 4 Streaming Inference Module: Real-Time Causal Step-by-Step Motion Estimator.

Implements the official Phase 4 runtime interface:
    motion = engine.update(sample_12d, timestamp)
    motion.speed
    motion.confidence
    motion.motion_state
    motion.uncertainty

Enforces zero future access: operates sample-by-sample via a causal rolling buffer.
"""

import json
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
import numpy as np
import torch
import torch.nn as nn


@dataclass
class MotionEstimate:
    """Standardized Phase 4 Motion Estimate Output for Phase 5 ESKF consumption."""
    timestamp: float
    forward_speed_mps: float
    speed_uncertainty: float
    motion_state: str
    motion_confidence: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": round(self.timestamp, 4),
            "forward_speed_mps": round(self.forward_speed_mps, 4),
            "speed_uncertainty": round(self.speed_uncertainty, 4),
            "motion_state": self.motion_state,
            "motion_confidence": round(self.motion_confidence, 4),
        }


class CausalStreamingInferenceEngine:
    """
    Real-time rolling-buffer streaming inference engine.
    Maintains the last W=30 normalized feature samples, feeds them to the model,
    and returns an updated motion estimate for each new sample.
    """

    STATE_LABELS = ["STANDSTILL", "CRUISING", "ACCELERATING", "BRAKING", "TURNING"]

    def __init__(
        self,
        model: nn.Module,
        normalization_path: str = "Data_details/outputs/phase4/models/normalization.json",
        window_length: int = 30,
        in_features: int = 12,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.model.eval()
        self.window_length = window_length
        self.in_features = in_features
        
        # Load train normalization parameters
        norm_file = Path(normalization_path)
        if not norm_file.exists():
            raise FileNotFoundError(f"Normalization config not found at: {norm_file}")
            
        with open(norm_file, "r") as f:
            norm_data = json.load(f)
            
        self.means = np.array(norm_data["mean"], dtype=np.float32)
        self.stds = np.array(norm_data["std"], dtype=np.float32)
        self.stds[self.stds < 1e-6] = 1.0
        
        # Initialize causal FIFO rolling buffer (W x C)
        self.buffer = np.zeros((self.window_length, self.in_features), dtype=np.float32)
        self.samples_received = 0

    def reset(self) -> None:
        """Resets the rolling buffer state."""
        self.buffer.fill(0.0)
        self.samples_received = 0

    def warm_up(self, samples_matrix: Union[np.ndarray, List[List[float]]]) -> None:
        """
        Pre-fills the causal FIFO buffer with historical samples (t < t_0)
        so that at t_0 the rolling buffer contains valid prior driving context.
        """
        samples_arr = np.asarray(samples_matrix, dtype=np.float32)
        if samples_arr.ndim == 1:
            samples_arr = samples_arr.reshape(1, -1)
        for sample in samples_arr:
            norm_sample = (sample - self.means) / self.stds
            self.buffer = np.roll(self.buffer, -1, axis=0)
            self.buffer[-1] = norm_sample
            self.samples_received += 1


    def update(self, raw_sample: Union[np.ndarray, List[float]], timestamp: float = 0.0) -> MotionEstimate:
        """
        Accepts one new 12-channel sample at current time t_k, updates the rolling buffer,
        runs causal inference, and returns the real-time MotionEstimate.
        """
        sample_arr = np.asarray(raw_sample, dtype=np.float32).ravel()
        if len(sample_arr) != self.in_features:
            raise ValueError(f"Expected sample length {self.in_features}, got {len(sample_arr)}")
            
        # 1. Normalize sample using frozen training statistics
        norm_sample = (sample_arr - self.means) / self.stds
        
        # 2. Shift rolling buffer left by 1 and insert new sample at the end (causal FIFO)
        self.buffer = np.roll(self.buffer, -1, axis=0)
        self.buffer[-1] = norm_sample
        self.samples_received += 1
        
        # 3. If buffer hasn't filled yet, warm up with zero speed or early estimation
        # Tensor shape: (1, window_length, in_features)
        x_tensor = torch.from_numpy(self.buffer).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            out = self.model(x_tensor)
            
        # Parse outputs
        speed_val = float(out["speed"].squeeze().item())
        speed_val = max(0.0, speed_val)  # Physics: non-negative forward speed
        
        # Uncertainty
        if "std" in out:
            uncertainty = float(out["std"].squeeze().item())
        else:
            # Heuristic residual uncertainty when model doesn't output variance
            uncertainty = 0.35  # Standard baseline residual sigma in m/s
            
        # Motion State
        if "state_probs" in out:
            probs = out["state_probs"].squeeze().cpu().numpy()
            best_idx = int(np.argmax(probs))
            motion_state = self.STATE_LABELS[best_idx]
            confidence = float(probs[best_idx])
        else:
            # Deterministic heuristic mapping if single-task model
            if speed_val < 0.2:
                motion_state = "STANDSTILL"
                confidence = 0.95
            else:
                motion_state = "CRUISING"
                confidence = max(0.5, min(0.99, 1.0 - (uncertainty / max(speed_val, 1.0))))
                
        return MotionEstimate(
            timestamp=timestamp,
            forward_speed_mps=speed_val,
            speed_uncertainty=uncertainty,
            motion_state=motion_state,
            motion_confidence=confidence,
        )
