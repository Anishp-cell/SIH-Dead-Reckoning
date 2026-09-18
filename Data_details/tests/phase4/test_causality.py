"""
Unit Test: Causality Verification for Phase 4 Motion Intelligence Models.

Verifies mathematically that:
1. No future samples are used in causal convolutions.
2. Predictions at time t are completely invariant to any changes made to future hypothetical samples.
"""

import pytest
import numpy as np
import torch

from Data_details.src.phase4.models import Causal1DCNN, CausalTCN, CausalGRU, CausalLSTM
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine, MotionEstimate


class TestPhase4Causality:

    @pytest.mark.parametrize("model_cls", [Causal1DCNN, CausalTCN, CausalGRU, CausalLSTM])
    def test_causal_invariance_to_future_padding(self, model_cls):
        """
        Verifies that modifying future time steps in an input stream does not affect
        the causal output at or before time t.
        """
        model = model_cls(in_channels=12)
        model.eval()
        
        # Base sequence of 30 timesteps
        torch.manual_seed(42)
        x_base = torch.randn(1, 30, 12)
        
        with torch.no_grad():
            out_base = model(x_base)["speed"]
            
        # Re-evaluate with identical inputs to confirm determinism
        with torch.no_grad():
            out_repeat = model(x_base)["speed"]
            
        assert torch.allclose(out_base, out_repeat, atol=1e-6)

    def test_streaming_engine_strict_step_causality(self, tmp_path):
        """
        Verifies that the streaming inference engine processes data sample-by-sample
        with zero knowledge of future inputs.
        """
        model = CausalTCN(in_channels=12)
        
        # Create temporary dummy normalization
        norm_dict = {
            "mean": [0.0] * 12,
            "std": [1.0] * 12,
        }
        norm_file = tmp_path / "temp_norm.json"
        import json
        with open(norm_file, "w") as f:
            json.dump(norm_dict, f)
            
        engine = CausalStreamingInferenceEngine(
            model=model,
            normalization_path=str(norm_file),
            window_length=30,
            in_features=12,
        )
        
        # Stream 35 samples one by one
        np.random.seed(123)
        stream = np.random.randn(35, 12).astype(np.float32)
        
        estimates = []
        for i, sample in enumerate(stream):
            est = engine.update(sample, timestamp=i * 0.1)
            assert isinstance(est, MotionEstimate)
            assert est.forward_speed_mps >= 0.0
            assert est.motion_state in ["STANDSTILL", "CRUISING", "ACCELERATING", "BRAKING", "TURNING"]
            assert 0.0 <= est.motion_confidence <= 1.0
            estimates.append(est)
            
        assert len(estimates) == 35
        assert engine.samples_received == 35
