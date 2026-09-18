"""
Unit Test: Model Forward Pass, Gradient Flow & Numerics.
"""

import pytest
import torch
from Data_details.src.phase4.models import get_model


class TestPhase4Models:

    @pytest.mark.parametrize("model_name", ["linear", "cnn1d", "gru", "lstm", "tcn", "uncertainty", "multitask"])
    def test_forward_pass_and_shape(self, model_name):
        model = get_model(model_name)
        model.eval()
        
        # Batch size 4, 30 timesteps, 12 features
        x = torch.randn(4, 30, 12)
        with torch.no_grad():
            out = model(x)
            
        assert isinstance(out, dict)
        assert "speed" in out
        speed = out["speed"]
        assert speed.shape == (4, 1)
        assert torch.all(speed >= 0.0)  # Physics: non-negative forward speed
        assert not torch.any(torch.isnan(speed))

    @pytest.mark.parametrize("model_name", ["linear", "cnn1d", "gru", "lstm", "tcn"])
    def test_backward_gradient_flow(self, model_name):
        model = get_model(model_name)
        model.train()
        
        x = torch.randn(2, 30, 12, requires_grad=True)
        out = model(x)["speed"]
        loss = out.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.any(torch.isnan(x.grad))
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None
                assert not torch.any(torch.isnan(p.grad))
