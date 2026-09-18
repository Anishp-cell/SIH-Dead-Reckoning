"""
Unit Test: Mobile Export and Runtime Parity Verification.
"""

import pytest
import numpy as np
import torch

from Data_details.src.phase4.models import get_model
from Data_details.src.phase4.export import export_phase4_model


class TestPhase4Export:

    def test_torchscript_and_onnx_parity(self, tmp_path):
        model = get_model("cnn1d")
        model.eval()
        
        export_info = export_phase4_model(
            model=model,
            model_name="test_cnn",
            output_dir=str(tmp_path),
        )
        
        assert "torchscript" in export_info
        ts_info = export_info["torchscript"]
        assert ts_info["parity_passed"] is True
        assert ts_info["max_difference"] < 1e-4
        assert ts_info["size_bytes"] > 1000
        
        assert "onnx" in export_info
        onnx_info = export_info["onnx"]
        assert onnx_info["parity_passed"] is True
        assert onnx_info["max_difference"] < 1e-4
        assert onnx_info["size_bytes"] > 1000
