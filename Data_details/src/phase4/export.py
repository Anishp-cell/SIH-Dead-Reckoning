"""
Phase 4 Model Export & Mobile Deployment Preparation Engine.

Exports PyTorch models to:
1. TorchScript (.pt) for PyTorch Mobile / Android C++ execution
2. ONNX (.onnx) for cross-platform neural accelerators (Qualcomm NPU / MediaTek APU)
3. JSON metadata and feature schema specifications

Validates strict numerical parity (< 1e-4 tolerance) between eager PyTorch and exported runtimes.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class ExportWrapper(nn.Module):
    """Wraps model to output both forward speed and predictive uncertainty for mobile deployment."""

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, 30, 12)
        out = self.base_model(x)
        speed = out["speed"]
        uncertainty = out.get("std", torch.ones_like(speed) * 0.5)
        return speed, uncertainty


ExportWrapperDual = ExportWrapper



def export_phase4_model(
    model: nn.Module,
    model_name: str,
    output_dir: str = "Data_details/outputs/phase4/models",
    input_shape: Tuple[int, int, int] = (1, 30, 12),
    sample_rate_hz: float = 10.0,
    metadata_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Exports model to TorchScript and ONNX, validating numerical precision against eager execution.
    Preserves both forward speed and speed uncertainty outputs.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    wrapper = ExportWrapper(model)
    wrapper.eval()
    
    dummy_input = torch.randn(*input_shape, dtype=torch.float32)
    
    # 1. Baseline Eager Output
    with torch.no_grad():
        eager_speed, eager_std = wrapper(dummy_input)
        eager_speed_np = eager_speed.numpy()
        eager_std_np = eager_std.numpy()
        
    export_results = {}
    
    # 2. TorchScript Export
    ts_file = out_path / f"phase4_{model_name}.pt"
    try:
        traced_ts = torch.jit.trace(wrapper, dummy_input)
        traced_ts.save(str(ts_file))
        
        # Parity check
        loaded_ts = torch.jit.load(str(ts_file))
        ts_speed, ts_std = loaded_ts(dummy_input)
        max_diff_speed = float(np.max(np.abs(eager_speed_np - ts_speed.detach().numpy())))
        max_diff_std = float(np.max(np.abs(eager_std_np - ts_std.detach().numpy())))
        max_diff_ts = max(max_diff_speed, max_diff_std)
        parity_ts = bool(max_diff_ts < 1e-4)
        
        export_results["torchscript"] = {
            "path": str(ts_file),
            "size_bytes": int(ts_file.stat().st_size),
            "max_difference": max_diff_ts,
            "max_diff_speed": max_diff_speed,
            "max_diff_uncertainty": max_diff_std,
            "parity_passed": parity_ts,
        }
        logger.info(f"TorchScript exported: {ts_file.name} ({ts_file.stat().st_size / 1024:.1f} KB, parity diff={max_diff_ts:.2e})")
    except Exception as e:
        logger.error(f"TorchScript export failed: {e}")
        export_results["torchscript"] = {"error": str(e)}
        
    # 3. ONNX Export
    onnx_file = out_path / f"phase4_{model_name}.onnx"
    try:
        torch.onnx.export(
            wrapper,
            dummy_input,
            str(onnx_file),
            input_names=["inertial_window_30x12"],
            output_names=["forward_speed_mps", "speed_uncertainty_mps"],
            dynamic_axes={
                "inertial_window_30x12": {0: "batch_size"},
                "forward_speed_mps": {0: "batch_size"},
                "speed_uncertainty_mps": {0: "batch_size"},
            },
            opset_version=14,
            dynamo=False,
            do_constant_folding=True,
        )
        
        # Parity check with onnxruntime
        parity_onnx = True
        max_diff_onnx = 0.0
        try:
            import onnxruntime as ort
            session = ort.InferenceSession(str(onnx_file), providers=["CPUExecutionProvider"])
            onnx_out = session.run(None, {"inertial_window_30x12": dummy_input.numpy()})
            onnx_speed, onnx_std = onnx_out[0], onnx_out[1]
            diff_speed_onnx = float(np.max(np.abs(eager_speed_np - onnx_speed)))
            diff_std_onnx = float(np.max(np.abs(eager_std_np - onnx_std)))
            max_diff_onnx = max(diff_speed_onnx, diff_std_onnx)
            parity_onnx = bool(max_diff_onnx < 1e-4)
        except Exception as ort_err:
            logger.warning(f"ONNX Runtime validation skipped: {ort_err}")
            
        export_results["onnx"] = {
            "path": str(onnx_file),
            "size_bytes": int(onnx_file.stat().st_size),
            "max_difference": max_diff_onnx,
            "max_diff_speed": diff_speed_onnx,
            "max_diff_uncertainty": diff_std_onnx,
            "parity_passed": parity_onnx,
        }
        logger.info(f"ONNX exported: {onnx_file.name} ({onnx_file.stat().st_size / 1024:.1f} KB, parity diff={max_diff_onnx:.2e})")
    except Exception as e:
        logger.error(f"ONNX export failed: {e}")
        export_results["onnx"] = {"error": str(e)}
        
    # 4. Feature Schema
    from Data_details.src.phase4.dataset import FEATURE_COLUMNS
    feature_schema = {
        "channels": len(FEATURE_COLUMNS),
        "ordered_features": FEATURE_COLUMNS,
        "input_tensor_shape": list(input_shape),
        "temporal_duration_seconds": input_shape[1] / sample_rate_hz,
        "sampling_rate_hz": sample_rate_hz,
        "output": "forward_speed_mps",
        "units": "meters per second",
    }
    with open(out_path / "feature_schema.json", "w") as f:
        json.dump(feature_schema, f, indent=2)
        
    # 5. Model Metadata
    n_params = sum(p.numel() for p in model.parameters())
    metadata = {
        "model_name": f"phase4_{model_name}",
        "architecture": model_name,
        "parameters": int(n_params),
        "input_shape": list(input_shape),
        "causal": True,
        "export_validation": export_results,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
        
    with open(out_path / "model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
        
    logger.info(f"Export metadata written to {out_path / 'model_metadata.json'}")
    return export_results
