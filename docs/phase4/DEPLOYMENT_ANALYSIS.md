# Phase 4 Mobile Deployment & Edge Runtime Analysis

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 4 — AI Motion Intelligence  
**Document Code**: `SIH26168-DEPLOY-01`  
**Classification**: MOBILE DEPLOYMENT SPECIFICATION  

---

## 1. Edge Deployment Constraints & Measured Budget

To deploy the AI Motion Intelligence module onto mobile hardware (Android smartphones, in-vehicle telematics units, and embedded edge compute boards), the architecture must satisfy strict resource bounds:

| Resource Metric | Target Limit (SIH Spec) | Measured (UNCERTAINTY Model) | Compliance Status |
|:----------------|:-----------------------:|:----------------------------:|:-----------------:|
| **TorchScript File Size** | $< 5.0\text{ MB}$ | **258.5 KB** (0.25 MB) | **PASSED (19.3x under limit)** |
| **ONNX File Size** | $< 5.0\text{ MB}$ | **199.8 KB** (0.20 MB) | **PASSED (25.0x under limit)** |
| **Active Parameters** | $< 250,000$ | **45,394 parameters** | **PASSED (5.5x under limit)** |
| **CPU Latency (Single Window)**| $< 10.0\text{ ms}$ | **2.703 ms** | **PASSED (3.7x faster than limit)** |
| **CPU Throughput** | $> 100\text{ Hz}$ | **370.0 Hz** | **PASSED (37x faster than 10 Hz IMU)** |
| **GPU Latency** | $< 5.0\text{ ms}$ | **2.654 ms** | **PASSED** |
| **Numerical Parity (ONNX)** | Diff $< 10^{-4}$ | **$2.86 \times 10^{-6}$** | **PASSED (Exact FP32 agreement)** |
| **Numerical Parity (TorchScript)**| Diff $< 10^{-4}$ | **$0.00$** | **PASSED (Identical execution)** |
| **Causality** | $100\%$ Causal | Strictly Causal ($K-1$ pad) | **PASSED (Verified by unit test)** |

---

## 2. Export Artifacts Specification

The Phase 4 compilation suite automatically generates and validates production deployment artifacts:

```text
Data_details/outputs/phase4/models/
├── phase4_uncertainty.pt        (258.5 KB) - Traced TorchScript binary
├── phase4_uncertainty.onnx      (199.8 KB) - Standardized ONNX Opset 14 graph
├── phase4_linear.pt             (7.3 KB)   - Lightweight fallback linear model
├── phase4_linear.onnx           (2.7 KB)   - Ultra-lightweight fallback model
├── normalization.json           (1.0 KB)   - Frozen training mean and std vectors (12-dim)
├── feature_schema.json          (0.5 KB)   - Channel order, names, and engineering units
├── model_metadata.json          (0.7 KB)   - Measured metrics and parity verification record
└── data_manifest.json           (0.7 KB)   - Dataset hash and partition boundary stamps
```

---

## 3. Streaming Inference Interface

For real-time on-device operation, the inference engine implements a causal FIFO sliding buffer (`StreamingMotionEngine`). It receives individual 12-channel IMU feature vectors at $10\text{ Hz}$ and outputs a structured motion estimate without reprocessing the full historical sequence:

```python
from Data_details.src.phase4.streaming import StreamingMotionEngine

# Initialize engine with exported weights and normalization stats
engine = StreamingMotionEngine(
    model_path="Data_details/outputs/phase4/models/phase4_uncertainty.pt",
    norm_path="Data_details/outputs/phase4/models/normalization.json",
    window_size=30,
    num_features=12
)

# On arrival of a new 10 Hz IMU sample from Phase 3:
# new_sample: array of shape (12,)
estimate = engine.update(new_sample, timestamp=t_now)

# Access clean estimates:
print(f"Speed: {estimate.forward_speed_mps:.2f} m/s")
print(f"Uncertainty (std): {estimate.speed_uncertainty:.2f} m/s")
print(f"Motion State: {estimate.motion_state}")
print(f"Is Ready: {estimate.is_ready}")
```

### Motion Estimate Interface Contract:
```python
@dataclass
class MotionEstimate:
    timestamp: float              # Sensor sample epoch timestamp (seconds)
    forward_speed_mps: float      # Predicted longitudinal speed (m/s)
    speed_uncertainty: float      # Dynamic standard deviation sigma (m/s)
    motion_state: str             # STANDSTILL, CRUISING, ACCELERATING, BRAKING, TURNING
    motion_confidence: float      # Normalized confidence score in [0, 1]
    is_ready: bool                # True once FIFO contains 30 samples (3.0 seconds)
```

---

## 4. Mobile Execution Recommendations

1. **Android Deployment (Phase 9 Target)**:
   - Use `phase4_uncertainty.onnx` with ONNX Runtime Mobile (`com.microsoft.onnxruntime:onnxruntime-android`).
   - Run inference on a dedicated low-priority background handler thread.
   - At $2.70\text{ ms}$ execution time every $100\text{ ms}$, CPU utilization is under $3.0\%$, leaving ample thermal and battery headroom for the device.
2. **ESKF Integration (Phase 5 Target)**:
   - Provide `forward_speed_mps` as a velocity measurement update $z_v = \hat{v}$.
   - Pass `speed_uncertainty**2` directly into the scalar measurement noise covariance matrix $\mathbf{R}_k = [\sigma_v^2]$. When the vehicle negotiates sharp bumps or violent turns, $\sigma_v$ expands dynamically, automatically preventing the ESKF from over-relying on speed measurements during disturbances.
