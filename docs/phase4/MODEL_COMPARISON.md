# Phase 4 Model Benchmark & Architecture Comparison

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 4 — AI Motion Intelligence  
**Document Code**: `SIH26168-EVAL-MODELS-01`  
**Classification**: BENCHMARK EVALUATION REPORT  

---

## 1. Candidate Architectures & Technical Rationale

We benchmark seven distinct causal neural network candidates to evaluate their capability to predict forward vehicle velocity $v_{\text{fwd}}$ directly from smartphone inertial features:

1. **Model A: Linear / Ridge Regression Baseline**
   - *Architecture*: Fully connected linear projection over flattened 360-element temporal window ($30 \times 12$).
   - *Rationale*: Establishes whether vehicle speed is linearly observable from raw kinematics.
   - *Parameters*: 361 | *Size*: 1.41 KB

2. **Model B: Causal 1D-CNN**
   - *Architecture*: 3-layer temporal convolutional network ($32 \to 64 \to 64$ channels), causal zero-padding ($K-1$), BatchNorm1d, LeakyReLU, adaptive pooling, and linear head.
   - *Rationale*: Compact local feature extraction with fixed low-latency execution.
   - *Parameters*: 22,177 | *Size*: 86.63 KB

3. **Model C: Causal Gated Recurrent Unit (GRU)**
   - *Architecture*: 2-layer unidirectional GRU ($H=48$) processing sequential timesteps $t=1 \dots 30$, projecting final hidden state $h_{30}$ to scalar velocity.
   - *Rationale*: Continuous hidden state updates matching physical vehicle momentum.
   - *Parameters*: 24,641 | *Size*: 96.25 KB

4. **Model D: Causal Long Short-Term Memory (LSTM)**
   - *Architecture*: 2-layer unidirectional LSTM ($H=48$) with separate cell state $c_t$ and hidden state $h_t$.
   - *Rationale*: Classical benchmark for long-term sequence modeling in inertial odometry literature.
   - *Parameters*: 32,321 | *Size*: 126.25 KB

5. **Model E: Causal Dilated Temporal Convolutional Network (TCN)**
   - *Architecture*: 4 residual blocks with exponential dilations ($d \in \{1, 2, 4, 8\}$), receptive field $R = 31 \ge 30$, residual $1\times 1$ skip projections.
   - *Rationale*: Receptive horizon covering full 3-second window with parallel execution and non-vanishing gradients.
   - *Parameters*: 70,321 | *Size*: 274.69 KB

6. **Model F: Heteroscedastic Uncertainty Speed Model**
   - *Architecture*: Causal dilated TCN backbone branching into dual linear heads: predictive mean $\hat{\mu} = \hat{v}$ and predictive log-variance $s = \log \sigma^2$, trained under Gaussian Negative Log-Likelihood (NLL).
   - *Rationale*: Delivers real-time measurement uncertainty $\sigma_v$ essential for dynamic Kalman gain weighting in Phase 5 ESKF.
   - *Parameters*: 45,394 | *Size*: 177.32 KB

7. **Model G: Multitask Dual-Head Model**
   - *Architecture*: Shared causal feature backbone simultaneously predicting continuous speed $\hat{v}$ and 5-class vehicle motion state (Standstill, Cruising, Accelerating, Braking, Turning).
   - *Rationale*: Joint learning enforces physical state awareness, improving standstill detection and maneuver transitions.
   - *Parameters*: 45,526 | *Size*: 177.84 KB

---

## 2. Experimental Benchmark Protocol

- **Dataset**: IO-VNBD Sequence S1 (Coventry UK, 51,746 samples = 86.2 min, 37.2 km).
- **Temporal Split**: Block split with 50-sample ($5.0\text{ s}$) Purge Gaps:
  - Training: First 70.0% of sequence (7,239 windows).
  - Validation: 15.0% intermediate block (1,552 windows).
  - Testing: Final 15.0% held-out block (1,533 windows).
- **Normalization**: Z-score parameters $(\boldsymbol{\mu}, \boldsymbol{\sigma})$ computed strictly on the Training set and frozen.
- **Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU (Blackwell `sm_120`), PyTorch 2.11.0+cu128.
- **Inference Benchmarking**: Measured across 200 consecutive single-window runs on CPU (Intel Core i7/i9) and GPU.

---

## 3. Quantitative Model Comparison Results

All metrics below are strictly **MEASURED** on the held-out test partition (1,533 windows, 15.0% of sequence S1):

| Model Architecture | Parameters | Model Size | Val RMSE (m/s) | Test RMSE (m/s) | Test RMSE (km/h) | Test MAE (m/s) | Test $R^2$ | Test Max Err (m/s) | Test P95 Err (m/s) | Test Bias (m/s) | CPU Latency (ms) | GPU Latency (ms) | CPU Throughput | Causality Verified |
|:-------------------|:----------:|:----------:|:--------------:|:---------------:|:----------------:|:--------------:|:----------:|:------------------:|:------------------:|:---------------:|:----------------:|:----------------:|:--------------:|:------------------:|
| **LINEAR**         | 361        | 1.41 KB    | 3.0975         | 3.2190          | 11.59            | 2.2297         | 0.5935     | 10.8376            | 7.0847             | -0.7632         | 0.015 ms         | 0.061 ms         | 66,368 Hz      | **YES (100%)**     |
| **CNN1D**          | 22,177     | 86.63 KB   | 1.9315         | 1.8683          | 6.73             | 1.2792         | 0.8631     | 6.9683             | 4.0976             | -0.1718         | 0.945 ms         | 1.344 ms         | 1,058 Hz       | **YES (100%)**     |
| **GRU**            | 24,641     | 96.25 KB   | 2.0559         | 1.7691          | 6.37             | 1.2029         | 0.8772     | 7.5079             | 3.7807             | -0.1047         | 4.229 ms         | 0.577 ms         | 237 Hz         | **YES (100%)**     |
| **LSTM**           | 32,321     | 126.25 KB  | 2.0228         | 1.8661          | 6.72             | 1.2395         | 0.8634     | 7.3793             | 4.1387             | -0.1510         | 0.910 ms         | 0.632 ms         | 1,099 Hz       | **YES (100%)**     |
| **TCN**            | 70,321     | 274.69 KB  | 1.9113         | 1.7841          | 6.42             | 1.1814         | 0.8751     | 8.3855             | 3.7912             | -0.0520         | 3.715 ms         | 3.326 ms         | 269 Hz         | **YES (100%)**     |
| **UNCERTAINTY**    | **45,394** | **177.32 KB** | **1.8883**   | **1.7669**      | **6.36**         | **1.1772**     | **0.8775** | **7.1649**         | **3.8100**         | **-0.2299**     | **2.703 ms**     | **2.654 ms**     | **370 Hz**     | **YES (100%)**     |
| **MULTITASK**      | 45,526     | 177.84 KB  | 1.9589         | 1.7856          | 6.43             | 1.1881         | 0.8749     | 7.4064             | 3.8802             | -0.2051         | 2.746 ms         | 3.028 ms         | 364 Hz         | **YES (100%)**     |

---

## 4. Speed-Regime Performance Analysis

The selected model (**UNCERTAINTY**) was evaluated across 5 distinct operational speed regimes on the held-out test partition:

| Speed Regime | Velocity Interval | Test Sample Count | MAE (m/s) | RMSE (m/s) | $R^2$ | P95 Absolute Error (m/s) |
|:-------------|:-----------------:|:-----------------:|:---------:|:----------:|:-----:|:------------------------:|
| **Stationary** | $< 0.5\text{ m/s}$ | 427 | **0.1413** | **0.5202** | -26.09 | **0.8013** |
| **Low Speed** | $0.5 - 5.0\text{ m/s}$ | 287 | 1.2246 | 1.7108 | -0.59 | 3.4926 |
| **Medium Speed** | $5.0 - 10.0\text{ m/s}$ | 411 | 1.6113 | 2.1043 | -1.28 | 4.5729 |
| **High Speed** | $10.0 - 15.0\text{ m/s}$ | 389 | 1.7620 | 2.1930 | -1.00 | 4.2602 |
| **Highway** | $15.0+\text{ m/s}$ | 19 | 2.3810 | 2.7081 | -760.01 | 3.9187 |

*Note on Local $R^2$*: When restricted to narrow sub-intervals (such as stationary points where variance $\text{Var}(y) \approx 0$), the denominator $\sum (y_i - \bar{y})^2$ becomes near-zero, rendering local $R^2$ artificially negative. Across the entire test distribution, the global $R^2$ is **0.8775**.

---

## 5. Motion-Condition Performance Analysis

Evaluating the model across distinct kinematic vehicle states reveals how well the network handles maneuvers:

| Motion Condition | Definition Rule | Test Sample Count | MAE (m/s) | RMSE (m/s) | Max Error (m/s) | P95 Error (m/s) |
|:-----------------|:---------------:|:-----------------:|:---------:|:----------:|:---------------:|:---------------:|
| **STANDSTILL**   | $\|\mathbf{a}\| < 0.2\text{ m/s}^2, \|\boldsymbol{\omega}\| < 0.05\text{ rad/s}$ | 393 | **0.1000** | **0.4300** | 4.8557 | **0.6845** |
| **CRUISING**     | $|a_{\text{fwd}}| \le 0.4\text{ m/s}^2, |\omega_{\text{yaw}}| \le 0.08\text{ rad/s}$ | 354 | 1.4623 | 1.9424 | 7.1212 | 3.7280 |
| **ACCELERATING** | $a_{\text{fwd}} > 0.4\text{ m/s}^2$ | 304 | 1.5389 | 2.0403 | 7.1649 | 4.3710 |
| **BRAKING**      | $a_{\text{fwd}} < -0.4\text{ m/s}^2$ | 334 | 1.5601 | 2.0458 | 6.9514 | 4.1401 |
| **TURNING**      | $|\omega_{\text{yaw}}| > 0.08\text{ rad/s}$ | 148 | 1.7488 | 2.1967 | 6.6223 | 4.3055 |

---

## 6. Technical Model Selection Justification

We officially select **Model F: Heteroscedastic Uncertainty Speed Model** (`phase4_uncertainty`) for Phase 5 navigation integration based on the following documented criteria:

1. **Top Predictive Accuracy**: Achieves the lowest Test RMSE (**1.7669 m/s** = 6.36 km/h) and lowest Test MAE (**1.1772 m/s**), alongside the highest coefficient of determination ($R^2 = \mathbf{0.8775}$).
2. **Zero-Latency Uncertainty Head**: Unlike standard point-prediction models, Model F predicts both forward speed $\hat{v}_k$ and its instantaneous variance $\sigma_{v, k}^2 = \exp(s_k)$. This variance directly provisions the measurement covariance matrix $\mathbf{R}_k = [\sigma_{v, k}^2]$ of the Phase 5 ESKF without requiring Monte Carlo dropout or ensemble sampling.
3. **Execution Efficiency**: At 45,394 parameters and a model footprint of 177.32 KB (TorchScript: 258.5 KB, ONNX: 199.8 KB), single-window CPU execution takes only **2.703 ms** (370 Hz throughput). This uses less than $3.0\%$ of available CPU budget on a 10 Hz IMU thread.
4. **Superior Standstill Precision**: Yields an MAE of **0.100 m/s** and RMSE of **0.430 m/s** during vehicle standstills, avoiding false phantom speed drift when stopped at intersections.
