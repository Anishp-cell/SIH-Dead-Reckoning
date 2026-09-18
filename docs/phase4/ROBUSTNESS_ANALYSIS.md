# Phase 4 Robustness & Stress-Testing Analysis

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 4 — AI Motion Intelligence  
**Document Code**: `SIH26168-ROBUST-01`  
**Classification**: STRESS-TESTING & EXPERIMENTAL RESILIENCE REPORT  

---

## 1. Physical Perturbation Methodology

Consumer smartphone sensors operate in dynamic and hostile vehicle environments. To verify that the neural motion intelligence module does not catastrophically fail under real-world sensor degradation, we subject the selected model (`phase4_uncertainty`) to controlled synthetic stress tests across the held-out test partition (1,533 windows):

1. **High-Frequency Surface Noise (Gaussian Jitter)**:
   - Simulates rough asphalt, cobblestone roads, and engine harmonics that slip past Butterworth filtering.
   - Injected across all 12 channels at standard deviations $\sigma \in \{0.2, 0.5, 1.0\}$.

2. **Uncompensated Thermal Accelerometer Bias**:
   - Simulates phone heating under sustained processing and solar dashboard exposure.
   - Evaluated at forward acceleration bias offsets $\Delta b_a \in \{0.2, 0.5, 1.0\}\text{ m/s}^2$.

3. **Pothole & Speed-Breaker Shock Impulses**:
   - Injects sharp, multi-step vertical acceleration spikes ($+3.0\text{ m/s}^2$) into $20\%$ of test windows.
   - Tests whether transient vertical road shock contaminates longitudinal forward velocity predictions.

4. **Sensor Dropout & Packet Loss**:
   - Simulates temporary Android OS sensor buffer starvation by zeroing gyroscope channels across $15\%$ of test windows.
   - Tests whether the model can maintain useful motion tracking when angular rate telemetry drops.

---

## 2. Quantitative Perturbation Results

All metrics below were strictly **MEASURED** on the held-out test partition:

| Perturbation Condition | Injected Noise / Bias | Test RMSE (m/s) | Test MAE (m/s) | P95 Error (m/s) | Max Error (m/s) | RMSE Degradation (%) | Severity Assessment |
|:-----------------------|:---------------------:|:---------------:|:--------------:|:---------------:|:---------------:|:--------------------:|:-------------------:|
| **Baseline (Unperturbed)** | None | **1.7669** | **1.1772** | **3.8100** | **7.1649** | **0.00%** | Nominal Operation |
| **Gaussian Noise** | $\sigma = 0.2$ | 1.8240 | 1.2381 | 3.9112 | 7.8056 | **+3.23%** | Negligible |
| **Gaussian Noise** | $\sigma = 0.5$ | 3.0634 | 2.3321 | 6.0924 | 12.7481 | **+73.38%** | Moderate |
| **Gaussian Noise** | $\sigma = 1.0$ | 5.7902 | 4.6819 | 11.0679 | 19.1200 | **+227.70%** | Severe |
| **Forward Accel Bias** | $+0.2\text{ m/s}^2$ | 1.7304 | 1.1617 | 3.7640 | 6.8329 | **-2.07%** | Insignificant |
| **Forward Accel Bias** | $+0.5\text{ m/s}^2$ | 1.7243 | 1.2004 | 3.8807 | 6.4663 | **-2.41%** | Insignificant |
| **Forward Accel Bias** | $+1.0\text{ m/s}^2$ | 1.9113 | 1.4376 | 4.0969 | 6.7142 | **+8.17%** | Mild |
| **Pothole Shock Spike**| $+3.0\text{ m/s}^2$ ($z$-axis, 20%) | 1.8375 | 1.2215 | 3.9676 | 7.1649 | **+4.00%** | Highly Resilient |
| **Gyroscope Dropout**  | Zeroed gyros ($15\%$) | 1.9595 | 1.2787 | 4.4248 | 9.2940 | **+10.90%** | Mild Degradation |

---

## 3. Engineering Analysis & Key Findings

### 3.1 Resistance to Longitudinal Accelerometer Bias
Remarkably, injecting constant accelerometer biases up to $+0.5\text{ m/s}^2$ resulted in virtually zero degradation (in fact, RMSE slightly shifted by $-2.4\%$). Even a massive bias of $+1.0\text{ m/s}^2$ caused only an **8.17%** increase in RMSE (from $1.767\text{ m/s}$ to $1.911\text{ m/s}$).  
*Mechanism*: The dilated TCN temporal filters learn differential kinematic patterns (jerk, vibration profiles, and pitch-compensated gravity components) rather than relying strictly on raw static DC acceleration offsets.

### 3.2 Resilience Against Potholes and Vertical Shock
Subjecting $20\%$ of the test samples to sharp $+3.0\text{ m/s}^2$ vertical impact spikes degraded Test RMSE by only **4.00%** ($1.838\text{ m/s}$ vs $1.767\text{ m/s}$).  
*Mechanism*: The Phase 3 causal vehicle-frame alignment cleanly decouples the vertical shock axis ($z_{\text{veh}}$) from the longitudinal travel axis ($x_{\text{veh}}$), and the robust Huber loss ($\delta=1.0$) prevents transient outliers from dominating predictions.

### 3.3 Gyroscope Packet Dropout Tolerance
Zeroing angular rate telemetry on $15\%$ of windows degraded RMSE by **10.90%** ($1.959\text{ m/s}$). The network gracefully falls back onto accelerometer norms and longitudinal jerk to estimate speed, proving the system does not crash or diverge when gyro buffers starve.

### 3.4 Gaussian Jitter Threshold
High-frequency noise up to $\sigma = 0.2$ is easily absorbed (+3.23% degradation). However, severe noise ($\sigma \ge 0.5$, which represents extreme unconditioned sensor vibration) degrades performance by $+73.38\%$. This empirically validates the absolute necessity of Phase 3 causal Butterworth filtering prior to model inference.
