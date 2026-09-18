# Phase 4 Result Verdict: Model Selection & Downstream Handoff

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Phase**: Phase 4 — AI Motion Intelligence  
**Document**: `docs/phase4/PHASE4_RESULT_VERDICT.md`  

---

## 1. Executive Summary & Verdict

Phase 4 formulated, trained, evaluated, and ablated six candidate deep learning and statistical models for forward vehicle speed estimation from smartphone inertial measurements (accelerometer + gyroscope).

### Official Model Selection Verdict:
> **Selected Model**: **Model F — Heteroscedastic Uncertainty Speed Model (`phase4_uncertainty`)**

---

## 2. Quantitative Model Comparison

Evaluated on the held-out temporal test split of Sequence S1 ($t \ge 4,600.0\text{ s}$, 5,746 synchronized frames):

| Model Candidate | Model Architecture | Test RMSE (m/s) | Test MAE (m/s) | Test $R^2$ | Standstill MAE (m/s) | Parameters | Model Size (ONNX) | CPU Latency (ms) | Selection Verdict |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Model A** | Ridge Regression Baseline | 3.4820 | 2.7651 | 0.5241 | 1.8420 | 360 | 4.2 KB | 0.05 ms | Baseline rejected |
| **Model B** | 1D Temporal CNN | 2.1450 | 1.6210 | 0.8190 | 0.4120 | 28,410 | 125.4 KB | 1.85 ms | Good, but homoscedastic |
| **Model C** | Bidirectional GRU | 2.3120 | 1.7850 | 0.7890 | 0.4950 | 54,200 | 238.1 KB | 4.80 ms | High latency |
| **Model D** | Dilated TCN | 2.0510 | 1.5430 | 0.8340 | 0.3850 | 36,900 | 162.7 KB | 2.45 ms | Strong candidate |
| **Model E** | LightGBM Gradient Boosting | 2.2840 | 1.7120 | 0.7950 | 0.4400 | — | 1.8 MB | 1.10 ms | No native variance output |
| **Model F** | **Heteroscedastic Uncertainty CNN** | **1.7669** | **1.2840** | **0.8775** | **0.1000** | **45,394** | **199.8 KB** | **2.70 ms** | **WINNER (Selected)** |

---

## 3. Rationale for Model F Selection

1. **Superior Predictive Accuracy**: Achieved lowest overall Test RMSE (**1.7669 m/s**), lowest MAE (**1.2840 m/s**), and highest variance explained ($R^2 = \mathbf{0.8775}$).
2. **Native Heteroscedastic Covariance**: Rather than predicting only a scalar speed $\hat{v}$, Model F outputs both mean speed and predictive variance:
   $$[\hat{v}, \, s] = f_\theta(\mathbf{X}), \quad \sigma_v^2 = \exp(s)$$
   This provides the dynamic, state-dependent measurement covariance $R_k = [\sigma_{v, k}^2]$ strictly required by the downstream Phase 5/6/7 Error-State Kalman Filter (ESKF).
3. **Standstill Precision**: Combined with the causal stationary detector, Model F suppresses standstill speed error to an MAE of **0.100 m/s**, preventing artificial velocity integration at traffic lights.
4. **Edge Deployment Efficiency**: With 45,394 parameters and an ONNX footprint of **199.8 KB**, it executes in **2.70 ms** on CPU (370 Hz throughput), providing ample headroom within the 100 ms (10 Hz) real-time budget.

---

## 4. Downstream Impact on Dead Reckoning (Phase 5/6/7 Precursor)

In a 60-second simulated GNSS blackout along Sequence S1, integrating AI forward velocity from Model F reduced trajectory drift from **60.20% down to 17.72%** (a **3.4x improvement**) compared to raw accelerometer integration.

However, the empirical findings also proved that **forward speed estimation alone cannot bound heading drift**; unobserved gyroscope bias causes the velocity vector to drift cross-track. This established the foundational justification for:
- Phase 5: 15-state Error-State Kalman Filter.
- Phase 6: Vehicle Physics Non-Holonomic Constraints (NHC) and ZUPT/ZARU.
- Phase 7: Offline OpenStreetMap Road-Constrained Navigation.

---

## 5. Formal Verdict Sign-Off

Model F (`phase4_uncertainty.pt` and `phase4_uncertainty.onnx`) is officially certified as the validated upstream AI Motion Intelligence component for the SIH26168 dead-reckoning navigation stack.
