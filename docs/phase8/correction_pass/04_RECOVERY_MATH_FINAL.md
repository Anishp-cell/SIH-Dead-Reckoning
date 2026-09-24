# Phase 8 Recovery Mathematics: Derivation & Proof of Consistency

**Document**: `docs/phase8/correction_pass/04_RECOVERY_MATH_FINAL.md`  
**Purpose**: Mathematical proof of covariance-consistent soft recovery.

---

## 1. Problem with Naive State Scaling
If an estimator updates state via:
$$\delta \mathbf{x} = K \boldsymbol{\nu}, \quad \delta \mathbf{x}_{\text{scaled}} = \alpha \delta \mathbf{x}$$
while updating covariance via standard Joseph form:
$$P^+ = (I - K H) P^- (I - K H)^T + K R K^T$$
the resulting posterior covariance $P^+$ reflects full information gain from $R$, but the state was only partially corrected. The filter becomes overconfident ($P^+$ too small relative to remaining state error), causing filter divergence on subsequent steps.

## 2. Mathematically Consistent Effective Covariance Inflation
To ensure mathematical consistency, the measurement noise covariance itself must be inflated:
$$R_{\text{eff}} = \frac{R}{\alpha}, \quad 0 < \alpha \le 1$$
Innovation covariance:
$$S_{\text{eff}} = H P^- H^T + R_{\text{eff}} = H P^- H^T + \frac{R}{\alpha}$$
Kalman gain:
$$K_{\text{eff}} = P^- H^T S_{\text{eff}}^{-1}$$
State correction:
$$\delta \mathbf{x} = K_{\text{eff}} \boldsymbol{\nu}$$
Joseph-form covariance update:
$$P^+ = (I - K_{\text{eff}} H) P^- (I - K_{\text{eff}} H)^T + K_{\text{eff}} R_{\text{eff}} K_{\text{eff}}^T$$

As $\alpha \to 0$ (heavy damping), $R_{\text{eff}} \to \infty$, $K_{\text{eff}} \to 0$, $\delta \mathbf{x} \to 0$, and $P^+ \to P^-$.  
As $\alpha \to 1$ (nominal healthy update), $R_{\text{eff}} \to R$, recovering the optimal minimum-variance Kalman estimator.
