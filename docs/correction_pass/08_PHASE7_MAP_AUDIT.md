# Pre-Phase-8 Audit: Phase 7 Map Matching & Observation Model Audit

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/correction_pass/08_PHASE7_MAP_AUDIT.md`  

---

## 1. Executive Summary

Phase 7 incorporates offline OpenStreetMap road network geometry to constrain cross-track dead-reckoning drift and directly observe vehicle heading. This audit evaluates the exactness of the observation models, the analytical measurement Jacobians, the innovation gating sequence, and the mathematical definition of map confidence $c_{\text{map}}$.

### Key Audit Findings:
1. **Heading Jacobian Approximation Discovered**: In `Data_details/src/phase7/matching/map_measurement.py`, the heading Jacobian was hardcoded as $H_\psi[0, 8] = 1.0$. This implicitly assumed a perfectly level vehicle ($R_{nb} \approx I$). Under non-zero roll ($\phi$) and pitch ($\theta$), this injected heading corrections incorrectly into the body pitch axis. We derive the exact analytical Jacobian matching finite differences to $1.11 \times 10^{-9}$.
2. **Innovation Gating Sequence Defect**: In `compute_cross_track_update`, the raw innovation was clamped before computing the Normalized Innovation Squared (NIS): `nu_clamped = clip(nu, -15, 15)` followed by `nis = nu_clamped^2 / S`. This artificially compressed the Chi-square test statistic, allowing large road outliers ($>15\text{ m}$) that should have been rejected to pass gating.
3. **Formal Definition of Map Confidence $c_{\text{map}}$**: Documented the mathematical synthesis combining Gaussian spatial distance, wrapped Gaussian heading alignment, and posterior belief probability.

---

## 2. Exact Analytical Derivation of the Map Heading Jacobian

In the local ENU frame, the vehicle yaw angle is extracted from the direction cosine matrix $R_{nb}$ via standard ZYX Euler convention:
$$\psi = \text{atan2}(R_{10}, R_{00})$$

Under the right-multiplicative body-frame attitude error convention:
$$R_{nb}(\delta\boldsymbol{\theta}) \approx \hat{R}_{nb} (I_3 + [\delta\boldsymbol{\theta}]_\times) = \hat{R}_{nb} + \hat{R}_{nb} [\delta\boldsymbol{\theta}]_\times$$
where $\delta\boldsymbol{\theta} = [\delta\theta_x, \delta\theta_y, \delta\theta_z]^T$ represents body-frame roll, pitch, and yaw perturbations.

Perturbing the rotation matrix elements:
$$\delta R_{00} = R_{01} \delta\theta_z - R_{02} \delta\theta_y$$
$$\delta R_{10} = R_{11} \delta\theta_z - R_{12} \delta\theta_y$$
Notice that $\delta\theta_x$ (rotation about the vehicle longitudinal roll axis) produces zero first-order variation in $R_{00}$ and $R_{10}$.

Applying the chain rule for $\text{atan2}(y, x)$ with $y = R_{10}, x = R_{00}$:
$$\delta\psi = \frac{x \, \delta y - y \, \delta x}{x^2 + y^2} = \frac{R_{00} \delta R_{10} - R_{10} \delta R_{00}}{R_{00}^2 + R_{10}^2}$$

Substituting the perturbations:
$$\delta\psi = \left(\frac{-R_{00} R_{12} + R_{10} R_{02}}{R_{00}^2 + R_{10}^2}\right) \delta\theta_y + \left(\frac{R_{00} R_{11} - R_{10} R_{01}}{R_{00}^2 + R_{10}^2}\right) \delta\theta_z$$

### Exact 1x15 Heading Measurement Jacobian:
$$\mathbf{H}_\psi = \begin{bmatrix} \mathbf{0}_{1\times 3} & \mathbf{0}_{1\times 3} & 0 & H_{\theta, y} & H_{\theta, z} & \mathbf{0}_{1\times 3} & \mathbf{0}_{1\times 3} \end{bmatrix}$$
where:
$$H_{\theta, x} = 0.0$$
$$H_{\theta, y} = \frac{-R_{00} R_{12} + R_{10} R_{02}}{R_{00}^2 + R_{10}^2}$$
$$H_{\theta, z} = \frac{R_{00} R_{11} - R_{10} R_{01}}{R_{00}^2 + R_{10}^2}$$

### Numerical Verification Against Central Finite Differences:
Tested at combined non-zero orientation ($\phi = 0.15\text{ rad}, \theta = -0.25\text{ rad}, \psi = 0.85\text{ rad}$):
- $\mathbf{H}_{\text{analytic}} = [0.000000, 0.154233, 1.020496]$
- $\mathbf{H}_{\text{numeric}} = [0.000000, 0.154233, 1.020496]$
- **Maximum Absolute Discrepancy**: **$1.11 \times 10^{-9}$**
- *Observation*: Hardcoding $[0, 0, 1.0]$ introduced a $15.4\%$ error on the pitch coupling term and a $2.0\%$ scale error on yaw. The exact closed-form Jacobian fixes this across all operating angles.

---

## 3. Innovation Gating Sequence Correction

### Flawed Previous Sequence:
```text
Raw Innovation (nu)
      ↓
Clip to [-15m, 15m]  <-- CRITICAL DEFECT: Masked gross outliers from NIS
      ↓
Compute NIS = (nu_clamped^2) / S
      ↓
Chi-Square Gate Test
```

### Corrected Scientific Sequence:
```text
Raw Innovation (nu = z - h(x))
      ↓
Innovation Covariance S = H * P * H^T + R
      ↓
Raw NIS = (nu^2) / S
      ↓
Chi-Square Gating (Accept if NIS <= chi2_threshold, Reject otherwise)
      ↓
If Accepted: Optional Soft Limiting -> State Correction dx = K * nu
```

By computing NIS on the unclipped raw innovation, genuine road-matching anomalies (e.g. vehicle driving onto an unmapped off-road lot $50\text{ m}$ away) trigger immediate Chi-square rejection ($d_M^2 \gg 6.635$) rather than being artificially clamped and admitted.

---

## 4. Mathematical Definition of Map Confidence Metric $c_{\text{map}}$

The continuous map confidence score $c_{\text{map}} \in [0, 1]$ is formally defined as the product of three distinct physical criteria:
$$c_{\text{map}} = B_t(c^*) \cdot \exp\left(-\frac{d_{\perp}^2}{2\sigma_{\text{dist}}^2}\right) \cdot \cos^2(\Delta\psi)$$
where:
1. $B_t(c^*) \in [0, 1]$: Posterior belief from the first-order Markov topology tracker for the primary candidate segment $c^*$. In ambiguous parallel roads or multi-way junctions, belief is split across multiple candidates, driving $B_t(c^*) \to 0.3 - 0.5$.
2. $\exp\left(-\frac{d_{\perp}^2}{2\sigma_{\text{dist}}^2}\right)$: Gaussian spatial distance proximity ($\sigma_{\text{dist}} = 4.0\text{ m}$). Drops to zero if the vehicle leaves the road corridor.
3. $\cos^2(\Delta\psi)$: Heading alignment factor. Drops to zero if the vehicle heading is perpendicular to the road segment.

Measurement noise covariance is adaptively modulated:
$$R = \frac{R_0}{c_{\text{map}}^2 + \epsilon_{\text{map}}}$$
When $c_{\text{map}} \to 0$, $R \to \infty$, automatically decoupling map updates and preventing filter corruption during off-road driving.
