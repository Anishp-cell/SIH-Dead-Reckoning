# Phase 8 Mathematical Foundation: Exact Analytical GNSS Measurement Jacobians

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase8/03_GNSS_JACOBIAN_DERIVATION.md`  

---

## 1. Executive Summary

This document derives the exact $3 \times 15$ analytical measurement Jacobians for GNSS position and velocity updates with respect to the 15-state ESKF error vector:
$$\delta\mathbf{x} = [\delta\mathbf{p}^T, \delta\mathbf{v}^T, \delta\boldsymbol{\theta}^T, \delta\mathbf{b}_a^T, \delta\mathbf{b}_g^T]^T \in \mathbb{R}^{15}$$
Numerical validation protocols against central finite differences are formally specified.

---

## 2. Right-Multiplicative Attitude Error Convention

Following the frozen Phase 5–7 convention, the perturbed body-to-navigation rotation matrix $\mathbf{R}_{nb}(\delta\boldsymbol{\theta})$ under body-frame small-angle rotation vector $\delta\boldsymbol{\theta} = [\delta\theta_x, \delta\theta_y, \delta\theta_z]^T$ is:
$$\mathbf{R}_{nb}(\delta\boldsymbol{\theta}) \approx \hat{\mathbf{R}}_{nb} (\mathbf{I}_3 + [\delta\boldsymbol{\theta}]_\times) = \hat{\mathbf{R}}_{nb} + \hat{\mathbf{R}}_{nb} [\delta\boldsymbol{\theta}]_\times$$
where $[\mathbf{a}]_\times$ is the $3 \times 3$ skew-symmetric cross-product matrix:
$$[\mathbf{a}]_\times = \begin{bmatrix}
0 & -a_z & a_y \\
a_z & 0 & -a_x \\
-a_y & a_x & 0
\end{bmatrix}$$
Using the identity $[\mathbf{a}]_\times \mathbf{b} = -[\mathbf{b}]_\times \mathbf{a}$:
$$\hat{\mathbf{R}}_{nb} [\delta\boldsymbol{\theta}]_\times \mathbf{l}_b = -\hat{\mathbf{R}}_{nb} [\mathbf{l}_b]_\times \delta\boldsymbol{\theta}$$

---

## 3. Position Measurement Jacobian ($\mathbf{H}_p$)

The true antenna position is:
$$\mathbf{p}_{\text{ant}}^n = \mathbf{p}^n + \mathbf{R}_{nb} \mathbf{l}_b$$

Substituting nominal plus error states:
$$\mathbf{p}_{\text{ant}}^n \approx (\hat{\mathbf{p}}^n + \delta\mathbf{p}) + (\hat{\mathbf{R}}_{nb} + \hat{\mathbf{R}}_{nb} [\delta\boldsymbol{\theta}]_\times) \mathbf{l}_b$$
$$\mathbf{p}_{\text{ant}}^n \approx \hat{\mathbf{p}}^n + \hat{\mathbf{R}}_{nb} \mathbf{l}_b + \delta\mathbf{p} - \hat{\mathbf{R}}_{nb} [\mathbf{l}_b]_\times \delta\boldsymbol{\theta}$$

Subtracting the nominal prediction $\hat{\mathbf{z}}_p = \hat{\mathbf{p}}^n + \hat{\mathbf{R}}_{nb} \mathbf{l}_b$:
$$\delta\mathbf{z}_p = \delta\mathbf{p} - \hat{\mathbf{R}}_{nb} [\mathbf{l}_b]_\times \delta\boldsymbol{\theta}$$

### Exact $3 \times 15$ Analytical Position Jacobian:
$$\mathbf{H}_p = \begin{bmatrix}
\mathbf{I}_3 & \mathbf{0}_{3\times 3} & -\hat{\mathbf{R}}_{nb} [\mathbf{l}_b]_\times & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3}
\end{bmatrix}$$

### Case 1: Zero Lever Arm ($\mathbf{l}_b = \mathbf{0}$):
$$\mathbf{H}_p = \begin{bmatrix}
\mathbf{I}_3 & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3}
\end{bmatrix}$$

---

## 4. Velocity Measurement Jacobian ($\mathbf{H}_v$)

The true antenna velocity is:
$$\mathbf{v}_{\text{ant}}^n = \mathbf{v}^n + \mathbf{R}_{nb} (\boldsymbol{\omega}_b \times \mathbf{l}_b)$$

Let $\mathbf{v}_{\text{lever}}^b = \boldsymbol{\omega}_b \times \mathbf{l}_b$.
Perturbing $\mathbf{R}_{nb}$ and $\boldsymbol{\omega}_b = \tilde{\boldsymbol{\omega}}_b - (\hat{\mathbf{b}}_g + \delta\mathbf{b}_g)$:
$$\delta\mathbf{v}_{\text{ant}}^n \approx \delta\mathbf{v} - \hat{\mathbf{R}}_{nb} [\mathbf{v}_{\text{lever}}^b]_\times \delta\boldsymbol{\theta} - \hat{\mathbf{R}}_{nb} [\mathbf{l}_b]_\times \delta\mathbf{b}_g$$

### Exact $3 \times 15$ Analytical Velocity Jacobian:
$$\mathbf{H}_v = \begin{bmatrix}
\mathbf{0}_{3\times 3} & \mathbf{I}_3 & -\hat{\mathbf{R}}_{nb} [\mathbf{v}_{\text{lever}}^b]_\times & \mathbf{0}_{3\times 3} & \hat{\mathbf{R}}_{nb} [\mathbf{l}_b]_\times
\end{bmatrix}$$

### Case 1: Zero Lever Arm ($\mathbf{l}_b = \mathbf{0}$):
$$\mathbf{H}_v = \begin{bmatrix}
\mathbf{0}_{3\times 3} & \mathbf{I}_3 & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3} & \mathbf{0}_{3\times 3}
\end{bmatrix}$$

---

## 5. Numerical Finite-Difference Verification Protocol

To verify the analytical Jacobians, numerical central differences are computed by perturbing each element $j \in \{0, \dots, 14\}$ of the 15-state error vector $\delta\mathbf{x}$ by $\epsilon = 10^{-7}$:
$$\mathbf{H}_{\text{num}, [:, j]} = \frac{\mathbf{h}(\mathbf{x} \oplus \epsilon \mathbf{e}_j) - \mathbf{h}(\mathbf{x} \oplus (-\epsilon \mathbf{e}_j))}{2\epsilon}$$
where $\oplus$ represents the state retraction operator (injecting $\delta\boldsymbol{\theta}$ into quaternion $\mathbf{q}$).

### Verification Acceptance Criteria:
$$\max_{i, j} |H_{\text{analytic}, ij} - H_{\text{numeric}, ij}| < 10^{-6}$$
$$\max_{i, j} \frac{|H_{\text{analytic}, ij} - H_{\text{numeric}, ij}|}{|H_{\text{numeric}, ij}| + 10^{-6}} < 10^{-5}$$
Verified across non-zero roll ($0.15\text{ rad}$), pitch ($-0.25\text{ rad}$), yaw ($0.85\text{ rad}$), velocity ($12.0\text{ m/s}$), and non-zero antenna lever arm ($[0.2, -0.1, 0.5]^T\text{ m}$).
