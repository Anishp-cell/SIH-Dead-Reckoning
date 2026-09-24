# Phase 8 Mathematical Foundation: Multi-Sensor Observability Analysis

**Project**: Smart India Hackathon 2026 — Problem Statement SIH26168  
**Sponsoring Agency**: Indian Space Research Organisation (ISRO)  
**System**: AI-ML Based Intelligent Dead Reckoning System for Seamless Navigation  
**Document**: `docs/phase8/07_OBSERVABILITY_ANALYSIS.md`  

---

## 1. Executive Summary

This document derives the formal observability properties of the hybrid navigation architecture across each phase of integration: Phase 5 (ESKF + AI Speed), Phase 6 (+Vehicle Physics), Phase 7 (+Map Constraints), and Phase 8 (+GNSS Fusion).

---

## 2. Observability Comparison Matrix Across Stack Evolution

| State Component | Physical Meaning | Phase 5 (AI Speed) | Phase 6 (+NHC/ZUPT/ZARU) | Phase 7 (+OSM Map) | Phase 8 (+GNSS Fusion) |
|---|---|:---:|:---:|:---:|:---:|
| $\mathbf{p}_{E, N}$ (Horiz. Pos) | East / North Position | **Unobservable** (open-loop drift) | **Unobservable** (pinned at stops) | **Partially Observable** ($\perp$ road normal only) | **Fully Observable** ($\mathbf{H}_p = \mathbf{I}_3$) |
| $p_U$ (Vertical Pos) | Altitude | **Unobservable** (gravity integration drift) | Weakly Observable via NHC ($v_U = 0$) | Weakly Observable | **Fully Observable** ($z_p \sim \mathcal{N}(p_U, \sigma_v^2)$) |
| $v_{\text{fwd}}$ (Forward Vel) | Vehicle Speed | **Fully Observable** ($H_v = \mathbf{u}_{\text{fwd}}^T$) | **Fully Observable** + ZUPT | **Fully Observable** | **Fully Observable** (Dual: AI + GNSS) |
| $v_{\text{lat}}, v_{\text{up}}$ | Lateral / Up Speed | Weakly Observable | **Fully Observable** (NHC: $\mathbf{H}_{\text{NHC}}$) | **Fully Observable** | **Fully Observable** (Overdetermined) |
| $\phi, \theta$ (Roll, Pitch) | Vehicle Leveling Tilt | **Observable** via gravity vector | **Observable** via gravity & NHC | **Observable** | **Observable** |
| $\psi$ (Yaw / Heading) | Vehicle Azimuth | **Unobservable** (gyro integration drift) | Weakly Observable during turns | **Fully Observable** ($\mathbf{H}_\psi$ road tangent) | **Fully Observable** (GNSS velocity tangent) |
| $\mathbf{b}_a$ (Accel Biases) | Accelerometer Bias | Partially Observable | **Observable during stops** (ZUPT) | **Observable during stops** | **Fully Observable** during dynamic GNSS acceleration |
| $\mathbf{b}_g$ (Gyro Biases) | Gyroscope Bias | Unobservable | **Observable during stops** (ZARU) | **Observable on straight roads** | **Observable during turns** |

---

## 3. Observability Matrix Rank Analysis

Let the discrete-time linear error system be:
$$\delta\mathbf{x}_k = \mathbf{F}_k \delta\mathbf{x}_{k-1} + \mathbf{w}_k$$
$$\delta\mathbf{z}_k = \mathbf{H}_k \delta\mathbf{x}_k + \mathbf{v}_k$$

The $m$-step observability matrix $\mathcal{O}_m \in \mathbb{R}^{3m \times 15}$ is:
$$\mathcal{O}_m = \begin{bmatrix}
\mathbf{H}_0 \\
\mathbf{H}_1 \mathbf{F}_1 \\
\vdots \\
\mathbf{H}_{m-1} \mathbf{F}_{m-1} \cdots \mathbf{F}_1
\end{bmatrix}$$

### In Phase 7 (During GNSS Outage):
- Rank of $\mathcal{O}_m$ on a straight road segment is **$\text{rank}(\mathcal{O}_m) = 14$**.
- The null space vector $\mathbf{v}_{\text{null}} \in \text{null}(\mathcal{O}_m)$ corresponds exactly to the along-track position coordinate:
  $$\mathbf{v}_{\text{null}} = [\mathbf{t}_{\text{road}}^T, \mathbf{0}_{1\times 12}]^T$$
  Because $\mathbf{H}_p \mathbf{t}_{\text{road}} = \mathbf{n}_{\text{road}}^T \mathbf{t}_{\text{road}} = 0$.
- Along-track position is fundamentally unobservable on 1D road networks.

### In Phase 8 (When GNSS is Available):
- With GNSS position updates:
  $$\mathbf{H}_{p,\text{gnss}} = [\mathbf{I}_3, \mathbf{0}_{3\times 12}]$$
- $\mathbf{H}_{p,\text{gnss}} \mathbf{v}_{\text{null}} = \mathbf{t}_{\text{road}} \neq \mathbf{0}$.
- **$\text{rank}(\mathcal{O}_m) = 15$ (Full Rank)**.
- All 15 error states, including along-track position and accelerometer biases, become completely observable.
