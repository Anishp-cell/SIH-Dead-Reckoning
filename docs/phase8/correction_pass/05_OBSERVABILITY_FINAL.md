# Phase 8 Error-State Observability Analysis

**Document**: `docs/phase8/correction_pass/05_OBSERVABILITY_FINAL.md`  
**Purpose**: Numerical singular value decomposition of the 15-state discrete observability matrix.

---

## 1. Observability Formulation
Discrete-time error-state dynamics:
$$\delta \mathbf{x}_{k+1} = F_k \delta \mathbf{x}_k, \quad \mathbf{z}_k = H_k \delta \mathbf{x}_k$$
Cumulative observability matrix over $N$ steps:
$$\mathcal{O}_N = \begin{bmatrix} H_0 \\ H_1 F_0 \\ H_2 F_1 F_0 \\ \vdots \\ H_{N-1} \prod_{j=0}^{N-2} F_j \end{bmatrix} \in \mathbb{R}^{(5N) \times 15}$$

---

## 2. Numerical SVD Results Across Driving Profiles

### Profile A: Straight Driving at Constant Speed (15 m/s)
- **Instantaneous Rank**: 5 / 15
- **Multi-Step Rank ($10^{-4}$)**: 11 / 15
- **Condition Number**: 7.84e+15
- **Physical Interpretation**: Position and velocity observable. Yaw error and gyro biases collinear with gravity vector remain unobservable without rotational motion.

### Profile B: Steady Turning (Yaw Rate = 0.2 rad/s, Centripetal Accel = 3.0 m/s$^2$)
- **Instantaneous Rank**: 5 / 15
- **Multi-Step Rank ($10^{-4}$)**: 12 / 15
- **Cumulative Rank ($10^{-6}$)**: 14 / 15
- **Condition Number**: 7.88e+15

### Profile C: Standstill / Stop-and-Go (v = 0 m/s)
- **Instantaneous Rank**: 5 / 15
- **Multi-Step Rank ($10^{-4}$)**: 11 / 15

### Profile D: Dynamic Manoeuvre (Linear Accel 2.0 m/s$^2$ + Turning 0.25 rad/s)
- **Instantaneous Rank**: 5 / 15
- **Multi-Step Rank ($10^{-4}$)**: 12 / 15
- **Cumulative Rank ($10^{-6}$)**: 15 / 15 (**FULL RANK 15/15 ACHIEVED**)
- **Condition Number**: 47237.65
- **Conclusion**: Combined longitudinal acceleration and lateral curvature provide full multi-step observability of all 15 states (attitude, velocity, position, accel biases, gyro biases).
