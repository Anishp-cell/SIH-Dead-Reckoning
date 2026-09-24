"""
Phase 8 Rigorous Numerical Observability Analysis
Computes discrete-time multi-epoch cumulative observability matrices,
singular values, condition numbers, and rank across:
A. Straight Driving (constant velocity)
B. Steady Turning (centripetal acceleration)
C. Stop-and-Go (standstill with ZUPT)
D. Dynamic Manoeuvres (simultaneous longitudinal accel & yaw rate)
Outputs machine-readable JSON artifact and comprehensive report.
"""

import sys
import json
from pathlib import Path
import numpy as np

workspace_root = Path(__file__).resolve().parent.parent.parent.parent
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

from Data_details.src.phase5.core.quaternion import (
    euler_zyx_to_quaternion,
    quaternion_to_rotation_matrix,
)
from Data_details.src.phase8.fusion.gnss_measurement import (
    GNSSMeasurementModel,
    GNSSMeasurementConfig,
    skew_symmetric,
)


def compute_discrete_transition_matrix(
    q: np.ndarray,
    acc_b: np.ndarray,
    omega_b: np.ndarray,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Computes discrete error-state transition matrix Phi = exp(F * dt) ~ I + F*dt + 0.5*(F*dt)^2.
    State vector: [delta_p(3), delta_v(3), delta_theta_b(3), delta_ba(3), delta_bg(3)]^T (15 states).
    """
    R_nb = quaternion_to_rotation_matrix(q)
    F = np.zeros((15, 15), dtype=np.float64)

    # d(delta_p)/dt = delta_v
    F[0:3, 3:6] = np.eye(3, dtype=np.float64)

    # d(delta_v)/dt = -R_nb * [acc_b]x * delta_theta_b - R_nb * delta_ba
    F[3:6, 6:9] = -R_nb @ skew_symmetric(acc_b)
    F[3:6, 9:12] = -R_nb

    # d(delta_theta_b)/dt = -[omega_b]x * delta_theta_b - delta_bg
    F[6:9, 6:9] = -skew_symmetric(omega_b)
    F[6:9, 12:15] = -np.eye(3, dtype=np.float64)

    # First-order / second-order Taylor expansion
    F_dt = F * dt
    Phi = np.eye(15, dtype=np.float64) + F_dt + 0.5 * (F_dt @ F_dt)
    return Phi


def build_cumulative_observability_matrix(
    trajectory_states: list,
    H_measurements: list,
    dt: float = 0.1,
) -> np.ndarray:
    """
    Constructs multi-epoch discrete-time observability matrix:
    O_K = [H_0; H_1*Phi_0; H_2*Phi_1*Phi_0; ...; H_{K-1}*Phi_{K-2}*...*Phi_0]
    """
    K = len(H_measurements)
    rows = []

    # Product of state transitions: Phi_cum = Phi_{k-1} * ... * Phi_0
    Phi_cum = np.eye(15, dtype=np.float64)

    for k in range(K):
        H_k = H_measurements[k]
        rows.append(H_k @ Phi_cum)

        if k < K - 1:
            q_k, a_k, w_k = trajectory_states[k]
            Phi_k = compute_discrete_transition_matrix(q_k, a_k, w_k, dt=dt)
            Phi_cum = Phi_k @ Phi_cum

    return np.vstack(rows)


def analyze_observability() -> dict:
    """
    Executes numerical SVD and rank analysis across 4 benchmark driving conditions.
    """
    dt = 0.1
    K_steps = 20  # 2.0 seconds window
    g = 9.80665

    lever_arm = np.array([0.25, -0.15, 0.40], dtype=np.float64)
    cfg = GNSSMeasurementConfig(lever_arm_b=lever_arm)
    model = GNSSMeasurementModel(cfg)

    # State scaling matrix for dimensionally consistent singular value analysis
    scale_diag = np.array([
        1.0, 1.0, 1.0,        # pos (m)
        1.0, 1.0, 1.0,        # vel (m/s)
        0.05, 0.05, 0.05,     # attitude (~3 deg)
        0.05, 0.05, 0.05,     # accel bias (0.05 m/s^2)
        0.001, 0.001, 0.001,  # gyro bias (1 mrad/s)
    ], dtype=np.float64)
    S_scale = np.diag(scale_diag)

    scenarios = {
        "A_straight_constant_speed": {
            "description": "Straight driving at constant speed (15 m/s, zero angular rate)",
            "speed": 15.0,
            "yaw_rate": 0.0,
            "acc_fwd": 0.0,
        },
        "B_steady_turning": {
            "description": "Steady turning at constant speed (15 m/s, yaw rate 0.2 rad/s, centripetal accel 3.0 m/s^2)",
            "speed": 15.0,
            "yaw_rate": 0.20,
            "acc_fwd": 0.0,
        },
        "C_stop_and_go_standstill": {
            "description": "Stop-and-go standstill (speed = 0 m/s, zero yaw rate, stationary)",
            "speed": 0.0,
            "yaw_rate": 0.0,
            "acc_fwd": 0.0,
        },
        "D_dynamic_manoeuvre": {
            "description": "Dynamic manoeuvre (accelerating 2.0 m/s^2 with turning yaw rate 0.25 rad/s)",
            "speed": 12.0,
            "yaw_rate": 0.25,
            "acc_fwd": 2.0,
        },
    }

    results = {}

    for sc_name, sc_data in scenarios.items():
        traj_states = []
        H_meas_list = []

        curr_yaw = 0.0
        curr_v = sc_data["speed"]

        for k in range(K_steps):
            curr_yaw += sc_data["yaw_rate"] * dt
            curr_v += sc_data["acc_fwd"] * dt

            q_k = euler_zyx_to_quaternion(curr_yaw, 0.0, 0.0)
            a_centripetal = curr_v * sc_data["yaw_rate"]
            a_b = np.array([sc_data["acc_fwd"], a_centripetal, -g], dtype=np.float64)
            w_b = np.array([0.0, 0.0, sc_data["yaw_rate"]], dtype=np.float64)

            # Combined position + 2D velocity measurement
            H_p = model.position_jacobian(q_k, lever_arm)
            H_v2d = model.velocity_2d_jacobian(q_k, w_b, lever_arm)
            H_k = np.vstack([H_p, H_v2d])  # (5, 15)

            traj_states.append((q_k, a_b, w_b))
            H_meas_list.append(H_k)

        # Build observability matrix
        O_mat = build_cumulative_observability_matrix(traj_states, H_meas_list, dt=dt)  # (100, 15)

        # Scale for unit consistency
        O_scaled = O_mat @ S_scale

        # Singular Value Decomposition
        U, S_vals, Vt = np.linalg.svd(O_scaled, full_matrices=False)

        # Rank thresholds
        tol_1e4 = 1e-4 * S_vals[0]
        tol_1e6 = 1e-6 * S_vals[0]
        rank_1e4 = int(np.sum(S_vals > tol_1e4))
        rank_1e6 = int(np.sum(S_vals > tol_1e6))
        cond_num = float(S_vals[0] / max(1e-15, S_vals[-1]))

        # Instantaneous measurement rank at single epoch
        rank_instant = int(np.linalg.matrix_rank(H_meas_list[0]))

        results[sc_name] = {
            "description": sc_data["description"],
            "instantaneous_rank": rank_instant,
            "cumulative_rank_1e4": rank_1e4,
            "cumulative_rank_1e6": rank_1e6,
            "condition_number": cond_num,
            "singular_values": [float(f"{s:.4e}") for s in S_vals],
            "max_singular_value": float(f"{S_vals[0]:.4e}"),
            "min_singular_value": float(f"{S_vals[-1]:.4e}"),
            "full_rank_achieved": bool(rank_1e6 == 15),
        }

    # Save to JSON
    out_dir = workspace_root / "Data_details" / "outputs" / "phase8" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "observability_analysis_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 75)
    print("PHASE 8 MULTI-EPOCH DISCRETE OBSERVABILITY & SVD RANK AUDIT")
    print("=" * 75)
    for name, r in results.items():
        print(f"\n[{name}]")
        print(f"  Description       : {r['description']}")
        print(f"  Instantaneous Rank: {r['instantaneous_rank']} / 15")
        print(f"  Multi-Step Rank   : {r['cumulative_rank_1e6']} / 15 (tol=1e-6) | {r['cumulative_rank_1e4']} / 15 (tol=1e-4)")
        print(f"  Condition Number  : {r['condition_number']:.2e}")
        print(f"  Singular Values   : max={r['max_singular_value']}, min={r['min_singular_value']}")
        print(f"  Full Rank (15/15) : {r['full_rank_achieved']}")
    print(f"\nSaved machine-readable audit artifact to: {json_path}")
    print("=" * 75)

    return results


if __name__ == "__main__":
    analyze_observability()
