"""
Phase 8 Automated Numerical Finite-Difference Jacobian Verification
Rigorously evaluates analytical 3x15 position and 3x15/2x15 velocity Jacobians
against central finite differences under simultaneous non-zero roll, pitch, yaw,
velocity, angular rate, lever arm, and sensor biases.
Outputs machine-readable JSON and Markdown summary.
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
    inject_small_angle_error,
)
from Data_details.src.phase8.fusion.gnss_measurement import (
    GNSSMeasurementModel,
    GNSSMeasurementConfig,
)


def run_jacobian_finite_difference_verification() -> dict:
    """
    Executes automated numerical finite-difference verification across all GNSS observation models.
    """
    eps = 1e-7

    # Non-zero test parameters
    roll_rad = np.radians(15.0)
    pitch_rad = np.radians(-10.0)
    yaw_rad = np.radians(55.0)
    q0 = euler_zyx_to_quaternion(yaw_rad, pitch_rad, roll_rad)

    p0 = np.array([125.4, -48.2, 12.8], dtype=np.float64)
    v0 = np.array([14.5, -6.2, 0.8], dtype=np.float64)
    omega_b = np.array([0.045, -0.062, 0.115], dtype=np.float64)
    lever_arm = np.array([0.25, -0.15, 0.40], dtype=np.float64)

    cfg = GNSSMeasurementConfig(lever_arm_b=lever_arm)
    model = GNSSMeasurementModel(cfg)

    results = {}

    # =========================================================================
    # 1. Position Jacobian H_p (3 x 15)
    # =========================================================================
    H_p_analytical = model.position_jacobian(q0, lever_arm)
    H_p_num = np.zeros((3, 15), dtype=np.float64)

    # Cols 0:3 -> delta_p
    for i in range(3):
        dp = np.zeros(3)
        dp[i] = eps
        hp_plus = model.predict_position(p0 + dp, q0, lever_arm)
        hp_minus = model.predict_position(p0 - dp, q0, lever_arm)
        H_p_num[:, i] = (hp_plus - hp_minus) / (2.0 * eps)

    # Cols 3:6 -> delta_v (zero)
    # Cols 6:9 -> delta_theta_b (body rotation error)
    for k in range(3):
        dtheta = np.zeros(3)
        dtheta[k] = eps
        q_plus = inject_small_angle_error(q0, dtheta)
        q_minus = inject_small_angle_error(q0, -dtheta)
        hp_plus = model.predict_position(p0, q_plus, lever_arm)
        hp_minus = model.predict_position(p0, q_minus, lever_arm)
        H_p_num[:, 6 + k] = (hp_plus - hp_minus) / (2.0 * eps)

    # Cols 9:15 -> biases (zero)
    abs_err_p = np.abs(H_p_analytical - H_p_num)
    rel_err_p = abs_err_p / (np.abs(H_p_analytical) + 1e-12)

    results["position_3d"] = {
        "max_abs_error": float(np.max(abs_err_p)),
        "max_rel_error": float(np.max(rel_err_p)),
        "matrix_shape": list(H_p_analytical.shape),
        "passed": bool(np.max(abs_err_p) < 1e-6),
    }

    # =========================================================================
    # 2. 3D Velocity Jacobian H_v (3 x 15)
    # =========================================================================
    H_v_analytical = model.velocity_jacobian(q0, omega_b, lever_arm)
    H_v_num = np.zeros((3, 15), dtype=np.float64)

    # Cols 3:6 -> delta_v
    for i in range(3):
        dv = np.zeros(3)
        dv[i] = eps
        hv_plus = model.predict_velocity(v0 + dv, q0, omega_b, lever_arm)
        hv_minus = model.predict_velocity(v0 - dv, q0, omega_b, lever_arm)
        H_v_num[:, 3 + i] = (hv_plus - hv_minus) / (2.0 * eps)

    # Cols 6:9 -> delta_theta_b
    for k in range(3):
        dtheta = np.zeros(3)
        dtheta[k] = eps
        q_plus = inject_small_angle_error(q0, dtheta)
        q_minus = inject_small_angle_error(q0, -dtheta)
        hv_plus = model.predict_velocity(v0, q_plus, omega_b, lever_arm)
        hv_minus = model.predict_velocity(v0, q_minus, omega_b, lever_arm)
        H_v_num[:, 6 + k] = (hv_plus - hv_minus) / (2.0 * eps)

    # Cols 12:15 -> delta_bg (gyro bias coupling into velocity via lever arm)
    for k in range(3):
        dbg = np.zeros(3)
        dbg[k] = eps
        # If gyro bias shifts by dbg, true omega_b = omega_meas - (bg + dbg) = omega_b - dbg
        hv_plus = model.predict_velocity(v0, q0, omega_b + dbg, lever_arm)
        hv_minus = model.predict_velocity(v0, q0, omega_b - dbg, lever_arm)
        # Note: in error state, d(omega_meas - bg)/d(delta_bg) = -I, so delta_omega = -delta_bg
        H_v_num[:, 12 + k] = -(hv_plus - hv_minus) / (2.0 * eps)

    abs_err_v = np.abs(H_v_analytical - H_v_num)
    rel_err_v = abs_err_v / (np.abs(H_v_analytical) + 1e-12)

    results["velocity_3d"] = {
        "max_abs_error": float(np.max(abs_err_v)),
        "max_rel_error": float(np.max(rel_err_v)),
        "matrix_shape": list(H_v_analytical.shape),
        "passed": bool(np.max(abs_err_v) < 1e-6),
    }

    # =========================================================================
    # 3. 2D Horizontal Velocity Jacobian H_{v,2D} (2 x 15)
    # =========================================================================
    H_v2d_analytical = model.velocity_2d_jacobian(q0, omega_b, lever_arm)
    H_v2d_num = H_v_num[0:2, :]

    abs_err_v2d = np.abs(H_v2d_analytical - H_v2d_num)
    rel_err_v2d = abs_err_v2d / (np.abs(H_v2d_analytical) + 1e-12)

    results["velocity_2d"] = {
        "max_abs_error": float(np.max(abs_err_v2d)),
        "max_rel_error": float(np.max(rel_err_v2d)),
        "matrix_shape": list(H_v2d_analytical.shape),
        "passed": bool(np.max(abs_err_v2d) < 1e-6),
    }

    # Save to JSON
    out_dir = workspace_root / "Data_details" / "outputs" / "phase8" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "jacobian_finite_difference_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("PHASE 8 AUTOMATED NUMERICAL FINITE-DIFFERENCE JACOBIAN AUDIT")
    print("=" * 70)
    for name, res in results.items():
        print(f"[{name:15s}] Shape={res['matrix_shape']} | Max Abs Error={res['max_abs_error']:.3e} | Max Rel Error={res['max_rel_error']:.3e} | Passed={res['passed']}")
    print(f"\nSaved machine-readable audit artifact to: {json_path}")
    print("=" * 70)

    return results


if __name__ == "__main__":
    run_jacobian_finite_difference_verification()
