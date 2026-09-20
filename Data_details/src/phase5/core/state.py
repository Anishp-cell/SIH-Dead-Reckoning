"""
Phase 5 Navigation State & 15-State Error State Data Structures.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import numpy as np


@dataclass
class NominalState:
    """
    16-dimensional continuous nominal navigation state:
        p: (3,) position in Navigation (ENU) frame [m]
        v: (3,) velocity in Navigation (ENU) frame [m/s]
        q: (4,) attitude quaternion [qw, qx, qy, qz] (Body -> Navigation)
        ba: (3,) accelerometer bias in Body frame [m/s^2]
        bg: (3,) gyroscope bias in Body frame [rad/s]
    """
    p: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    v: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    q: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64))
    ba: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    bg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def copy(self) -> "NominalState":
        return NominalState(
            p=self.p.copy(),
            v=self.v.copy(),
            q=self.q.copy(),
            ba=self.ba.copy(),
            bg=self.bg.copy(),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "p": self.p.tolist(),
            "v": self.v.tolist(),
            "q": self.q.tolist(),
            "ba": self.ba.tolist(),
            "bg": self.bg.tolist(),
        }


@dataclass
class ErrorState:
    """
    15-dimensional indirect error state vector:
        dp: (3,) position error in Navigation (ENU) frame [m]
        dv: (3,) velocity error in Navigation (ENU) frame [m/s]
        dtheta: (3,) small-angle attitude error in Body frame [rad]
        dba: (3,) accelerometer bias error in Body frame [m/s^2]
        dbg: (3,) gyroscope bias error in Body frame [rad/s]
    """
    dp: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    dv: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    dtheta: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    dba: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    dbg: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))

    def to_vector(self) -> np.ndarray:
        """Flattens 5 error blocks into a contiguous 15x1 vector."""
        return np.concatenate([self.dp, self.dv, self.dtheta, self.dba, self.dbg])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> "ErrorState":
        """Reconstructs structured error state from 15x1 vector."""
        v = np.asarray(vec, dtype=np.float64).ravel()
        if len(v) != 15:
            raise ValueError(f"Expected 15-element error vector, got {len(v)}")
        return cls(
            dp=v[0:3].copy(),
            dv=v[3:6].copy(),
            dtheta=v[6:9].copy(),
            dba=v[9:12].copy(),
            dbg=v[12:15].copy(),
        )

    def reset(self) -> None:
        """Resets expectation of error state to zero after injection."""
        self.dp.fill(0.0)
        self.dv.fill(0.0)
        self.dtheta.fill(0.0)
        self.dba.fill(0.0)
        self.dbg.fill(0.0)


@dataclass
class NavigationEstimate:
    """Comprehensive navigation estimate output produced at every filter step."""
    timestamp: float
    p: np.ndarray                 # (3,) Position in ENU (m)
    v: np.ndarray                 # (3,) Velocity in ENU (m/s)
    q: np.ndarray                 # (4,) Quaternion [qw, qx, qy, qz]
    ba: np.ndarray                # (3,) Accel bias (m/s^2)
    bg: np.ndarray                # (3,) Gyro bias (rad/s)
    cov_diagonal: np.ndarray      # (15,) Diagonal elements of P
    forward_speed_mps: float      # Projected body forward speed (m/s)
    speed_innovation: float       # Measurement residual nu = z - h(x)
    innovation_variance: float    # S = H P H^T + R
    nis: float                    # Normalized Innovation Squared nu^2 / S
    measurement_accepted: bool    # Whether speed measurement passed NIS gate
    filter_healthy: bool          # Whether covariance is finite and positive
