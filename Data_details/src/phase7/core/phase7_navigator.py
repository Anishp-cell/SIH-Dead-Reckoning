"""
Phase 7 Integrated Navigator:
Unifies Phase 6 Vehicle-Physics ESKF with Phase 7 Offline OSM Map Matching.
Preserves the raw Phase 6 navigation solution while simultaneously generating
the map-matched candidate state and optionally applying soft Kalman updates
(cross-track position and road heading) with adaptive covariance and NIS gating.
Strictly causal, 100% offline, zero internet or future trajectory dependencies.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, Tuple
import numpy as np

from Data_details.src.phase5.core.quaternion import quaternion_to_euler_zyx, quaternion_to_rotation_matrix
from Data_details.src.phase5.core.frames import enu_yaw_to_geographic_rad
from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF
from ..matching.temporal_matcher import TemporalMapMatcher, MapMatchResult
from ..matching.map_measurement import MapMeasurementModel, MapMeasurementConfig
from ..matching.map_update import apply_map_cross_track_update, apply_map_heading_update


@dataclass
class Phase7NavigationEstimate:
    """
    Comprehensive structured output of the Phase 7 Navigation System
    compliant with Sections 29 & 30 of the Phase 7 specification.
    """
    timestamp: float
    # Phase 6 ESKF State
    p_eskf: np.ndarray                  # [East, North, Up] in meters
    v_eskf: np.ndarray                  # [East, North, Up] in m/s
    q_eskf: np.ndarray                  # Quaternion [w, x, y, z]
    yaw_eskf_rad: float                 # Cartesian ENU yaw in rad
    yaw_gps_deg: float                  # Geographic azimuth in deg [0, 360)
    v_b: np.ndarray                     # Body-frame velocity [v_fwd, v_lat, v_up]
    ba: np.ndarray                      # Accel bias [b_ax, b_ay, b_az]
    bg: np.ndarray                      # Gyro bias [b_gx, b_gy, b_gz]
    cov_diagonal: np.ndarray            # 15-element error covariance diagonal

    # Phase 7 Map Match State
    p_map_matched: np.ndarray           # Projected point on road [East, North]
    matched_segment_id: Optional[str]   # Matched road segment ID
    matched_way_id: Optional[int]       # OSM parent way ID
    road_heading_enu_rad: float         # Matched road heading in Cartesian ENU rad
    cross_track_error_m: float          # Signed cross-track error to road (m)
    along_track_position_m: float       # Distance along segment (m)
    map_confidence: float               # Continuous confidence score in [0, 1]
    candidate_count: int                # Number of candidate roads in window
    candidate_probability: float        # Top candidate posterior belief B(c*)

    # Map Constraint Updates
    map_cross_track_accepted: bool      # Whether cross-track Kalman update was applied
    map_heading_accepted: bool          # Whether heading Kalman update was applied
    cross_track_nis: float              # NIS of cross-track measurement
    heading_nis: float                  # NIS of road heading measurement
    cross_track_nu: float               # Innovation of cross-track measurement
    heading_nu: float                   # Innovation of heading measurement

    # Physical Constraints
    is_stationary: bool                 # Stationary detector state
    nhc_confidence: float               # Disturbance-adaptive NHC confidence


class Phase7Navigator:
    """
    Core navigation manager coordinating Phase 6 ESKF and Phase 7 Map Matching.
    """

    def __init__(
        self,
        eskf: Phase6ESKF,
        map_matcher: TemporalMapMatcher,
        map_model: Optional[MapMeasurementModel] = None,
        enable_map_updates: bool = True,
        enable_cross_track_update: bool = True,
        enable_heading_update: bool = True,
    ):
        self.eskf = eskf
        self.matcher = map_matcher
        self.map_model = map_model or MapMeasurementModel()
        self.enable_map_updates = bool(enable_map_updates)
        self.enable_cross_track_update = bool(enable_cross_track_update)
        self.enable_heading_update = bool(enable_heading_update)

        self.last_match: Optional[MapMatchResult] = None

    def initialize(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        q0: np.ndarray,
        ba0: Optional[np.ndarray] = None,
        bg0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
    ) -> None:
        """Initializes ESKF and resets map matcher."""
        self.eskf.initialize(p0, v0, q0, ba0, bg0, P0)
        self.matcher.reset()
        self.last_match = None

    def step(
        self,
        f_b: np.ndarray,
        omega_b: np.ndarray,
        dt: float,
        timestamp: float,
        v_speed_ai: Optional[float] = None,
        sigma_speed_ai: Optional[float] = None,
        enable_ai_speed: bool = True,
        enable_nhc: bool = True,
        enable_zupt: bool = True,
        enable_zaru: bool = True,
    ) -> Phase7NavigationEstimate:
        """
        Executes one complete navigation cycle:
        1. ESKF Prediction (IMU strapdown propagation)
        2. Stationary detection & ZUPT/ZARU updates
        3. Phase 4 AI forward speed update
        4. Disturbance-adaptive Non-Holonomic Constraint (NHC) update
        5. Phase 7 Map Candidate query & Temporal Matching
        6. Soft Map Kalman measurement updates (Cross-track & Road Heading)
        """
        # --- Steps 1 to 4: Phase 6 ESKF Navigation Cycle ---
        self.eskf.predict(f_b, omega_b, dt)

        # Stationary detection and physics updates
        is_stat, stat_dur, stat_conf = self.eskf.stationary_detector.update(
            f_b, omega_b, ai_speed_mps=v_speed_ai if enable_ai_speed else None, dt=dt
        )
        if (enable_zupt or enable_zaru) and is_stat:
            self.eskf.update_stationary_constraints(omega_b)

        # Phase 4 AI Speed update
        if enable_ai_speed and v_speed_ai is not None:
            sig_v = sigma_speed_ai if sigma_speed_ai is not None else 1.0
            self.eskf.update_speed(float(v_speed_ai), float(sig_v))

        # Disturbance-aware NHC update
        dist_report = self.eskf.disturbance_detector.evaluate(f_b, omega_b)
        nhc_conf = dist_report.c_nhc
        if enable_nhc:
            self.eskf.update_nhc(d_lat=dist_report.d_lat, d_up=dist_report.d_up)

        # Extract current state after ESKF propagation & vehicle physics
        p_curr = self.eskf.state.p.copy()
        v_curr = self.eskf.state.v.copy()
        q_curr = self.eskf.state.q.copy()
        R_nb_curr = quaternion_to_rotation_matrix(q_curr)
        v_b_curr = R_nb_curr.T @ v_curr
        v_fwd = float(v_b_curr[0])

        _, _, yaw_enu = quaternion_to_euler_zyx(q_curr)
        yaw_gps_deg = float(np.degrees(enu_yaw_to_geographic_rad(yaw_enu)))
        pos_cov_2x2 = self.eskf.P[0:2, 0:2].copy()
        yaw_std = float(np.sqrt(max(1e-6, self.eskf.P[8, 8])))

        # --- Step 5: Phase 7 Map Matching ---
        match = self.matcher.step(
            timestamp=timestamp,
            p_enu=p_curr,
            yaw_enu_rad=yaw_enu,
            pos_cov_2x2=pos_cov_2x2,
            yaw_std_rad=yaw_std,
            v_forward_mps=v_fwd,
            is_stationary=is_stat,
        )
        self.last_match = match

        # --- Step 6: Soft Map Kalman Measurement Updates ---
        ct_accepted = False
        hd_accepted = False
        ct_nis = 0.0
        hd_nis = 0.0
        ct_nu = 0.0
        hd_nu = 0.0

        if self.enable_map_updates and match.accepted:
            # 6a. Cross-track constraint update
            if self.enable_cross_track_update:
                (
                    self.eskf.state,
                    self.eskf.P,
                    ct_nu,
                    _,
                    ct_nis,
                    ct_accepted,
                ) = apply_map_cross_track_update(
                    self.eskf.state,
                    self.eskf.P,
                    match,
                    self.map_model,
                )

            # 6b. Road heading constraint update
            if self.enable_heading_update:
                (
                    self.eskf.state,
                    self.eskf.P,
                    hd_nu,
                    _,
                    hd_nis,
                    hd_accepted,
                ) = apply_map_heading_update(
                    self.eskf.state,
                    self.eskf.P,
                    match,
                    self.map_model,
                    v_forward_mps=v_fwd,
                )

        # Refresh state variables after map updates
        p_final = self.eskf.state.p.copy()
        v_final = self.eskf.state.v.copy()
        q_final = self.eskf.state.q.copy()
        R_nb_final = quaternion_to_rotation_matrix(q_final)
        v_b_final = R_nb_final.T @ v_final
        _, _, yaw_final = quaternion_to_euler_zyx(q_final)
        yaw_gps_final = float(np.degrees(enu_yaw_to_geographic_rad(yaw_final)))

        return Phase7NavigationEstimate(
            timestamp=timestamp,
            p_eskf=p_final,
            v_eskf=v_final,
            q_eskf=q_final,
            yaw_eskf_rad=yaw_final,
            yaw_gps_deg=yaw_gps_final,
            v_b=v_b_final,
            ba=self.eskf.state.ba.copy(),
            bg=self.eskf.state.bg.copy(),
            cov_diagonal=np.diag(self.eskf.P).copy(),
            p_map_matched=match.projected_position_enu.copy(),
            matched_segment_id=match.matched_segment_id,
            matched_way_id=match.matched_way_id,
            road_heading_enu_rad=match.road_heading_enu,
            cross_track_error_m=match.cross_track_error,
            along_track_position_m=match.along_track_position,
            map_confidence=match.map_confidence,
            candidate_count=match.candidate_count,
            candidate_probability=match.candidate_probability,
            map_cross_track_accepted=ct_accepted,
            map_heading_accepted=hd_accepted,
            cross_track_nis=ct_nis,
            heading_nis=hd_nis,
            cross_track_nu=ct_nu,
            heading_nu=hd_nu,
            is_stationary=is_stat,
            nhc_confidence=nhc_conf,
        )
