"""
Phase 7 Streaming Navigation Engine:
Executes sequential sample-by-sample 10 Hz real-time dead-reckoning navigation
fusing Phase 3 preconditioned IMU, Phase 4 AI motion intelligence, Phase 6 vehicle physics constraints
(NHC, ZUPT, ZARU, disturbance-adaptive weighting), and Phase 7 offline OSM map matching.
100% causal, zero future trajectory information, zero online internet dependencies.
"""

from typing import Optional, Union, List, Dict, Any
import numpy as np

from Data_details.src.phase5.core.quaternion import quaternion_to_euler_zyx, quaternion_to_rotation_matrix
from Data_details.src.phase5.core.frames import enu_yaw_to_geographic_rad
from Data_details.src.phase6.core.phase6_eskf import Phase6ESKF
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine
from .matching.temporal_matcher import TemporalMapMatcher, MapMatchResult
from .matching.map_measurement import MapMeasurementModel
from .matching.map_update import apply_map_cross_track_update, apply_map_heading_update
from .core.phase7_navigator import Phase7NavigationEstimate


class Phase7StreamingEngine:
    """
    Real-time sequential streaming navigation engine for Phase 7.
    """

    def __init__(
        self,
        eskf: Optional[Phase6ESKF] = None,
        ai_engine: Optional[CausalStreamingInferenceEngine] = None,
        map_matcher: Optional[TemporalMapMatcher] = None,
        map_model: Optional[MapMeasurementModel] = None,
        enable_map_updates: bool = True,
        enable_cross_track_update: bool = True,
        enable_heading_update: bool = True,
    ):
        self.eskf = eskf or Phase6ESKF()
        self.ai_engine = ai_engine
        self.matcher = map_matcher
        self.map_model = map_model or MapMeasurementModel()
        self.enable_map_updates = bool(enable_map_updates)
        self.enable_cross_track_update = bool(enable_cross_track_update)
        self.enable_heading_update = bool(enable_heading_update)

        self.last_timestamp: Optional[float] = None
        self.step_count = 0
        self.last_match: Optional[MapMatchResult] = None

    def initialize(
        self,
        p0: np.ndarray,
        v0: np.ndarray,
        q0: np.ndarray,
        ba0: Optional[np.ndarray] = None,
        bg0: Optional[np.ndarray] = None,
        P0: Optional[np.ndarray] = None,
        initial_timestamp: float = 0.0,
    ) -> None:
        """Initializes ESKF state and clears detectors and temporal matcher history."""
        self.eskf.initialize(p0, v0, q0, ba0, bg0, P0)
        self.last_timestamp = initial_timestamp
        self.step_count = 0
        self.eskf.stationary_detector.reset()
        if self.ai_engine is not None:
            self.ai_engine.reset()
        if self.matcher is not None:
            self.matcher.reset()
        self.last_match = None

    def step(
        self,
        imu_sample_12ch: Union[np.ndarray, List[float]],
        timestamp: float,
        dt_override: Optional[float] = None,
        enable_ai_speed: bool = True,
        enable_nhc: bool = True,
        enable_zupt: bool = True,
        enable_zaru: bool = True,
        fixed_sigma_override: Optional[float] = None,
    ) -> Phase7NavigationEstimate:
        """
        Executes one full streaming navigation cycle at 10 Hz:
        1. Extract specific force and angular rate.
        2. IMU Kinematic Prediction.
        3. Phase 4 AI Speed Inference & Update.
        4. Disturbance & Stationary Detection.
        5. ZUPT & ZARU Updates (if stationary).
        6. Non-Holonomic Constraints (NHC) Update.
        7. Phase 7 Map Candidate Query & Temporal Matching.
        8. Soft Map Kalman Measurement Updates.
        """
        sample_arr = np.asarray(imu_sample_12ch, dtype=np.float64).ravel()
        if len(sample_arr) < 6:
            raise ValueError(f"Expected at least 6 IMU channels, got {len(sample_arr)}")

        if dt_override is not None:
            dt = dt_override
        elif self.last_timestamp is not None:
            dt = max(1e-4, timestamp - self.last_timestamp)
        else:
            dt = 0.100

        self.last_timestamp = timestamp
        self.step_count += 1

        f_meas = np.array([sample_arr[0], sample_arr[1], -sample_arr[2]], dtype=np.float64)
        omega_meas = sample_arr[3:6]
        vibe_energy = float(sample_arr[10]) if len(sample_arr) > 10 else 0.0

        # 1. IMU Prediction step
        self.eskf.predict(f_meas, omega_meas, dt)

        # 2. Phase 4 AI Speed update
        ai_speed_val = 0.0
        if enable_ai_speed and self.ai_engine is not None:
            motion = self.ai_engine.update(sample_arr, timestamp=timestamp)
            ai_speed_val = motion.forward_speed_mps
            sigma_val = fixed_sigma_override if fixed_sigma_override is not None else motion.speed_uncertainty
            self.eskf.update_speed(ai_speed_val, sigma_val)

        # 3. Disturbance Detection
        dist_report = self.eskf.disturbance_detector.evaluate(f_meas, omega_meas, vibration_energy=vibe_energy)
        self.eskf.last_nhc_conf = dist_report.c_nhc

        # 4. Stationary Detection
        is_stat, stat_dur, stat_conf = self.eskf.stationary_detector.update(
            f_meas, omega_meas, ai_speed_mps=ai_speed_val if enable_ai_speed else None, dt=dt
        )
        self.eskf.last_is_stationary = is_stat
        self.eskf.last_stat_duration = stat_dur

        # 5. Stationary Updates (ZUPT + ZARU)
        if enable_zupt and is_stat:
            self.eskf.update_stationary_constraints(omega_meas)
        else:
            self.eskf.last_zupt_accepted = False
            self.eskf.last_zaru_accepted = False

        # 6. Non-Holonomic Constraints (NHC)
        if enable_nhc:
            self.eskf.update_nhc(d_lat=dist_report.d_lat, d_up=dist_report.d_up)

        # Capture Phase 6 state before map update
        p_pre = self.eskf.state.p.copy()
        v_pre = self.eskf.state.v.copy()
        q_pre = self.eskf.state.q.copy()
        R_nb_pre = quaternion_to_rotation_matrix(q_pre)
        v_b_pre = R_nb_pre.T @ v_pre
        v_fwd = float(v_b_pre[0])

        _, _, yaw_enu = quaternion_to_euler_zyx(q_pre)
        pos_cov_2x2 = self.eskf.P[0:2, 0:2].copy()
        yaw_std = float(np.sqrt(max(1e-6, self.eskf.P[8, 8])))

        # 7. Map Matching
        match: Optional[MapMatchResult] = None
        if self.matcher is not None:
            match = self.matcher.step(
                timestamp=timestamp,
                p_enu=p_pre,
                yaw_enu_rad=yaw_enu,
                pos_cov_2x2=pos_cov_2x2,
                yaw_std_rad=yaw_std,
                v_forward_mps=v_fwd,
                is_stationary=is_stat,
            )
        self.last_match = match

        # 8. Soft Map Kalman Updates
        ct_accepted = False
        hd_accepted = False
        ct_nis = 0.0
        hd_nis = 0.0
        ct_nu = 0.0
        hd_nu = 0.0

        if self.enable_map_updates and match is not None and match.accepted:
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

        p_post = self.eskf.state.p.copy()
        v_post = self.eskf.state.v.copy()
        q_post = self.eskf.state.q.copy()
        R_nb_post = quaternion_to_rotation_matrix(q_post)
        v_b_post = R_nb_post.T @ v_post
        _, _, yaw_post = quaternion_to_euler_zyx(q_post)
        yaw_gps_post = float(np.degrees(enu_yaw_to_geographic_rad(yaw_post)))

        return Phase7NavigationEstimate(
            timestamp=timestamp,
            p_eskf=p_post,
            v_eskf=v_post,
            q_eskf=q_post,
            yaw_eskf_rad=yaw_post,
            yaw_gps_deg=yaw_gps_post,
            v_b=v_b_post,
            ba=self.eskf.state.ba.copy(),
            bg=self.eskf.state.bg.copy(),
            cov_diagonal=np.diag(self.eskf.P).copy(),
            p_map_matched=match.projected_position_enu.copy() if match else np.array([np.nan, np.nan]),
            matched_segment_id=match.matched_segment_id if match else None,
            matched_way_id=match.matched_way_id if match else None,
            road_heading_enu_rad=match.road_heading_enu if match else 0.0,
            cross_track_error_m=match.cross_track_error if match else 0.0,
            along_track_position_m=match.along_track_position if match else 0.0,
            map_confidence=match.map_confidence if match else 0.0,
            candidate_count=match.candidate_count if match else 0,
            candidate_probability=match.candidate_probability if match else 0.0,
            map_cross_track_accepted=ct_accepted,
            map_heading_accepted=hd_accepted,
            cross_track_nis=ct_nis,
            heading_nis=hd_nis,
            cross_track_nu=ct_nu,
            heading_nu=hd_nu,
            is_stationary=is_stat,
            nhc_confidence=dist_report.c_nhc,
        )
