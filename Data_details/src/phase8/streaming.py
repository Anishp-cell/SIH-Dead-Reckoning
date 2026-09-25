"""
Phase 8 Streaming Navigation Engine:
Unified sequential 10 Hz real-time dead-reckoning and hybrid GNSS fusion.
Integrates:
1. Phase 3 Preconditioned IMU Strapdown Propagation
2. Phase 4 AI Motion Intelligence Speed Model
3. Phase 6 Disturbance-Adaptive Vehicle Physics (NHC, ZUPT, ZARU)
4. Phase 7 Offline OSM Map Matching Constraints
5. Phase 8 GNSS Position & Velocity Fusion, Quality Pre-Filter,
   4-State Outage Automaton, and Soft Zero-Teleportation Recovery.
100% causal, strictly offline, zero internet or future trajectory dependencies.
"""

from dataclasses import dataclass
from typing import Optional, Union, List, Dict, Any, Tuple
import numpy as np

from Data_details.src.phase5.core.quaternion import quaternion_to_euler_zyx, quaternion_to_rotation_matrix
from Data_details.src.phase5.core.frames import enu_yaw_to_geographic_rad
from Data_details.src.phase4.streaming import CausalStreamingInferenceEngine
from Data_details.src.phase7.matching.temporal_matcher import TemporalMapMatcher, MapMatchResult
from Data_details.src.phase7.matching.map_measurement import MapMeasurementModel
from Data_details.src.phase7.matching.map_update import (
    apply_map_cross_track_update,
    apply_map_heading_update,
    apply_map_along_track_update,
)
from Data_details.src.phase7.matching.spline_odometry import RoadSplineOdometry, SplineOdometryConfig

from .core.phase8_eskf import Phase8ESKF
from .gnss.coordinate import geodetic_to_enu, gps_vel_to_enu_velocity
from .gnss.quality import GNSSQualityAssessor, GNSSQualityReport
from .gnss.state_machine import GNSSOutageStateMachine, GNSSState
from .gnss.synthetic import Provenance


@dataclass
class Phase8NavigationEstimate:
    """
    Comprehensive structured output of Phase 8 Hybrid Navigation System.
    """
    timestamp: float
    # Fused Hybrid State
    p_fused: np.ndarray                  # [East, North, Up] (m)
    v_fused: np.ndarray                  # [v_East, v_North, v_Up] (m/s)
    q_fused: np.ndarray                  # Quaternion [w, x, y, z]
    yaw_fused_rad: float                 # Cartesian ENU yaw in rad
    yaw_gps_deg: float                   # Geographic azimuth in deg [0, 360)
    v_b: np.ndarray                      # Body velocity [v_fwd, v_lat, v_up]
    ba: np.ndarray                       # Accel bias (m/s^2)
    bg: np.ndarray                       # Gyro bias (rad/s)
    cov_diagonal: np.ndarray             # 15-element error covariance diagonal

    # GNSS Telemetry & State
    gnss_state: str                      # HEALTHY, SUSPECT, OUTAGE, RECOVERING
    gnss_pos_raw_enu: np.ndarray         # Raw GNSS position [East, North, Up]
    gnss_vel_raw_enu: np.ndarray         # Raw GNSS velocity [v_East, v_North, v_Up]
    gnss_pos_accepted: bool              # Position Kalman update applied
    gnss_vel_accepted: bool              # Velocity Kalman update applied
    gnss_pos_nis: float                  # 3-DOF position NIS
    gnss_vel_nis: float                  # 3-DOF velocity NIS
    gnss_p_step_m: float                 # Applied position step (m)
    recovery_alpha: float                # Soft recovery damping factor [0.2, 1.0]
    gnss_acc_m: float                    # Receiver stated horizontal accuracy
    gnss_sats: int                       # Number of satellites

    # Map Match Telemetry
    p_map_matched: np.ndarray            # Projected point on road [East, North]
    matched_segment_id: Optional[str]    # Matched road segment ID
    matched_way_id: Optional[int]        # OSM parent way ID
    road_heading_enu_rad: float          # Matched road heading (ENU rad)
    cross_track_error_m: float           # Cross-track error (m)
    along_track_position_m: float        # Distance along segment (m)
    map_confidence: float                # Map confidence score [0, 1]
    candidate_count: int                 # Number of candidate roads
    candidate_probability: float = 0.0   # Top candidate posterior
    map_cross_track_accepted: bool = False # Cross-track Kalman update applied
    map_heading_accepted: bool = False   # Heading Kalman update applied
    map_along_track_accepted: bool = False # Along-track road-spline Kalman update applied
    cross_track_nis: float = 0.0
    heading_nis: float = 0.0
    along_track_nis: float = 0.0

    # Physical Constraints
    is_stationary: bool = False
    nhc_confidence: float = 1.0
    filter_healthy: bool = True


class Phase8StreamingEngine:
    """
    Real-time sequential streaming navigation engine for Phase 8.
    """

    def __init__(
        self,
        ref_lat_deg: float,
        ref_lon_deg: float,
        ref_alt_m: float = 0.0,
        eskf: Optional[Phase8ESKF] = None,
        ai_engine: Optional[CausalStreamingInferenceEngine] = None,
        map_matcher: Optional[TemporalMapMatcher] = None,
        map_model: Optional[MapMeasurementModel] = None,
        enable_map_updates: bool = True,
        enable_cross_track_update: bool = True,
        enable_heading_update: bool = True,
        enable_along_track_update: bool = True,
        enable_gnss_pos: bool = True,
        enable_gnss_vel: bool = True,
        spline_odometry: Optional[RoadSplineOdometry] = None,
    ):
        self.ref_lat_deg = ref_lat_deg
        self.ref_lon_deg = ref_lon_deg
        self.ref_alt_m = ref_alt_m

        self.eskf = eskf or Phase8ESKF()
        self.ai_engine = ai_engine
        self.matcher = map_matcher
        self.map_model = map_model or MapMeasurementModel()
        self.spline_odometry = spline_odometry or RoadSplineOdometry()

        self.enable_map_updates = bool(enable_map_updates)
        self.enable_cross_track_update = bool(enable_cross_track_update)
        self.enable_heading_update = bool(enable_heading_update)
        self.enable_along_track_update = bool(enable_along_track_update)
        self.enable_gnss_pos = bool(enable_gnss_pos)
        self.enable_gnss_vel = bool(enable_gnss_vel)

        self.quality_assessor = GNSSQualityAssessor()
        self.state_machine = GNSSOutageStateMachine()

        self.last_timestamp: Optional[float] = None
        self.last_gnss_timestamp: Optional[float] = None
        self.gnss_timeout_s: float = 1.5
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
        """Initializes ESKF state, clears detectors, state machine, and temporal matcher."""
        self.eskf.initialize(p0, v0, q0, ba0, bg0, P0)
        self.last_timestamp = initial_timestamp
        self.step_count = 0
        self.eskf.stationary_detector.reset()
        if self.ai_engine is not None:
            self.ai_engine.reset()
        if self.matcher is not None:
            self.matcher.reset()
        self.quality_assessor.reset()
        self.state_machine.reset(initial_state=GNSSState.HEALTHY, initial_time=initial_timestamp)
        self.spline_odometry.reset()
        self.last_gnss_timestamp = initial_timestamp
        self.last_match = None

    def step(
        self,
        imu_sample_12ch: Union[np.ndarray, List[float]],
        timestamp: float,
        gps_lat: Optional[float] = None,
        gps_lon: Optional[float] = None,
        gps_alt: Optional[float] = None,
        gps_acc_m: Optional[float] = None,
        gps_sats: Optional[int] = None,
        gps_speed_mps: Optional[float] = None,
        gps_heading_deg: Optional[float] = None,
        gps_timestamp: Optional[float] = None,
        is_synthetic_outage: bool = False,
        dt_override: Optional[float] = None,
        enable_ai_speed: bool = True,
        enable_nhc: bool = True,
        enable_zupt: bool = True,
        enable_zaru: bool = True,
        fixed_sigma_override: Optional[float] = None,
    ) -> Phase8NavigationEstimate:
        """
        Executes one full streaming navigation cycle at 10 Hz:
        1. IMU Kinematic Prediction.
        2. AI Speed Update.
        3. Stationary Detection & ZUPT/ZARU Updates.
        4. Disturbance-Adaptive NHC Update.
        5. Map Candidate Query & Soft Map Updates (Phase 7).
        6. GNSS Coordinate Transformation & Quality Assessment.
        7. Outage State Machine Step & Recovery Damping.
        8. Soft GNSS Kalman Updates (Position & Velocity).
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

        # 4. Stationary Detection & Updates (ZUPT + ZARU)
        is_stat, stat_dur, stat_conf = self.eskf.stationary_detector.update(
            f_meas, omega_meas, ai_speed_mps=ai_speed_val if enable_ai_speed else None, dt=dt
        )
        self.eskf.last_is_stationary = is_stat
        self.eskf.last_stat_duration = stat_dur

        if enable_zupt and is_stat:
            self.eskf.update_stationary_constraints(omega_meas)
        else:
            self.eskf.last_zupt_accepted = False
            self.eskf.last_zaru_accepted = False

        # 5. Non-Holonomic Constraints (NHC)
        if enable_nhc:
            self.eskf.update_nhc(d_lat=dist_report.d_lat, d_up=dist_report.d_up)

        # Capture State before Map Matching
        p_pre_map = self.eskf.state.p.copy()
        v_pre_map = self.eskf.state.v.copy()
        q_pre_map = self.eskf.state.q.copy()
        R_nb_pre = quaternion_to_rotation_matrix(q_pre_map)
        v_b_pre = R_nb_pre.T @ v_pre_map
        v_fwd = float(v_b_pre[0])

        _, _, yaw_enu = quaternion_to_euler_zyx(q_pre_map)
        pos_cov_2x2 = self.eskf.P[0:2, 0:2].copy()
        yaw_std = float(np.sqrt(max(1e-6, self.eskf.P[8, 8])))

        # 6. Map Matching (Phase 7)
        match: Optional[MapMatchResult] = None
        if self.matcher is not None:
            match = self.matcher.step(
                timestamp=timestamp,
                p_enu=p_pre_map,
                yaw_enu_rad=yaw_enu,
                pos_cov_2x2=pos_cov_2x2,
                yaw_std_rad=yaw_std,
                v_forward_mps=v_fwd,
                is_stationary=is_stat,
            )
        self.last_match = match

        # 7. Soft Map Updates
        ct_accepted = False
        hd_accepted = False
        al_accepted = False
        ct_nis = 0.0
        hd_nis = 0.0
        al_nis = 0.0

        if self.enable_map_updates and match is not None and match.accepted:
            if self.enable_cross_track_update:
                (
                    self.eskf.state,
                    self.eskf.P,
                    _,
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
                    _,
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

            # Along-track road-spline update during outage
            is_outage_mode = is_synthetic_outage or (self.state_machine.current_state in [GNSSState.OUTAGE, GNSSState.SUSPECT])
            if is_outage_mode:
                if not self.spline_odometry.is_active:
                    self.spline_odometry.start_outage(match, timestamp)
                target_along, var_along, valid = self.spline_odometry.step(
                    ai_speed_val,
                    sigma_val,
                    timestamp,
                    match,
                    is_stationary=is_stat,
                    dt_override=dt,
                )
                if valid and self.enable_along_track_update:
                    (
                        self.eskf.state,
                        self.eskf.P,
                        _,
                        _,
                        al_nis,
                        al_accepted,
                    ) = apply_map_along_track_update(
                        self.eskf.state,
                        self.eskf.P,
                        match,
                        self.map_model,
                        target_along_m=target_along,
                        variance_along_m2=var_along,
                    )
        else:
            if not is_synthetic_outage and self.state_machine.current_state == GNSSState.HEALTHY:
                self.spline_odometry.reset()

        # 8. GNSS Processing & Fusion (Phase 8)
        gnss_pos_enu = np.array([np.nan, np.nan, np.nan])
        gnss_vel_enu = np.array([np.nan, np.nan, np.nan])
        has_gnss_coords = (
            gps_lat is not None and gps_lon is not None and
            not np.isnan(gps_lat) and not np.isnan(gps_lon) and
            not is_synthetic_outage
        )

        acc_m = float(gps_acc_m if gps_acc_m is not None else 999.0)
        num_sats = int(gps_sats if gps_sats is not None else 0)
        speed_mps = float(gps_speed_mps if gps_speed_mps is not None else 0.0)
        bearing_deg = float(gps_heading_deg if gps_heading_deg is not None else 0.0)

        # Determine if this epoch brings a new GNSS fix or is within normal interval
        is_new_gnss = False
        if has_gnss_coords:
            if gps_timestamp is not None:
                is_new_gnss = (self.last_gnss_timestamp is None) or (gps_timestamp > self.last_gnss_timestamp + 1e-4)
            else:
                is_new_gnss = (self.last_gnss_timestamp is None) or (timestamp - self.last_gnss_timestamp >= 0.95)

        gnss_pos_accepted = False
        gnss_vel_accepted = False
        gnss_pos_nis = 0.0
        gnss_vel_nis = 0.0
        actual_p_step = 0.0
        quality_report: Optional[GNSSQualityReport] = None

        if has_gnss_coords and is_new_gnss:
            alt_m = float(gps_alt if gps_alt is not None else self.ref_alt_m)
            gnss_pos_enu = geodetic_to_enu(
                lat_deg=gps_lat,
                lon_deg=gps_lon,
                alt_m=alt_m,
                ref_lat_deg=self.ref_lat_deg,
                ref_lon_deg=self.ref_lon_deg,
                ref_alt_m=self.ref_alt_m,
            )
            gnss_vel_enu = gps_vel_to_enu_velocity(speed_mps, bearing_deg)

            quality_report = self.quality_assessor.evaluate(
                timestamp=timestamp,
                p_enu=gnss_pos_enu,
                stated_acc_m=acc_m,
                num_sats=num_sats,
                speed_mps=speed_mps,
                bearing_deg=bearing_deg,
            )
            is_sample_valid = quality_report.is_valid

            # Evaluate gate on raw unclipped innovation before state machine transition
            if is_sample_valid:
                stated_acc = quality_report.scaled_acc_m if quality_report else acc_m
                gate_accepted, pos_nis = self.eskf.evaluate_gnss_position_gate(
                    z_pos_enu=gnss_pos_enu,
                    stated_acc_m=stated_acc,
                )
                is_gating_accepted = gate_accepted
                gnss_pos_nis = pos_nis
            else:
                is_gating_accepted = False
                gnss_pos_nis = 999.0
                stated_acc = acc_m

            # Step state machine with the ACTUAL gate decision
            sm_status = self.state_machine.step(
                timestamp=timestamp,
                is_sample_valid=is_sample_valid,
                is_gating_accepted=is_gating_accepted,
            )
            current_gnss_state = sm_status.state
            recovery_alpha = sm_status.recovery_alpha
            self.last_gnss_timestamp = timestamp if gps_timestamp is None else gps_timestamp

            # Apply updates if not in full OUTAGE and gate accepted
            p_pre_gnss = self.eskf.state.p.copy()
            v_pre_gnss = self.eskf.state.v.copy()
            q_pre_gnss = self.eskf.state.q.copy()

            if current_gnss_state != GNSSState.OUTAGE and is_gating_accepted:
                if self.enable_gnss_pos:
                    gnss_pos_accepted = self.eskf.update_gnss_position(
                        z_pos_enu=gnss_pos_enu,
                        stated_acc_m=stated_acc,
                        recovery_alpha=recovery_alpha,
                    )
                    actual_p_step = self.eskf.last_gnss_p_step

                # 2D horizontal velocity update (never fabricates vertical velocity)
                if self.enable_gnss_vel and speed_mps > 0.5:
                    gnss_vel_accepted = self.eskf.update_gnss_velocity_2d(
                        z_vel_2d=gnss_vel_enu[:2],
                        omega_b=omega_meas,
                        vel_acc_mps=max(0.3, stated_acc * 0.1),
                        recovery_alpha=recovery_alpha,
                    )
                    gnss_vel_nis = self.eskf.last_gnss_vel_nis

                if current_gnss_state == GNSSState.RECOVERING and (gnss_pos_accepted or gnss_vel_accepted):
                    self.eskf.recovery_manager.evaluate_transition(
                        timestamp=timestamp,
                        p_pre=p_pre_gnss,
                        p_post=self.eskf.state.p,
                        v_pre=v_pre_gnss,
                        v_post=self.eskf.state.v,
                        q_pre=q_pre_gnss,
                        q_post=self.eskf.state.q,
                        dt=dt,
                    )
        else:
            # No new GNSS arrival at this 10 Hz epoch
            time_since_gnss = (timestamp - self.last_gnss_timestamp) if self.last_gnss_timestamp is not None else 999.0
            if is_synthetic_outage or time_since_gnss >= self.gnss_timeout_s:
                sm_status = self.state_machine.step(
                    timestamp=timestamp,
                    is_sample_valid=False,
                    is_gating_accepted=False,
                )
            else:
                sm_status = self.state_machine.get_status(timestamp=timestamp)

            current_gnss_state = sm_status.state
            recovery_alpha = sm_status.recovery_alpha

        # Assemble Final Phase 8 Output
        p_final = self.eskf.state.p.copy()
        v_final = self.eskf.state.v.copy()
        q_final = self.eskf.state.q.copy()
        R_nb_final = quaternion_to_rotation_matrix(q_final)
        v_b_final = R_nb_final.T @ v_final
        _, _, yaw_final = quaternion_to_euler_zyx(q_final)
        yaw_gps_final = float(np.degrees(enu_yaw_to_geographic_rad(yaw_final)))

        cov_diag = np.diag(self.eskf.P).copy()
        filter_healthy = bool(np.all(np.isfinite(cov_diag)) and np.all(cov_diag > 0))

        return Phase8NavigationEstimate(
            timestamp=timestamp,
            p_fused=p_final,
            v_fused=v_final,
            q_fused=q_final,
            yaw_fused_rad=yaw_final,
            yaw_gps_deg=yaw_gps_final,
            v_b=v_b_final,
            ba=self.eskf.state.ba.copy(),
            bg=self.eskf.state.bg.copy(),
            cov_diagonal=cov_diag,
            gnss_state=current_gnss_state.value,
            gnss_pos_raw_enu=gnss_pos_enu,
            gnss_vel_raw_enu=gnss_vel_enu,
            gnss_pos_accepted=gnss_pos_accepted,
            gnss_vel_accepted=gnss_vel_accepted,
            gnss_pos_nis=gnss_pos_nis,
            gnss_vel_nis=gnss_vel_nis,
            gnss_p_step_m=actual_p_step,
            recovery_alpha=recovery_alpha,
            gnss_acc_m=acc_m,
            gnss_sats=num_sats,
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
            map_along_track_accepted=al_accepted,
            cross_track_nis=ct_nis,
            heading_nis=hd_nis,
            along_track_nis=al_nis,
            is_stationary=is_stat,
            nhc_confidence=dist_report.c_nhc,
            filter_healthy=filter_healthy,
        )
