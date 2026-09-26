"""
Topological Route-Aware Dead Reckoning (RADR) Maneuver Guidance Engine.

This module provides offline, real-time topological guidance during GNSS blackout
conditions (tunnels, underpasses, urban canyons, flyover splits):
1. Pre-caches route maneuver waypoints (turn angles, flyover ramps, underpasses, lane merges).
2. Computes continuous along-track distance and time-to-maneuver using dead-reckoned positions.
3. Issues progressive driver prompts (Advance Notice -> Lane Selection -> Immediate Turn -> Execution).
4. Monitors IMU gyroscope angular rates to verify maneuver execution:
   - When executed: triggers heading bias alignment (pseudo-measurement) to clamp yaw drift.
   - When missed: triggers divergence alert, calculates cross-track drift penalty, and re-routes.
"""

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Dict, Any, List, Optional, Tuple
import numpy as np


class ManeuverType(str, Enum):
    TURN_RIGHT_90 = "TURN_RIGHT_90"
    TURN_LEFT_90 = "TURN_LEFT_90"
    TURN_RIGHT_SLIGHT = "TURN_RIGHT_SLIGHT"
    TURN_LEFT_SLIGHT = "TURN_LEFT_SLIGHT"
    BRANCH_FLYOVER = "BRANCH_FLYOVER"
    BRANCH_UNDERPASS_TUNNEL = "BRANCH_UNDERPASS_TUNNEL"
    KEEP_LEFT_FORK = "KEEP_LEFT_FORK"
    KEEP_RIGHT_FORK = "KEEP_RIGHT_FORK"
    LANE_CHANGE_LEFT = "LANE_CHANGE_LEFT"
    LANE_CHANGE_RIGHT = "LANE_CHANGE_RIGHT"
    ROUNDABOUT_EXIT = "ROUNDABOUT_EXIT"
    STRAIGHT_CONTINUE = "STRAIGHT_CONTINUE"


class GuidancePhase(str, Enum):
    EN_ROUTE = "EN_ROUTE"                         # > 150m: Cruising on road
    ADVANCE_ALERT = "ADVANCE_ALERT"               # 50m - 150m: Prepare for maneuver
    PREPARE_MANEUVER = "PREPARE_MANEUVER"         # 15m - 50m: Lane positioning
    EXECUTE_NOW = "EXECUTE_NOW"                   # 0m - 15m: Turn execution window
    MANEUVER_COMPLETED = "MANEUVER_COMPLETED"     # Turn verified via Gyro -> Heading Reset
    MISSED_MANEUVER = "MISSED_MANEUVER"           # Vehicle failed to turn -> Divergence risk


@dataclass
class WaypointManeuver:
    """Represents a topological maneuver node cached along the pre-planned route."""
    maneuver_id: str
    type: ManeuverType
    pos_enu: np.ndarray                        # [East, North, Up] in meters
    ingress_heading_rad: float                 # Road heading before maneuver (rad, ENU)
    egress_heading_rad: float                  # Target road heading after maneuver (rad, ENU)
    turn_angle_deg: float                      # Signed relative turn angle (+CW/-CCW or standard)
    road_name: str                             # Name of the egress road / branch
    action_instruction: str                    # User-facing prompt (e.g., "In 80m, take underpass on left")
    recommended_lanes: List[int] = field(default_factory=lambda: [1])  # 1-indexed lane recommendations
    total_lanes: int = 2
    trigger_radius_m: float = 14.0             # Radius around pos_enu where turn execution occurs
    warning_distance_m: float = 150.0          # Distance to begin advance alerts
    elevation_delta_m: float = 0.0             # Expected Z change (+ for flyover, - for tunnel)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "maneuver_id": self.maneuver_id,
            "type": self.type.value,
            "pos_enu": [round(float(x), 2) for x in self.pos_enu[:2]],
            "pos_z": round(float(self.pos_enu[2]), 2) if len(self.pos_enu) > 2 else 0.0,
            "turn_angle_deg": round(float(self.turn_angle_deg), 1),
            "road_name": self.road_name,
            "action_instruction": self.action_instruction,
            "recommended_lanes": self.recommended_lanes,
            "total_lanes": self.total_lanes,
            "trigger_radius_m": self.trigger_radius_m,
            "elevation_delta_m": round(float(self.elevation_delta_m), 1),
        }


@dataclass
class ManeuverGuidanceState:
    """Current state of topological route guidance at a specific time step."""
    time_s: float
    active_maneuver: Optional[WaypointManeuver]
    phase: GuidancePhase
    distance_to_maneuver_m: float
    time_to_maneuver_s: float
    turn_executed: bool
    missed_turn_detected: bool
    heading_calibration_rad: Optional[float]   # Corrected heading if maneuver executed
    cross_track_divergence_m: float            # Penalty if maneuver was missed
    prompt_text: str                           # e.g., "In 45m: Take Right 90° Turn"
    lane_guidance_display: str                 # Visual lane indicator (e.g. "[ ⮰ ] [   ]")
    maneuver_icon: str                         # Icon identifier for UI (e.g., "turn_right", "tunnel")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "time_s": round(float(self.time_s), 2),
            "maneuver_id": self.active_maneuver.maneuver_id if self.active_maneuver else None,
            "type": self.active_maneuver.type.value if self.active_maneuver else None,
            "phase": self.phase.value,
            "dist_m": round(float(self.distance_to_maneuver_m), 1),
            "ttm_s": round(float(self.time_to_maneuver_s), 1),
            "turn_executed": self.turn_executed,
            "missed_turn": self.missed_turn_detected,
            "calibrated_yaw": round(float(self.heading_calibration_rad), 3) if self.heading_calibration_rad is not None else None,
            "divergence_m": round(float(self.cross_track_divergence_m), 2),
            "prompt": self.prompt_text,
            "lane_display": self.lane_guidance_display,
            "icon": self.maneuver_icon,
        }


class TopologicalManeuverEngine:
    """
    Core engine managing topological route guidance and maneuver verification
    during GNSS blackout.
    """

    def __init__(self, waypoints: Optional[List[WaypointManeuver]] = None):
        self.waypoints: List[WaypointManeuver] = waypoints or []
        self.current_wp_idx: int = 0
        self.last_phase: GuidancePhase = GuidancePhase.EN_ROUTE
        self.accumulated_turn_rad: float = 0.0
        self.in_turn_execution: bool = False
        self.missed_warning_issued: bool = False
        self.last_pos_enu: Optional[np.ndarray] = None
        self.maneuver_start_yaw: Optional[float] = None

    def add_waypoint(self, wp: WaypointManeuver):
        self.waypoints.append(wp)

    def set_route(self, waypoints: List[WaypointManeuver]):
        self.waypoints = sorted(waypoints, key=lambda w: w.pos_enu[0]) # or natural order
        self.current_wp_idx = 0
        self.last_phase = GuidancePhase.EN_ROUTE
        self.accumulated_turn_rad = 0.0
        self.in_turn_execution = False

    def update(
        self,
        time_s: float,
        pos_enu: np.ndarray,
        vel_enu: np.ndarray,
        yaw_rad: float,
        gyro_z_radps: float,
        dt: float = 0.1,
    ) -> ManeuverGuidanceState:
        """
        Executes one step of topological guidance.
        Args:
            time_s: Current simulation/epoch time
            pos_enu: Dead-reckoned 3D position [East, North, Up]
            vel_enu: Dead-reckoned 3D velocity [vE, vN, vU]
            yaw_rad: Dead-reckoned heading in ENU radians
            gyro_z_radps: Vertical axis gyroscope rate (rad/s)
            dt: Time step duration
        """
        if not self.waypoints or self.current_wp_idx >= len(self.waypoints):
            # No upcoming maneuver on route
            return ManeuverGuidanceState(
                time_s=time_s,
                active_maneuver=None,
                phase=GuidancePhase.EN_ROUTE,
                distance_to_maneuver_m=9999.0,
                time_to_maneuver_s=999.0,
                turn_executed=False,
                missed_turn_detected=False,
                heading_calibration_rad=None,
                cross_track_divergence_m=0.0,
                prompt_text="Continue on route (Clear Sky / Blackout Dead Reckoning)",
                lane_guidance_display="[   ] [   ]",
                maneuver_icon="straight",
            )

        wp = self.waypoints[self.current_wp_idx]

        # 1. Compute vector and along-track distance to the waypoint
        d_vec = wp.pos_enu[:2] - pos_enu[:2]
        dist_euclidean = float(np.linalg.norm(d_vec))

        # Project along the ingress heading to get signed along-track distance
        u_track = np.array([np.cos(wp.ingress_heading_rad), np.sin(wp.ingress_heading_rad)])
        dist_along = float(np.dot(d_vec, u_track))

        # Vehicle 2D speed
        speed_2d = float(np.linalg.norm(vel_enu[:2]))
        speed_safe = max(1.5, speed_2d)
        time_to_maneuver = max(0.0, dist_along / speed_safe) if dist_along > 0 else 0.0

        # Lane guidance display string
        lane_str = self._format_lane_display(wp.recommended_lanes, wp.total_lanes, wp.type)
        icon_str = self._get_icon_for_type(wp.type)

        # 2. Determine Guidance Phase
        heading_calibration: Optional[float] = None
        turn_executed = False
        missed_turn = False
        divergence_m = 0.0

        if dist_along > wp.warning_distance_m:
            phase = GuidancePhase.EN_ROUTE
            prompt = f"In {dist_along:.0f}m: {wp.action_instruction}"

        elif dist_along > 50.0:
            phase = GuidancePhase.ADVANCE_ALERT
            prompt = f"In {dist_along:.0f}m: {wp.action_instruction}"

        elif dist_along > wp.trigger_radius_m:
            phase = GuidancePhase.PREPARE_MANEUVER
            prompt = f"In {dist_along:.0f}m: Prepare to {self._get_action_verb(wp.type)} onto {wp.road_name}"

        elif dist_along >= -wp.trigger_radius_m:
            # Inside the maneuver execution window!
            phase = GuidancePhase.EXECUTE_NOW
            prompt = f"{self._get_execution_prompt(wp.type)} NOW ({wp.road_name})"

            if not self.in_turn_execution:
                self.in_turn_execution = True
                self.accumulated_turn_rad = 0.0
                self.maneuver_start_yaw = yaw_rad

            # Accumulate angular rate
            self.accumulated_turn_rad += gyro_z_radps * dt

            # Check if relative turn angle matches expected turn
            expected_turn_rad = math.radians(wp.turn_angle_deg)
            turn_diff = abs(self.accumulated_turn_rad - expected_turn_rad)

            # Tolerance for completed turn: within 22 degrees (0.38 rad)
            if abs(self.accumulated_turn_rad) > 0.4 * abs(expected_turn_rad) and turn_diff < 0.40:
                phase = GuidancePhase.MANEUVER_COMPLETED
                turn_executed = True
                prompt = f"Maneuver Completed: Heading Calibrated to {wp.road_name}"
                heading_calibration = wp.egress_heading_rad
                self.current_wp_idx += 1
                self.in_turn_execution = False

        else:
            # Passed the waypoint (dist_along < -trigger_radius_m)
            # Check if vehicle actually turned or went straight (missed the maneuver)
            expected_turn_rad = math.radians(wp.turn_angle_deg)
            actual_heading_delta = abs(self.accumulated_turn_rad)

            if abs(expected_turn_rad) > math.radians(30.0) and actual_heading_delta < math.radians(20.0):
                # Major turn was expected, but driver kept going straight!
                phase = GuidancePhase.MISSED_MANEUVER
                missed_turn = True
                # Divergence error: distance from intended egress road
                divergence_m = abs(dist_along) * math.sin(abs(expected_turn_rad))
                prompt = f"ALERT: Missed {self._get_action_verb(wp.type)}! Drift penalty: +{divergence_m:.1f}m. Re-routing..."
                self.current_wp_idx += 1
                self.in_turn_execution = False
            else:
                # Driver did turn or it was a minor fork
                phase = GuidancePhase.MANEUVER_COMPLETED
                turn_executed = True
                prompt = f"Maneuver Completed on {wp.road_name}"
                heading_calibration = wp.egress_heading_rad
                self.current_wp_idx += 1
                self.in_turn_execution = False

        self.last_phase = phase
        self.last_pos_enu = pos_enu.copy()

        return ManeuverGuidanceState(
            time_s=time_s,
            active_maneuver=wp,
            phase=phase,
            distance_to_maneuver_m=max(0.0, dist_along),
            time_to_maneuver_s=time_to_maneuver,
            turn_executed=turn_executed,
            missed_turn_detected=missed_turn,
            heading_calibration_rad=heading_calibration,
            cross_track_divergence_m=divergence_m,
            prompt_text=prompt,
            lane_guidance_display=lane_str,
            maneuver_icon=icon_str,
        )

    def _format_lane_display(self, rec_lanes: List[int], total_lanes: int, m_type: ManeuverType) -> str:
        """Formats an intuitive multi-lane ASCII/Unicode visualizer."""
        total = max(1, min(6, total_lanes))
        indicators = []
        for i in range(1, total + 1):
            if i in rec_lanes:
                if m_type in [ManeuverType.TURN_RIGHT_90, ManeuverType.TURN_RIGHT_SLIGHT, ManeuverType.KEEP_RIGHT_FORK]:
                    indicators.append("[ ⮱ ]")
                elif m_type in [ManeuverType.TURN_LEFT_90, ManeuverType.TURN_LEFT_SLIGHT, ManeuverType.KEEP_LEFT_FORK]:
                    indicators.append("[ ⮰ ]")
                elif m_type == ManeuverType.BRANCH_UNDERPASS_TUNNEL:
                    indicators.append("[ ⮷ ]")
                elif m_type == ManeuverType.BRANCH_FLYOVER:
                    indicators.append("[ ⮵ ]")
                else:
                    indicators.append("[ ↑ ]")
            else:
                indicators.append("[   ]")
        return " ".join(indicators)

    def _get_icon_for_type(self, m_type: ManeuverType) -> str:
        mapping = {
            ManeuverType.TURN_RIGHT_90: "turn_right_90",
            ManeuverType.TURN_LEFT_90: "turn_left_90",
            ManeuverType.TURN_RIGHT_SLIGHT: "turn_right_slight",
            ManeuverType.TURN_LEFT_SLIGHT: "turn_left_slight",
            ManeuverType.BRANCH_FLYOVER: "flyover_ramp",
            ManeuverType.BRANCH_UNDERPASS_TUNNEL: "underpass_tunnel",
            ManeuverType.KEEP_LEFT_FORK: "fork_left",
            ManeuverType.KEEP_RIGHT_FORK: "fork_right",
            ManeuverType.LANE_CHANGE_LEFT: "lane_left",
            ManeuverType.LANE_CHANGE_RIGHT: "lane_right",
            ManeuverType.ROUNDABOUT_EXIT: "roundabout",
            ManeuverType.STRAIGHT_CONTINUE: "straight",
        }
        return mapping.get(m_type, "straight")

    def _get_action_verb(self, m_type: ManeuverType) -> str:
        if m_type == ManeuverType.TURN_RIGHT_90:
            return "turn sharp right (90°)"
        if m_type == ManeuverType.TURN_LEFT_90:
            return "turn sharp left (90°)"
        if m_type == ManeuverType.BRANCH_UNDERPASS_TUNNEL:
            return "descend into underpass / tunnel"
        if m_type == ManeuverType.BRANCH_FLYOVER:
            return "take elevated flyover ramp"
        if m_type == ManeuverType.KEEP_LEFT_FORK:
            return "keep left at fork"
        if m_type == ManeuverType.KEEP_RIGHT_FORK:
            return "keep right at fork"
        return "maneuver"

    def _get_execution_prompt(self, m_type: ManeuverType) -> str:
        if m_type == ManeuverType.TURN_RIGHT_90:
            return "TURN RIGHT 90°"
        if m_type == ManeuverType.TURN_LEFT_90:
            return "TURN LEFT 90°"
        if m_type == ManeuverType.BRANCH_UNDERPASS_TUNNEL:
            return "ENTER TUNNEL / UNDERPASS"
        if m_type == ManeuverType.BRANCH_FLYOVER:
            return "TAKE FLYOVER RAMP"
        if m_type == ManeuverType.KEEP_LEFT_FORK:
            return "KEEP LEFT"
        if m_type == ManeuverType.KEEP_RIGHT_FORK:
            return "KEEP RIGHT"
        return "TURN"
