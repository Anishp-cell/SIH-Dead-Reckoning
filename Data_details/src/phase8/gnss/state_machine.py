"""
Phase 8 GNSS Outage and Recovery 4-State Machine with Temporal Hysteresis
States: GNSS_HEALTHY, GNSS_SUSPECT, GNSS_OUTAGE, GNSS_RECOVERING.
"""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class GNSSState(str, Enum):
    HEALTHY = "GNSS_HEALTHY"
    SUSPECT = "GNSS_SUSPECT"
    OUTAGE = "GNSS_OUTAGE"
    RECOVERING = "GNSS_RECOVERING"


@dataclass
class StateMachineStatus:
    state: GNSSState
    fail_counter: int
    valid_counter: int
    recovery_alpha: float
    state_duration_sec: float
    transition_event: Optional[str] = None


class GNSSOutageStateMachine:
    """
    Causal state machine orchestrating GNSS availability, outage gating,
    and soft recovery damping.
    """

    def __init__(
        self,
        n_fail_to_outage: int = 3,       # Consecutive failures to declare OUTAGE
        m_valid_to_recover: int = 3,     # Consecutive valid passes to begin RECOVERING
        recovery_duration_sec: float = 2.0,  # Duration of recovery damping phase
        min_recovery_alpha: float = 0.2,     # Initial recovery Kalman gain scale
    ):
        self.n_fail_to_outage = n_fail_to_outage
        self.m_valid_to_recover = m_valid_to_recover
        self.recovery_duration_sec = recovery_duration_sec
        self.min_recovery_alpha = min_recovery_alpha

        self.current_state: GNSSState = GNSSState.HEALTHY
        self.fail_counter: int = 0
        self.valid_counter: int = 0

        self.state_enter_time: float = 0.0
        self.recovery_start_time: float = 0.0
        self.last_timestamp: float = 0.0

    @property
    def state(self) -> GNSSState:
        return self.current_state

    @property
    def allow_measurement_update(self) -> bool:
        return self.current_state in (GNSSState.HEALTHY, GNSSState.RECOVERING)

    def reset(self, initial_state: GNSSState = GNSSState.HEALTHY, initial_time: float = 0.0) -> None:
        """Resets counters and sets state."""
        self.current_state = initial_state
        self.fail_counter = 0
        self.valid_counter = 0
        self.state_enter_time = initial_time
        self.recovery_start_time = initial_time
        self.last_timestamp = initial_time

    def step(
        self,
        timestamp: float,
        is_sample_valid: bool,
        is_gating_accepted: bool,
    ) -> StateMachineStatus:
        """
        Updates the state machine based on the current sample's validity and gating result.
        """
        self.last_timestamp = timestamp
        dt_in_state = max(0.0, timestamp - self.state_enter_time)
        transition_event: Optional[str] = None

        sample_clean = is_sample_valid and is_gating_accepted

        if sample_clean:
            self.valid_counter += 1
            self.fail_counter = 0
        else:
            self.fail_counter += 1
            self.valid_counter = 0

        # State transition logic
        if self.current_state == GNSSState.HEALTHY:
            if not sample_clean:
                self.current_state = GNSSState.SUSPECT
                self.state_enter_time = timestamp
                transition_event = "HEALTHY_TO_SUSPECT"

        elif self.current_state == GNSSState.SUSPECT:
            if sample_clean:
                self.current_state = GNSSState.HEALTHY
                self.state_enter_time = timestamp
                transition_event = "SUSPECT_TO_HEALTHY"
            elif self.fail_counter >= self.n_fail_to_outage:
                self.current_state = GNSSState.OUTAGE
                self.state_enter_time = timestamp
                transition_event = f"SUSPECT_TO_OUTAGE_FAIL_COUNT_{self.fail_counter}"

        elif self.current_state == GNSSState.OUTAGE:
            if sample_clean and self.valid_counter >= self.m_valid_to_recover:
                self.current_state = GNSSState.RECOVERING
                self.state_enter_time = timestamp
                self.recovery_start_time = timestamp
                transition_event = f"OUTAGE_TO_RECOVERING_VALID_COUNT_{self.valid_counter}"

        elif self.current_state == GNSSState.RECOVERING:
            if not sample_clean:
                # Any fault during recovery drops back to OUTAGE immediately
                self.current_state = GNSSState.OUTAGE
                self.state_enter_time = timestamp
                transition_event = "RECOVERING_RELAPSE_TO_OUTAGE"
            else:
                elapsed_rec = timestamp - self.recovery_start_time
                if elapsed_rec >= self.recovery_duration_sec:
                    self.current_state = GNSSState.HEALTHY
                    self.state_enter_time = timestamp
                    transition_event = f"RECOVERING_TO_HEALTHY_ELAPSED_{elapsed_rec:.2f}s"

        # Compute recovery damping factor alpha in [0.2, 1.0]
        if self.current_state == GNSSState.HEALTHY:
            recovery_alpha = 1.0
        elif self.current_state == GNSSState.RECOVERING:
            elapsed = max(0.0, timestamp - self.recovery_start_time)
            progress = min(1.0, elapsed / max(1e-3, self.recovery_duration_sec))
            # Smooth linear interpolation from min_alpha to 1.0
            recovery_alpha = self.min_recovery_alpha + (1.0 - self.min_recovery_alpha) * progress
        elif self.current_state == GNSSState.SUSPECT:
            recovery_alpha = 0.5
        else:  # OUTAGE
            recovery_alpha = 0.0

        return StateMachineStatus(
            state=self.current_state,
            fail_counter=self.fail_counter,
            valid_counter=self.valid_counter,
            recovery_alpha=float(recovery_alpha),
            state_duration_sec=float(timestamp - self.state_enter_time),
            transition_event=transition_event,
        )

    def get_status(self, timestamp: float) -> StateMachineStatus:
        """
        Returns the current state machine status without updating counters or triggering transitions.
        Used during intra-epoch IMU steps between 1 Hz GNSS arrivals.
        """
        if self.current_state == GNSSState.HEALTHY:
            recovery_alpha = 1.0
        elif self.current_state == GNSSState.RECOVERING:
            elapsed = max(0.0, timestamp - self.recovery_start_time)
            progress = min(1.0, elapsed / max(1e-3, self.recovery_duration_sec))
            recovery_alpha = self.min_recovery_alpha + (1.0 - self.min_recovery_alpha) * progress
        elif self.current_state == GNSSState.SUSPECT:
            recovery_alpha = 0.5
        else:
            recovery_alpha = 0.0

        return StateMachineStatus(
            state=self.current_state,
            fail_counter=self.fail_counter,
            valid_counter=self.valid_counter,
            recovery_alpha=float(recovery_alpha),
            state_duration_sec=float(timestamp - self.state_enter_time),
            transition_event=None,
        )

