"""
Unit tests for Phase 8 4-state causal GNSS Outage State Machine with hysteresis.
"""

import pytest
from Data_details.src.phase8.gnss.state_machine import GNSSOutageStateMachine, GNSSState


def test_state_machine_hysteresis_transitions():
    """
    Verifies state transitions:
    HEALTHY -> SUSPECT -> OUTAGE -> RECOVERING -> HEALTHY
    """
    sm = GNSSOutageStateMachine(
        n_fail_to_outage=3,
        m_valid_to_recover=3,
        recovery_duration_sec=2.0,
        min_recovery_alpha=0.2,
    )
    sm.reset(initial_state=GNSSState.HEALTHY, initial_time=0.0)

    # 1. Clean step
    st = sm.step(timestamp=0.1, is_sample_valid=True, is_gating_accepted=True)
    assert st.state == GNSSState.HEALTHY
    assert st.recovery_alpha == 1.0

    # 2. First failure -> SUSPECT
    st = sm.step(timestamp=0.2, is_sample_valid=False, is_gating_accepted=False)
    assert st.state == GNSSState.SUSPECT
    assert sm.fail_counter == 1

    # 3. Second failure -> stays SUSPECT
    st = sm.step(timestamp=0.3, is_sample_valid=False, is_gating_accepted=False)
    assert st.state == GNSSState.SUSPECT
    assert sm.fail_counter == 2

    # 4. Third failure -> OUTAGE
    st = sm.step(timestamp=0.4, is_sample_valid=False, is_gating_accepted=False)
    assert st.state == GNSSState.OUTAGE
    assert st.recovery_alpha == 0.0

    # 5. Outage continues
    for t in [0.5, 0.6, 0.7]:
        st = sm.step(timestamp=t, is_sample_valid=False, is_gating_accepted=False)
        assert st.state == GNSSState.OUTAGE

    # 6. Recovery hysteresis: 1st valid pass -> still OUTAGE
    st = sm.step(timestamp=0.8, is_sample_valid=True, is_gating_accepted=True)
    assert st.state == GNSSState.OUTAGE
    assert sm.valid_counter == 1

    # 7. 2nd valid pass -> still OUTAGE
    st = sm.step(timestamp=0.9, is_sample_valid=True, is_gating_accepted=True)
    assert st.state == GNSSState.OUTAGE
    assert sm.valid_counter == 2

    # 8. 3rd valid pass -> enters RECOVERING
    st = sm.step(timestamp=1.0, is_sample_valid=True, is_gating_accepted=True)
    assert st.state == GNSSState.RECOVERING
    assert st.recovery_alpha >= 0.2

    # 9. Damping ramp over 2.0 seconds: from t=1.0 to t=3.0
    st_mid = sm.step(timestamp=2.0, is_sample_valid=True, is_gating_accepted=True)
    assert st_mid.state == GNSSState.RECOVERING
    assert 0.5 <= st_mid.recovery_alpha < 1.0

    # 10. Elapsed >= 2.0s -> enters HEALTHY
    st_healthy = sm.step(timestamp=3.1, is_sample_valid=True, is_gating_accepted=True)
    assert st_healthy.state == GNSSState.HEALTHY
    assert st_healthy.recovery_alpha == 1.0


def test_state_machine_recovery_relapse():
    """Verifies that any fault encountered during RECOVERING immediately falls back to OUTAGE."""
    sm = GNSSOutageStateMachine(n_fail_to_outage=3, m_valid_to_recover=2)
    sm.reset(initial_state=GNSSState.OUTAGE, initial_time=0.0)

    # 2 valid passes to enter RECOVERING
    sm.step(timestamp=0.1, is_sample_valid=True, is_gating_accepted=True)
    st = sm.step(timestamp=0.2, is_sample_valid=True, is_gating_accepted=True)
    assert st.state == GNSSState.RECOVERING

    # Fault during recovery
    st_relapse = sm.step(timestamp=0.3, is_sample_valid=False, is_gating_accepted=True)
    assert st_relapse.state == GNSSState.OUTAGE
    assert st_relapse.transition_event == "RECOVERING_RELAPSE_TO_OUTAGE"
