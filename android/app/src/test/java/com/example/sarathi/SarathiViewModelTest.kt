package com.example.sarathi

import com.example.sarathi.model.NavigationMode
import com.example.sarathi.model.VehicleType
import com.example.sarathi.viewmodel.AppTab
import com.example.sarathi.viewmodel.SarathiViewModel
import junit.framework.TestCase.assertEquals
import junit.framework.TestCase.assertFalse
import junit.framework.TestCase.assertTrue
import kotlinx.coroutines.test.runTest
import org.junit.Test

class SarathiViewModelTest {

    @Test
    fun initialState_isHomeTabAndCarVehicle() = runTest {
        val viewModel = SarathiViewModel()
        assertEquals(AppTab.HOME, viewModel.currentTab.value)
        assertEquals(VehicleType.CAR_4W, viewModel.vehicleState.value.vehicleType)
        assertFalse(viewModel.vehicleState.value.isRunning)
        assertEquals(NavigationMode.READY, viewModel.vehicleState.value.mode)
    }

    @Test
    fun setVehicleType_updatesState() = runTest {
        val viewModel = SarathiViewModel()
        viewModel.setVehicleType(VehicleType.BIKE_2W)
        assertEquals(VehicleType.BIKE_2W, viewModel.vehicleState.value.vehicleType)
    }

    @Test
    fun startNavigation_switchesToNavigationTabAndRuns() = runTest {
        val viewModel = SarathiViewModel()
        viewModel.startNavigation()
        assertEquals(AppTab.NAVIGATION, viewModel.currentTab.value)
        assertTrue(viewModel.vehicleState.value.isRunning)
        assertTrue(viewModel.vehicleState.value.isStarted)
        assertEquals(NavigationMode.GNSS_LOCKED, viewModel.vehicleState.value.mode)
    }

    @Test
    fun selectTab_updatesCurrentTab() = runTest {
        val viewModel = SarathiViewModel()
        viewModel.selectTab(AppTab.AUDIT)
        assertEquals(AppTab.AUDIT, viewModel.currentTab.value)
    }

    @Test
    fun auditResults_meetISROBenchmarks() = runTest {
        val viewModel = SarathiViewModel()
        val audit = viewModel.auditResults.value
        assertTrue("Drift must be under target", audit.driftPercentAchieved < audit.driftPercentTarget)
        assertTrue("Cross track must be within lane", audit.crossTrackAchievedMeters < audit.crossTrackTargetMeters)
        assertTrue("Recovery step must be under safety limit", audit.recoveryStepAchievedMeters < audit.recoveryStepTargetMeters)
        assertTrue("Drift flag must be passed", audit.driftPassed)
        assertTrue("Cross track flag must be passed", audit.crossTrackPassed)
        assertTrue("Recovery flag must be passed", audit.recoveryStepPassed)
    }
}
