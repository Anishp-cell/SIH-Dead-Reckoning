package com.example.sarathi.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.sarathi.model.ActiveAlgorithm
import com.example.sarathi.model.AuditResults
import com.example.sarathi.model.NavigationMode
import com.example.sarathi.model.SimulationStage
import com.example.sarathi.model.VehicleState
import com.example.sarathi.model.VehicleType
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlin.math.sin

enum class AppTab {
    SPLASH,
    HOME,
    NAVIGATION,
    AUDIT
}

class SarathiViewModel : ViewModel() {

    private val _currentTab = MutableStateFlow(AppTab.SPLASH)
    val currentTab: StateFlow<AppTab> = _currentTab.asStateFlow()

    private val _vehicleState = MutableStateFlow(VehicleState())
    val vehicleState: StateFlow<VehicleState> = _vehicleState.asStateFlow()

    private val _auditResults = MutableStateFlow(AuditResults())
    val auditResults: StateFlow<AuditResults> = _auditResults.asStateFlow()

    private val _preFlightStep = MutableStateFlow<Int?>(null)
    val preFlightStep: StateFlow<Int?> = _preFlightStep.asStateFlow()

    private var simulationJob: Job? = null

    fun selectTab(tab: AppTab) {
        _currentTab.value = tab
    }

    fun goBack() {
        when (_currentTab.value) {
            AppTab.AUDIT -> _currentTab.value = AppTab.NAVIGATION
            AppTab.NAVIGATION -> {
                stopSimulation()
                _currentTab.value = AppTab.HOME
            }
            AppTab.HOME -> resetToIntro()
            AppTab.SPLASH -> {}
        }
    }

    fun resetToIntro() {
        stopSimulation()
        _vehicleState.value = VehicleState()
        _preFlightStep.value = null
        _currentTab.value = AppTab.SPLASH
    }

    fun setVehicleType(type: VehicleType) {
        _vehicleState.update { it.copy(vehicleType = type) }
    }

    fun toggleDiagnosticsPanel() {
        _vehicleState.update { it.copy(showDiagnosticsPanel = !it.showDiagnosticsPanel) }
    }

    fun runPreFlightChecksAndNavigate() {
        viewModelScope.launch {
            for (step in 1..5) {
                _preFlightStep.value = step
                delay(400L)
            }
            _preFlightStep.value = 6
            delay(500L)
            _preFlightStep.value = null
            startNavigation()
        }
    }

    fun dismissPreFlightChecks() {
        _preFlightStep.value = null
    }

    fun startNavigation() {
        _currentTab.value = AppTab.NAVIGATION
        startSimulation()
    }

    fun startSimulation() {
        simulationJob?.cancel()
        _vehicleState.update {
            VehicleState(
                vehicleType = it.vehicleType,
                stage = SimulationStage.NORMAL_GNSS,
                mode = NavigationMode.GNSS_LOCKED,
                activeAlgorithm = ActiveAlgorithm.CLOSED_LOOP_GNSS,
                isStarted = true,
                isRunning = true,
                isPaused = false,
                isCompleted = false
            )
        }
        startSimulationTicker()
    }

    fun stopSimulation() {
        simulationJob?.cancel()
        _vehicleState.update {
            VehicleState(
                vehicleType = it.vehicleType,
                stage = SimulationStage.READY,
                mode = NavigationMode.READY,
                activeAlgorithm = ActiveAlgorithm.STANDBY,
                isStarted = false,
                isRunning = false,
                isPaused = false,
                isCompleted = false
            )
        }
    }

    fun restartSimulation() {
        startSimulation()
    }

    fun togglePauseResume() {
        val currentlyRunning = _vehicleState.value.isRunning
        if (currentlyRunning) {
            simulationJob?.cancel()
            _vehicleState.update { it.copy(isRunning = false, isPaused = true) }
        } else {
            if (!_vehicleState.value.isStarted) {
                startSimulation()
            } else {
                _vehicleState.update { it.copy(isRunning = true, isPaused = false) }
                startSimulationTicker()
            }
        }
    }

    private fun startSimulationTicker() {
        simulationJob = viewModelScope.launch {
            val tickIntervalMs = 30L
            val tickDeltaSec = tickIntervalMs / 1000.0f

            while (_vehicleState.value.timeSeconds < _vehicleState.value.totalScenarioSeconds) {
                delay(tickIntervalMs)

                _vehicleState.update { state ->
                    val newTime = state.timeSeconds + tickDeltaSec
                    val normalizedProgress = (newTime / state.totalScenarioSeconds).coerceIn(0.0f, 1.0f)

                    // Story stage progression
                    val story = computeStoryStage(newTime)
                    val (carOffset, ghostOffset, curveDeg) = computeDynamics(newTime, story.inTunnel, state.vehicleType)

                    // Speeds per story stage:
                    val targetCruiseSpeed = when {
                        newTime < 7.5f  -> 62.0f + 0.8f * sin(newTime * 0.5f)
                        newTime < 11.5f -> 58.5f + 0.5f * sin(newTime * 0.6f)
                        newTime < 20.0f -> 54.0f + 0.6f * sin(newTime * 0.8f) // DR live cruise 54 km/h
                        newTime < 28.0f -> 51.5f + 0.4f * sin(newTime * 0.7f) // Pothole zone
                        newTime < 35.0f -> 48.0f + 0.5f * sin(newTime * 0.7f) // Gently turns curve
                        newTime < 38.0f -> 52.0f + 0.6f * sin(newTime * 0.6f) // Turn verified
                        newTime < 43.0f -> 58.0f + 0.6f * sin(newTime * 0.5f) // Seamless fusion
                        newTime < 48.0f -> 62.0f + 0.8f * sin(newTime * 0.5f) // Normal restored
                        else            -> 0.0f
                    }

                    val rampFactor = (newTime / 1.5f).coerceIn(0.0f, 1.0f)
                    val currentSpeedKmh = if (newTime >= state.totalScenarioSeconds) 0.0f else targetCruiseSpeed * rampFactor
                    val currentSpeedMs = currentSpeedKmh / 3.6f

                    val newDistanceTraveled = state.distanceTraveledMeters + (currentSpeedMs * tickDeltaSec)
                    val distanceRemaining = (1400.0f - newDistanceTraveled).coerceAtLeast(0.0f)

                    // Uncertainty Halo
                    val haloRadius = if (story.inTunnel) {
                        (15.0f + 8.0f * sin((newTime - 11.5f) * 0.45f)).coerceIn(15.0f, 25.0f)
                    } else {
                        14.0f
                    }

                    // Trust percentages
                    val gnssTrust = when {
                        newTime < 7.5f  -> 100
                        newTime < 11.5f -> (100 - ((newTime - 7.5f) / 4.0f * 75f)).toInt().coerceIn(25, 100)
                        newTime < 38.0f -> 0
                        newTime < 43.0f -> (25 + ((newTime - 38.0f) / 5.0f * 70f)).toInt().coerceIn(25, 95)
                        else            -> 100
                    }

                    val insTrust = when {
                        newTime < 7.5f  -> 40
                        newTime < 11.5f -> (40 + ((newTime - 7.5f) / 4.0f * 55f)).toInt().coerceIn(40, 95)
                        newTime < 38.0f -> 98
                        newTime < 43.0f -> (98 - ((newTime - 38.0f) / 5.0f * 53f)).toInt().coerceIn(45, 98)
                        else            -> 40
                    }

                    state.copy(
                        timeSeconds = newTime,
                        routeProgress = normalizedProgress,
                        stage = story.stage,
                        mode = story.mode,
                        inTunnel = story.inTunnel,
                        tunnelProgress = story.tunnelProgress,
                        activeAlgorithm = story.algo,
                        potholePulsing = story.potholePulsing,
                        showPotholeHazard = story.showPotholeHazard,
                        showGpsLostAlert = story.alert,
                        turnInstruction = story.stage.instruction,
                        speedKmh = currentSpeedKmh,
                        speedMs = currentSpeedMs,
                        speedUncertaintyMs = if (story.inTunnel) 0.38f else 0.16f,
                        distanceTraveledMeters = newDistanceTraveled,
                        distanceRemainingMeters = distanceRemaining,
                        crossTrackMeters = if (story.inTunnel) (0.28f + 0.12f * sin(newTime * 0.28f)) else 0.16f,
                        recoveryStepMeters = if (story.mode == NavigationMode.RECOVERING) 0.006f else 0.0f,
                        roadCurveDegrees = curveDeg,
                        carLateralOffsetRatio = carOffset,
                        ghostLateralOffsetRatio = ghostOffset,
                        covarianceHaloRadiusDp = haloRadius,
                        gnssTrustPercent = gnssTrust,
                        insTrustPercent = insTrust,
                        aiSpeedEstimateKmh = if (story.inTunnel) (52.8f + 0.4f * sin(newTime * 1.1f)) else currentSpeedKmh,
                        speedVarianceSigma2 = if (story.inTunnel) 0.84f else 0.18f,
                        headingSigmaDeg = if (story.inTunnel) 1.7f else 0.3f,
                        isMapMatchingLocked = true,
                        isNhcActive = true,
                        isDisturbanceGated = story.potholePulsing,
                        isCompleted = newTime >= state.totalScenarioSeconds
                    )
                }
            }

            _vehicleState.update {
                it.copy(
                    isRunning = false,
                    isCompleted = true,
                    stage = SimulationStage.DESTINATION_REACHED,
                    mode = NavigationMode.COMPLETED,
                    speedKmh = 0.0f,
                    speedMs = 0.0f,
                    distanceRemainingMeters = 0.0f,
                    routeProgress = 1.0f,
                    potholePulsing = false,
                    showPotholeHazard = false
                )
            }
        }
    }

    private fun computeStoryStage(t: Float): StoryState {
        return when {
            // Stage 1: Normal GNSS navigation (0.0s - 7.5s) - 100% Signal Integrity
            t < 7.5f -> StoryState(
                stage = SimulationStage.NORMAL_GNSS,
                mode = NavigationMode.GNSS_LOCKED,
                inTunnel = false,
                algo = ActiveAlgorithm.CLOSED_LOOP_GNSS,
                tunnelProgress = 0.0f,
                alert = false,
                showPotholeHazard = false,
                potholePulsing = false
            )
            // Stage 2: GNSS degradation begins (7.5s - 11.5s)
            t < 11.5f -> StoryState(
                stage = SimulationStage.GNSS_DEGRADING,
                mode = NavigationMode.GNSS_LOCKED,
                inTunnel = false,
                algo = ActiveAlgorithm.CLOSED_LOOP_GNSS,
                tunnelProgress = 0.0f,
                alert = false,
                showPotholeHazard = false,
                potholePulsing = false
            )
            // Stage 3: GNSS Lost -> DR Detected & Active (11.5s - 16.0s)
            t < 16.0f -> StoryState(
                stage = SimulationStage.GNSS_LOST_DR_ACTIVE,
                mode = NavigationMode.DEAD_RECKONING,
                inTunnel = true,
                algo = ActiveAlgorithm.AI_SPEED,
                tunnelProgress = (t - 11.5f) / 26.5f,
                alert = t < 14.5f,
                showPotholeHazard = false,
                potholePulsing = false
            )
            // Stage 4: DR Navigation with NHC & Map Matching (16.0s - 20.0s)
            t < 20.0f -> StoryState(
                stage = SimulationStage.DR_NAVIGATING,
                mode = NavigationMode.DEAD_RECKONING,
                inTunnel = true,
                algo = ActiveAlgorithm.NHC_PHYSICS,
                tunnelProgress = (t - 11.5f) / 26.5f,
                alert = false,
                showPotholeHazard = false,
                potholePulsing = false
            )
            // Stage 5: Pothole Hazard appears, approached and crossed (20.0s - 28.0s)
            t < 28.0f -> StoryState(
                stage = SimulationStage.DISTURBANCE_POTHOLE,
                mode = NavigationMode.DEAD_RECKONING,
                inTunnel = true,
                algo = ActiveAlgorithm.POTHOLE_GATING,
                tunnelProgress = (t - 11.5f) / 26.5f,
                alert = false,
                showPotholeHazard = true,
                potholePulsing = (t in 23.5f..27.0f)
            )
            // Stage 6: Approaching and executing turn under DR (28.0s - 35.0s)
            t < 35.0f -> StoryState(
                stage = SimulationStage.DR_APPROACH_TURN,
                mode = NavigationMode.DEAD_RECKONING,
                inTunnel = true,
                algo = ActiveAlgorithm.OSM_SPLINE,
                tunnelProgress = (t - 11.5f) / 26.5f,
                alert = false,
                showPotholeHazard = false,
                potholePulsing = false
            )
            // Stage 7: Turn verified via topological spline (35.0s - 38.0s)
            t < 38.0f -> StoryState(
                stage = SimulationStage.DR_TURN_VERIFIED,
                mode = NavigationMode.DEAD_RECKONING,
                inTunnel = true,
                algo = ActiveAlgorithm.OSM_SPLINE,
                tunnelProgress = (t - 11.5f) / 26.5f,
                alert = false,
                showPotholeHazard = false,
                potholePulsing = false
            )
            // Stage 8: DR Ends -> GNSS Signal Restored & Seamless Fusion (38.0s - 43.0s)
            t < 43.0f -> StoryState(
                stage = SimulationStage.SEAMLESS_FUSION,
                mode = NavigationMode.RECOVERING,
                inTunnel = false,
                algo = ActiveAlgorithm.SOFT_RECOVERY,
                tunnelProgress = 1.0f,
                alert = false,
                showPotholeHazard = false,
                potholePulsing = false
            )
            // Stage 9: Normal Navigation Restored (43.0s - 48.0s)
            t < 48.0f -> StoryState(
                stage = SimulationStage.NORMAL_RESTORED,
                mode = NavigationMode.GNSS_LOCKED,
                inTunnel = false,
                algo = ActiveAlgorithm.CLOSED_LOOP_GNSS,
                tunnelProgress = 1.0f,
                alert = false,
                showPotholeHazard = false,
                potholePulsing = false
            )
            // Stage 10: Destination Reached
            else -> StoryState(
                stage = SimulationStage.DESTINATION_REACHED,
                mode = NavigationMode.COMPLETED,
                inTunnel = false,
                algo = ActiveAlgorithm.STANDBY,
                tunnelProgress = 1.0f,
                alert = false,
                showPotholeHazard = false,
                potholePulsing = false
            )
        }
    }

    private fun computeDynamics(t: Float, inTunnel: Boolean, vehicleType: VehicleType): Triple<Float, Float, Float> {
        // Curve of the road during the turn section (28.0s to 35.0s)
        val baseCurve = when {
            t in 28.0f..35.0f -> {
                val curveNorm = sin((t - 28.0f) / 7.0f * Math.PI.toFloat())
                curveNorm * 18.0f
            }
            else -> 0.0f
        }

        val carLateral = (0.02f * sin(t * 0.35f)).coerceIn(-0.06f, 0.06f)

        // Raw IMU drift: unconstrained sensor drifts sideways without map matching / NHC
        val ghostLateral = if (inTunnel) {
            val blackoutElapsed = (t - 11.5f).coerceAtLeast(0.0f)
            (0.038f * blackoutElapsed + 0.0028f * blackoutElapsed * blackoutElapsed).coerceAtMost(2.6f)
        } else {
            0.0f
        }

        val effectiveCarLateral = if (vehicleType == VehicleType.BIKE_2W && inTunnel) {
            carLateral * 0.85f
        } else {
            carLateral
        }

        return Triple(effectiveCarLateral, ghostLateral, baseCurve)
    }

    override fun onCleared() {
        super.onCleared()
        simulationJob?.cancel()
    }
}

data class StoryState(
    val stage: SimulationStage,
    val mode: NavigationMode,
    val inTunnel: Boolean,
    val algo: ActiveAlgorithm,
    val tunnelProgress: Float,
    val alert: Boolean,
    val showPotholeHazard: Boolean,
    val potholePulsing: Boolean
)
