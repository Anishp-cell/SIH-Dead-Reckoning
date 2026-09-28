package com.example.sarathi.viewmodel

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.example.sarathi.model.ActiveAlgorithm
import com.example.sarathi.model.AuditResults
import com.example.sarathi.model.NavigationMode
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
    HOME,
    NAVIGATION,
    AUDIT
}

class SarathiViewModel : ViewModel() {

    private val _currentTab = MutableStateFlow(AppTab.HOME)
    val currentTab: StateFlow<AppTab> = _currentTab.asStateFlow()

    private val _vehicleState = MutableStateFlow(VehicleState())
    val vehicleState: StateFlow<VehicleState> = _vehicleState.asStateFlow()

    private val _auditResults = MutableStateFlow(AuditResults())
    val auditResults: StateFlow<AuditResults> = _auditResults.asStateFlow()

    private var simulationJob: Job? = null

    fun selectTab(tab: AppTab) {
        _currentTab.value = tab
    }

    fun setVehicleType(type: VehicleType) {
        _vehicleState.update { it.copy(vehicleType = type) }
    }

    fun startNavigation() {
        _currentTab.value = AppTab.NAVIGATION
        restartSimulation()
    }

    fun restartSimulation() {
        simulationJob?.cancel()
        _vehicleState.update {
            VehicleState(
                vehicleType = it.vehicleType,
                isRunning = true,
                isCompleted = false
            )
        }
        startSimulationTicker()
    }

    fun togglePauseResume() {
        val currentlyRunning = _vehicleState.value.isRunning
        if (currentlyRunning) {
            simulationJob?.cancel()
            _vehicleState.update { it.copy(isRunning = false) }
        } else {
            _vehicleState.update { it.copy(isRunning = true) }
            startSimulationTicker()
        }
    }

    private fun startSimulationTicker() {
        simulationJob = viewModelScope.launch {
            val tickIntervalMs = 50L
            val tickDeltaSec = tickIntervalMs / 1000.0f

            while (_vehicleState.value.timeSeconds < _vehicleState.value.totalScenarioSeconds) {
                delay(tickIntervalMs)

                _vehicleState.update { state ->
                    val newTime = state.timeSeconds + tickDeltaSec
                    val (mode, inTunnel, algo, tunnelProgress) = computeStage(newTime)
                    val (carOffset, ghostOffset, curveDeg) = computeDynamics(newTime, inTunnel, state.vehicleType)
                    val haloRadius = if (inTunnel) {
                        (16.0f + 10.0f * sin((newTime - 15.0f) * 0.35f)).coerceIn(16.0f, 26.0f)
                    } else {
                        16.0f
                    }
                    val isPothole = inTunnel && (newTime in 31.0f..33.5f)

                    val currentSpeedKmh = if (inTunnel) {
                        47.5f + 1.2f * sin(newTime * 0.8f)
                    } else {
                        48.5f + 0.8f * sin(newTime * 0.5f)
                    }

                    val currentSpeedMs = currentSpeedKmh / 3.6f
                    val newDistance = state.distanceTraveledMeters + (currentSpeedMs * tickDeltaSec)

                    state.copy(
                        timeSeconds = newTime,
                        mode = mode,
                        inTunnel = inTunnel,
                        tunnelProgress = tunnelProgress,
                        activeAlgorithm = if (isPothole) ActiveAlgorithm.POTHOLE_GATING else algo,
                        potholePulsing = isPothole,
                        speedKmh = currentSpeedKmh,
                        speedMs = currentSpeedMs,
                        speedUncertaintyMs = if (inTunnel) 0.38f else 0.18f,
                        distanceTraveledMeters = newDistance,
                        crossTrackMeters = if (inTunnel) (0.35f + 0.25f * sin(newTime * 0.25f)) else 0.22f,
                        recoveryStepMeters = if (mode == NavigationMode.RECOVERING) 0.006f else 0.0f,
                        roadCurveDegrees = curveDeg,
                        carLateralOffsetRatio = carOffset,
                        ghostLateralOffsetRatio = ghostOffset,
                        covarianceHaloRadiusDp = haloRadius,
                        isCompleted = newTime >= 50.0f
                    )
                }
            }

            _vehicleState.update {
                it.copy(
                    isRunning = false,
                    isCompleted = true,
                    mode = NavigationMode.COMPLETED
                )
            }
        }
    }

    private fun computeStage(t: Float): Quadruple<NavigationMode, Boolean, ActiveAlgorithm, Float> {
        return when {
            t < 15.0f -> {
                Quadruple(
                    NavigationMode.GNSS_LOCKED,
                    false,
                    ActiveAlgorithm.CLOSED_LOOP_GNSS,
                    0.0f
                )
            }
            t in 15.0f..45.0f -> {
                val progress = (t - 15.0f) / 30.0f
                val algo = when {
                    t < 25.0f -> ActiveAlgorithm.AI_SPEED
                    t < 33.0f -> ActiveAlgorithm.NHC_PHYSICS
                    t < 37.0f -> ActiveAlgorithm.POTHOLE_GATING
                    else -> ActiveAlgorithm.OSM_SPLINE
                }
                Quadruple(
                    NavigationMode.DEAD_RECKONING,
                    true,
                    algo,
                    progress
                )
            }
            t in 45.0f..48.0f -> {
                Quadruple(
                    NavigationMode.RECOVERING,
                    false,
                    ActiveAlgorithm.SOFT_RECOVERY,
                    1.0f
                )
            }
            else -> {
                Quadruple(
                    NavigationMode.GNSS_LOCKED,
                    false,
                    ActiveAlgorithm.CLOSED_LOOP_GNSS,
                    1.0f
                )
            }
        }
    }

    private fun computeDynamics(t: Float, inTunnel: Boolean, vehicleType: VehicleType): Triple<Float, Float, Float> {
        val baseCurve = when {
            t in 22.0f..38.0f -> {
                val curveNorm = sin((t - 22.0f) / 16.0f * Math.PI.toFloat())
                curveNorm * 18.0f
            }
            else -> 0.0f
        }

        // Cyan vehicle is locked to lane center with tiny realistic steering noise (<0.10f)
        val carLateral = (0.05f * sin(t * 0.4f)).coerceIn(-0.15f, 0.15f)

        // Red Ghost (Raw uncorrected IMU drift) drifts monotonically off to the right barrier during blackout
        val ghostLateral = if (inTunnel) {
            val blackoutElapsed = t - 15.0f
            // Exponential quadratic drift curve representing uncorrected double integration
            (0.02f * blackoutElapsed + 0.0028f * blackoutElapsed * blackoutElapsed).coerceAtMost(2.6f)
        } else if (t < 15.0f) {
            carLateral
        } else {
            0.0f // Hidden post recovery
        }

        // 2-Wheeler Lean Compensation adjusts slight banking bias
        val effectiveCarLateral = if (vehicleType == VehicleType.BIKE_2W && inTunnel) {
            carLateral * 0.85f // Extra stability from lean compensation
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

data class Quadruple<A, B, C, D>(val first: A, val second: B, val third: C, val fourth: D)
