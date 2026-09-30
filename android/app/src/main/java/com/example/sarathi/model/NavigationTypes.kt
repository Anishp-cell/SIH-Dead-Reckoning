package com.example.sarathi.model

enum class VehicleType(
    val title: String,
    val subtitle: String,
    val badge: String
) {
    CAR_4W(
        title = "4-Wheeler (Car / Sedan)",
        subtitle = "Standard Non-Holonomic Constraints (v_lat = 0, v_up = 0)",
        badge = "4W CAR"
    ),
    BIKE_2W(
        title = "2-Wheeler (Scooter / Motorcycle)",
        subtitle = "Banked Lean-Angle Compensation (v_contact-lat)",
        badge = "2W SCOOTER"
    )
}

enum class NavigationMode(
    val label: String,
    val badgeText: String
) {
    READY(
        label = "GPS READY (NavIC Dual-Band L5/S)",
        badgeText = "STANDBY"
    ),
    GNSS_LOCKED(
        label = "GNSS LOCKED (NavIC + GPS)",
        badgeText = "HEALTHY"
    ),
    DEAD_RECKONING(
        label = "AI DEAD RECKONING ACTIVE",
        badgeText = "OUTAGE ACTIVE"
    ),
    RECOVERING(
        label = "RECOVERING (Soft Damping Alpha=0.85)",
        badgeText = "SOFT RECOVERY"
    ),
    COMPLETED(
        label = "DESTINATION REACHED",
        badgeText = "COMPLETED"
    )
}

enum class ActiveAlgorithm(
    val chipLabel: String,
    val detailText: String
) {
    STANDBY(
        chipLabel = "NavIC Standby",
        detailText = "System calibrated and ready to navigate"
    ),
    CLOSED_LOOP_GNSS(
        chipLabel = "NavIC Dual-Band L5/S",
        detailText = "15-State ESKF with 3D Pseudorange Innovation Gating"
    ),
    AI_SPEED(
        chipLabel = "AI Speedometer (204 KB)",
        detailText = "Heteroscedastic Kinematic Velocity Prediction"
    ),
    NHC_PHYSICS(
        chipLabel = "NHC Physics: Locked",
        detailText = "Non-Holonomic Body-Frame Zero Lateral Constraints"
    ),
    OSM_SPLINE(
        chipLabel = "OSM Spline: Snapped",
        detailText = "KD-Tree Road Topology & 1D Spline Odometry"
    ),
    POTHOLE_GATING(
        chipLabel = "Shock Gated (>45 m/s3)",
        detailText = "Dynamic Pothole Jerk Suppression Active"
    ),
    ZUPT_STANDSTILL(
        chipLabel = "ZUPT: Standstill Anchor",
        detailText = "Zero Velocity Update Anchoring Creep"
    ),
    SOFT_RECOVERY(
        chipLabel = "Zero-Teleportation Glide",
        detailText = "Continuous Kalman Gain Damping (Delta p = 0.006m)"
    )
}

data class VehicleState(
    val timeSeconds: Float = 0.0f,
    val totalScenarioSeconds: Float = 55.0f,
    val mode: NavigationMode = NavigationMode.READY,
    val vehicleType: VehicleType = VehicleType.CAR_4W,
    val speedKmh: Float = 0.0f,
    val speedMs: Float = 0.0f,
    val speedUncertaintyMs: Float = 0.15f,
    val distanceTraveledMeters: Float = 0.0f,
    val crossTrackMeters: Float = 0.22f,
    val recoveryStepMeters: Float = 0.006f,
    val activeAlgorithm: ActiveAlgorithm = ActiveAlgorithm.STANDBY,
    val inTunnel: Boolean = false,
    val tunnelProgress: Float = 0.0f,
    val roadCurveDegrees: Float = 0.0f,
    val carLateralOffsetRatio: Float = 0.0f,
    val ghostLateralOffsetRatio: Float = 0.0f,
    val covarianceHaloRadiusDp: Float = 14.0f,
    val potholePulsing: Boolean = false,
    val showGpsLostAlert: Boolean = false,
    val gpsLostAlertMessage: String = "GPS Signal Lost • Dead Reckoning Autonomous Mode Active",
    val isStarted: Boolean = false,
    val isRunning: Boolean = false,
    val isPaused: Boolean = false,
    val isCompleted: Boolean = false
)

data class AuditResults(
    val distanceTraveledMeters: Float = 113.54f,
    val driftPercentAchieved: Float = 3.50f,
    val driftPercentTarget: Float = 10.00f,
    val driftPassed: Boolean = true,
    val crossTrackAchievedMeters: Float = 0.68f,
    val crossTrackTargetMeters: Float = 3.50f,
    val crossTrackPassed: Boolean = true,
    val recoveryStepAchievedMeters: Float = 0.006f,
    val recoveryStepTargetMeters: Float = 3.50f,
    val recoveryStepPassed: Boolean = true,
    val filterFrequencyHz: Float = 261.6f,
    val aiInferenceLatencyUs: Float = 35.8f,
    val aiModelSizeBytes: Long = 208896L,
    val navicSatellitesLocked: Int = 7,
    val potholesDampedCount: Int = 3
)
