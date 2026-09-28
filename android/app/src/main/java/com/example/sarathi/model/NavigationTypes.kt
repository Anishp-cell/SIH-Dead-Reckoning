package com.example.sarathi.model

enum class VehicleType(
    val title: String,
    val subtitle: String,
    val badge: String
) {
    CAR_4W(
        title = "4-Wheeler (Car / Truck)",
        subtitle = "Standard Non-Holonomic Constraints (v_lat = 0, v_up = 0)",
        badge = "4-WHEELER NHC"
    ),
    BIKE_2W(
        title = "2-Wheeler (Motorcycle / Scooter)",
        subtitle = "Banked Lean-Angle Compensation (v_contact-lat)",
        badge = "2-WHEELER LEAN"
    )
}

enum class NavigationMode(
    val label: String,
    val badgeText: String
) {
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
        label = "RUN COMPLETED (Audit Ready)",
        badgeText = "COMPLETED"
    )
}

enum class ActiveAlgorithm(
    val chipLabel: String,
    val detailText: String
) {
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
    val mode: NavigationMode = NavigationMode.GNSS_LOCKED,
    val vehicleType: VehicleType = VehicleType.CAR_4W,
    val speedKmh: Float = 48.2f,
    val speedMs: Float = 13.38f,
    val speedUncertaintyMs: Float = 0.22f,
    val distanceTraveledMeters: Float = 0.0f,
    val crossTrackMeters: Float = 0.42f,
    val recoveryStepMeters: Float = 0.006f,
    val activeAlgorithm: ActiveAlgorithm = ActiveAlgorithm.CLOSED_LOOP_GNSS,
    val inTunnel: Boolean = false,
    val tunnelProgress: Float = 0.0f,
    val roadCurveDegrees: Float = 0.0f,
    val carLateralOffsetRatio: Float = 0.0f,
    val ghostLateralOffsetRatio: Float = 0.0f,
    val covarianceHaloRadiusDp: Float = 16.0f,
    val potholePulsing: Boolean = false,
    val isRunning: Boolean = false,
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
