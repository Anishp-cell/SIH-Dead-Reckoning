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

enum class SimulationStage(
    val title: String,
    val subtitle: String,
    val instruction: String
) {
    READY(
        "SYSTEM READY",
        "NavIC Dual-Band L5/S Standby",
        "Ready to Navigate • Route Loaded"
    ),
    NORMAL_GNSS(
        "GNSS + INS FUSION",
        "Dual-Band NavIC Locked • 100% Signal Integrity",
        "In 650 m • Keep Left on Highway"
    ),
    GNSS_DEGRADING(
        "GNSS SIGNAL DEGRADED",
        "Multipath Detected • GNSS Trust ↓  INS Trust ↑",
        "In 420 m • Keep Left on Highway"
    ),
    GNSS_LOST_DR_ACTIVE(
        "DEAD RECKONING ACTIVE",
        "GNSS Lost • AI Velocity + 15-State ESKF Active",
        "In 250 m • Left Turn Ahead (DR Active)"
    ),
    DR_NAVIGATING(
        "DR MODE • ROAD LOCKED",
        "NHC Physics & Map Constraint Locked • Zero Lateral Drift",
        "In 180 m • Approach Left Curve (DR Active)"
    ),
    DISTURBANCE_POTHOLE(
        "DISTURBANCE DETECTED",
        "Pothole Shock (>45 m/s³) Gated • Trajectory Stable",
        "In 120 m • Pothole Shock Spike Gated"
    ),
    DR_APPROACH_TURN(
        "APPROACHING TURN (DR)",
        "Route-Aware DR • Gyroscope Heading Integration",
        "In 80 m • Turning Left into Curve"
    ),
    DR_TURN_VERIFIED(
        "TURN VERIFIED",
        "Topological Spline Snapped • Curve Negotiated in DR",
        "Curve Completed • Proceed on Route"
    ),
    GNSS_RESTORED(
        "GNSS SIGNAL RESTORED",
        "Comparing GNSS vs DR • Estimating Offset",
        "GNSS Reacquired • Merging Solution"
    ),
    SEAMLESS_FUSION(
        "SEAMLESS FUSION ACTIVE",
        "Continuous Kalman Glide Active (Δp = 0.006 m/step)",
        "Zero-Teleportation Recovery (0.006 m/step)"
    ),
    NORMAL_RESTORED(
        "GNSS + INS RESTORED",
        "All Sensors Healthy • Optimal Navigation State",
        "Proceed 100 m to Destination"
    ),
    DESTINATION_REACHED(
        "DESTINATION REACHED",
        "Route Completed • 100% Lane Integrity Preserved",
        "Arrived at Destination Point"
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
    val totalScenarioSeconds: Float = 48.0f,
    val routeProgress: Float = 0.0f, // 0.0f to 1.0f physical progress along route
    val stage: SimulationStage = SimulationStage.READY,
    val mode: NavigationMode = NavigationMode.READY,
    val vehicleType: VehicleType = VehicleType.CAR_4W,
    val speedKmh: Float = 0.0f,
    val speedMs: Float = 0.0f,
    val speedUncertaintyMs: Float = 0.15f,
    val distanceTraveledMeters: Float = 0.0f,
    val distanceRemainingMeters: Float = 1400.0f,
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
    val showPotholeHazard: Boolean = false,
    val showGpsLostAlert: Boolean = false,
    val gpsLostAlertMessage: String = "GPS Signal Lost • Dead Reckoning Autonomous Mode Active",
    val turnInstruction: String = "In 650 m • Keep Left on Highway",
    val gnssTrustPercent: Int = 100,
    val insTrustPercent: Int = 40,
    val aiSpeedEstimateKmh: Float = 62.0f,
    val speedVarianceSigma2: Float = 0.84f,
    val headingSigmaDeg: Float = 1.7f,
    val isMapMatchingLocked: Boolean = true,
    val isNhcActive: Boolean = true,
    val isDisturbanceGated: Boolean = false,
    val showDiagnosticsPanel: Boolean = false,
    val isStarted: Boolean = false,
    val isRunning: Boolean = false,
    val isPaused: Boolean = false,
    val isCompleted: Boolean = false
)

data class AuditResults(
    val distanceTraveledMeters: Float = 1420.0f,
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
