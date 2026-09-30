package com.example.sarathi.ui.nav

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.slideOutVertically
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sarathi.model.NavigationMode
import com.example.sarathi.model.VehicleState
import com.example.sarathi.theme.BorderLight
import com.example.sarathi.theme.NavRouteBlue
import com.example.sarathi.theme.StatusGnssBgLight
import com.example.sarathi.theme.StatusGnssGreen
import com.example.sarathi.theme.StatusOutageAmber
import com.example.sarathi.theme.StatusOutageBgLight
import com.example.sarathi.theme.StatusRecoveryBgLight
import com.example.sarathi.theme.StatusRecoveryBlue
import com.example.sarathi.theme.SurfaceCardLight
import com.example.sarathi.theme.SurfaceCardSubtle
import com.example.sarathi.theme.SurfaceManeuverGreen
import com.example.sarathi.theme.SurfaceManeuverTunnel
import com.example.sarathi.theme.TextMutedDark
import com.example.sarathi.theme.TextOnGreen
import com.example.sarathi.theme.TextPrimaryDark
import com.example.sarathi.theme.TextSecondaryDark
import java.util.Locale

@Composable
fun NavigationScreen(
    state: VehicleState,
    onStartSimulation: () -> Unit,
    onStopSimulation: () -> Unit,
    onTogglePause: () -> Unit,
    onRestart: () -> Unit,
    onOpenAudit: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(modifier = modifier.fillMaxSize()) {
        // 1. Vector Road Canvas (100% of background)
        VectorMapCanvas(state = state)

        // 2. Top Maneuver Banner & Dynamic Status Pill
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .padding(horizontal = 14.dp, vertical = 10.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Maneuver Banner (Google Maps Emerald Green or Dark Tunnel)
            ManeuverBanner(state = state)

            Spacer(modifier = Modifier.height(6.dp))

            // Dynamic Status Pill
            DynamicStatusPill(state = state)

            // Non-Intrusive Smooth GPS Lost Alert
            AnimatedVisibility(
                visible = state.showGpsLostAlert,
                enter = slideInVertically(initialOffsetY = { -it }) + fadeIn(),
                exit = slideOutVertically(targetOffsetY = { -it }) + fadeOut()
            ) {
                GpsLostToast(message = state.gpsLostAlertMessage)
            }
        }

        // 3. Bottom Floating Speedometer, Control Buttons & Telemetry Card
        BottomTelemetryCard(
            state = state,
            onStartSimulation = onStartSimulation,
            onStopSimulation = onStopSimulation,
            onTogglePause = onTogglePause,
            onRestart = onRestart,
            onOpenAudit = onOpenAudit,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(horizontal = 14.dp, vertical = 10.dp)
        )
    }
}

@Composable
private fun ManeuverBanner(state: VehicleState) {
    val bannerBg = if (state.inTunnel) SurfaceManeuverTunnel else SurfaceManeuverGreen

    val maneuverTitle = when {
        !state.isStarted -> "Ready to Navigate • Tap Start Below"
        state.timeSeconds < 15.0f -> "In 500m Enter Coventry Underpass"
        state.inTunnel -> "Inside Underpass • Dead Reckoning Active"
        state.mode == NavigationMode.RECOVERING -> "Exiting Underpass • Soft Recovery Damping"
        else -> "Route Complete • Destination Reached"
    }

    val maneuverSub = when {
        !state.isStarted -> "A45 Dunchurch Highway (ISRO NavIC Dual-Band)"
        state.timeSeconds < 15.0f -> "Straight on Highway • Satellite Lock Active"
        state.inTunnel -> "Sub-surface Outage (${(state.timeSeconds - 15.0f).toInt()}s blackout elapsed)"
        state.mode == NavigationMode.RECOVERING -> "Continuous Kalman Gain Damping (Zero Teleportation)"
        else -> "All Benchmarks Verified Passed"
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .shadow(4.dp, RoundedCornerShape(12.dp))
            .clip(RoundedCornerShape(12.dp))
            .background(bannerBg)
            .padding(horizontal = 16.dp, vertical = 10.dp)
    ) {
        Column {
            Text(
                text = maneuverTitle,
                color = TextOnGreen,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = maneuverSub,
                color = TextOnGreen.copy(alpha = 0.85f),
                fontSize = 11.sp
            )
        }
    }
}

@Composable
private fun DynamicStatusPill(state: VehicleState) {
    val (bgColor, dotColor, textColor, pillText) = when (state.mode) {
        NavigationMode.READY -> Quadruple(
            StatusGnssBgLight,
            StatusGnssGreen,
            StatusGnssGreen,
            "GPS READY (NavIC Dual-Band L5/S)"
        )
        NavigationMode.GNSS_LOCKED -> Quadruple(
            StatusGnssBgLight,
            StatusGnssGreen,
            StatusGnssGreen,
            "GNSS LOCKED (NavIC + GPS)"
        )
        NavigationMode.DEAD_RECKONING -> Quadruple(
            StatusOutageBgLight,
            StatusOutageAmber,
            StatusOutageAmber,
            "AI DEAD RECKONING ACTIVE [Tunnel: ${(state.timeSeconds - 15.0f).toInt()}s]"
        )
        NavigationMode.RECOVERING -> Quadruple(
            StatusRecoveryBgLight,
            StatusRecoveryBlue,
            StatusRecoveryBlue,
            "RECOVERING (Soft Damping Alpha=0.85)"
        )
        NavigationMode.COMPLETED -> Quadruple(
            StatusGnssBgLight,
            StatusGnssGreen,
            StatusGnssGreen,
            "DESTINATION REACHED (Audit Ready)"
        )
    }

    Box(
        modifier = Modifier
            .shadow(2.dp, RoundedCornerShape(20.dp))
            .clip(RoundedCornerShape(20.dp))
            .background(bgColor)
            .border(1.dp, dotColor.copy(alpha = 0.35f), RoundedCornerShape(20.dp))
            .padding(horizontal = 12.dp, vertical = 5.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(7.dp)
                    .clip(CircleShape)
                    .background(dotColor)
            )
            Spacer(modifier = Modifier.width(7.dp))
            Text(
                text = pillText,
                color = textColor,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                fontFamily = FontFamily.Monospace
            )
        }
    }
}

@Composable
private fun GpsLostToast(message: String) {
    Box(
        modifier = Modifier
            .padding(top = 6.dp)
            .shadow(4.dp, RoundedCornerShape(8.dp))
            .clip(RoundedCornerShape(8.dp))
            .background(Color(0xFFFEF3C7))
            .border(1.dp, Color(0xFFF59E0B), RoundedCornerShape(8.dp))
            .padding(horizontal = 12.dp, vertical = 6.dp)
    ) {
        Text(
            text = message,
            color = Color(0xFF92400E),
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace
        )
    }
}

@Composable
private fun BottomTelemetryCard(
    state: VehicleState,
    onStartSimulation: () -> Unit,
    onStopSimulation: () -> Unit,
    onTogglePause: () -> Unit,
    onRestart: () -> Unit,
    onOpenAudit: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .shadow(6.dp, RoundedCornerShape(16.dp))
            .clip(RoundedCornerShape(16.dp))
            .background(SurfaceCardLight)
            .border(1.dp, BorderLight, RoundedCornerShape(16.dp))
            .padding(14.dp)
    ) {
        Column {
            // Speedometer & Primary Metrics Row
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically
            ) {
                // Large Digital Speedometer
                Column {
                    Row(verticalAlignment = Alignment.Bottom) {
                        Text(
                            text = String.format(Locale.US, "%.1f", state.speedKmh),
                            color = NavRouteBlue,
                            fontSize = 36.sp,
                            fontWeight = FontWeight.Black,
                            fontFamily = FontFamily.Monospace
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "km/h",
                            color = TextSecondaryDark,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.SemiBold,
                            modifier = Modifier.padding(bottom = 5.dp)
                        )
                    }
                    Text(
                        text = "AI SPEED: " + String.format(Locale.US, "%.1f", state.speedMs) + " m/s",
                        color = TextMutedDark,
                        fontSize = 10.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Medium
                    )
                }

                // Distance & Lane Offset Telemetry
                Column(horizontalAlignment = Alignment.End) {
                    Text(
                        text = "DIST: " + String.format(Locale.US, "%.0f", state.distanceTraveledMeters) + " m",
                        color = TextPrimaryDark,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "LANE OFFSET: " + String.format(Locale.US, "%.2f", state.crossTrackMeters) + " m",
                        color = TextSecondaryDark,
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "VEHICLE: " + state.vehicleType.badge,
                        color = NavRouteBlue,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Contextual Algorithm Status Ribbon
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(SurfaceCardSubtle)
                    .border(1.dp, BorderLight, RoundedCornerShape(8.dp))
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "ACTIVE ALGORITHM:",
                        color = TextMutedDark,
                        fontSize = 9.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = state.activeAlgorithm.chipLabel,
                        color = NavRouteBlue,
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            // Interactive Simulation Controls (Start, Pause, Stop, Restart)
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.spacedBy(8.dp)
            ) {
                if (!state.isStarted || !state.isRunning) {
                    Button(
                        onClick = if (!state.isStarted) onStartSimulation else onTogglePause,
                        colors = ButtonDefaults.buttonColors(containerColor = NavRouteBlue, contentColor = Color.White),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.weight(1.0f)
                    ) {
                        Text(
                            text = if (!state.isStarted) "START" else "RESUME",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                } else {
                    Button(
                        onClick = onTogglePause,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFE2E8F0), contentColor = TextPrimaryDark),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.weight(1.0f)
                    ) {
                        Text(
                            text = "PAUSE",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }

                if (state.isStarted) {
                    Button(
                        onClick = onStopSimulation,
                        colors = ButtonDefaults.buttonColors(containerColor = Color(0xFFFEE2E2), contentColor = Color(0xFFDC2626)),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.weight(1.0f)
                    ) {
                        Text(
                            text = "STOP",
                            fontSize = 12.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                    }

                    Button(
                        onClick = onRestart,
                        colors = ButtonDefaults.buttonColors(containerColor = SurfaceCardSubtle, contentColor = TextSecondaryDark),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.border(1.dp, BorderLight, RoundedCornerShape(8.dp))
                    ) {
                        Text(
                            text = "RESTART",
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }

            // Post-Completion Button Prompt
            AnimatedVisibility(
                visible = state.isCompleted,
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                Column {
                    Spacer(modifier = Modifier.height(8.dp))
                    Button(
                        onClick = onOpenAudit,
                        colors = ButtonDefaults.buttonColors(containerColor = StatusGnssGreen, contentColor = Color.White),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = "VIEW PERFORMANCE AUDIT REPORT",
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp,
                            fontFamily = FontFamily.Monospace
                        )
                    }
                }
            }
        }
    }
}

data class Quadruple<A, B, C, D>(val first: A, val second: B, val third: C, val fourth: D)
