package com.example.sarathi.ui.nav

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sarathi.model.NavigationMode
import com.example.sarathi.model.VehicleState
import com.example.sarathi.theme.AccentCyan
import com.example.sarathi.theme.BgCard
import com.example.sarathi.theme.BgSurface
import com.example.sarathi.theme.BorderHighlight
import com.example.sarathi.theme.BorderSubtle
import com.example.sarathi.theme.StatusGnss
import com.example.sarathi.theme.StatusOutage
import com.example.sarathi.theme.StatusRecovery
import com.example.sarathi.theme.TextMuted
import com.example.sarathi.theme.TextPrimary
import com.example.sarathi.theme.TextSecondary
import java.util.Locale

@Composable
fun NavigationScreen(
    state: VehicleState,
    onRestart: () -> Unit,
    onTogglePause: () -> Unit,
    onOpenAudit: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(modifier = modifier.fillMaxSize().background(Color(0xFF080D1A))) {
        // 1. Vector Map Canvas (100% of background)
        VectorMapCanvas(state = state)

        // 2. Top Maneuver Banner & Dynamic Status Pill
        Column(
            modifier = Modifier
                .fillMaxWidth()
                .align(Alignment.TopCenter)
                .padding(horizontal = 16.dp, vertical = 12.dp),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            // Maneuver Banner
            ManeuverBanner(state = state)

            Spacer(modifier = Modifier.height(8.dp))

            // Hero Dynamic Status Pill
            DynamicStatusPill(state = state)
        }

        // 3. Floating Quick Control Buttons (Restart, Pause)
        Row(
            modifier = Modifier
                .align(Alignment.CenterStart)
                .padding(start = 16.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = onRestart,
                colors = ButtonDefaults.buttonColors(containerColor = BgSurface, contentColor = TextPrimary),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
            ) {
                Text(text = "RESTART", fontSize = 11.sp, fontWeight = FontWeight.Bold)
            }

            Button(
                onClick = onTogglePause,
                colors = ButtonDefaults.buttonColors(containerColor = BgSurface, contentColor = TextPrimary),
                shape = RoundedCornerShape(8.dp),
                modifier = Modifier.border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
            ) {
                Text(
                    text = if (state.isRunning) "PAUSE" else "RESUME",
                    fontSize = 11.sp,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        // 4. Bottom Floating Telemetry & Algorithm Ribbon Card
        BottomTelemetryCard(
            state = state,
            onOpenAudit = onOpenAudit,
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .padding(horizontal = 16.dp, vertical = 12.dp)
        )
    }
}

@Composable
private fun ManeuverBanner(state: VehicleState) {
    val maneuverText = when {
        state.timeSeconds < 15.0f -> "In 500m Enter Coventry Underpass"
        state.inTunnel -> "Inside Underpass • Dead Reckoning Active"
        state.mode == NavigationMode.RECOVERING -> "Exiting Underpass • Soft Recovery Damping"
        else -> "Route Complete • Destination Reached"
    }

    val maneuverSub = when {
        state.timeSeconds < 15.0f -> "A45 Dunchurch Highway (ISRO NavIC Dual-Band)"
        state.inTunnel -> "Sub-surface GNSS Blackout (${(state.timeSeconds - 15.0f).toInt()}s elapsed)"
        state.mode == NavigationMode.RECOVERING -> "Continuous Kalman Gain Damping (Zero Teleportation)"
        else -> "All Benchmarks Verified Passed"
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(BgSurface)
            .border(1.dp, BorderHighlight, RoundedCornerShape(12.dp))
            .padding(horizontal = 14.dp, vertical = 10.dp)
    ) {
        Column {
            Text(
                text = maneuverText,
                color = TextPrimary,
                fontSize = 14.sp,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = maneuverSub,
                color = TextSecondary,
                fontSize = 11.sp
            )
        }
    }
}

@Composable
private fun DynamicStatusPill(state: VehicleState) {
    val (dotColor, pillText) = when (state.mode) {
        NavigationMode.GNSS_LOCKED -> Pair(StatusGnss, "GNSS LOCKED (NavIC + GPS)")
        NavigationMode.DEAD_RECKONING -> Pair(
            StatusOutage,
            "AI DEAD RECKONING ACTIVE [Tunnel: ${(state.timeSeconds - 15.0f).toInt()}s]"
        )
        NavigationMode.RECOVERING -> Pair(StatusRecovery, "RECOVERING (Soft Damping Alpha=0.85)")
        NavigationMode.COMPLETED -> Pair(StatusGnss, "SESSION COMPLETED (Audit Ready)")
    }

    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(20.dp))
            .background(BgSurface)
            .border(1.dp, dotColor.copy(alpha = 0.5f), RoundedCornerShape(20.dp))
            .padding(horizontal = 12.dp, vertical = 6.dp)
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(8.dp)
                    .clip(CircleShape)
                    .background(dotColor)
            )
            Spacer(modifier = Modifier.width(8.dp))
            Text(
                text = pillText,
                color = TextPrimary,
                fontSize = 11.sp,
                fontWeight = FontWeight.SemiBold,
                fontFamily = FontFamily.Monospace
            )
        }
    }
}

@Composable
private fun BottomTelemetryCard(
    state: VehicleState,
    onOpenAudit: () -> Unit,
    modifier: Modifier = Modifier
) {
    Box(
        modifier = modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(16.dp))
            .background(BgSurface)
            .border(1.dp, BorderSubtle, RoundedCornerShape(16.dp))
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
                            color = AccentCyan,
                            fontSize = 38.sp,
                            fontWeight = FontWeight.Bold,
                            fontFamily = FontFamily.Monospace
                        )
                        Spacer(modifier = Modifier.width(4.dp))
                        Text(
                            text = "km/h",
                            color = TextSecondary,
                            fontSize = 14.sp,
                            modifier = Modifier.padding(bottom = 6.dp)
                        )
                    }
                    Text(
                        text = "AI SPEED: " + String.format(Locale.US, "%.1f", state.speedMs) + " m/s",
                        color = TextMuted,
                        fontSize = 10.sp,
                        fontFamily = FontFamily.Monospace
                    )
                }

                // Distance & Lane Offset Telemetry
                Column(horizontalAlignment = Alignment.End) {
                    Text(
                        text = "DIST: " + String.format(Locale.US, "%.0f", state.distanceTraveledMeters) + " m",
                        color = TextPrimary,
                        fontSize = 12.sp,
                        fontWeight = FontWeight.SemiBold,
                        fontFamily = FontFamily.Monospace
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "LANE OFFSET: " + String.format(Locale.US, "%.2f", state.crossTrackMeters) + " m",
                        color = TextSecondary,
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace
                    )
                    Spacer(modifier = Modifier.height(2.dp))
                    Text(
                        text = "VEHICLE: " + state.vehicleType.badge,
                        color = AccentCyan,
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            Spacer(modifier = Modifier.height(10.dp))

            // The Hero Live Algorithm Ribbon
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .clip(RoundedCornerShape(8.dp))
                    .background(BgCard)
                    .border(1.dp, BorderSubtle, RoundedCornerShape(8.dp))
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Text(
                        text = "ACTIVE ALGORITHM:",
                        color = TextMuted,
                        fontSize = 9.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = state.activeAlgorithm.chipLabel,
                        color = AccentCyan,
                        fontSize = 11.sp,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold
                    )
                }
            }

            // Post-Completion Button Prompt
            AnimatedVisibility(
                visible = state.isCompleted,
                enter = fadeIn(),
                exit = fadeOut()
            ) {
                Column {
                    Spacer(modifier = Modifier.height(10.dp))
                    Button(
                        onClick = onOpenAudit,
                        colors = ButtonDefaults.buttonColors(containerColor = StatusGnss, contentColor = Color(0xFF080D1A)),
                        shape = RoundedCornerShape(8.dp),
                        modifier = Modifier.fillMaxWidth()
                    ) {
                        Text(
                            text = "VIEW PERFORMANCE AUDIT SCORECARD",
                            fontWeight = FontWeight.Bold,
                            fontSize = 12.sp
                        )
                    }
                }
            }
        }
    }
}
