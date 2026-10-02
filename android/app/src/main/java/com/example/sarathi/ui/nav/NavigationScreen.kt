package com.example.sarathi.ui.nav

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxHeight
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
import androidx.compose.material3.Icon
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sarathi.model.ActiveAlgorithm
import com.example.sarathi.model.NavigationMode
import com.example.sarathi.model.VehicleState
import com.example.sarathi.theme.ChipBg
import com.example.sarathi.theme.ChipBorder
import com.example.sarathi.theme.DividerColor
import com.example.sarathi.theme.MapBg
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.theme.SheetBg
import com.example.sarathi.theme.StatusAmber
import com.example.sarathi.theme.StatusBlue
import com.example.sarathi.theme.StatusGreen
import com.example.sarathi.theme.StatusRed
import com.example.sarathi.theme.TextChip
import com.example.sarathi.theme.TextPrimary
import com.example.sarathi.theme.TextSecondary
import com.example.sarathi.theme.TextStat
import com.example.sarathi.theme.TextStatLabel

@Composable
fun NavigationScreen(
    state: VehicleState,
    onStartSimulation: () -> Unit,
    onStopSimulation: () -> Unit,
    onTogglePause: () -> Unit,
    onRestart: () -> Unit,
    onOpenAudit: () -> Unit
) {
    Box(modifier = Modifier.fillMaxSize()) {

        // ── Full-screen map canvas ───────────────────────────────────────────
        SarathiMapCanvas(
            state = state,
            modifier = Modifier.fillMaxSize()
        )

        // ── Top header bar: SAARTHI + status chips ───────────────────────────
        TopStatusBar(state = state)

        // ── Compass rose (top-right) ─────────────────────────────────────────
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = 116.dp, end = 16.dp)
                .size(40.dp)
                .clip(CircleShape)
                .background(ChipBg.copy(alpha = 0.9f)),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "N",
                color = TextPrimary,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold,
                textAlign = TextAlign.Center
            )
        }

        // ── GPS Lost toast ───────────────────────────────────────────────────
        AnimatedVisibility(
            visible = state.showGpsLostAlert,
            enter = fadeIn(),
            exit = fadeOut(),
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 130.dp)
        ) {
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(24.dp))
                    .background(Color(0xDD1C2333))
                    .padding(horizontal = 20.dp, vertical = 10.dp)
            ) {
                Text(
                    text = "GPS Signal Lost  •  Dead Reckoning Active",
                    color = StatusAmber,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.SemiBold
                )
            }
        }

        // ── Bottom sheet ─────────────────────────────────────────────────────
        Column(
            modifier = Modifier
                .align(Alignment.BottomCenter)
                .fillMaxWidth()
        ) {
            BottomSheet(
                state = state,
                onStartSimulation = onStartSimulation,
                onStopSimulation = onStopSimulation,
                onTogglePause = onTogglePause,
                onRestart = onRestart,
                onOpenAudit = onOpenAudit
            )
        }
    }
}

// ── Top status bar ─────────────────────────────────────────────────────────────
@Composable
private fun TopStatusBar(state: VehicleState) {
    val gnssAvailable = !state.inTunnel && state.isRunning
    val drActive      = state.inTunnel || state.mode == NavigationMode.DEAD_RECKONING

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(MapBg.copy(alpha = 0.92f))
            .padding(horizontal = 16.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        // Logo + name
        Text(
            text = "A  SAARTHI",
            color = TextPrimary,
            fontSize = 16.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.sp,
            modifier = Modifier.weight(1f)
        )

        // Status chips (GNSS, NETWORK, DR MODE)
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
            StatusChip(
                label = "GNSS",
                status = if (gnssAvailable) "ACTIVE" else "UNAVAIL.",
                dotColor = if (gnssAvailable) StatusGreen else StatusRed
            )
            StatusChip(
                label = "NETWORK",
                status = if (state.isRunning) "ONLINE" else "OFFLINE",
                dotColor = if (state.isRunning) StatusGreen else StatusRed
            )
            StatusChip(
                label = "DR MODE",
                status = if (drActive) "ACTIVE" else "STANDBY",
                dotColor = if (drActive) StatusGreen else StatusAmber
            )
        }
    }
}

@Composable
private fun StatusChip(label: String, status: String, dotColor: Color) {
    Column(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(ChipBg)
            .padding(horizontal = 10.dp, vertical = 6.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        Text(text = label, color = TextSecondary, fontSize = 8.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 0.5.sp)
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(dotColor)
        )
        Text(text = status, color = dotColor, fontSize = 8.sp, fontWeight = FontWeight.Bold)
    }
}

// ── Bottom stats sheet ─────────────────────────────────────────────────────────
@Composable
private fun BottomSheet(
    state: VehicleState,
    onStartSimulation: () -> Unit,
    onStopSimulation: () -> Unit,
    onTogglePause: () -> Unit,
    onRestart: () -> Unit,
    onOpenAudit: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(topStart = 24.dp, topEnd = 24.dp))
            .background(SheetBg)
            .padding(horizontal = 24.dp, vertical = 20.dp)
    ) {

        // Algorithm badge (active state indicator)
        if (state.isRunning) {
            AlgorithmBadge(state = state)
            Spacer(modifier = Modifier.height(16.dp))
        }

        // Three stat columns: Speed | Heading | Confidence
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceAround,
            verticalAlignment = Alignment.CenterVertically
        ) {
            StatColumn(
                icon = "speed",
                value = "%.0f".format(state.speedKmh),
                unit = "km/h"
            )

            // Divider
            Box(
                modifier = Modifier
                    .width(1.dp)
                    .height(48.dp)
                    .background(DividerColor)
            )

            StatColumn(
                icon = "heading",
                value = "%.1f".format(128.4f + (state.roadCurveDegrees * 0.3f)),
                unit = "Heading"
            )

            Box(
                modifier = Modifier
                    .width(1.dp)
                    .height(48.dp)
                    .background(DividerColor)
            )

            StatColumn(
                icon = "confidence",
                value = "%.0f%%".format(confidenceFromState(state)),
                unit = "Confidence"
            )
        }

        Spacer(modifier = Modifier.height(20.dp))

        // Control buttons
        when {
            !state.isStarted -> {
                Button(
                    onClick = onStartSimulation,
                    modifier = Modifier.fillMaxWidth().height(52.dp),
                    shape = RoundedCornerShape(14.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = PuckBlue)
                ) {
                    Text("START SIMULATION", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 14.sp, letterSpacing = 1.sp)
                }
            }
            state.isCompleted -> {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(
                        onClick = onRestart,
                        modifier = Modifier.weight(1f).height(52.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = PuckBlue)
                    ) {
                        Text("RESTART", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                    Button(
                        onClick = onOpenAudit,
                        modifier = Modifier.weight(1f).height(52.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = StatusGreen)
                    ) {
                        Text("VIEW AUDIT", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                }
            }
            else -> {
                Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    Button(
                        onClick = onTogglePause,
                        modifier = Modifier.weight(1f).height(52.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(
                            containerColor = if (state.isRunning) StatusAmber else StatusGreen
                        )
                    ) {
                        Text(
                            if (state.isRunning) "PAUSE" else "RESUME",
                            color = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize = 13.sp
                        )
                    }
                    Button(
                        onClick = onStopSimulation,
                        modifier = Modifier.weight(1f).height(52.dp),
                        shape = RoundedCornerShape(14.dp),
                        colors = ButtonDefaults.buttonColors(containerColor = StatusRed.copy(alpha = 0.85f))
                    ) {
                        Text("STOP", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun StatColumn(icon: String, value: String, unit: String) {
    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        // Icon placeholder (text-based since no vector resources referenced)
        Text(
            text = when (icon) {
                "speed"      -> "~"
                "heading"    -> "o"
                "confidence" -> "||"
                else         -> "+"
            },
            color = TextStatLabel,
            fontSize = 18.sp,
            fontWeight = FontWeight.Bold
        )
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = value,
            color = TextStat,
            fontSize = 28.sp,
            fontWeight = FontWeight.Bold
        )
        Text(
            text = unit,
            color = TextStatLabel,
            fontSize = 11.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

@Composable
private fun AlgorithmBadge(state: VehicleState) {
    val (label, color) = when (state.mode) {
        NavigationMode.GNSS_LOCKED    -> "GNSS LOCKED  •  NavIC Dual-Band" to StatusGreen
        NavigationMode.DEAD_RECKONING -> "DEAD RECKONING  •  ${state.activeAlgorithm.chipLabel}" to StatusAmber
        NavigationMode.RECOVERING     -> "SOFT RECOVERY  •  Kalman Glide" to StatusBlue
        NavigationMode.COMPLETED      -> "ROUTE COMPLETE" to StatusGreen
        else                          -> "READY  •  NavIC Standby" to TextSecondary
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(10.dp))
            .background(color.copy(alpha = 0.10f))
            .padding(horizontal = 14.dp, vertical = 8.dp)
    ) {
        Text(
            text = label,
            color = color,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 0.5.sp
        )
    }
}

private fun confidenceFromState(state: VehicleState): Float = when {
    !state.isRunning               -> 0f
    state.mode == NavigationMode.DEAD_RECKONING -> (91f - state.tunnelProgress * 18f).coerceAtLeast(72f)
    state.mode == NavigationMode.RECOVERING     -> 87f
    else                                        -> 96f
}
