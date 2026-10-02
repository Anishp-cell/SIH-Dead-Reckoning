package com.example.sarathi.ui.nav

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.example.sarathi.model.SimulationStage
import com.example.sarathi.model.VehicleState
import com.example.sarathi.theme.AlertBg
import com.example.sarathi.theme.AppBarBg
import com.example.sarathi.theme.AppBarText
import com.example.sarathi.theme.ChipBarBg
import com.example.sarathi.theme.ChipBorder
import com.example.sarathi.theme.ChipLabelText
import com.example.sarathi.theme.CompassBg
import com.example.sarathi.theme.HandleColor
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.theme.SheetBg
import com.example.sarathi.theme.SheetDivider
import com.example.sarathi.theme.SheetLabel
import com.example.sarathi.theme.SheetText
import com.example.sarathi.theme.StatusAmber
import com.example.sarathi.theme.StatusBlue
import com.example.sarathi.theme.StatusGreen
import com.example.sarathi.theme.StatusRed

@Composable
fun NavigationScreen(
    state: VehicleState,
    onStartSimulation: () -> Unit,
    onStopSimulation: () -> Unit,
    onTogglePause: () -> Unit,
    onRestart: () -> Unit,
    onOpenAudit: () -> Unit,
    onToggleDiagnostics: () -> Unit = {},
    onBack: () -> Unit = {},
    onLogoClick: () -> Unit = {}
) {
    Box(modifier = Modifier.fillMaxSize()) {

        // 1. Full-screen light map canvas with dynamic continuous vehicle forward motion
        SarathiMapCanvas(
            state    = state,
            modifier = Modifier.fillMaxSize()
        )

        // 2. Top section: App Bar + Floating Status Card + Turn Guidance
        Column(
            modifier = Modifier
                .align(Alignment.TopCenter)
                .fillMaxWidth()
        ) {
            TopAppBar(
                onBack = onBack,
                onLogoClick = onLogoClick
            )
            FloatingStatusCard(state = state)

            // Turn-by-Turn Instruction Banner
            if (state.isRunning || state.isCompleted) {
                TurnInstructionBanner(state = state)
            }
        }

        // 3. Floating North Compass (top-right)
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = if (state.isRunning) 210.dp else 165.dp, end = 16.dp)
                .size(44.dp)
                .shadow(8.dp, CircleShape)
                .clip(CircleShape)
                .background(CompassBg),
            contentAlignment = Alignment.Center
        ) {
            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                Text("▲", color = Color.White, fontSize = 11.sp, lineHeight = 12.sp)
                Text("N", color = Color.White, fontSize = 10.sp, fontWeight = FontWeight.Bold, lineHeight = 12.sp)
            }
        }

        // 4. Floating Clutter-Free Telemetry Button (opens dedicated math & algorithm modal)
        Box(
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(top = if (state.isRunning) 264.dp else 218.dp, end = 16.dp)
                .shadow(8.dp, RoundedCornerShape(20.dp))
                .clip(RoundedCornerShape(20.dp))
                .background(Color(0xFF0F172A))
                .border(1.dp, Color(0xFF38BDF8), RoundedCornerShape(20.dp))
                .clickable { onToggleDiagnostics() }
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Row(
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.spacedBy(5.dp)
            ) {
                Text("⚡", fontSize = 11.sp)
                Text(
                    text = "TELEMETRY",
                    color = Color.White,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    letterSpacing = 0.5.sp
                )
            }
        }

        // 5. GPS Lost Non-Intrusive Alert Toast
        AnimatedVisibility(
            visible  = state.showGpsLostAlert,
            enter    = fadeIn(),
            exit     = fadeOut(),
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 216.dp)
        ) {
            Box(
                modifier = Modifier
                    .shadow(12.dp, RoundedCornerShape(24.dp))
                    .clip(RoundedCornerShape(24.dp))
                    .background(AlertBg.copy(alpha = 0.95f))
                    .border(1.dp, StatusRed.copy(alpha = 0.5f), RoundedCornerShape(24.dp))
                    .padding(horizontal = 18.dp, vertical = 8.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Box(modifier = Modifier.size(8.dp).clip(CircleShape).background(StatusRed))
                    Text(
                        "GPS Lost • Dead Reckoning Active",
                        color = Color.White,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.SemiBold
                    )
                }
            }
        }

        // 6. Pothole Disturbance Alert Toast
        AnimatedVisibility(
            visible  = state.potholePulsing,
            enter    = fadeIn(),
            exit     = fadeOut(),
            modifier = Modifier
                .align(Alignment.TopCenter)
                .padding(top = 216.dp)
        ) {
            Box(
                modifier = Modifier
                    .shadow(12.dp, RoundedCornerShape(24.dp))
                    .clip(RoundedCornerShape(24.dp))
                    .background(Color(0xFF0F172A).copy(alpha = 0.95f))
                    .border(1.dp, StatusAmber, RoundedCornerShape(24.dp))
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Text("🛡️", fontSize = 12.sp)
                    Text(
                        "SHOCK GATED (>45 m/s³) • Trajectory Protected",
                        color = StatusAmber,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                }
            }
        }

        // 7. KEY / LEGEND OF THE DOTS (Right Side Bottom Corner Above the Score Card)
        if (state.inTunnel && state.isRunning) {
            Box(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(bottom = 195.dp, end = 16.dp)
                    .shadow(6.dp, RoundedCornerShape(10.dp))
                    .clip(RoundedCornerShape(10.dp))
                    .background(Color(0xDD0F172A))
                    .border(1.dp, Color(0xFF334155), RoundedCornerShape(10.dp))
                    .padding(horizontal = 10.dp, vertical = 6.dp)
            ) {
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(7.dp).clip(CircleShape).background(Color(0xFFEF4444)))
                        Spacer(Modifier.width(6.dp))
                        Text("Red: Naive IMU Drift", color = Color(0xFFFCA5A5), fontSize = 9.sp, fontWeight = FontWeight.SemiBold)
                    }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Box(Modifier.size(7.dp).clip(CircleShape).background(Color(0xFF38BDF8)))
                        Spacer(Modifier.width(6.dp))
                        Text("Blue: SAARTHI Road Lock", color = Color(0xFFBAE6FD), fontSize = 9.sp, fontWeight = FontWeight.SemiBold)
                    }
                }
            }
        }

        // 8. Compact Floating Bottom Sheet (Clutter-Free, Never Overlaps Nav Bar)
        BottomStatsSheet(
            state             = state,
            onStartSimulation = onStartSimulation,
            onStopSimulation  = onStopSimulation,
            onTogglePause     = onTogglePause,
            onRestart         = onRestart,
            onOpenAudit       = onOpenAudit,
            modifier          = Modifier.align(Alignment.BottomCenter)
        )

        // 9. Dedicated Telemetry & Algorithm Deep-Dive Dialog Modal
        if (state.showDiagnosticsPanel) {
            TelemetryModal(
                state = state,
                onDismiss = onToggleDiagnostics
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Top App Bar with Back Arrow and Clickable SAARTHI Logo
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun TopAppBar(
    onBack: () -> Unit,
    onLogoClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(AppBarBg)
            .padding(horizontal = 16.dp, vertical = 12.dp),
        verticalAlignment     = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Box(
                modifier = Modifier
                    .size(36.dp)
                    .clip(CircleShape)
                    .background(Color(0xFFF1F5F9))
                    .clickable { onBack() },
                contentAlignment = Alignment.Center
            ) {
                Text("←", color = AppBarText, fontSize = 18.sp, fontWeight = FontWeight.Bold)
            }
            Spacer(modifier = Modifier.width(10.dp))
            Row(
                modifier = Modifier.clickable { onLogoClick() },
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("▲", color = PuckBlue, fontSize = 20.sp, fontWeight = FontWeight.Bold)
                Spacer(modifier = Modifier.width(8.dp))
                Text(
                    "SAARTHI",
                    color         = AppBarText,
                    fontSize      = 21.sp,
                    fontWeight    = FontWeight.Black,
                    letterSpacing = 1.2.sp
                )
            }
        }
        Box(
            modifier = Modifier
                .size(36.dp)
                .clip(CircleShape)
                .background(Color(0xFFF1F5F9))
                .border(1.dp, Color(0xFFE2E8F0), CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Text("⚙", color = AppBarText, fontSize = 16.sp)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Floating Dark Status Card: GNSS STATUS | NETWORK | DR MODE
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun FloatingStatusCard(state: VehicleState) {
    val (gnssStatus, gnssColor) = when {
        state.stage == SimulationStage.READY               -> "AVAILABLE" to StatusGreen
        state.stage == SimulationStage.GNSS_DEGRADING      -> "DEGRADED" to StatusAmber
        state.inTunnel                                     -> "UNAVAILABLE" to StatusRed
        state.stage == SimulationStage.SEAMLESS_FUSION     -> "RESTORED" to StatusBlue
        else                                               -> "AVAILABLE" to StatusGreen
    }

    val netOnline = !state.inTunnel
    val (netStatus, netColor) = if (netOnline) "ONLINE" to StatusGreen else "OFFLINE" to StatusRed

    val (drStatus, drColor) = when {
        state.stage == SimulationStage.READY               -> "STANDBY" to StatusAmber
        state.stage == SimulationStage.GNSS_DEGRADING      -> "ENGAGING" to StatusAmber
        state.inTunnel                                     -> "ACTIVE" to StatusGreen
        state.stage == SimulationStage.SEAMLESS_FUSION     -> "FUSING" to StatusBlue
        else                                               -> "STANDBY" to StatusAmber
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 4.dp)
            .shadow(12.dp, RoundedCornerShape(22.dp))
            .clip(RoundedCornerShape(22.dp))
            .background(ChipBarBg)
            .border(1.dp, ChipBorder, RoundedCornerShape(22.dp))
            .padding(vertical = 10.dp, horizontal = 8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
            StatusColumn(
                modifier = Modifier.weight(1f),
                label    = "GNSS STATUS",
                iconType = "gnss",
                status   = gnssStatus,
                color    = gnssColor
            )

            Box(modifier = Modifier.width(1.dp).height(40.dp).background(ChipBorder))

            StatusColumn(
                modifier = Modifier.weight(1f),
                label    = "NETWORK",
                iconType = if (netOnline) "net_on" else "net_off",
                status   = netStatus,
                color    = netColor
            )

            Box(modifier = Modifier.width(1.dp).height(40.dp).background(ChipBorder))

            StatusColumn(
                modifier = Modifier.weight(1f),
                label    = "DR MODE",
                iconType = "dr",
                status   = drStatus,
                color    = drColor
            )
        }
    }
}

@Composable
private fun StatusColumn(
    modifier: Modifier,
    label: String,
    iconType: String,
    status: String,
    color: Color
) {
    Column(
        modifier            = modifier,
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(3.dp)
    ) {
        Text(
            text       = label,
            color      = ChipLabelText,
            fontSize   = 9.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 0.5.sp
        )

        Row(
            verticalAlignment     = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.spacedBy(5.dp)
        ) {
            when (iconType) {
                "gnss"    -> SatelliteIcon(color = color)
                "net_off" -> NetworkOffIcon(color = color)
                "net_on"  -> NetworkOnIcon(color = color)
                "dr"      -> DrModeIcon(color = color)
            }
            Text(
                text       = status,
                color      = color,
                fontSize   = 11.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Turn-by-Turn Instruction Banner
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun TurnInstructionBanner(state: VehicleState) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical = 2.dp)
            .shadow(6.dp, RoundedCornerShape(14.dp))
            .clip(RoundedCornerShape(14.dp))
            .background(SheetBg)
            .border(1.dp, SheetDivider, RoundedCornerShape(14.dp))
            .padding(horizontal = 14.dp, vertical = 7.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                Text(
                    text = if (state.stage == SimulationStage.DR_APPROACH_TURN || state.stage == SimulationStage.DR_TURN_VERIFIED) "↰" else "↑",
                    color = PuckBlue,
                    fontSize = 18.sp,
                    fontWeight = FontWeight.Black
                )
                Column {
                    Text(
                        text = state.turnInstruction,
                        color = SheetText,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold
                    )
                    Text(
                        text = if (state.inTunnel) "Dead Reckoning Mode • Map Spline Snapped" else "NavIC Dual-Band L5/S Locked",
                        color = SheetLabel,
                        fontSize = 9.5.sp
                    )
                }
            }
            Text(
                text = "%.1f km".format((state.distanceRemainingMeters / 1000f).coerceAtLeast(0f)),
                color = SheetText,
                fontSize = 11.sp,
                fontWeight = FontWeight.Bold
            )
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Custom Status Icons
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun SatelliteIcon(color: Color) {
    Canvas(modifier = Modifier.size(18.dp)) {
        val w = size.width
        val h = size.height
        drawArc(
            color       = color,
            startAngle  = 200f,
            sweepAngle  = 140f,
            useCenter   = false,
            topLeft     = Offset(w * 0.15f, h * 0.15f),
            size        = Size(w * 0.7f, h * 0.5f),
            style       = Stroke(width = 1.6f, cap = StrokeCap.Round)
        )
        drawLine(
            color       = color,
            start       = Offset(w * 0.5f, h * 0.40f),
            end         = Offset(w * 0.5f, h * 0.15f),
            strokeWidth = 1.6f,
            cap         = StrokeCap.Round
        )
        drawCircle(color = color, center = Offset(w * 0.5f, h * 0.15f), radius = 1.8f)
        drawLine(
            color       = color,
            start       = Offset(w * 0.5f, h * 0.40f),
            end         = Offset(w * 0.25f, h * 0.85f),
            strokeWidth = 1.6f,
            cap         = StrokeCap.Round
        )
        drawLine(
            color       = color,
            start       = Offset(w * 0.5f, h * 0.40f),
            end         = Offset(w * 0.75f, h * 0.85f),
            strokeWidth = 1.6f,
            cap         = StrokeCap.Round
        )
    }
}

@Composable
private fun NetworkOffIcon(color: Color) {
    Canvas(modifier = Modifier.size(18.dp)) {
        val w = size.width
        val h = size.height
        drawLine(color.copy(alpha = 0.6f), Offset(w * 0.25f, h * 0.75f), Offset(w * 0.25f, h * 0.60f), strokeWidth = 1.8f, cap = StrokeCap.Round)
        drawLine(color.copy(alpha = 0.6f), Offset(w * 0.50f, h * 0.75f), Offset(w * 0.50f, h * 0.40f), strokeWidth = 1.8f, cap = StrokeCap.Round)
        drawLine(color.copy(alpha = 0.6f), Offset(w * 0.75f, h * 0.75f), Offset(w * 0.75f, h * 0.20f), strokeWidth = 1.8f, cap = StrokeCap.Round)
        drawLine(
            color       = Color(0xFFEF4444),
            start       = Offset(w * 0.15f, h * 0.15f),
            end         = Offset(w * 0.85f, h * 0.85f),
            strokeWidth = 2.0f,
            cap         = StrokeCap.Round
        )
    }
}

@Composable
private fun NetworkOnIcon(color: Color) {
    Canvas(modifier = Modifier.size(18.dp)) {
        val w = size.width
        val h = size.height
        drawLine(color, Offset(w * 0.25f, h * 0.75f), Offset(w * 0.25f, h * 0.60f), strokeWidth = 1.8f, cap = StrokeCap.Round)
        drawLine(color, Offset(w * 0.50f, h * 0.75f), Offset(w * 0.50f, h * 0.40f), strokeWidth = 1.8f, cap = StrokeCap.Round)
        drawLine(color, Offset(w * 0.75f, h * 0.75f), Offset(w * 0.75f, h * 0.20f), strokeWidth = 1.8f, cap = StrokeCap.Round)
    }
}

@Composable
private fun DrModeIcon(color: Color) {
    Canvas(modifier = Modifier.size(18.dp)) {
        val w = size.width
        val h = size.height
        val arrow = Path().apply {
            moveTo(w * 0.5f, h * 0.15f)
            lineTo(w * 0.85f, h * 0.85f)
            lineTo(w * 0.5f, h * 0.65f)
            lineTo(w * 0.15f, h * 0.85f)
            close()
        }
        drawPath(arrow, color = color)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Compact Bottom Stats Sheet (Slim ~140dp height, Zero Clutter)
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun BottomStatsSheet(
    state: VehicleState,
    onStartSimulation: () -> Unit,
    onStopSimulation: () -> Unit,
    onTogglePause: () -> Unit,
    onRestart: () -> Unit,
    onOpenAudit: () -> Unit,
    modifier: Modifier = Modifier
) {
    Column(
        modifier = modifier
            .fillMaxWidth()
            .shadow(16.dp, RoundedCornerShape(topStart = 26.dp, topEnd = 26.dp))
            .clip(RoundedCornerShape(topStart = 26.dp, topEnd = 26.dp))
            .background(SheetBg)
            .padding(horizontal = 20.dp, vertical = 10.dp),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        // Drag handle indicator
        Box(
            modifier = Modifier
                .width(36.dp)
                .height(4.dp)
                .clip(RoundedCornerShape(2.dp))
                .background(HandleColor)
        )

        Spacer(modifier = Modifier.height(8.dp))

        // Compact Stage Status Pill
        if (state.isRunning || state.isCompleted) {
            StoryStagePill(state = state)
            Spacer(modifier = Modifier.height(8.dp))
        }

        // 3 Key Stats: Speed | Heading | Confidence
        Row(
            modifier              = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceAround,
            verticalAlignment     = Alignment.CenterVertically
        ) {
            StatColumn(
                icon = { SpeedometerIcon() },
                value = if (state.isRunning) "%.0f".format(state.speedKmh) else "62",
                unit = "km/h"
            )

            Box(modifier = Modifier.width(1.dp).height(44.dp).background(SheetDivider))

            val headingVal = if (state.isRunning) {
                128.4f + state.roadCurveDegrees * 0.35f
            } else {
                128.4f
            }
            StatColumn(
                icon = { CrosshairIcon() },
                value = "%.1f°".format(headingVal),
                unit = "Heading"
            )

            Box(modifier = Modifier.width(1.dp).height(44.dp).background(SheetDivider))

            val confidenceVal = confidenceFor(state)
            StatColumn(
                icon = { SignalBarsIcon() },
                value = "%.0f%%".format(confidenceVal),
                unit = "Confidence"
            )
        }

        Spacer(modifier = Modifier.height(10.dp))

        // Action Buttons
        when {
            !state.isStarted -> {
                Button(
                    onClick  = onStartSimulation,
                    modifier = Modifier.fillMaxWidth().height(48.dp),
                    shape    = RoundedCornerShape(14.dp),
                    colors   = ButtonDefaults.buttonColors(containerColor = PuckBlue)
                ) {
                    Text(
                        "START SIMULATION",
                        color         = Color.White,
                        fontWeight    = FontWeight.Bold,
                        fontSize      = 14.sp,
                        letterSpacing = 1.sp
                    )
                }
            }
            state.isCompleted -> {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Button(
                        onClick  = onOpenAudit,
                        modifier = Modifier.weight(1.3f).height(48.dp),
                        shape    = RoundedCornerShape(14.dp),
                        colors   = ButtonDefaults.buttonColors(containerColor = StatusGreen)
                    ) {
                        Text("📊 CHECK AUDIT REPORT", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                    Button(
                        onClick  = onRestart,
                        modifier = Modifier.weight(0.7f).height(48.dp),
                        shape    = RoundedCornerShape(14.dp),
                        colors   = ButtonDefaults.buttonColors(containerColor = Color(0xFF334155))
                    ) {
                        Text("RESTART", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                }
            }
            else -> {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.spacedBy(10.dp)
                ) {
                    Button(
                        onClick  = onTogglePause,
                        modifier = Modifier.weight(1f).height(48.dp),
                        shape    = RoundedCornerShape(14.dp),
                        colors   = ButtonDefaults.buttonColors(
                            containerColor = if (state.isRunning) StatusAmber else StatusGreen
                        )
                    ) {
                        Text(
                            if (state.isRunning) "PAUSE" else "RESUME",
                            color      = Color.White,
                            fontWeight = FontWeight.Bold,
                            fontSize   = 13.sp
                        )
                    }
                    Button(
                        onClick  = onStopSimulation,
                        modifier = Modifier.weight(1f).height(48.dp),
                        shape    = RoundedCornerShape(14.dp),
                        colors   = ButtonDefaults.buttonColors(containerColor = StatusRed)
                    ) {
                        Text("STOP", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 13.sp)
                    }
                }
            }
        }
    }
}

@Composable
private fun StatColumn(icon: @Composable () -> Unit, value: String, unit: String) {
    Column(
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(1.dp)
    ) {
        Box(modifier = Modifier.size(20.dp), contentAlignment = Alignment.Center) {
            icon()
        }
        Text(
            text       = value,
            color      = SheetText,
            fontSize   = 24.sp,
            fontWeight = FontWeight.Bold,
            lineHeight = 26.sp
        )
        Text(
            text       = unit,
            color      = SheetLabel,
            fontSize   = 10.sp,
            fontWeight = FontWeight.Medium
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Story Stage Pill
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun StoryStagePill(state: VehicleState) {
    val (color, badgeTitle) = when (state.stage) {
        SimulationStage.NORMAL_GNSS         -> StatusGreen to "● GNSS LOCKED • 100% SIGNAL INTEGRITY"
        SimulationStage.GNSS_DEGRADING      -> StatusAmber to "● GNSS DEGRADING • Trust Transfer to DR"
        SimulationStage.GNSS_LOST_DR_ACTIVE -> StatusRed   to "● DR ACTIVE • 15-State ESKF Engaged"
        SimulationStage.DR_NAVIGATING       -> StatusAmber to "● DR MODE • NHC Physics & Road Spline Locked"
        SimulationStage.DISTURBANCE_POTHOLE -> StatusAmber to "● POTHOLE SHOCK (>45 m/s³) GATED"
        SimulationStage.DR_APPROACH_TURN    -> StatusAmber to "● ROUTE-AWARE DR • Negotiating Turn"
        SimulationStage.DR_TURN_VERIFIED    -> StatusGreen to "● TURN VERIFIED • Zero Lateral Drift"
        SimulationStage.GNSS_RESTORED       -> StatusBlue  to "● GNSS RESTORED • Evaluating Innovation"
        SimulationStage.SEAMLESS_FUSION     -> StatusBlue  to "● SEAMLESS FUSION • Continuous Kalman Glide"
        SimulationStage.NORMAL_RESTORED     -> StatusGreen to "● GNSS + INS RESTORED • Normal Nav"
        SimulationStage.DESTINATION_REACHED -> StatusGreen to "● DESTINATION REACHED • Mission Succeeded"
        else                                -> SheetLabel  to "● SYSTEM READY • Standby"
    }

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(8.dp))
            .background(color.copy(alpha = 0.10f))
            .border(1.dp, color.copy(alpha = 0.35f), RoundedCornerShape(8.dp))
            .padding(horizontal = 10.dp, vertical = 5.dp)
    ) {
        Text(
            text          = badgeTitle,
            color         = color,
            fontSize      = 10.sp,
            fontWeight    = FontWeight.Bold,
            letterSpacing = 0.4.sp
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dedicated Telemetry & Algorithm Modal Dialog
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun TelemetryModal(
    state: VehicleState,
    onDismiss: () -> Unit
) {
    Dialog(onDismissRequest = onDismiss) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .shadow(24.dp, RoundedCornerShape(20.dp))
                .clip(RoundedCornerShape(20.dp))
                .background(Color(0xFF0F172A))
                .border(1.dp, Color(0xFF334155), RoundedCornerShape(20.dp))
                .padding(18.dp)
        ) {
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .verticalScroll(rememberScrollState()),
                verticalArrangement = Arrangement.spacedBy(12.dp)
            ) {
                // Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween,
                    verticalAlignment = Alignment.CenterVertically
                ) {
                    Column {
                        Text(
                            "DR ENGINE LIVE TELEMETRY",
                            color = Color.White,
                            fontSize = 13.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        )
                        Text(
                            "Kinematics • ESKF Covariance • Constraints",
                            color = Color(0xFF94A3B8),
                            fontSize = 10.sp
                        )
                    }
                    Box(
                        modifier = Modifier
                            .size(28.dp)
                            .clip(CircleShape)
                            .background(Color(0xFF1E293B))
                            .clickable { onDismiss() },
                        contentAlignment = Alignment.Center
                    ) {
                        Text("✕", color = Color.White, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                    }
                }

                Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(Color(0xFF1E293B)))

                // Row 1: AI Speedometer & Heteroscedastic Variance
                TelemetryRow(
                    label1 = "AI Speed Est",
                    val1 = "%.1f km/h".format(state.aiSpeedEstimateKmh),
                    label2 = "Speed Variance σ²",
                    val2 = "%.2f (m/s)²".format(state.speedVarianceSigma2)
                )

                // Row 2: Heading Error & Cross-Track Spline Offset
                TelemetryRow(
                    label1 = "Heading StdDev σ_θ",
                    val1 = "%.1f°".format(state.headingSigmaDeg),
                    label2 = "Spline Cross-Track",
                    val2 = "%.2f m".format(state.crossTrackMeters)
                )

                // Row 3: Trust Allocation
                TelemetryRow(
                    label1 = "GNSS Trust",
                    val1 = "${state.gnssTrustPercent}%",
                    label2 = "INS DR Trust",
                    val2 = "${state.insTrustPercent}%"
                )

                // Row 4: Physics & Constraints
                TelemetryRow(
                    label1 = "NHC Body Frame",
                    val1 = "v_lat = 0, v_up = 0",
                    label2 = "Recovery Step Δp",
                    val2 = "0.006 m/step"
                )

                Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(Color(0xFF1E293B)))

                // Algorithm Pill Summary
                Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                    Text("ACTIVE FILTER ALGORITHM", color = Color(0xFF64748B), fontSize = 9.sp, fontWeight = FontWeight.Bold)
                    Text(
                        state.activeAlgorithm.chipLabel + " — " + state.activeAlgorithm.detailText,
                        color = Color(0xFF38BDF8),
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Medium
                    )
                }

                Spacer(modifier = Modifier.height(4.dp))

                Button(
                    onClick = onDismiss,
                    modifier = Modifier.fillMaxWidth().height(42.dp),
                    shape = RoundedCornerShape(10.dp),
                    colors = ButtonDefaults.buttonColors(containerColor = Color(0xFF1E293B))
                ) {
                    Text("CLOSE TELEMETRY", color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
            }
        }
    }
}

@Composable
private fun TelemetryRow(label1: String, val1: String, label2: String, val2: String) {
    Row(
        modifier = Modifier.fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceBetween
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(label1, color = Color(0xFF64748B), fontSize = 9.sp)
            Text(val1, color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        }
        Spacer(modifier = Modifier.width(8.dp))
        Column(modifier = Modifier.weight(1f)) {
            Text(label2, color = Color(0xFF64748B), fontSize = 9.sp)
            Text(val2, color = Color.White, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Icons
// ─────────────────────────────────────────────────────────────────────────────
@Composable
private fun SpeedometerIcon() {
    Canvas(modifier = Modifier.size(20.dp)) {
        val w = size.width
        val h = size.height
        drawArc(
            color       = SheetLabel,
            startAngle  = 150f,
            sweepAngle  = 240f,
            useCenter   = false,
            topLeft     = Offset(w * 0.1f, h * 0.1f),
            size        = Size(w * 0.8f, h * 0.8f),
            style       = Stroke(width = 2.0f, cap = StrokeCap.Round)
        )
        drawLine(
            color       = PuckBlue,
            start       = Offset(w * 0.5f, h * 0.5f),
            end         = Offset(w * 0.72f, h * 0.30f),
            strokeWidth = 2.2f,
            cap         = StrokeCap.Round
        )
        drawCircle(color = SheetText, radius = 2.5f, center = Offset(w * 0.5f, h * 0.5f))
    }
}

@Composable
private fun CrosshairIcon() {
    Canvas(modifier = Modifier.size(20.dp)) {
        val w = size.width
        val h = size.height
        drawCircle(
            color  = SheetLabel,
            center = Offset(w * 0.5f, h * 0.5f),
            radius = w * 0.38f,
            style  = Stroke(width = 1.8f)
        )
        drawLine(SheetLabel, Offset(w * 0.5f, h * 0.08f), Offset(w * 0.5f, h * 0.28f), strokeWidth = 1.8f)
        drawLine(SheetLabel, Offset(w * 0.5f, h * 0.72f), Offset(w * 0.5f, h * 0.92f), strokeWidth = 1.8f)
        drawLine(SheetLabel, Offset(w * 0.08f, h * 0.5f), Offset(w * 0.28f, h * 0.5f), strokeWidth = 1.8f)
        drawLine(SheetLabel, Offset(w * 0.72f, h * 0.5f), Offset(w * 0.92f, h * 0.5f), strokeWidth = 1.8f)
        drawCircle(color = PuckBlue, center = Offset(w * 0.5f, h * 0.5f), radius = 2.5f)
    }
}

@Composable
private fun SignalBarsIcon() {
    Canvas(modifier = Modifier.size(20.dp)) {
        val w = size.width
        val h = size.height
        val bars = 4
        val barW = w * 0.16f
        val gap = w * 0.08f
        val heights = listOf(0.35f, 0.55f, 0.75f, 0.95f)

        for (i in 0 until bars) {
            val barH = h * heights[i]
            val x = w * 0.05f + i * (barW + gap)
            val y = h - barH
            drawRoundRect(
                color = StatusGreen,
                topLeft = Offset(x, y),
                size = Size(barW, barH),
                cornerRadius = androidx.compose.ui.geometry.CornerRadius(2f, 2f)
            )
        }
    }
}

private fun confidenceFor(state: VehicleState): Float {
    return when {
        !state.isRunning                                   -> 98.0f
        state.stage == SimulationStage.NORMAL_GNSS         -> 98.0f
        state.stage == SimulationStage.GNSS_DEGRADING      -> (98.0f - (state.timeSeconds - 7.5f) * 1.8f).coerceAtLeast(88.0f)
        state.inTunnel                                     -> (92.0f - (state.timeSeconds - 11.5f) * 0.22f).coerceAtLeast(82.0f)
        state.stage == SimulationStage.SEAMLESS_FUSION     -> 94.0f
        state.stage == SimulationStage.NORMAL_RESTORED     -> 98.0f
        else                                               -> 98.0f
    }
}
