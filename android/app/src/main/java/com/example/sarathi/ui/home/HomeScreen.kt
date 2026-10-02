package com.example.sarathi.ui.home

import androidx.compose.animation.AnimatedVisibility
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
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
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.window.Dialog
import com.example.sarathi.model.VehicleType
import com.example.sarathi.theme.HomeBg
import com.example.sarathi.theme.HomeBorder
import com.example.sarathi.theme.HomeCardBg
import com.example.sarathi.theme.HomeSubtext
import com.example.sarathi.theme.HomeText
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.theme.StatusAmber
import com.example.sarathi.theme.StatusGreen

@Composable
fun HomeScreen(
    selectedVehicleType: VehicleType,
    onSelectVehicleType: (VehicleType) -> Unit,
    onStartNavigation: () -> Unit,
    preFlightStep: Int? = null,
    onDismissPreFlight: () -> Unit = {},
    onBack: () -> Unit = {},
    onLogoClick: () -> Unit = {}
) {
    Box(modifier = Modifier.fillMaxSize().background(HomeBg)) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 20.dp, vertical = 16.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            // Header: Back Arrow + SAARTHI Logo + Settings
            Row(
                modifier = Modifier.fillMaxWidth(),
                verticalAlignment = Alignment.CenterVertically,
                horizontalArrangement = Arrangement.SpaceBetween
            ) {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(36.dp)
                            .clip(CircleShape)
                            .background(Color(0xFFE2E8F0))
                            .clickable { onBack() },
                        contentAlignment = Alignment.Center
                    ) {
                        Text("←", color = HomeText, fontSize = 18.sp, fontWeight = FontWeight.Bold)
                    }
                    Spacer(Modifier.width(10.dp))
                    Column(
                        modifier = Modifier.clickable { onLogoClick() }
                    ) {
                        Row(verticalAlignment = Alignment.CenterVertically) {
                            Text("▲", color = PuckBlue, fontSize = 22.sp, fontWeight = FontWeight.ExtraBold)
                            Spacer(Modifier.width(8.dp))
                            Text(
                                "SAARTHI",
                                color = HomeText,
                                fontSize = 24.sp,
                                fontWeight = FontWeight.Black,
                                letterSpacing = 1.2.sp
                            )
                        }
                        Text(
                            "AI Dead Reckoning • ISRO NavIC L5/S",
                            color = Color(0xFF1E293B),
                            fontSize = 12.sp,
                            fontWeight = FontWeight.SemiBold
                        )
                        Text(
                            "SAC Ahmedabad • Team SIH26168",
                            color = HomeSubtext,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Medium
                        )
                    }
                }

                Box(
                    modifier = Modifier
                        .size(42.dp)
                        .clip(CircleShape)
                        .background(HomeCardBg)
                        .border(1.dp, HomeBorder, CircleShape),
                    contentAlignment = Alignment.Center
                ) {
                    Text("⚙", color = HomeText, fontSize = 18.sp)
                }
            }

            // Section 1: Vehicle Dynamics Profile
            SectionLabel("VEHICLE DYNAMICS PROFILE")
            Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
                VehicleType.values().forEach { type ->
                    VehicleCard(
                        type = type,
                        isSelected = selectedVehicleType == type,
                        onSelect = { onSelectVehicleType(type) }
                    )
                }
            }

            // Section 2: Sensor Health
            SectionLabel("SENSOR INTEGRITY STATUS")
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                InfoTile("NavIC Dual-Band", "L5/S LOCKED", StatusGreen, Modifier.weight(1f))
                InfoTile("IMU 100 Hz", "CALIBRATED", StatusGreen, Modifier.weight(1f))
                InfoTile("15-State ESKF", "READY", StatusAmber, Modifier.weight(1f))
            }

            // Section 3: Scenario Overview
            SectionLabel("MISSION SCENARIO")
            ScenarioCard()

            Spacer(Modifier.height(4.dp))

            // Start Navigation Button
            Button(
                onClick = onStartNavigation,
                modifier = Modifier
                    .fillMaxWidth()
                    .height(56.dp)
                    .shadow(8.dp, RoundedCornerShape(16.dp)),
                shape = RoundedCornerShape(16.dp),
                colors = ButtonDefaults.buttonColors(containerColor = PuckBlue)
            ) {
                Row(
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.spacedBy(8.dp)
                ) {
                    Text("▲", color = Color.White, fontSize = 14.sp)
                    Text(
                        "START NAVIGATION",
                        color = Color.White,
                        fontWeight = FontWeight.Bold,
                        fontSize = 15.sp,
                        letterSpacing = 1.sp
                    )
                }
            }

            Spacer(Modifier.height(8.dp))
        }

        // Pre-Flight Verification Checklist Dialog
        if (preFlightStep != null) {
            PreFlightCheckDialog(currentStep = preFlightStep)
        }
    }
}

@Composable
private fun SectionLabel(text: String) {
    Text(
        text = text,
        color = Color(0xFF334155),
        fontSize = 11.sp,
        fontWeight = FontWeight.Bold,
        letterSpacing = 1.5.sp
    )
}

@Composable
private fun VehicleCard(type: VehicleType, isSelected: Boolean, onSelect: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .shadow(if (isSelected) 4.dp else 1.dp, RoundedCornerShape(14.dp))
            .clip(RoundedCornerShape(14.dp))
            .background(HomeCardBg)
            .border(
                width = if (isSelected) 2.dp else 1.dp,
                color = if (isSelected) PuckBlue else HomeBorder,
                shape = RoundedCornerShape(14.dp)
            )
            .clickable { onSelect() }
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(Modifier.weight(1f)) {
            Row(verticalAlignment = Alignment.CenterVertically) {
                Text(
                    text = type.title,
                    color = HomeText,
                    fontSize = 14.sp,
                    fontWeight = FontWeight.Bold
                )
            }
            Spacer(Modifier.height(3.dp))
            Text(
                text = type.subtitle,
                color = Color(0xFF475569),
                fontSize = 11.sp,
                fontWeight = FontWeight.Medium,
                lineHeight = 14.sp
            )
        }
        Spacer(Modifier.width(10.dp))
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(8.dp))
                .background(if (isSelected) PuckBlue else Color(0xFFE2E8F0))
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Text(
                text = if (isSelected) "ACTIVE" else "SELECT",
                color = if (isSelected) Color.White else Color(0xFF475569),
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.5.sp
            )
        }
    }
}

@Composable
private fun InfoTile(label: String, status: String, statusColor: Color, modifier: Modifier = Modifier) {
    Column(
        modifier = modifier
            .shadow(1.dp, RoundedCornerShape(12.dp))
            .clip(RoundedCornerShape(12.dp))
            .background(HomeCardBg)
            .border(1.dp, HomeBorder, RoundedCornerShape(12.dp))
            .padding(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Box(Modifier.size(8.dp).clip(CircleShape).background(statusColor))
        Text(
            text = label,
            color = Color(0xFF1E293B),
            fontSize = 10.sp,
            fontWeight = FontWeight.SemiBold
        )
        Text(
            text = status,
            color = statusColor,
            fontSize = 10.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

@Composable
private fun ScenarioCard() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .shadow(1.dp, RoundedCornerShape(14.dp))
            .clip(RoundedCornerShape(14.dp))
            .background(HomeCardBg)
            .border(1.dp, HomeBorder, RoundedCornerShape(14.dp))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        ScenarioRow("ROUTE", "Dunchurch Highway Outage Corridor")
        ScenarioRow("GNSS OUTAGE", "Multi-Stage Simulated Tunnel & Flyover")
        ScenarioRow("GROUND TRUTH", "Racelogic VBOX RTK 100 Hz Sub-cm")
        ScenarioRow("SATELLITE SYS", "ISRO NavIC Dual L5/S + GPS L1/L5")
    }
}

@Composable
private fun ScenarioRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(
            text = label,
            color = Color(0xFF64748B),
            fontSize = 11.sp,
            fontWeight = FontWeight.SemiBold
        )
        Text(
            text = value,
            color = Color(0xFF0F172A),
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pre-Flight Verification Checklist Dialog
// ─────────────────────────────────────────────────────────────────────────────
@Composable
fun PreFlightCheckDialog(currentStep: Int) {
    Dialog(onDismissRequest = {}) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .shadow(24.dp, RoundedCornerShape(22.dp))
                .clip(RoundedCornerShape(22.dp))
                .background(Color(0xFF0F172A))
                .border(1.dp, Color(0xFF334155), RoundedCornerShape(22.dp))
                .padding(20.dp)
        ) {
            Column(
                modifier = Modifier.fillMaxWidth(),
                verticalArrangement = Arrangement.spacedBy(14.dp)
            ) {
                // Header
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    verticalAlignment = Alignment.CenterVertically,
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Column {
                        Text(
                            text = "PRE-FLIGHT SENSOR CHECK",
                            color = Color.White,
                            fontSize = 14.sp,
                            fontWeight = FontWeight.Bold,
                            letterSpacing = 1.sp
                        )
                        Text(
                            text = "Verifying Dead Reckoning Hardware",
                            color = Color(0xFF94A3B8),
                            fontSize = 11.sp
                        )
                    }
                    Box(
                        modifier = Modifier
                            .clip(RoundedCornerShape(6.dp))
                            .background(PuckBlue.copy(alpha = 0.2f))
                            .border(1.dp, PuckBlue, RoundedCornerShape(6.dp))
                            .padding(horizontal = 8.dp, vertical = 4.dp)
                    ) {
                        Text(
                            text = if (currentStep >= 6) "READY" else "CHECKING...",
                            color = if (currentStep >= 6) StatusGreen else PuckBlue,
                            fontSize = 9.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                }

                Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(Color(0xFF1E293B)))

                // Checklist Items
                ChecklistItem(
                    stepNumber = 1,
                    title = "Phone Mount & Orientation",
                    subtitle = "Rigid body dock verified (Pitch -2.1°, Roll 0.4°)",
                    currentStep = currentStep
                )
                ChecklistItem(
                    stepNumber = 2,
                    title = "High-Rate IMU (100 Hz)",
                    subtitle = "SO(3) Static Gyro Bias & Accel Calibrated",
                    currentStep = currentStep
                )
                ChecklistItem(
                    stepNumber = 3,
                    title = "ISRO NavIC Dual-Band L5/S",
                    subtitle = "7 NavIC Satellites Locked • Carrier-to-Noise 44 dB",
                    currentStep = currentStep
                )
                ChecklistItem(
                    stepNumber = 4,
                    title = "AI Speedometer Kinematic Model",
                    subtitle = "204 KB Heteroscedastic ONNX Engine Loaded",
                    currentStep = currentStep
                )
                ChecklistItem(
                    stepNumber = 5,
                    title = "HD Map Topological Splines",
                    subtitle = "OSM Highway KD-Tree Snapped & Cached",
                    currentStep = currentStep
                )

                Box(modifier = Modifier.fillMaxWidth().height(1.dp).background(Color(0xFF1E293B)))

                // Bottom Launch Progress
                if (currentStep >= 6) {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        Text("🚀", fontSize = 16.sp)
                        Spacer(Modifier.width(8.dp))
                        Text(
                            "ALL SENSORS VERIFIED • LAUNCHING...",
                            color = StatusGreen,
                            fontSize = 11.sp,
                            fontWeight = FontWeight.Bold
                        )
                    }
                } else {
                    Row(
                        modifier = Modifier.fillMaxWidth(),
                        horizontalArrangement = Arrangement.Center,
                        verticalAlignment = Alignment.CenterVertically
                    ) {
                        CircularProgressIndicator(
                            modifier = Modifier.size(16.dp),
                            color = PuckBlue,
                            strokeWidth = 2.dp
                        )
                        Spacer(Modifier.width(10.dp))
                        Text(
                            "Calibrating step $currentStep of 5...",
                            color = Color(0xFF94A3B8),
                            fontSize = 11.sp
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ChecklistItem(
    stepNumber: Int,
    title: String,
    subtitle: String,
    currentStep: Int
) {
    val isPassed = currentStep > stepNumber
    val isCurrent = currentStep == stepNumber

    Row(
        modifier = Modifier.fillMaxWidth(),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Box(
            modifier = Modifier
                .size(24.dp)
                .clip(CircleShape)
                .background(
                    when {
                        isPassed -> StatusGreen.copy(alpha = 0.2f)
                        isCurrent -> PuckBlue.copy(alpha = 0.2f)
                        else -> Color(0xFF1E293B)
                    }
                )
                .border(
                    width = 1.dp,
                    color = when {
                        isPassed -> StatusGreen
                        isCurrent -> PuckBlue
                        else -> Color(0xFF334155)
                    },
                    shape = CircleShape
                ),
            contentAlignment = Alignment.Center
        ) {
            when {
                isPassed -> Text("✓", color = StatusGreen, fontSize = 12.sp, fontWeight = FontWeight.Bold)
                isCurrent -> CircularProgressIndicator(
                    modifier = Modifier.size(12.dp),
                    color = PuckBlue,
                    strokeWidth = 1.5.dp
                )
                else -> Text("$stepNumber", color = Color(0xFF64748B), fontSize = 10.sp)
            }
        }

        Spacer(Modifier.width(12.dp))

        Column(Modifier.weight(1f)) {
            Text(
                text = title,
                color = if (isPassed || isCurrent) Color.White else Color(0xFF94A3B8),
                fontSize = 12.sp,
                fontWeight = if (isPassed || isCurrent) FontWeight.Bold else FontWeight.Normal
            )
            Text(
                text = subtitle,
                color = if (isPassed) Color(0xFF94A3B8) else Color(0xFF64748B),
                fontSize = 10.sp
            )
        }
    }
}
