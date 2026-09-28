package com.example.sarathi.ui.home

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
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sarathi.model.VehicleType
import com.example.sarathi.theme.AccentCyan
import com.example.sarathi.theme.BgBase
import com.example.sarathi.theme.BgCard
import com.example.sarathi.theme.BgSurface
import com.example.sarathi.theme.BorderHighlight
import com.example.sarathi.theme.BorderSubtle
import com.example.sarathi.theme.StatusGnss
import com.example.sarathi.theme.TextMuted
import com.example.sarathi.theme.TextPrimary
import com.example.sarathi.theme.TextSecondary

@Composable
fun HomeScreen(
    selectedVehicleType: VehicleType,
    onSelectVehicleType: (VehicleType) -> Unit,
    onStartNavigation: () -> Unit,
    modifier: Modifier = Modifier
) {
    val scrollState = rememberScrollState()

    Column(
        modifier = modifier
            .fillMaxSize()
            .background(BgBase)
            .verticalScroll(scrollState)
            .padding(horizontal = 20.dp, vertical = 24.dp)
    ) {
        // App Header
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column {
                Text(
                    text = "SARATHI",
                    color = AccentCyan,
                    fontSize = 28.sp,
                    fontWeight = FontWeight.Black,
                    letterSpacing = 2.sp,
                    fontFamily = FontFamily.Monospace
                )
                Text(
                    text = "AI-ML Intelligent Dead Reckoning",
                    color = TextSecondary,
                    fontSize = 12.sp,
                    fontWeight = FontWeight.Medium
                )
            }

            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(6.dp))
                    .background(BgCard)
                    .border(1.dp, BorderSubtle, RoundedCornerShape(6.dp))
                    .padding(horizontal = 8.dp, vertical = 4.dp)
            ) {
                Text(
                    text = "ISRO NavIC L5/S",
                    color = StatusGnss,
                    fontSize = 10.sp,
                    fontFamily = FontFamily.Monospace,
                    fontWeight = FontWeight.Bold
                )
            }
        }

        Spacer(modifier = Modifier.height(6.dp))
        Text(
            text = "Space Applications Centre (SAC), ISRO Ahmedabad • SIH26168",
            color = TextMuted,
            fontSize = 10.sp,
            fontFamily = FontFamily.Monospace
        )

        Spacer(modifier = Modifier.height(24.dp))

        // 1. Vehicle Profile Selection Card
        Text(
            text = "VEHICLE DYNAMICS PROFILE",
            color = TextPrimary,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            letterSpacing = 1.sp
        )
        Spacer(modifier = Modifier.height(8.dp))

        VehicleTypeOption(
            type = VehicleType.CAR_4W,
            isSelected = selectedVehicleType == VehicleType.CAR_4W,
            onSelect = { onSelectVehicleType(VehicleType.CAR_4W) }
        )

        Spacer(modifier = Modifier.height(10.dp))

        VehicleTypeOption(
            type = VehicleType.BIKE_2W,
            isSelected = selectedVehicleType == VehicleType.BIKE_2W,
            onSelect = { onSelectVehicleType(VehicleType.BIKE_2W) }
        )

        Spacer(modifier = Modifier.height(20.dp))

        // 2. Sensor Mount & Dynamic Alignment Card
        Text(
            text = "SENSOR MOUNT & CALIBRATION",
            color = TextPrimary,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            letterSpacing = 1.sp
        )
        Spacer(modifier = Modifier.height(8.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(BgSurface)
                .border(1.dp, BorderSubtle, RoundedCornerShape(12.dp))
                .padding(14.dp)
        ) {
            Column {
                Row(verticalAlignment = Alignment.CenterVertically) {
                    Box(
                        modifier = Modifier
                            .size(8.dp)
                            .clip(CircleShape)
                            .background(StatusGnss)
                    )
                    Spacer(modifier = Modifier.width(8.dp))
                    Text(
                        text = "Dynamic SO(3) Calibration: READY",
                        color = StatusGnss,
                        fontSize = 11.sp,
                        fontWeight = FontWeight.Bold,
                        fontFamily = FontFamily.Monospace
                    )
                }
                Spacer(modifier = Modifier.height(6.dp))
                Text(
                    text = "Automatic phone-to-vehicle pitch, roll, and yaw determination. Real-time mount slip detection & pothole shock dampening active.",
                    color = TextSecondary,
                    fontSize = 11.sp,
                    lineHeight = 16.sp
                )
            }
        }

        Spacer(modifier = Modifier.height(20.dp))

        // 3. Evaluation Scenario Card
        Text(
            text = "AUTOMATED EVALUATION SCENARIO",
            color = TextPrimary,
            fontSize = 11.sp,
            fontWeight = FontWeight.Bold,
            fontFamily = FontFamily.Monospace,
            letterSpacing = 1.sp
        )
        Spacer(modifier = Modifier.height(8.dp))

        Box(
            modifier = Modifier
                .fillMaxWidth()
                .clip(RoundedCornerShape(12.dp))
                .background(BgSurface)
                .border(1.dp, BorderSubtle, RoundedCornerShape(12.dp))
                .padding(14.dp)
        ) {
            Column {
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(text = "ROUTE:", color = TextMuted, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
                    Text(text = "Dunchurch Hwy Underpass", color = TextPrimary, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(text = "BLACKOUT DURATION:", color = TextMuted, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
                    Text(text = "60-Second Full Outage", color = AccentCyan, fontSize = 11.sp, fontWeight = FontWeight.Bold)
                }
                Spacer(modifier = Modifier.height(4.dp))
                Row(
                    modifier = Modifier.fillMaxWidth(),
                    horizontalArrangement = Arrangement.SpaceBetween
                ) {
                    Text(text = "GROUND TRUTH:", color = TextMuted, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
                    Text(text = "Racelogic VBOX RTK 100 Hz", color = TextSecondary, fontSize = 10.sp, fontFamily = FontFamily.Monospace)
                }
            }
        }

        Spacer(modifier = Modifier.height(28.dp))

        // 4. Large Action Button: START NAVIGATION
        Button(
            onClick = onStartNavigation,
            colors = ButtonDefaults.buttonColors(containerColor = AccentCyan, contentColor = BgBase),
            shape = RoundedCornerShape(12.dp),
            modifier = Modifier
                .fillMaxWidth()
                .height(54.dp)
        ) {
            Text(
                text = "START NAVIGATION",
                fontSize = 14.sp,
                fontWeight = FontWeight.Black,
                letterSpacing = 1.5.sp,
                fontFamily = FontFamily.Monospace
            )
        }
    }
}

@Composable
private fun VehicleTypeOption(
    type: VehicleType,
    isSelected: Boolean,
    onSelect: () -> Unit
) {
    val borderColor = if (isSelected) AccentCyan else BorderSubtle
    val bgColor = if (isSelected) BgCard else BgSurface

    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(bgColor)
            .border(1.5.dp, borderColor, RoundedCornerShape(12.dp))
            .clickable { onSelect() }
            .padding(14.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically
        ) {
            Column(modifier = Modifier.weight(1.0f)) {
                Text(
                    text = type.title,
                    color = if (isSelected) AccentCyan else TextPrimary,
                    fontSize = 13.sp,
                    fontWeight = FontWeight.Bold
                )
                Spacer(modifier = Modifier.height(2.dp))
                Text(
                    text = type.subtitle,
                    color = TextSecondary,
                    fontSize = 10.sp
                )
            }

            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(4.dp))
                    .background(if (isSelected) AccentCyan else BgBase)
                    .border(1.dp, if (isSelected) AccentCyan else BorderHighlight, RoundedCornerShape(4.dp))
                    .padding(horizontal = 6.dp, vertical = 2.dp)
            ) {
                Text(
                    text = if (isSelected) "ACTIVE" else "SELECT",
                    color = if (isSelected) BgBase else TextMuted,
                    fontSize = 9.sp,
                    fontWeight = FontWeight.Bold,
                    fontFamily = FontFamily.Monospace
                )
            }
        }
    }
}
