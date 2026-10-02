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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sarathi.model.VehicleType
import com.example.sarathi.theme.ChipBg
import com.example.sarathi.theme.ChipBorder
import com.example.sarathi.theme.MapBg
import com.example.sarathi.theme.MapTerrain
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.theme.StatusAmber
import com.example.sarathi.theme.StatusGreen
import com.example.sarathi.theme.TextChip
import com.example.sarathi.theme.TextPrimary
import com.example.sarathi.theme.TextSecondary

@Composable
fun HomeScreen(
    selectedVehicleType: VehicleType,
    onSelectVehicleType: (VehicleType) -> Unit,
    onStartNavigation: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MapBg)
            .verticalScroll(rememberScrollState())
            .padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(20.dp)
    ) {
        Spacer(modifier = Modifier.height(12.dp))

        // Header
        Column {
            Text(
                text = "A  SAARTHI",
                color = TextPrimary,
                fontSize = 22.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 2.sp
            )
            Text(
                text = "AI-Enhanced Dead Reckoning  •  ISRO NavIC",
                color = TextSecondary,
                fontSize = 12.sp,
                fontWeight = FontWeight.Normal
            )
            Spacer(modifier = Modifier.height(2.dp))
            Text(
                text = "Space Applications Centre (SAC)  •  SIH26168",
                color = TextSecondary.copy(alpha = 0.6f),
                fontSize = 11.sp
            )
        }

        // Section: Vehicle
        SectionLabel("VEHICLE PROFILE")
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            VehicleType.values().forEach { type ->
                VehicleCard(
                    type = type,
                    isSelected = selectedVehicleType == type,
                    onSelect = { onSelectVehicleType(type) }
                )
            }
        }

        // Section: System status
        SectionLabel("SYSTEM STATUS")
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            InfoTile("NavIC L5/S", "READY", StatusGreen, modifier = Modifier.weight(1f))
            InfoTile("SO(3) Cal.", "ACTIVE", StatusGreen, modifier = Modifier.weight(1f))
            InfoTile("15-State ESKF", "ONLINE", StatusAmber, modifier = Modifier.weight(1f))
        }

        // Section: Scenario
        SectionLabel("DEMO SCENARIO")
        ScenarioCard()

        Spacer(modifier = Modifier.height(8.dp))

        // CTA button
        Button(
            onClick = onStartNavigation,
            modifier = Modifier
                .fillMaxWidth()
                .height(56.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = PuckBlue)
        ) {
            Text(
                text = "START NAVIGATION",
                color = Color.White,
                fontWeight = FontWeight.Bold,
                fontSize = 15.sp,
                letterSpacing = 1.5.sp
            )
        }

        Spacer(modifier = Modifier.height(8.dp))
    }
}

@Composable
private fun SectionLabel(text: String) {
    Text(
        text = text,
        color = TextSecondary,
        fontSize = 10.sp,
        fontWeight = FontWeight.Bold,
        letterSpacing = 1.5.sp
    )
}

@Composable
private fun VehicleCard(type: VehicleType, isSelected: Boolean, onSelect: () -> Unit) {
    val borderColor = if (isSelected) PuckBlue else ChipBorder
    val bgColor     = if (isSelected) PuckBlue.copy(alpha = 0.12f) else ChipBg

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(bgColor)
            .border(1.dp, borderColor, RoundedCornerShape(14.dp))
            .clickable { onSelect() }
            .padding(horizontal = 16.dp, vertical = 14.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(text = type.title, color = TextPrimary, fontSize = 13.sp, fontWeight = FontWeight.SemiBold)
            Spacer(modifier = Modifier.height(2.dp))
            Text(text = type.subtitle, color = TextSecondary, fontSize = 10.sp)
        }
        Spacer(modifier = Modifier.width(12.dp))
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(8.dp))
                .background(if (isSelected) PuckBlue else ChipBorder)
                .padding(horizontal = 10.dp, vertical = 4.dp)
        ) {
            Text(
                text = if (isSelected) "ACTIVE" else "SELECT",
                color = Color.White,
                fontSize = 9.sp,
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
            .clip(RoundedCornerShape(12.dp))
            .background(ChipBg)
            .padding(12.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(6.dp)
    ) {
        Box(
            modifier = Modifier
                .size(8.dp)
                .clip(CircleShape)
                .background(statusColor)
        )
        Text(text = label, color = TextSecondary, fontSize = 9.sp, fontWeight = FontWeight.Medium, letterSpacing = 0.3.sp)
        Text(text = status, color = statusColor, fontSize = 9.sp, fontWeight = FontWeight.Bold)
    }
}

@Composable
private fun ScenarioCard() {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(ChipBg)
            .border(1.dp, ChipBorder, RoundedCornerShape(14.dp))
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        ScenarioRow("ROUTE", "Dunchurch Highway Underpass")
        ScenarioRow("BLACKOUT", "60-Second GPS Full Outage (tunnel)")
        ScenarioRow("GROUND TRUTH", "Racelogic VBOX RTK  100 Hz")
        ScenarioRow("SATELLITE SYS", "ISRO NavIC L5/S + GPS L1/L5")
    }
}

@Composable
private fun ScenarioRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(text = label, color = TextSecondary, fontSize = 10.sp, fontWeight = FontWeight.Medium, letterSpacing = 0.5.sp)
        Text(text = value, color = TextChip, fontSize = 10.sp, fontWeight = FontWeight.SemiBold)
    }
}
