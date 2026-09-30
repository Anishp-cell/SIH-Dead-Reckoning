package com.example.sarathi.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.sarathi.theme.BorderLight
import com.example.sarathi.theme.MapTerrainBg
import com.example.sarathi.theme.NavRouteBlue
import com.example.sarathi.theme.SurfaceCardLight
import com.example.sarathi.theme.TextMutedDark
import com.example.sarathi.ui.audit.AuditScreen
import com.example.sarathi.ui.home.HomeScreen
import com.example.sarathi.ui.nav.NavigationScreen
import com.example.sarathi.viewmodel.AppTab
import com.example.sarathi.viewmodel.SarathiViewModel

@Composable
fun SarathiApp(
    viewModel: SarathiViewModel = viewModel()
) {
    val currentTab by viewModel.currentTab.collectAsState()
    val vehicleState by viewModel.vehicleState.collectAsState()
    val auditResults by viewModel.auditResults.collectAsState()

    Scaffold(
        bottomBar = {
            BottomTabBar(
                currentTab = currentTab,
                onSelectTab = { viewModel.selectTab(it) }
            )
        },
        containerColor = MapTerrainBg
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            when (currentTab) {
                AppTab.HOME -> {
                    HomeScreen(
                        selectedVehicleType = vehicleState.vehicleType,
                        onSelectVehicleType = { viewModel.setVehicleType(it) },
                        onStartNavigation = { viewModel.startNavigation() }
                    )
                }
                AppTab.NAVIGATION -> {
                    NavigationScreen(
                        state = vehicleState,
                        onStartSimulation = { viewModel.startSimulation() },
                        onStopSimulation = { viewModel.stopSimulation() },
                        onTogglePause = { viewModel.togglePauseResume() },
                        onRestart = { viewModel.restartSimulation() },
                        onOpenAudit = { viewModel.selectTab(AppTab.AUDIT) }
                    )
                }
                AppTab.AUDIT -> {
                    AuditScreen(
                        auditResults = auditResults,
                        onRunAgain = { viewModel.startNavigation() }
                    )
                }
            }
        }
    }
}

@Composable
private fun BottomTabBar(
    currentTab: AppTab,
    onSelectTab: (AppTab) -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .shadow(8.dp)
            .background(SurfaceCardLight)
            .border(1.dp, BorderLight)
            .padding(horizontal = 14.dp, vertical = 8.dp)
    ) {
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.SpaceAround,
            verticalAlignment = Alignment.CenterVertically
        ) {
            TabItem(
                label = "HOME",
                isSelected = currentTab == AppTab.HOME,
                onClick = { onSelectTab(AppTab.HOME) }
            )

            TabItem(
                label = "NAVIGATION",
                isSelected = currentTab == AppTab.NAVIGATION,
                onClick = { onSelectTab(AppTab.NAVIGATION) }
            )

            TabItem(
                label = "AUDIT",
                isSelected = currentTab == AppTab.AUDIT,
                onClick = { onSelectTab(AppTab.AUDIT) }
            )
        }
    }
}

@Composable
private fun TabItem(
    label: String,
    isSelected: Boolean,
    onClick: () -> Unit
) {
    val bgColor = if (isSelected) Color(0xFFE8F0FE) else Color.Transparent
    val textColor = if (isSelected) NavRouteBlue else TextMutedDark
    val borderColor = if (isSelected) NavRouteBlue.copy(alpha = 0.4f) else Color.Transparent

    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(bgColor)
            .border(1.dp, borderColor, RoundedCornerShape(8.dp))
            .clickable { onClick() }
            .padding(horizontal = 18.dp, vertical = 7.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = label,
            color = textColor,
            fontSize = 11.sp,
            fontWeight = if (isSelected) FontWeight.Bold else FontWeight.Medium,
            fontFamily = FontFamily.Monospace,
            letterSpacing = 1.sp
        )
    }
}
