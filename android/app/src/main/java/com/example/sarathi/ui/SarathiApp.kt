package com.example.sarathi.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
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
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.sarathi.theme.AccentCyan
import com.example.sarathi.theme.AccentCyanSubtle
import com.example.sarathi.theme.BgBase
import com.example.sarathi.theme.BgSurface
import com.example.sarathi.theme.BorderSubtle
import com.example.sarathi.theme.TextMuted
import com.example.sarathi.theme.TextPrimary
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
        containerColor = BgBase
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
                        onRestart = { viewModel.restartSimulation() },
                        onTogglePause = { viewModel.togglePauseResume() },
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
            .background(BgSurface)
            .border(1.dp, BorderSubtle)
            .padding(horizontal = 16.dp, vertical = 8.dp)
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
    val bgColor = if (isSelected) AccentCyanSubtle else BgSurface
    val textColor = if (isSelected) AccentCyan else TextMuted
    val borderColor = if (isSelected) AccentCyan.copy(alpha = 0.5f) else BorderSubtle

    Box(
        modifier = Modifier
            .clip(RoundedCornerShape(8.dp))
            .background(bgColor)
            .border(1.dp, borderColor, RoundedCornerShape(8.dp))
            .clickable { onClick() }
            .padding(horizontal = 18.dp, vertical = 8.dp),
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
