package com.example.sarathi.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.sarathi.theme.ChipBg
import com.example.sarathi.theme.ChipBorder
import com.example.sarathi.theme.MapBg
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.theme.TextPrimary
import com.example.sarathi.theme.TextSecondary
import com.example.sarathi.ui.audit.AuditScreen
import com.example.sarathi.ui.home.HomeScreen
import com.example.sarathi.ui.nav.NavigationScreen
import com.example.sarathi.viewmodel.AppTab
import com.example.sarathi.viewmodel.SarathiViewModel

@Composable
fun SarathiApp(
    viewModel: SarathiViewModel = viewModel()
) {
    val currentTab   by viewModel.currentTab.collectAsState()
    val vehicleState by viewModel.vehicleState.collectAsState()
    val auditResults by viewModel.auditResults.collectAsState()

    Scaffold(
        bottomBar = {
            BottomTabBar(
                currentTab    = currentTab,
                onSelectTab   = { viewModel.selectTab(it) }
            )
        },
        containerColor = MapBg
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
        ) {
            when (currentTab) {
                AppTab.HOME -> HomeScreen(
                    selectedVehicleType  = vehicleState.vehicleType,
                    onSelectVehicleType  = { viewModel.setVehicleType(it) },
                    onStartNavigation    = { viewModel.startNavigation() }
                )
                AppTab.NAVIGATION -> NavigationScreen(
                    state              = vehicleState,
                    onStartSimulation  = { viewModel.startSimulation() },
                    onStopSimulation   = { viewModel.stopSimulation() },
                    onTogglePause      = { viewModel.togglePauseResume() },
                    onRestart          = { viewModel.restartSimulation() },
                    onOpenAudit        = { viewModel.selectTab(AppTab.AUDIT) }
                )
                AppTab.AUDIT -> AuditScreen(
                    auditResults = auditResults,
                    onRunAgain   = { viewModel.startNavigation() }
                )
            }
        }
    }
}

@Composable
private fun BottomTabBar(
    currentTab: AppTab,
    onSelectTab: (AppTab) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .background(MapBg)
            .padding(horizontal = 20.dp, vertical = 10.dp),
        horizontalArrangement = Arrangement.SpaceAround,
        verticalAlignment = Alignment.CenterVertically
    ) {
        listOf(
            AppTab.HOME       to "HOME",
            AppTab.NAVIGATION to "NAV",
            AppTab.AUDIT      to "AUDIT"
        ).forEach { (tab, label) ->
            val selected = currentTab == tab
            Box(
                modifier = Modifier
                    .clip(RoundedCornerShape(10.dp))
                    .background(if (selected) PuckBlue else Color.Transparent)
                    .clickable { onSelectTab(tab) }
                    .padding(horizontal = 24.dp, vertical = 10.dp),
                contentAlignment = Alignment.Center
            ) {
                Text(
                    text = label,
                    color = if (selected) TextPrimary else TextSecondary,
                    fontSize = 11.sp,
                    fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                    letterSpacing = 1.sp
                )
            }
        }
    }
}
