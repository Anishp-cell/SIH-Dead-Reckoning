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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.example.sarathi.theme.HomeBg
import com.example.sarathi.theme.HomeBorder
import com.example.sarathi.theme.HomeCardBg
import com.example.sarathi.theme.HomeSubtext
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.ui.audit.AuditScreen
import com.example.sarathi.ui.home.HomeScreen
import com.example.sarathi.ui.nav.NavigationScreen
import com.example.sarathi.ui.splash.SplashScreen
import com.example.sarathi.viewmodel.AppTab
import com.example.sarathi.viewmodel.SarathiViewModel

@Composable
fun SarathiApp(viewModel: SarathiViewModel = viewModel()) {
    val currentTab     by viewModel.currentTab.collectAsState()
    val vehicleState   by viewModel.vehicleState.collectAsState()
    val auditResults   by viewModel.auditResults.collectAsState()
    val preFlightStep  by viewModel.preFlightStep.collectAsState()

    Scaffold(
        bottomBar = {
            if (currentTab != AppTab.SPLASH) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(HomeCardBg)
                        .border(1.dp, HomeBorder, RoundedCornerShape(topStart = 0.dp, topEnd = 0.dp))
                        .padding(horizontal = 20.dp, vertical = 8.dp),
                    horizontalArrangement = Arrangement.SpaceAround,
                    verticalAlignment     = Alignment.CenterVertically
                ) {
                    listOf(AppTab.HOME to "HOME", AppTab.NAVIGATION to "NAV", AppTab.AUDIT to "AUDIT")
                        .forEach { (tab, label) ->
                            val selected = currentTab == tab
                            Box(
                                modifier = Modifier
                                    .clip(RoundedCornerShape(10.dp))
                                    .background(if (selected) PuckBlue else Color.Transparent)
                                    .clickable { viewModel.selectTab(tab) }
                                    .padding(horizontal = 28.dp, vertical = 8.dp),
                                contentAlignment = Alignment.Center
                            ) {
                                Text(
                                    text       = label,
                                    color      = if (selected) Color.White else HomeSubtext,
                                    fontSize   = 11.sp,
                                    fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                                    letterSpacing = 1.sp
                                )
                            }
                        }
                }
            }
        },
        containerColor = HomeBg
    ) { innerPadding ->
        Box(
            Modifier
                .fillMaxSize()
                .padding(if (currentTab == AppTab.SPLASH) androidx.compose.foundation.layout.PaddingValues(0.dp) else innerPadding)
        ) {
            when (currentTab) {
                AppTab.SPLASH ->
                    SplashScreen(
                        onSplashFinished = { viewModel.selectTab(AppTab.HOME) }
                    )
                AppTab.HOME ->
                    HomeScreen(
                        selectedVehicleType = vehicleState.vehicleType,
                        onSelectVehicleType = { viewModel.setVehicleType(it) },
                        onStartNavigation   = { viewModel.runPreFlightChecksAndNavigate() },
                        preFlightStep       = preFlightStep,
                        onDismissPreFlight  = { viewModel.dismissPreFlightChecks() },
                        onBack              = { viewModel.goBack() },
                        onLogoClick         = { viewModel.resetToIntro() }
                    )
                AppTab.NAVIGATION ->
                    NavigationScreen(
                        state               = vehicleState,
                        onStartSimulation   = { viewModel.startSimulation() },
                        onStopSimulation    = { viewModel.stopSimulation() },
                        onTogglePause       = { viewModel.togglePauseResume() },
                        onRestart           = { viewModel.restartSimulation() },
                        onOpenAudit         = { viewModel.selectTab(AppTab.AUDIT) },
                        onToggleDiagnostics = { viewModel.toggleDiagnosticsPanel() },
                        onBack              = { viewModel.goBack() },
                        onLogoClick         = { viewModel.resetToIntro() }
                    )
                AppTab.AUDIT ->
                    AuditScreen(
                        auditResults = auditResults,
                        onRunAgain   = { viewModel.startNavigation() },
                        onBack       = { viewModel.goBack() },
                        onLogoClick  = { viewModel.resetToIntro() }
                    )
            }
        }
    }
}
