package com.example.sarathi.ui.audit

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
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.rememberScrollState
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
import com.example.sarathi.model.AuditResults
import com.example.sarathi.theme.HomeBg
import com.example.sarathi.theme.HomeBorder
import com.example.sarathi.theme.HomeCardBg
import com.example.sarathi.theme.HomeSubtext
import com.example.sarathi.theme.HomeText
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.theme.StatusGreen
import com.example.sarathi.theme.StatusRed

@Composable
fun AuditScreen(
    auditResults: AuditResults,
    onRunAgain: () -> Unit,
    onBack: () -> Unit = {},
    onLogoClick: () -> Unit = {}
) {
    Column(
        modifier = Modifier.fillMaxSize().background(HomeBg)
            .verticalScroll(rememberScrollState()).padding(horizontal = 20.dp, vertical = 14.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp)
    ) {
        // Top Bar: Back Button + Clickable Logo
        Row(
            modifier = Modifier.fillMaxWidth(),
            verticalAlignment = Alignment.CenterVertically
        ) {
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
            Spacer(Modifier.width(12.dp))
            Row(
                modifier = Modifier.clickable { onLogoClick() },
                verticalAlignment = Alignment.CenterVertically
            ) {
                Text("▲", color = PuckBlue, fontSize = 20.sp, fontWeight = FontWeight.Bold)
                Spacer(Modifier.width(8.dp))
                Column {
                    Text("SAARTHI", color = HomeText, fontSize = 20.sp, fontWeight = FontWeight.Black, letterSpacing = 1.2.sp)
                    Text("MISSION AUDIT REPORT", color = HomeSubtext, fontSize = 10.sp, fontWeight = FontWeight.SemiBold, letterSpacing = 1.sp)
                }
            }
        }

        val allPassed = auditResults.driftPassed && auditResults.crossTrackPassed && auditResults.recoveryStepPassed
        Box(
            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp))
                .background((if (allPassed) StatusGreen else StatusRed).copy(alpha = 0.12f))
                .border(1.dp, (if (allPassed) StatusGreen else StatusRed).copy(alpha = 0.4f), RoundedCornerShape(14.dp))
                .padding(16.dp),
            contentAlignment = Alignment.Center
        ) {
            Text(
                if (allPassed) "✓  ALL BENCHMARKS PASSED" else "✗  SOME BENCHMARKS FAILED",
                color = if (allPassed) StatusGreen else StatusRed,
                fontSize = 14.sp, fontWeight = FontWeight.Bold
            )
        }

        SectionLabel("ACCURACY BENCHMARKS")
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            MetricCard("Cumulative Position Drift", "%.2f%%".format(auditResults.driftPercentAchieved),
                "< %.1f%%".format(auditResults.driftPercentTarget), auditResults.driftPassed)
            MetricCard("Cross-Track Lane Error", "%.2f m".format(auditResults.crossTrackAchievedMeters),
                "< %.2f m".format(auditResults.crossTrackTargetMeters), auditResults.crossTrackPassed)
            MetricCard("Recovery Step Offset", "%.3f m".format(auditResults.recoveryStepAchievedMeters),
                "< %.2f m".format(auditResults.recoveryStepTargetMeters), auditResults.recoveryStepPassed)
        }

        SectionLabel("SYSTEM TELEMETRY")
        Column(
            modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(14.dp))
                .background(HomeCardBg).border(1.dp, HomeBorder, RoundedCornerShape(14.dp)).padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(10.dp)
        ) {
            TelRow("Distance Traveled", "%.2f m".format(auditResults.distanceTraveledMeters))
            TelRow("Filter Frequency", "%.1f Hz".format(auditResults.filterFrequencyHz))
            TelRow("AI Inference Latency", "%.1f µs".format(auditResults.aiInferenceLatencyUs))
            TelRow("AI Model Size", "${auditResults.aiModelSizeBytes / 1024} KB")
            TelRow("NavIC Satellites Locked", "${auditResults.navicSatellitesLocked}")
            TelRow("Potholes Dampened", "${auditResults.potholesDampedCount}")
        }

        Spacer(Modifier.height(4.dp))

        Button(
            onClick = onRunAgain,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = PuckBlue)
        ) {
            Text("RUN AGAIN", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 15.sp, letterSpacing = 1.sp)
        }

        Spacer(Modifier.height(8.dp))
    }
}

@Composable
private fun SectionLabel(text: String) {
    Text(text, color = HomeSubtext, fontSize = 10.sp, fontWeight = FontWeight.Bold, letterSpacing = 1.5.sp)
}

@Composable
private fun MetricCard(label: String, achieved: String, target: String, passed: Boolean) {
    Row(
        modifier = Modifier.fillMaxWidth().clip(RoundedCornerShape(12.dp))
            .background(HomeCardBg).border(1.dp, HomeBorder, RoundedCornerShape(12.dp)).padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(Modifier.weight(1f)) {
            Text(label, color = HomeSubtext, fontSize = 10.sp, fontWeight = FontWeight.Medium)
            Spacer(Modifier.height(4.dp))
            Text(achieved, color = HomeText, fontSize = 20.sp, fontWeight = FontWeight.Bold)
            Text("Target: $target", color = HomeSubtext, fontSize = 10.sp)
        }
        Box(
            modifier = Modifier.clip(RoundedCornerShape(8.dp))
                .background((if (passed) StatusGreen else StatusRed).copy(alpha = 0.12f))
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Text(
                if (passed) "PASSED" else "FAILED",
                color = if (passed) StatusGreen else StatusRed,
                fontSize = 10.sp, fontWeight = FontWeight.Bold
            )
        }
    }
}

@Composable
private fun TelRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, color = HomeSubtext, fontSize = 11.sp)
        Text(value, color = HomeText, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
    }
}
