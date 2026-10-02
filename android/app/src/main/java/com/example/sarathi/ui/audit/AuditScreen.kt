package com.example.sarathi.ui.audit

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
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
import com.example.sarathi.model.AuditResults
import com.example.sarathi.theme.ChipBg
import com.example.sarathi.theme.ChipBorder
import com.example.sarathi.theme.MapBg
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.theme.StatusGreen
import com.example.sarathi.theme.StatusRed
import com.example.sarathi.theme.TextChip
import com.example.sarathi.theme.TextPrimary
import com.example.sarathi.theme.TextSecondary

@Composable
fun AuditScreen(
    auditResults: AuditResults,
    onRunAgain: () -> Unit
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
                text = "A  SAARTHI  AUDIT",
                color = TextPrimary,
                fontSize = 20.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 2.sp
            )
            Text(
                text = "ISRO SIH26168 Benchmark Results",
                color = TextSecondary,
                fontSize = 12.sp
            )
        }

        // Overall pass banner
        val allPassed = auditResults.driftPassed && auditResults.crossTrackPassed && auditResults.recoveryStepPassed
        PassBanner(allPassed)

        // Benchmark metrics
        SectionLabel("ACCURACY BENCHMARKS")
        Column(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            MetricCard(
                label = "Cumulative Position Drift",
                achieved = "%.2f%%".format(auditResults.driftPercentAchieved),
                target = "< %.1f%%".format(auditResults.driftPercentTarget),
                passed = auditResults.driftPassed
            )
            MetricCard(
                label = "Cross-Track Lane Error",
                achieved = "%.2f m".format(auditResults.crossTrackAchievedMeters),
                target = "< %.2f m".format(auditResults.crossTrackTargetMeters),
                passed = auditResults.crossTrackPassed
            )
            MetricCard(
                label = "Recovery Step Offset",
                achieved = "%.3f m".format(auditResults.recoveryStepAchievedMeters),
                target = "< %.2f m".format(auditResults.recoveryStepTargetMeters),
                passed = auditResults.recoveryStepPassed
            )
        }

        // System telemetry
        SectionLabel("SYSTEM TELEMETRY")
        TelemetryGrid(auditResults)

        Spacer(modifier = Modifier.height(8.dp))

        Button(
            onClick = onRunAgain,
            modifier = Modifier.fillMaxWidth().height(56.dp),
            shape = RoundedCornerShape(16.dp),
            colors = ButtonDefaults.buttonColors(containerColor = PuckBlue)
        ) {
            Text("RUN AGAIN", color = Color.White, fontWeight = FontWeight.Bold, fontSize = 15.sp, letterSpacing = 1.5.sp)
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
private fun PassBanner(passed: Boolean) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(14.dp))
            .background(if (passed) StatusGreen.copy(alpha = 0.15f) else StatusRed.copy(alpha = 0.15f))
            .padding(16.dp),
        contentAlignment = Alignment.Center
    ) {
        Text(
            text = if (passed) "ALL BENCHMARKS PASSED" else "SOME BENCHMARKS FAILED",
            color = if (passed) StatusGreen else StatusRed,
            fontSize = 14.sp,
            fontWeight = FontWeight.Bold,
            letterSpacing = 1.sp
        )
    }
}

@Composable
private fun MetricCard(label: String, achieved: String, target: String, passed: Boolean) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(ChipBg)
            .padding(16.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(text = label, color = TextSecondary, fontSize = 10.sp, fontWeight = FontWeight.Medium, letterSpacing = 0.3.sp)
            Spacer(modifier = Modifier.height(4.dp))
            Text(text = achieved, color = TextPrimary, fontSize = 18.sp, fontWeight = FontWeight.Bold)
            Text(text = "Target: $target", color = TextSecondary, fontSize = 10.sp)
        }
        Box(
            modifier = Modifier
                .clip(RoundedCornerShape(8.dp))
                .background(if (passed) StatusGreen.copy(alpha = 0.15f) else StatusRed.copy(alpha = 0.15f))
                .padding(horizontal = 12.dp, vertical = 6.dp)
        ) {
            Text(
                text = if (passed) "PASSED" else "FAILED",
                color = if (passed) StatusGreen else StatusRed,
                fontSize = 10.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 0.5.sp
            )
        }
    }
}

@Composable
private fun TelemetryGrid(r: AuditResults) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .clip(RoundedCornerShape(12.dp))
            .background(ChipBg)
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(10.dp)
    ) {
        TelemetryRow("Distance Traveled", "%.2f m".format(r.distanceTraveledMeters))
        TelemetryRow("Filter Frequency", "%.1f Hz".format(r.filterFrequencyHz))
        TelemetryRow("AI Inference Latency", "%.1f us".format(r.aiInferenceLatencyUs))
        TelemetryRow("AI Model Size", "${r.aiModelSizeBytes / 1024} KB")
        TelemetryRow("NavIC Satellites Locked", "${r.navicSatellitesLocked}")
        TelemetryRow("Potholes Dampened", "${r.potholesDampedCount}")
    }
}

@Composable
private fun TelemetryRow(label: String, value: String) {
    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(text = label, color = TextSecondary, fontSize = 11.sp)
        Text(text = value, color = TextChip, fontSize = 11.sp, fontWeight = FontWeight.SemiBold)
    }
}
