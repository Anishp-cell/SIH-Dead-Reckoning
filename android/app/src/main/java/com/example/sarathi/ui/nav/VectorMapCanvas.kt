package com.example.sarathi.ui.nav

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.unit.dp
import com.example.sarathi.model.NavigationMode
import com.example.sarathi.model.VehicleState
import com.example.sarathi.theme.AccentCyan
import com.example.sarathi.theme.AccentCyanGlow
import com.example.sarathi.theme.GhostDriftLine
import com.example.sarathi.theme.GhostDriftRed
import com.example.sarathi.theme.RoadAsphalt
import com.example.sarathi.theme.RoadLaneActive
import com.example.sarathi.theme.RoadMarking
import com.example.sarathi.theme.RoadTunnelAsphalt
import com.example.sarathi.theme.StatusGnss
import com.example.sarathi.theme.TunnelCanopy

@Composable
fun VectorMapCanvas(
    state: VehicleState,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition(label = "roadLines")
    val dashPhase by infiniteTransition.animateFloat(
        initialValue = 0.0f,
        targetValue = 120.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 650, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "dashMovement"
    )

    Canvas(modifier = modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height

        val roadWidth = w * 0.72f
        val roadLeft = (w - roadWidth) / 2.0f
        val roadRight = roadLeft + roadWidth
        val roadCenterX = w / 2.0f
        val laneWidth = roadWidth / 2.0f

        // 1. Draw Road Asphalt Background
        val roadColor = if (state.inTunnel) RoadTunnelAsphalt else RoadAsphalt
        drawRect(
            color = roadColor,
            topLeft = Offset(roadLeft, 0.0f),
            size = Size(roadWidth, h)
        )

        // 2. Translucent Lane Corridor (OSM Map Clamping Visualization)
        val activeLaneLeft = roadCenterX - (laneWidth / 2.0f)
        drawRect(
            color = RoadLaneActive,
            topLeft = Offset(activeLaneLeft, 0.0f),
            size = Size(laneWidth, h)
        )

        // 3. Road Side Barriers / Curbs
        val barrierWidth = 6.dp.toPx()
        val barrierColor = if (state.inTunnel) Color(0xFF1E293B) else Color(0xFF334155)
        drawRect(color = barrierColor, topLeft = Offset(roadLeft - barrierWidth, 0.0f), size = Size(barrierWidth, h))
        drawRect(color = barrierColor, topLeft = Offset(roadRight, 0.0f), size = Size(barrierWidth, h))

        // 4. Moving Center Dashed Line (Simulates forward velocity)
        val strokeWidth = 3.dp.toPx()
        val dashEffect = PathEffect.dashPathEffect(floatArrayOf(45.0f, 35.0f), dashPhase)
        drawLine(
            color = RoadMarking,
            start = Offset(roadCenterX, 0.0f),
            end = Offset(roadCenterX, h),
            strokeWidth = strokeWidth,
            pathEffect = dashEffect
        )

        // Lane Left and Right boundary lines
        drawLine(
            color = RoadMarking.copy(alpha = 0.4f),
            start = Offset(roadLeft + 12.0f, 0.0f),
            end = Offset(roadLeft + 12.0f, h),
            strokeWidth = 2.dp.toPx()
        )
        drawLine(
            color = RoadMarking.copy(alpha = 0.4f),
            start = Offset(roadRight - 12.0f, 0.0f),
            end = Offset(roadRight - 12.0f, h),
            strokeWidth = 2.dp.toPx()
        )

        // 5. Tunnel Shading & Canopy Vignette
        if (state.inTunnel) {
            drawRect(
                brush = Brush.verticalGradient(
                    colors = listOf(
                        TunnelCanopy,
                        Color(0xB3050811),
                        Color(0xCC050811)
                    )
                ),
                topLeft = Offset(0.0f, 0.0f),
                size = size
            )

            // Overhead tunnel portal arch lines
            for (i in 1..4) {
                val archY = h * (i * 0.22f)
                drawLine(
                    color = Color(0x2238BDF8),
                    start = Offset(roadLeft - 20.0f, archY),
                    end = Offset(roadRight + 20.0f, archY),
                    strokeWidth = 1.5f
                )
            }
        }

        // Vehicle Base Position
        val carCenterY = h * 0.70f
        val carX = roadCenterX + (state.carLateralOffsetRatio * (laneWidth * 0.5f))

        // 6. Algorithmic Proof 1: Red Ghost Marker (Raw Uncorrected IMU Drift)
        if (state.inTunnel && state.ghostLateralOffsetRatio > 0.05f) {
            val ghostX = roadCenterX + (state.ghostLateralOffsetRatio * (laneWidth * 0.5f))
            drawGhostDrift(ghostX, carCenterY)
        }

        // 7. Algorithmic Proof 2: 15-State ESKF Covariance Halo
        val haloRadiusPx = state.covarianceHaloRadiusDp.dp.toPx()
        drawCircle(
            brush = Brush.radialGradient(
                colors = listOf(AccentCyanGlow, Color.Transparent),
                center = Offset(carX, carCenterY),
                radius = haloRadiusPx * 1.5f
            ),
            center = Offset(carX, carCenterY),
            radius = haloRadiusPx * 1.5f
        )
        drawCircle(
            color = AccentCyan.copy(alpha = 0.5f),
            center = Offset(carX, carCenterY),
            radius = haloRadiusPx,
            style = Stroke(width = 1.5.dp.toPx())
        )

        // 8. Algorithmic Proof 3: Pothole Shock Ripple
        if (state.potholePulsing) {
            drawCircle(
                color = Color(0x66F59E0B),
                center = Offset(carX, carCenterY),
                radius = 48.dp.toPx(),
                style = Stroke(width = 2.dp.toPx())
            )
        }

        // 9. Algorithmic Proof 4: Soft Recovery Magnetic Guide (At tunnel exit)
        if (state.mode == NavigationMode.RECOVERING) {
            val targetGpsX = roadCenterX
            drawLine(
                color = StatusGnss.copy(alpha = 0.7f),
                start = Offset(carX, carCenterY),
                end = Offset(targetGpsX, carCenterY - 40.0f),
                strokeWidth = 2.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(10.0f, 10.0f))
            )
            drawCircle(
                color = StatusGnss,
                center = Offset(targetGpsX, carCenterY - 40.0f),
                radius = 6.dp.toPx()
            )
        }

        // 10. The Hero Vehicle Puck (Neon Cyan Heading-Interpolated Arrow)
        rotate(degrees = state.roadCurveDegrees, pivot = Offset(carX, carCenterY)) {
            drawVehiclePuck(carX, carCenterY, state.inTunnel)
        }
    }
}

private fun DrawScope.drawVehiclePuck(x: Float, y: Float, inTunnel: Boolean) {
    val puckHeight = 34.0f
    val puckHalfWidth = 14.0f

    val path = Path().apply {
        moveTo(x, y - puckHeight)
        lineTo(x + puckHalfWidth, y + (puckHeight * 0.4f))
        lineTo(x, y + (puckHeight * 0.15f))
        lineTo(x - puckHalfWidth, y + (puckHeight * 0.4f))
        close()
    }

    val primaryColor = if (inTunnel) AccentCyan else Color(0xFF0284C7)
    drawPath(path = path, color = primaryColor)

    // Inner dark aerodynamic core
    val corePath = Path().apply {
        moveTo(x, y - (puckHeight * 0.6f))
        lineTo(x + (puckHalfWidth * 0.5f), y + (puckHeight * 0.25f))
        lineTo(x, y + (puckHeight * 0.05f))
        lineTo(x - (puckHalfWidth * 0.5f), y + (puckHeight * 0.25f))
        close()
    }
    drawPath(path = corePath, color = Color(0xFF080D1A))
}

private fun DrawScope.drawGhostDrift(ghostX: Float, y: Float) {
    // Dotted red trail connecting center to drifting ghost
    drawLine(
        color = GhostDriftLine,
        start = Offset(ghostX - 20.0f, y + 25.0f),
        end = Offset(ghostX, y),
        strokeWidth = 2.0f,
        pathEffect = PathEffect.dashPathEffect(floatArrayOf(6.0f, 6.0f))
    )

    // Translucent ghost dot
    drawCircle(
        color = GhostDriftRed,
        center = Offset(ghostX, y),
        radius = 11.0f
    )
    drawCircle(
        color = Color(0xFFEF4444),
        center = Offset(ghostX, y),
        radius = 11.0f,
        style = Stroke(width = 1.5f)
    )
}
