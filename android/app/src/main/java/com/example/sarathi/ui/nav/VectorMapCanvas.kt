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
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.RoundRect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.unit.dp
import com.example.sarathi.model.NavigationMode
import com.example.sarathi.model.VehicleState
import com.example.sarathi.model.VehicleType
import com.example.sarathi.theme.GhostDriftLine
import com.example.sarathi.theme.GhostDriftRed
import com.example.sarathi.theme.MapLaneActive
import com.example.sarathi.theme.MapLaneBorder
import com.example.sarathi.theme.MapRoadAsphalt
import com.example.sarathi.theme.MapRoadBorder
import com.example.sarathi.theme.MapRoadMarking
import com.example.sarathi.theme.MapRoadTunnel
import com.example.sarathi.theme.MapTerrainBg
import com.example.sarathi.theme.NavRouteBlue
import com.example.sarathi.theme.StatusGnssGreen
import com.example.sarathi.theme.TunnelOverlayLight

@Composable
fun VectorMapCanvas(
    state: VehicleState,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition(label = "roadMovement")
    
    // Only animate road motion when the vehicle is moving forward
    val animationDuration = if (state.isRunning && state.speedKmh > 1.0f) {
        (650.0f * (48.0f / state.speedKmh)).toInt().coerceIn(300, 1500)
    } else {
        10000000 // Stationary
    }

    val dashPhase by infiniteTransition.animateFloat(
        initialValue = 0.0f,
        targetValue = 120.0f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = animationDuration, easing = LinearEasing),
            repeatMode = RepeatMode.Restart
        ),
        label = "roadDashMovement"
    )

    Canvas(modifier = modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height

        val roadWidth = w * 0.76f
        val roadLeft = (w - roadWidth) / 2.0f
        val roadRight = roadLeft + roadWidth
        val roadCenterX = w / 2.0f
        val laneWidth = roadWidth / 2.0f

        // 1. Map Terrain Canvas (Google Maps Soft Light Terrain)
        drawRect(
            color = MapTerrainBg,
            topLeft = Offset(0.0f, 0.0f),
            size = size
        )

        // Surrounding subtle building / city block grid
        for (i in 0..5) {
            val blockY = h * (i * 0.18f)
            drawRoundRect(
                color = Color(0xFFE2E8F0),
                topLeft = Offset(12.0f, blockY + 10.0f),
                size = Size(roadLeft - 24.0f, h * 0.12f),
                cornerRadius = CornerRadius(8.0f, 8.0f)
            )
            drawRoundRect(
                color = Color(0xFFE2E8F0),
                topLeft = Offset(roadRight + 12.0f, blockY + 10.0f),
                size = Size(roadLeft - 24.0f, h * 0.12f),
                cornerRadius = CornerRadius(8.0f, 8.0f)
            )
        }

        // 2. Road Surface (Clean White Road or Soft Blue-Gray in Tunnel)
        val roadSurfaceColor = if (state.inTunnel) MapRoadTunnel else MapRoadAsphalt
        drawRect(
            color = roadSurfaceColor,
            topLeft = Offset(roadLeft, 0.0f),
            size = Size(roadWidth, h)
        )

        // 3. Road Outer Borders (Google Maps Clean Curb Lines)
        val curbBorderColor = if (state.inTunnel) Color(0xFF94A3B8) else MapRoadBorder
        drawLine(
            color = curbBorderColor,
            start = Offset(roadLeft, 0.0f),
            end = Offset(roadLeft, h),
            strokeWidth = 2.dp.toPx()
        )
        drawLine(
            color = curbBorderColor,
            start = Offset(roadRight, 0.0f),
            end = Offset(roadRight, h),
            strokeWidth = 2.dp.toPx()
        )

        // 4. Translucent Active Lane Corridor (OSM Map Clamping)
        val activeLaneLeft = roadCenterX - (laneWidth / 2.0f)
        drawRect(
            color = MapLaneActive,
            topLeft = Offset(activeLaneLeft, 0.0f),
            size = Size(laneWidth, h)
        )
        drawLine(
            color = MapLaneBorder.copy(alpha = 0.35f),
            start = Offset(activeLaneLeft, 0.0f),
            end = Offset(activeLaneLeft, h),
            strokeWidth = 1.dp.toPx()
        )
        drawLine(
            color = MapLaneBorder.copy(alpha = 0.35f),
            start = Offset(activeLaneLeft + laneWidth, 0.0f),
            end = Offset(activeLaneLeft + laneWidth, h),
            strokeWidth = 1.dp.toPx()
        )

        // 5. Forward Moving Center Dashed Line (Flows downward to simulate forward vehicle movement)
        val strokeWidth = 3.dp.toPx()
        val effectiveDashPhase = if (state.isRunning) dashPhase else 0.0f
        val dashEffect = PathEffect.dashPathEffect(floatArrayOf(40.0f, 32.0f), effectiveDashPhase)
        drawLine(
            color = MapRoadMarking,
            start = Offset(roadCenterX, 0.0f),
            end = Offset(roadCenterX, h),
            strokeWidth = strokeWidth,
            pathEffect = dashEffect
        )

        // 6. Tunnel Shading & Overhead Arch Portals
        if (state.inTunnel) {
            drawRect(
                color = TunnelOverlayLight,
                topLeft = Offset(0.0f, 0.0f),
                size = size
            )

            // Overhead tunnel portal lines
            for (i in 1..4) {
                val archY = h * (i * 0.22f)
                drawLine(
                    color = Color(0xFF64748B).copy(alpha = 0.4f),
                    start = Offset(roadLeft - 10.0f, archY),
                    end = Offset(roadRight + 10.0f, archY),
                    strokeWidth = 2.0f
                )
            }
        }

        // Vehicle Base Position on Screen
        val carCenterY = h * 0.68f
        val carX = roadCenterX + (state.carLateralOffsetRatio * (laneWidth * 0.5f))

        // 7. Algorithmic Proof 1: Red Ghost Marker (Raw Uncorrected IMU Drift)
        if (state.inTunnel && state.ghostLateralOffsetRatio > 0.05f) {
            val ghostX = roadCenterX + (state.ghostLateralOffsetRatio * (laneWidth * 0.5f))
            drawGhostDrift(ghostX, carCenterY)
        }

        // 8. Algorithmic Proof 2: 15-State ESKF Covariance Halo
        val haloRadiusPx = state.covarianceHaloRadiusDp.dp.toPx()
        drawCircle(
            color = NavRouteBlue.copy(alpha = 0.18f),
            center = Offset(carX, carCenterY),
            radius = haloRadiusPx * 1.5f
        )
        drawCircle(
            color = NavRouteBlue.copy(alpha = 0.6f),
            center = Offset(carX, carCenterY),
            radius = haloRadiusPx,
            style = Stroke(width = 1.5.dp.toPx())
        )

        // 9. Algorithmic Proof 3: Pothole Shock Pulse
        if (state.potholePulsing) {
            drawCircle(
                color = Color(0xFFD97706).copy(alpha = 0.4f),
                center = Offset(carX, carCenterY),
                radius = 42.dp.toPx(),
                style = Stroke(width = 2.dp.toPx())
            )
        }

        // 10. Algorithmic Proof 4: Soft Recovery Magnetic Guide
        if (state.mode == NavigationMode.RECOVERING) {
            val targetGpsX = roadCenterX
            drawLine(
                color = StatusGnssGreen,
                start = Offset(carX, carCenterY),
                end = Offset(targetGpsX, carCenterY - 35.0f),
                strokeWidth = 2.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(8.0f, 8.0f))
            )
            drawCircle(
                color = StatusGnssGreen,
                center = Offset(targetGpsX, carCenterY - 35.0f),
                radius = 5.dp.toPx()
            )
        }

        // 11. Vehicle Custom Icon (Car vs 2-Wheeler Scooter)
        rotate(degrees = state.roadCurveDegrees, pivot = Offset(carX, carCenterY)) {
            when (state.vehicleType) {
                VehicleType.CAR_4W -> drawCarIcon(carX, carCenterY, state.inTunnel)
                VehicleType.BIKE_2W -> drawScooterIcon(carX, carCenterY, state.inTunnel)
            }
        }
    }
}

private fun DrawScope.drawCarIcon(x: Float, y: Float, inTunnel: Boolean) {
    val carW = 26.0f
    val carH = 50.0f
    val primaryColor = if (inTunnel) Color(0xFF0284C7) else Color(0xFF1A73E8)

    // Car Shadow
    drawRoundRect(
        color = Color(0x33000000),
        topLeft = Offset(x - (carW / 2.0f) + 2.0f, y - (carH / 2.0f) + 4.0f),
        size = Size(carW, carH),
        cornerRadius = CornerRadius(7.0f, 7.0f)
    )

    // Car Body (Chassis)
    drawRoundRect(
        color = primaryColor,
        topLeft = Offset(x - (carW / 2.0f), y - (carH / 2.0f)),
        size = Size(carW, carH),
        cornerRadius = CornerRadius(6.0f, 6.0f)
    )

    // Windshield (Front glass)
    val glassColor = Color(0xFFE0F2FE)
    drawRoundRect(
        color = glassColor,
        topLeft = Offset(x - (carW * 0.38f), y - (carH * 0.32f)),
        size = Size(carW * 0.76f, carH * 0.22f),
        cornerRadius = CornerRadius(3.0f, 3.0f)
    )

    // Roof
    drawRoundRect(
        color = primaryColor.copy(alpha = 0.9f),
        topLeft = Offset(x - (carW * 0.34f), y - (carH * 0.08f)),
        size = Size(carW * 0.68f, carH * 0.24f),
        cornerRadius = CornerRadius(3.0f, 3.0f)
    )

    // Rear Window
    drawRoundRect(
        color = glassColor,
        topLeft = Offset(x - (carW * 0.34f), y + (carH * 0.18f)),
        size = Size(carW * 0.68f, carH * 0.14f),
        cornerRadius = CornerRadius(2.0f, 2.0f)
    )

    // Front Headlights
    drawCircle(color = Color(0xFFFEF08A), center = Offset(x - (carW * 0.35f), y - (carH * 0.46f)), radius = 2.5f)
    drawCircle(color = Color(0xFFFEF08A), center = Offset(x + (carW * 0.35f), y - (carH * 0.46f)), radius = 2.5f)
}

private fun DrawScope.drawScooterIcon(x: Float, y: Float, inTunnel: Boolean) {
    val scooterW = 18.0f
    val scooterH = 46.0f
    val primaryColor = if (inTunnel) Color(0xFF0284C7) else Color(0xFF1E8E3E)

    // Scooter Shadow
    drawRoundRect(
        color = Color(0x33000000),
        topLeft = Offset(x - (scooterW / 2.0f) + 2.0f, y - (scooterH / 2.0f) + 3.0f),
        size = Size(scooterW, scooterH),
        cornerRadius = CornerRadius(6.0f, 6.0f)
    )

    // Scooter Body (Slim chassis)
    drawRoundRect(
        color = primaryColor,
        topLeft = Offset(x - (scooterW * 0.32f), y - (scooterH * 0.40f)),
        size = Size(scooterW * 0.64f, scooterH * 0.80f),
        cornerRadius = CornerRadius(5.0f, 5.0f)
    )

    // Handlebars
    drawLine(
        color = Color(0xFF1E293B),
        start = Offset(x - (scooterW * 0.60f), y - (scooterH * 0.35f)),
        end = Offset(x + (scooterW * 0.60f), y - (scooterH * 0.35f)),
        strokeWidth = 3.0f
    )

    // Seat
    drawRoundRect(
        color = Color(0xFF0F172A),
        topLeft = Offset(x - (scooterW * 0.28f), y - (scooterH * 0.05f)),
        size = Size(scooterW * 0.56f, scooterH * 0.36f),
        cornerRadius = CornerRadius(4.0f, 4.0f)
    )

    // Rider Helmet Outline
    drawCircle(
        color = Color(0xFFF8FAFC),
        center = Offset(x, y - (scooterH * 0.12f)),
        radius = 5.5f
    )
    drawCircle(
        color = Color(0xFF0F172A),
        center = Offset(x, y - (scooterH * 0.12f)),
        radius = 5.5f,
        style = Stroke(width = 1.2f)
    )

    // Front Headlight
    drawCircle(color = Color(0xFFFEF08A), center = Offset(x, y - (scooterH * 0.46f)), radius = 2.8f)
}

private fun DrawScope.drawGhostDrift(ghostX: Float, y: Float) {
    drawLine(
        color = GhostDriftLine,
        start = Offset(ghostX - 18.0f, y + 20.0f),
        end = Offset(ghostX, y),
        strokeWidth = 2.0f,
        pathEffect = PathEffect.dashPathEffect(floatArrayOf(6.0f, 6.0f))
    )

    drawCircle(
        color = GhostDriftRed,
        center = Offset(ghostX, y),
        radius = 10.0f
    )
    drawCircle(
        color = Color(0xFFDC2626),
        center = Offset(ghostX, y),
        radius = 10.0f,
        style = Stroke(width = 1.5f)
    )
}
