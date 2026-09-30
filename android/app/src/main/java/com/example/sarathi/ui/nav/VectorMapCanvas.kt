package com.example.sarathi.ui.nav

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
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
    Canvas(modifier = modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height

        val roadWidth = w * 0.74f
        val roadLeft = (w - roadWidth) / 2.0f
        val roadRight = roadLeft + roadWidth
        val roadCenterX = w / 2.0f
        val laneWidth = roadWidth / 2.0f

        // Pixels traveled tied to real sensor fusion distance (35 px per meter)
        val distancePx = state.distanceTraveledMeters * 35.0f

        // 1. Map Terrain Canvas (Google Maps Soft Light Terrain)
        drawRect(
            color = MapTerrainBg,
            topLeft = Offset(0.0f, 0.0f),
            size = size
        )

        // 2. Dynamic Moving Scenery: City Blocks & Roadside Features scrolling downward
        val blockHeight = 110.0f
        val blockSpacing = 170.0f
        val sceneryScroll = distancePx % blockSpacing
        val numBlocks = (h / blockSpacing).toInt() + 3

        for (i in -1..numBlocks) {
            val blockY = (i * blockSpacing) + sceneryScroll
            val buildingWidth = roadLeft - 28.0f
            if (buildingWidth > 20.0f) {
                // Left building block
                drawRoundRect(
                    color = Color(0xFFE2E8F0),
                    topLeft = Offset(14.0f, blockY),
                    size = Size(buildingWidth, blockHeight),
                    cornerRadius = CornerRadius(10.0f, 10.0f)
                )
                // Left roadside green patch
                drawCircle(
                    color = Color(0xFFC6E7CE),
                    center = Offset(roadLeft - 10.0f, blockY + (blockHeight / 2.0f)),
                    radius = 8.0f
                )

                // Right building block
                drawRoundRect(
                    color = Color(0xFFE2E8F0),
                    topLeft = Offset(roadRight + 14.0f, blockY),
                    size = Size(buildingWidth, blockHeight),
                    cornerRadius = CornerRadius(10.0f, 10.0f)
                )
                // Right roadside green patch
                drawCircle(
                    color = Color(0xFFC6E7CE),
                    center = Offset(roadRight + 10.0f, blockY + (blockHeight / 2.0f)),
                    radius = 8.0f
                )
            }
        }

        // 3. Road Surface (Clean White Asphalt or Soft Tunnel Slate)
        val roadSurfaceColor = if (state.inTunnel) MapRoadTunnel else MapRoadAsphalt
        drawRect(
            color = roadSurfaceColor,
            topLeft = Offset(roadLeft, 0.0f),
            size = Size(roadWidth, h)
        )

        // 4. Moving Road Texture / Joints (subtle downward motion)
        val jointSpacing = 280.0f
        val jointScroll = distancePx % jointSpacing
        val numJoints = (h / jointSpacing).toInt() + 2
        for (j in -1..numJoints) {
            val jointY = (j * jointSpacing) + jointScroll
            drawLine(
                color = Color(0xFFE2E8F0).copy(alpha = 0.6f),
                start = Offset(roadLeft, jointY),
                end = Offset(roadRight, jointY),
                strokeWidth = 1.0f
            )
        }

        // 5. Road Outer Borders (Clean Curbs)
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

        // 6. Active Lane Clamping Corridor (Translucent blue guidance corridor)
        val activeLaneLeft = roadCenterX - (laneWidth / 2.0f)
        drawRect(
            color = MapLaneActive,
            topLeft = Offset(activeLaneLeft, 0.0f),
            size = Size(laneWidth, h)
        )
        drawLine(
            color = MapLaneBorder.copy(alpha = 0.4f),
            start = Offset(activeLaneLeft, 0.0f),
            end = Offset(activeLaneLeft, h),
            strokeWidth = 1.5.dp.toPx()
        )
        drawLine(
            color = MapLaneBorder.copy(alpha = 0.4f),
            start = Offset(activeLaneLeft + laneWidth, 0.0f),
            end = Offset(activeLaneLeft + laneWidth, h),
            strokeWidth = 1.5.dp.toPx()
        )

        // 7. Continuous Forward-Moving Center Dashed Line (Flows downward)
        val dashPeriod = 72.0f
        val dashPhase = distancePx % dashPeriod
        val dashEffect = PathEffect.dashPathEffect(floatArrayOf(40.0f, 32.0f), dashPhase)
        drawLine(
            color = MapRoadMarking,
            start = Offset(roadCenterX, -dashPeriod),
            end = Offset(roadCenterX, h + dashPeriod),
            strokeWidth = 3.dp.toPx(),
            pathEffect = dashEffect
        )

        // 8. Tunnel Shading & Overhead Arch Portals
        if (state.inTunnel) {
            drawRect(
                color = TunnelOverlayLight,
                topLeft = Offset(0.0f, 0.0f),
                size = size
            )

            // Dynamic Overhead Arch Portals moving downward
            val archSpacing = 220.0f
            val archScroll = distancePx % archSpacing
            val numArches = (h / archSpacing).toInt() + 2
            for (a in -1..numArches) {
                val archY = (a * archSpacing) + archScroll
                drawLine(
                    color = Color(0xFF64748B).copy(alpha = 0.45f),
                    start = Offset(roadLeft - 8.0f, archY),
                    end = Offset(roadRight + 8.0f, archY),
                    strokeWidth = 2.5f
                )
            }
        }

        // 9. Vehicle Base Position on Screen (Smooth lateral sway & lane position)
        val carCenterY = h * 0.65f
        val carX = roadCenterX + (state.carLateralOffsetRatio * (laneWidth * 0.48f))

        // 10. Forward Headlight Illumination Beam
        if (state.isRunning) {
            drawHeadlightBeam(carX, carCenterY, state.roadCurveDegrees)
        }

        // 11. Algorithmic Proof 1: Red Ghost Marker (Raw Uncorrected IMU Drift)
        if (state.inTunnel && state.ghostLateralOffsetRatio > 0.04f) {
            val ghostX = roadCenterX + (state.ghostLateralOffsetRatio * (laneWidth * 0.48f))
            drawGhostDrift(ghostX, carCenterY)
        }

        // 12. Algorithmic Proof 2: 15-State ESKF Covariance Halo
        val haloRadiusPx = state.covarianceHaloRadiusDp.dp.toPx()
        drawCircle(
            color = NavRouteBlue.copy(alpha = 0.16f),
            center = Offset(carX, carCenterY),
            radius = haloRadiusPx * 1.4f
        )
        drawCircle(
            color = NavRouteBlue.copy(alpha = 0.55f),
            center = Offset(carX, carCenterY),
            radius = haloRadiusPx,
            style = Stroke(width = 1.8.dp.toPx())
        )

        // 13. Algorithmic Proof 3: Pothole Shock Pulse
        if (state.potholePulsing) {
            drawCircle(
                color = Color(0xFFD97706).copy(alpha = 0.45f),
                center = Offset(carX, carCenterY),
                radius = 38.dp.toPx(),
                style = Stroke(width = 2.2.dp.toPx())
            )
        }

        // 14. Algorithmic Proof 4: Soft Recovery Magnetic Guide
        if (state.mode == NavigationMode.RECOVERING) {
            val targetGpsX = roadCenterX
            drawLine(
                color = StatusGnssGreen,
                start = Offset(carX, carCenterY),
                end = Offset(targetGpsX, carCenterY - 40.0f),
                strokeWidth = 2.dp.toPx(),
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(8.0f, 8.0f))
            )
            drawCircle(
                color = StatusGnssGreen,
                center = Offset(targetGpsX, carCenterY - 40.0f),
                radius = 5.dp.toPx()
            )
        }

        // 15. Vehicle Custom Icon (Car vs 2-Wheeler Scooter) with smooth tilt/curve
        rotate(degrees = state.roadCurveDegrees, pivot = Offset(carX, carCenterY)) {
            when (state.vehicleType) {
                VehicleType.CAR_4W -> drawCarIcon(carX, carCenterY, state.inTunnel)
                VehicleType.BIKE_2W -> drawScooterIcon(carX, carCenterY, state.inTunnel)
            }
        }
    }
}

private fun DrawScope.drawHeadlightBeam(x: Float, y: Float, headingDeg: Float) {
    rotate(degrees = headingDeg, pivot = Offset(x, y)) {
        val beamPath = Path().apply {
            moveTo(x - 10.0f, y - 24.0f)
            lineTo(x - 42.0f, y - 140.0f)
            lineTo(x + 42.0f, y - 140.0f)
            lineTo(x + 10.0f, y - 24.0f)
            close()
        }
        drawPath(
            path = beamPath,
            brush = Brush.verticalGradient(
                colors = listOf(
                    Color(0x38FEF08A),
                    Color(0x00FEF08A)
                ),
                startY = y - 24.0f,
                endY = y - 140.0f
            )
        )
    }
}

private fun DrawScope.drawCarIcon(x: Float, y: Float, inTunnel: Boolean) {
    val carW = 28.0f
    val carH = 52.0f
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
        color = primaryColor.copy(alpha = 0.92f),
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
    drawCircle(color = Color(0xFFFEF08A), center = Offset(x - (carW * 0.35f), y - (carH * 0.46f)), radius = 2.8f)
    drawCircle(color = Color(0xFFFEF08A), center = Offset(x + (carW * 0.35f), y - (carH * 0.46f)), radius = 2.8f)

    // Rear Tail Lights
    drawCircle(color = Color(0xFFEF4444), center = Offset(x - (carW * 0.35f), y + (carH * 0.44f)), radius = 2.2f)
    drawCircle(color = Color(0xFFEF4444), center = Offset(x + (carW * 0.35f), y + (carH * 0.44f)), radius = 2.2f)
}

private fun DrawScope.drawScooterIcon(x: Float, y: Float, inTunnel: Boolean) {
    val scooterW = 20.0f
    val scooterH = 48.0f
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
        start = Offset(x - (scooterW * 0.65f), y - (scooterH * 0.35f)),
        end = Offset(x + (scooterW * 0.65f), y - (scooterH * 0.35f)),
        strokeWidth = 3.2f
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
        radius = 5.8f
    )
    drawCircle(
        color = Color(0xFF0F172A),
        center = Offset(x, y - (scooterH * 0.12f)),
        radius = 5.8f,
        style = Stroke(width = 1.4f)
    )

    // Front Headlight
    drawCircle(color = Color(0xFFFEF08A), center = Offset(x, y - (scooterH * 0.46f)), radius = 3.0f)

    // Rear Tail Light
    drawCircle(color = Color(0xFFEF4444), center = Offset(x, y + (scooterH * 0.42f)), radius = 2.2f)
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
