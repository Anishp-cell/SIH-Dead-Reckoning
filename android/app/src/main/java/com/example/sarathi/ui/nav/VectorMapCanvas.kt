package com.example.sarathi.ui.nav

import androidx.compose.animation.core.LinearEasing
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
import androidx.compose.ui.graphics.drawscope.clipRect
import com.example.sarathi.model.NavigationMode
import com.example.sarathi.model.VehicleState
import com.example.sarathi.theme.ChipBg
import com.example.sarathi.theme.GhostDriftLine
import com.example.sarathi.theme.GhostDriftRed
import com.example.sarathi.theme.MapBg
import com.example.sarathi.theme.MapCurb
import com.example.sarathi.theme.MapGrass
import com.example.sarathi.theme.MapRoadDark
import com.example.sarathi.theme.MapRoadLight
import com.example.sarathi.theme.MapRouteLine
import com.example.sarathi.theme.MapRouteGlow
import com.example.sarathi.theme.MapTerrain
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.theme.PuckGlow
import com.example.sarathi.theme.StatusBlue
import com.example.sarathi.theme.StatusGreen
import com.example.sarathi.theme.TunnelDark
import com.example.sarathi.theme.TunnelWall
import kotlin.math.sin
import kotlin.math.cos

@Composable
fun SarathiMapCanvas(
    state: VehicleState,
    modifier: Modifier = Modifier
) {
    // Animate scroll offset continuously when running - this drives forward motion
    val infiniteTransition = rememberInfiniteTransition(label = "mapScroll")

    // Duration inversely proportional to speed — higher speed = faster scroll
    val scrollDuration = if (state.isRunning && state.speedKmh > 0.5f) {
        // At 50 km/h, one scroll cycle takes ~800ms (tuned for visual feel)
        (800.0f * (50.0f / state.speedKmh.coerceAtLeast(1f))).toInt().coerceIn(200, 3000)
    } else {
        10_000_000 // effectively frozen
    }

    // scrollT goes 0..1 continuously, repeating — each cycle = 1 road "tile"
    val scrollT by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = scrollDuration, easing = LinearEasing)
        ),
        label = "scrollT"
    )

    // Lateral sway animation — smooth left/right gentle curve
    val swayT by infiniteTransition.animateFloat(
        initialValue = 0f,
        targetValue = (2.0 * Math.PI).toFloat(),
        animationSpec = infiniteRepeatable(
            animation = tween(durationMillis = 7000, easing = LinearEasing)
        ),
        label = "swayT"
    )

    Canvas(modifier = modifier.fillMaxSize()) {
        val w = size.width
        val h = size.height

        // Draw the full navigation map
        drawNavigationMap(
            state = state,
            scrollT = scrollT,
            swayT = swayT,
            w = w,
            h = h
        )
    }
}

private fun DrawScope.drawNavigationMap(
    state: VehicleState,
    scrollT: Float,
    swayT: Float,
    w: Float,
    h: Float
) {
    val cx = w / 2f

    // ── 1. Background ──────────────────────────────────────────────────────────
    drawRect(
        color = if (state.inTunnel) TunnelDark else MapBg,
        size = Size(w, h)
    )

    // ── 2. Terrain blocks (grass areas beside road) ────────────────────────────
    if (!state.inTunnel) {
        drawTerrainBlocks(cx, w, h)
    } else {
        drawTunnelWalls(cx, w, h, scrollT)
    }

    // ── 3. Perspective road (the key: vanishes at horizon, wide at bottom) ─────
    // Slight lateral curve based on state
    val curveLean = state.carLateralOffsetRatio * 0.08f  // subtle steering lean

    val vanishX = cx + curveLean * w * 0.4f  // vanishing point on horizon
    val vanishY = h * 0.30f                   // horizon line at 30% from top

    // Road boundaries at bottom of screen
    val roadHalfWidthBottom = w * 0.44f
    val roadLeftBottom  = cx - roadHalfWidthBottom
    val roadRightBottom = cx + roadHalfWidthBottom

    drawPerspectiveRoad(
        vanishX = vanishX,
        vanishY = vanishY,
        roadLeftBottom = roadLeftBottom,
        roadRightBottom = roadRightBottom,
        h = h,
        w = w,
        state = state
    )

    // ── 4. Center dashed lane markings — scrolling with scrollT ───────────────
    drawScrollingDashes(
        vanishX = vanishX,
        vanishY = vanishY,
        cx = cx,
        h = h,
        scrollT = scrollT,
        state = state
    )

    // ── 5. Blue route overlay line ─────────────────────────────────────────────
    drawRouteOverlay(
        vanishX = vanishX,
        vanishY = vanishY,
        cx = cx,
        h = h,
        state = state
    )

    // ── 6. Ghost drift marker (dead reckoning uncorrected path) ───────────────
    if (state.inTunnel && state.ghostLateralOffsetRatio > 0.02f) {
        val ghostX = cx + state.ghostLateralOffsetRatio * w * 0.18f
        drawGhostMarker(ghostX, h * 0.60f)
    }

    // ── 7. Recovery guide line ─────────────────────────────────────────────────
    if (state.mode == NavigationMode.RECOVERING) {
        drawLine(
            color = StatusGreen.copy(alpha = 0.7f),
            start = Offset(cx, h * 0.60f),
            end = Offset(cx, h * 0.45f),
            strokeWidth = 2f,
            pathEffect = PathEffect.dashPathEffect(floatArrayOf(10f, 8f))
        )
    }

    // ── 8. Vehicle puck — always at fixed position (world scrolls, puck stays) ─
    val puckY = h * 0.60f
    drawVehiclePuck(cx, puckY, state)

    // ── 9. Covariance halo (ESKF 15-state uncertainty ring) ───────────────────
    if (state.isRunning) {
        val haloR = (state.covarianceHaloRadiusDp * density).coerceIn(20f, 60f)
        drawCircle(
            color = PuckGlow,
            center = Offset(cx, puckY),
            radius = haloR * 1.6f
        )
        if (state.inTunnel) {
            // Larger growing halo during dead reckoning outage
            drawCircle(
                color = GhostDriftRed.copy(alpha = 0.15f),
                center = Offset(cx, puckY),
                radius = haloR * 2.4f,
                style = Stroke(width = 1.5f * density)
            )
        }
    }
}

// ── Terrain blocks: grass patches flanking the road ──────────────────────────
private fun DrawScope.drawTerrainBlocks(cx: Float, w: Float, h: Float) {
    // Left grass
    drawRect(
        color = MapGrass,
        size = Size(cx * 0.45f, h)
    )
    // Right grass
    drawRect(
        color = MapGrass,
        topLeft = Offset(w - cx * 0.45f, 0f),
        size = Size(cx * 0.45f, h)
    )
    // Darker terrain behind road
    drawRect(
        color = MapTerrain,
        topLeft = Offset(cx * 0.35f, 0f),
        size = Size(w - cx * 0.70f, h * 0.35f)
    )
}

// ── Tunnel wall effect ─────────────────────────────────────────────────────────
private fun DrawScope.drawTunnelWalls(cx: Float, w: Float, h: Float, scrollT: Float) {
    // Left wall
    drawRect(color = TunnelWall, size = Size(cx * 0.30f, h))
    // Right wall
    drawRect(color = TunnelWall, topLeft = Offset(w - cx * 0.30f, 0f), size = Size(cx * 0.30f, h))

    // Moving tunnel lights (overhead strip lights)
    val lightSpacing = h * 0.22f
    repeat(7) { i ->
        val rawY = (i * lightSpacing) + scrollT * lightSpacing
        val lightY = rawY % (h + lightSpacing) - lightSpacing * 0.5f
        // Left strip
        drawRect(
            color = Color(0xFFD4A017).copy(alpha = 0.5f),
            topLeft = Offset(cx * 0.06f, lightY),
            size = Size(cx * 0.12f, 6f)
        )
        // Right strip
        drawRect(
            color = Color(0xFFD4A017).copy(alpha = 0.5f),
            topLeft = Offset(w - cx * 0.18f, lightY),
            size = Size(cx * 0.12f, 6f)
        )
    }
}

// ── Perspective road ──────────────────────────────────────────────────────────
private fun DrawScope.drawPerspectiveRoad(
    vanishX: Float,
    vanishY: Float,
    roadLeftBottom: Float,
    roadRightBottom: Float,
    h: Float,
    w: Float,
    state: VehicleState
) {
    // Main road fill
    val roadPath = Path().apply {
        moveTo(vanishX, vanishY)
        lineTo(roadRightBottom, h)
        lineTo(roadLeftBottom, h)
        close()
    }
    drawPath(
        path = roadPath,
        color = if (state.inTunnel) MapRoadDark.copy(alpha = 0.92f) else MapRoadDark
    )

    // Road surface highlight (lighter band in center)
    val innerW = (roadRightBottom - roadLeftBottom) * 0.42f
    val innerLeft = (roadLeftBottom + roadRightBottom) / 2f - innerW / 2f
    val innerPath = Path().apply {
        moveTo(vanishX, vanishY + 2f)
        lineTo(innerLeft + innerW, h)
        lineTo(innerLeft, h)
        close()
    }
    drawPath(
        path = innerPath,
        brush = Brush.verticalGradient(
            colors = listOf(MapRoadLight.copy(alpha = 0f), MapRoadLight.copy(alpha = 0.45f)),
            startY = vanishY,
            endY = h
        )
    )

    // Left shoulder curb
    drawLine(
        color = MapCurb.copy(alpha = 0.8f),
        start = Offset(vanishX, vanishY),
        end = Offset(roadLeftBottom, h),
        strokeWidth = 2f
    )
    // Right shoulder curb
    drawLine(
        color = MapCurb.copy(alpha = 0.8f),
        start = Offset(vanishX, vanishY),
        end = Offset(roadRightBottom, h),
        strokeWidth = 2f
    )
}

// ── Scrolling dashed center lane markings ─────────────────────────────────────
private fun DrawScope.drawScrollingDashes(
    vanishX: Float,
    vanishY: Float,
    cx: Float,
    h: Float,
    scrollT: Float,
    state: VehicleState
) {
    if (!state.isRunning && state.speedKmh < 0.5f) return

    // Number of dashes visible on screen
    val numDashes = 12
    for (i in 0..numDashes) {
        // tNorm = normalized position on road, 0=vanish, 1=bottom
        // We offset by scrollT to make them scroll downward
        val rawT = (i.toFloat() / numDashes) + scrollT / numDashes
        val tNorm = rawT % 1f

        // Skip small tNorm values near horizon (too small to see)
        if (tNorm < 0.05f) continue

        // Perspective interpolation: items near horizon are small, near camera are large
        val tPow = tNorm * tNorm  // quadratic perspective

        val x = lerp(vanishX, cx, tPow)
        val y = lerp(vanishY, h, tPow)

        // Dash length and width scale with perspective
        val dashLen = lerp(2f, 32f, tPow)
        val dashW   = lerp(0.5f, 3.5f, tPow)

        val yEnd = lerp(vanishY, h, (tNorm + 0.025f).coerceAtMost(1f) * (tNorm + 0.025f).coerceAtMost(1f))

        // Only draw every other "slot" to create dash/gap pattern
        if (i % 2 == 0) {
            drawLine(
                color = Color.White.copy(alpha = lerp(0.2f, 0.7f, tPow)),
                start = Offset(x, y),
                end = Offset(x, (y + dashLen).coerceAtMost(h)),
                strokeWidth = dashW
            )
        }
    }
}

// ── Blue route overlay ─────────────────────────────────────────────────────────
private fun DrawScope.drawRouteOverlay(
    vanishX: Float,
    vanishY: Float,
    cx: Float,
    h: Float,
    state: VehicleState
) {
    val routeHalfWidthBottom = (h - vanishY) * 0.22f

    val routePath = Path().apply {
        moveTo(vanishX, vanishY + 4f)
        lineTo(cx + routeHalfWidthBottom, h)
        lineTo(cx - routeHalfWidthBottom, h)
        close()
    }

    // Glow layer
    drawPath(
        path = routePath,
        brush = Brush.verticalGradient(
            colors = listOf(MapRouteGlow.copy(alpha = 0f), MapRouteGlow),
            startY = vanishY,
            endY = h
        )
    )

    // Solid route line
    drawLine(
        color = MapRouteLine.copy(alpha = 0.85f),
        start = Offset(vanishX, vanishY + 2f),
        end = Offset(cx, h),
        strokeWidth = 4f
    )

    // Route border highlight
    drawLine(
        color = Color.White.copy(alpha = 0.3f),
        start = Offset(vanishX, vanishY + 2f),
        end = Offset(cx - 2f, h),
        strokeWidth = 1.5f
    )
}

// ── Vehicle puck (the blue arrow/circle at center) ────────────────────────────
private fun DrawScope.drawVehiclePuck(x: Float, y: Float, state: VehicleState) {
    val r = 22f * density

    // Outer glow ring
    drawCircle(color = PuckGlow, center = Offset(x, y), radius = r * 1.55f)

    // White ring
    drawCircle(
        color = Color.White.copy(alpha = 0.9f),
        center = Offset(x, y),
        radius = r + 3f,
        style = Stroke(width = 2.5f)
    )

    // Solid blue puck
    drawCircle(color = PuckBlue, center = Offset(x, y), radius = r)

    // Navigation arrow on top of puck
    val arrowPath = Path().apply {
        moveTo(x, y - r * 0.65f)
        lineTo(x + r * 0.35f, y + r * 0.40f)
        lineTo(x, y + r * 0.15f)
        lineTo(x - r * 0.35f, y + r * 0.40f)
        close()
    }
    drawPath(path = arrowPath, color = Color.White)
}

// ── Ghost marker (red uncorrected IMU drift position) ─────────────────────────
private fun DrawScope.drawGhostMarker(x: Float, y: Float) {
    drawCircle(color = GhostDriftRed, center = Offset(x, y), radius = 10f * density)
    drawCircle(
        color = Color(0xFFEF4444),
        center = Offset(x, y),
        radius = 10f * density,
        style = Stroke(width = 1.5f * density)
    )
    drawLine(
        color = GhostDriftLine,
        start = Offset(x, y + 14f * density),
        end = Offset(x, y + 32f * density),
        strokeWidth = 1.5f * density,
        pathEffect = PathEffect.dashPathEffect(floatArrayOf(5f, 4f))
    )
}

private fun lerp(a: Float, b: Float, t: Float): Float = a + (b - a) * t
