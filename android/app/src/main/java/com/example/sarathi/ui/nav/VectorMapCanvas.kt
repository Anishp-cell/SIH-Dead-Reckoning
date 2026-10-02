package com.example.sarathi.ui.nav

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
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
import com.example.sarathi.model.NavigationMode
import com.example.sarathi.model.VehicleState
import com.example.sarathi.theme.GhostLine
import com.example.sarathi.theme.GhostRed
import com.example.sarathi.theme.PuckAura
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.theme.PuckShadow
import com.example.sarathi.theme.PuckWhite
import com.example.sarathi.theme.RoadCurb
import com.example.sarathi.theme.RoadDash
import com.example.sarathi.theme.RoadSurface
import com.example.sarathi.theme.RouteBlue
import com.example.sarathi.theme.RouteGlow
import com.example.sarathi.theme.SecondaryRoad
import com.example.sarathi.theme.SecondaryRoadEdge
import com.example.sarathi.theme.StatusAmber
import com.example.sarathi.theme.StatusGreen
import com.example.sarathi.theme.StatusRed
import com.example.sarathi.theme.TerrainBg
import com.example.sarathi.theme.TerrainGrass
import com.example.sarathi.theme.TerrainWater
import kotlin.math.atan2
import kotlin.math.sqrt

@Composable
fun SarathiMapCanvas(
    state: VehicleState,
    modifier: Modifier = Modifier
) {
    val infiniteTransition = rememberInfiniteTransition(label = "mapMotion")

    // Pulse animation for puck aura, covariance halo, and pothole ripples
    val auraPulse by infiniteTransition.animateFloat(
        initialValue = 0.92f,
        targetValue  = 1.16f,
        animationSpec = infiniteRepeatable(tween(1200, easing = LinearEasing)),
        label        = "aura"
    )

    // Smooth continuous vehicle progress interpolation (60/120 fps continuous glide)
    val animatedProgress by animateFloatAsState(
        targetValue = state.routeProgress,
        animationSpec = tween(durationMillis = 35, easing = LinearEasing),
        label = "puckProgress"
    )

    // Fluid forward streaming lane dashes tied directly to real distance traveled
    val dashOffset = if (state.isRunning) {
        ((state.distanceTraveledMeters * 0.035f) % 1.0f)
    } else {
        0.0f
    }

    Canvas(modifier = modifier.fillMaxSize()) {
        val W = size.width
        val H = size.height

        // 1. Light clean map background
        drawRect(color = TerrainBg, size = size)

        // 2. Scenic water lakes
        drawTerrainLakes(W, H)

        // 3. Soft sage-green landscape patches
        drawTerrainGrassZones(W, H)

        // 4. Secondary interchange ramp branching off right
        drawSecondaryRamps(W, H)

        // 5. Curved Highway Geometry (240 precision spline samples)
        val curvePoints = computeHighwayCurve(W, H, state.roadCurveDegrees)
        drawMainHighway(curvePoints, W, H, state.inTunnel)

        // 6. Glowing Blue Navigation Route Line
        drawGlowingRoutePath(curvePoints, W, H)

        // 7. Start and Destination Checkpoints on the Route
        drawRouteEndpoints(curvePoints, W, H)

        // 8. Forward Scrolling Lane Markings
        drawFlowingDashes(curvePoints, W, H, dashOffset, state.isRunning)

        // 9. Tunnel / Outage Canopy (Shaded zone in the middle section of highway)
        drawTunnelCanopy(curvePoints, W, H, state.inTunnel)

        // 10. Pothole Hazard on Roadbed (Only displayed during Stage 5: 20s - 28s)
        if (state.showPotholeHazard) {
            drawRoadPotholeHazard(curvePoints, state.potholePulsing, auraPulse)
        }

        // 11. DYNAMIC CONTINUOUS VEHICLE PUCK MOVEMENT ALONG HIGHWAY
        // Smooth continuous float lerp from start (t = 0.62) to destination (t = 0.16)
        val startCurveT = 0.62f
        val endCurveT   = 0.16f
        val currentCurveT = if (state.isStarted) {
            lerp(startCurveT, endCurveT, animatedProgress)
        } else {
            startCurveT
        }

        val puckPos = getLanePoint(curvePoints, currentCurveT, laneOffsetRatio = -0.22f)
        val puckTangent = getLaneTangent(curvePoints, currentCurveT)
        // Tangent angle so navigation arrow turns naturally with the road curve
        val puckAngleDeg = (atan2(puckTangent.y, puckTangent.x) * 180f / Math.PI.toFloat()) - 90f

        // 12. Pothole Shock Gating Shield Aura (Active during disturbance)
        if (state.potholePulsing) {
            drawPotholeActiveShield(puckPos, auraPulse)
        }

        // 13. Raw IMU Drift Marker vs SAARTHI Road-Constrained Position (Clean, No On-Canvas Text)
        if (state.inTunnel && state.ghostLateralOffsetRatio > 0.01f) {
            val ghostPos = Offset(puckPos.x + state.ghostLateralOffsetRatio * W * 0.15f, puckPos.y + 12f)
            drawGhostTrajectory(actualPos = puckPos, ghostPos = ghostPos)
        }

        // 14. ESKF Covariance Uncertainty Halo
        if (state.isRunning) {
            val haloRadius = if (state.inTunnel) {
                (state.covarianceHaloRadiusDp * density * auraPulse * 1.3f).coerceIn(30f, 72f)
            } else {
                25f * density
            }
            drawCircle(
                color = PuckAura,
                center = puckPos,
                radius = haloRadius
            )
        }

        // 15. Soft Recovery Guide Line (Zero-teleportation Kalman glide)
        if (state.mode == NavigationMode.RECOVERING) {
            drawLine(
                color = StatusGreen.copy(alpha = 0.90f),
                start = puckPos,
                end = Offset(puckPos.x, puckPos.y - 75f),
                strokeWidth = 3.5f * density,
                pathEffect = PathEffect.dashPathEffect(floatArrayOf(12f, 8f))
            )
        }

        // 16. The Navigation Vehicle Puck (smooth continuous 60fps movement)
        drawNavigationPuck(puckPos, puckAngleDeg, auraPulse)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Curve Geometry Calculations & Continuous Float Interpolation
// ─────────────────────────────────────────────────────────────────────────────
data class CurveSample(
    val center: Offset,
    val normal: Offset,
    val tangent: Offset,
    val width: Float,
    val t: Float
)

private fun computeHighwayCurve(W: Float, H: Float, curveDeg: Float): List<CurveSample> {
    val steps = 240 // High-resolution spline for continuous interpolation
    val samples = ArrayList<CurveSample>(steps)

    val curveMod = curveDeg * 1.8f

    val p0 = Offset(W * -0.04f, H * 0.33f)
    val p1 = Offset(W * 0.20f, H * 0.38f)
    val p2 = Offset(W * 0.44f, H * 0.58f)
    val p3 = Offset(W * 0.68f + curveMod, H * 1.05f)

    for (i in 0..steps) {
        val t = i.toFloat() / steps.toFloat()
        val pt = cubicBezier(p0, p1, p2, p3, t)
        val tan = cubicBezierTangent(p0, p1, p2, p3, t)
        val norm = Offset(-tan.y, tan.x)

        val w = lerp(22f, W * 0.58f, t * t)
        samples.add(CurveSample(pt, norm, tan, w, t))
    }
    return samples
}

private fun cubicBezier(p0: Offset, p1: Offset, p2: Offset, p3: Offset, t: Float): Offset {
    val u = 1f - t
    val tt = t * t
    val uu = u * u
    val uuu = uu * u
    val ttt = tt * t

    val x = uuu * p0.x + 3f * uu * t * p1.x + 3f * u * tt * p2.x + ttt * p3.x
    val y = uuu * p0.y + 3f * uu * t * p1.y + 3f * u * tt * p2.y + ttt * p3.y
    return Offset(x, y)
}

private fun cubicBezierTangent(p0: Offset, p1: Offset, p2: Offset, p3: Offset, t: Float): Offset {
    val u = 1f - t
    val dx = 3f * u * u * (p1.x - p0.x) + 6f * u * t * (p2.x - p1.x) + 3f * t * t * (p3.x - p2.x)
    val dy = 3f * u * u * (p1.y - p0.y) + 6f * u * t * (p2.y - p1.y) + 3f * t * t * (p3.y - p2.y)
    val len = sqrt(dx * dx + dy * dy).coerceAtLeast(0.0001f)
    return Offset(dx / len, dy / len)
}

// Continuous float interpolation solves the discrete frame stepping bug completely!
private fun getLanePoint(samples: List<CurveSample>, t: Float, laneOffsetRatio: Float): Offset {
    val maxIdx = (samples.size - 1).toFloat()
    val exactIdx = (maxIdx * t).coerceIn(0f, maxIdx - 1.0001f)
    val i0 = exactIdx.toInt()
    val i1 = (i0 + 1).coerceAtMost(samples.size - 1)
    val frac = exactIdx - i0

    val s0 = samples[i0]
    val s1 = samples[i1]

    val c0 = s0.center + s0.normal * (s0.width * laneOffsetRatio)
    val c1 = s1.center + s1.normal * (s1.width * laneOffsetRatio)

    return Offset(lerp(c0.x, c1.x, frac), lerp(c0.y, c1.y, frac))
}

private fun getLaneTangent(samples: List<CurveSample>, t: Float): Offset {
    val maxIdx = (samples.size - 1).toFloat()
    val exactIdx = (maxIdx * t).coerceIn(0f, maxIdx - 1.0001f)
    val i0 = exactIdx.toInt()
    val i1 = (i0 + 1).coerceAtMost(samples.size - 1)
    val frac = exactIdx - i0

    val t0 = samples[i0].tangent
    val t1 = samples[i1].tangent

    val tx = lerp(t0.x, t1.x, frac)
    val ty = lerp(t0.y, t1.y, frac)
    val len = sqrt(tx * tx + ty * ty).coerceAtLeast(0.0001f)
    return Offset(tx / len, ty / len)
}

// ─────────────────────────────────────────────────────────────────────────────
// Terrain Water & Landscapes
// ─────────────────────────────────────────────────────────────────────────────
private fun DrawScope.drawTerrainLakes(W: Float, H: Float) {
    drawOval(
        color = TerrainWater,
        topLeft = Offset(W * 0.06f, H * 0.28f),
        size = Size(W * 0.35f, H * 0.085f)
    )
    drawOval(
        color = TerrainWater,
        topLeft = Offset(W * 0.46f, H * 0.30f),
        size = Size(W * 0.42f, H * 0.08f)
    )
}

private fun DrawScope.drawTerrainGrassZones(W: Float, H: Float) {
    val grass1 = Path().apply {
        moveTo(0f, H * 0.44f)
        cubicTo(W * 0.16f, H * 0.43f, W * 0.24f, H * 0.49f, W * 0.20f, H * 0.60f)
        cubicTo(W * 0.16f, H * 0.67f, W * 0.08f, H * 0.70f, 0f, H * 0.68f)
        close()
    }
    drawPath(grass1, color = TerrainGrass)

    val grass2 = Path().apply {
        moveTo(0f, H * 0.74f)
        cubicTo(W * 0.20f, H * 0.75f, W * 0.28f, H * 0.83f, W * 0.16f, H * 0.95f)
        lineTo(0f, H * 0.95f)
        close()
    }
    drawPath(grass2, color = TerrainGrass)

    val grass3 = Path().apply {
        moveTo(W, H * 0.50f)
        cubicTo(W * 0.74f, H * 0.52f, W * 0.66f, H * 0.62f, W * 0.72f, H * 0.74f)
        lineTo(W, H * 0.74f)
        close()
    }
    drawPath(grass3, color = TerrainGrass)
}

private fun DrawScope.drawSecondaryRamps(W: Float, H: Float) {
    val ramp = Path().apply {
        moveTo(W * 0.36f, H * 0.50f)
        cubicTo(W * 0.58f, H * 0.56f, W * 0.82f, H * 0.59f, W * 1.05f, H * 0.63f)
    }
    drawPath(ramp, color = SecondaryRoadEdge, style = Stroke(width = 30f))
    drawPath(ramp, color = SecondaryRoad, style = Stroke(width = 24f))

    val ramp2 = Path().apply {
        moveTo(W * 0.62f, H * 0.57f)
        cubicTo(W * 0.78f, H * 0.67f, W * 0.88f, H * 0.77f, W * 1.05f, H * 0.82f)
    }
    drawPath(ramp2, color = SecondaryRoadEdge, style = Stroke(width = 26f))
    drawPath(ramp2, color = SecondaryRoad, style = Stroke(width = 20f))
}

// ─────────────────────────────────────────────────────────────────────────────
// Highway Surface & Curbs
// ─────────────────────────────────────────────────────────────────────────────
private fun DrawScope.drawMainHighway(
    samples: List<CurveSample>,
    W: Float,
    H: Float,
    inTunnel: Boolean
) {
    val leftEdge = ArrayList<Offset>(samples.size)
    val rightEdge = ArrayList<Offset>(samples.size)

    for (s in samples) {
        leftEdge.add(s.center - s.normal * (s.width * 0.5f))
        rightEdge.add(s.center + s.normal * (s.width * 0.5f))
    }

    val roadPath = Path().apply {
        moveTo(leftEdge[0].x, leftEdge[0].y)
        for (i in 1 until leftEdge.size) {
            lineTo(leftEdge[i].x, leftEdge[i].y)
        }
        for (i in rightEdge.indices.reversed()) {
            lineTo(rightEdge[i].x, rightEdge[i].y)
        }
        close()
    }

    val roadColor = if (inTunnel) RoadSurface.copy(red = 0.22f, green = 0.26f, blue = 0.33f) else RoadSurface
    drawPath(
        path = roadPath,
        brush = Brush.verticalGradient(
            colors = listOf(
                roadColor.copy(alpha = 0.90f),
                roadColor,
                roadColor
            ),
            startY = samples.first().center.y,
            endY = H
        )
    )

    for (i in 0 until leftEdge.size - 1) {
        drawLine(RoadCurb, leftEdge[i], leftEdge[i + 1], strokeWidth = 3f)
        drawLine(RoadCurb, rightEdge[i], rightEdge[i + 1], strokeWidth = 3f)
    }

    val rightLine = ArrayList<Offset>(samples.size)
    for (s in samples) {
        rightLine.add(s.center + s.normal * (s.width * 0.44f))
    }
    for (i in 0 until rightLine.size - 1) {
        val t = samples[i].t
        val w = lerp(0.8f, 3.5f, t * t)
        val alpha = lerp(0.2f, 0.75f, t)
        drawLine(RoadDash.copy(alpha = alpha), rightLine[i], rightLine[i + 1], strokeWidth = w)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Glowing Route Path & Start / Destination Markers
// ─────────────────────────────────────────────────────────────────────────────
private fun DrawScope.drawGlowingRoutePath(samples: List<CurveSample>, W: Float, H: Float) {
    val routePts = ArrayList<Offset>(samples.size)
    for (s in samples) {
        routePts.add(s.center - s.normal * (s.width * 0.22f))
    }

    val routePath = Path().apply {
        if (routePts.isNotEmpty()) {
            moveTo(routePts[0].x, routePts[0].y)
            for (i in 1 until routePts.size) {
                lineTo(routePts[i].x, routePts[i].y)
            }
        }
    }

    drawPath(
        path = routePath,
        brush = Brush.verticalGradient(
            colors = listOf(RouteGlow.copy(alpha = 0.05f), RouteGlow.copy(alpha = 0.60f)),
            startY = samples.first().center.y,
            endY = H
        ),
        style = Stroke(width = 34f)
    )

    drawPath(
        path = routePath,
        brush = Brush.verticalGradient(
            colors = listOf(RouteBlue.copy(alpha = 0.15f), RouteBlue.copy(alpha = 0.90f)),
            startY = samples.first().center.y,
            endY = H
        ),
        style = Stroke(width = 16f)
    )

    drawPath(
        path = routePath,
        color = Color.White.copy(alpha = 0.92f),
        style = Stroke(width = 4.5f)
    )
}

private fun DrawScope.drawRouteEndpoints(samples: List<CurveSample>, W: Float, H: Float) {
    // Destination pin at top of route (t = 0.16)
    val destPt = getLanePoint(samples, 0.16f, -0.22f)
    drawCircle(color = StatusRed, center = destPt, radius = 9f * density)
    drawCircle(color = Color.White, center = destPt, radius = 4f * density)

    // Start checkpoint at start of route (t = 0.62)
    val startPt = getLanePoint(samples, 0.62f, -0.22f)
    drawCircle(color = StatusGreen.copy(alpha = 0.85f), center = startPt, radius = 7f * density)
    drawCircle(color = Color.White, center = startPt, radius = 3f * density)
}

// ─────────────────────────────────────────────────────────────────────────────
// Flowing Dashed Lane Markings
// ─────────────────────────────────────────────────────────────────────────────
private fun DrawScope.drawFlowingDashes(
    samples: List<CurveSample>,
    W: Float,
    H: Float,
    scrollOffset: Float,
    isRunning: Boolean
) {
    val totalSlots = 20
    val activeOffset = if (isRunning) scrollOffset else 0f

    for (slot in 0 until totalSlots) {
        val rawT = (slot.toFloat() + activeOffset) / totalSlots.toFloat()
        val tNorm = rawT % 1.0f

        if (tNorm < 0.04f) continue
        if (slot % 2 == 1) continue

        val tStart = (tNorm * tNorm).coerceIn(0f, 1f)
        val tEnd = ((tNorm + 0.55f / totalSlots) * (tNorm + 0.55f / totalSlots)).coerceIn(0f, 1f)

        val p1 = getLanePoint(samples, tStart, laneOffsetRatio = 0.02f)
        val p2 = getLanePoint(samples, tEnd, laneOffsetRatio = 0.02f)

        val strokeW = lerp(0.8f, 5.2f, tStart)
        val alpha = lerp(0.2f, 0.85f, tStart)

        drawLine(
            color = RoadDash.copy(alpha = alpha),
            start = p1,
            end = p2,
            strokeWidth = strokeW
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tunnel Overlay during Dead Reckoning Outage
// ─────────────────────────────────────────────────────────────────────────────
private fun DrawScope.drawTunnelCanopy(samples: List<CurveSample>, W: Float, H: Float, inTunnel: Boolean) {
    val tunnelStartIdx = (samples.size * 0.30f).toInt()
    val tunnelEndIdx = (samples.size * 0.68f).toInt()

    val tunnelPath = Path().apply {
        val sStart = samples[tunnelStartIdx]
        val leftTop = sStart.center - sStart.normal * (sStart.width * 0.54f)
        moveTo(leftTop.x, leftTop.y)

        for (i in tunnelStartIdx..tunnelEndIdx) {
            val s = samples[i]
            val pt = s.center - s.normal * (s.width * 0.54f)
            lineTo(pt.x, pt.y)
        }
        for (i in tunnelEndIdx downTo tunnelStartIdx) {
            val s = samples[i]
            val pt = s.center + s.normal * (s.width * 0.54f)
            lineTo(pt.x, pt.y)
        }
        close()
    }

    val canopyAlpha = if (inTunnel) 0.38f else 0.18f
    drawPath(path = tunnelPath, color = Color(0xFF0F172A).copy(alpha = canopyAlpha))

    for (i in tunnelStartIdx..tunnelEndIdx step 12) {
        val s = samples[i]
        val wallPt = s.center - s.normal * (s.width * 0.50f)
        drawCircle(color = StatusAmber.copy(alpha = if (inTunnel) 0.85f else 0.40f), center = wallPt, radius = 3f)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Distinct Road Pothole Hazard on Asphalt (Only drawn when showPotholeHazard is true)
// ─────────────────────────────────────────────────────────────────────────────
private fun DrawScope.drawRoadPotholeHazard(
    samples: List<CurveSample>,
    isDisturbanceActive: Boolean,
    pulse: Float
) {
    val potholePos = getLanePoint(samples, 0.38f, -0.22f)

    // 1. Dark asphalt depression crater
    drawOval(
        color = Color(0xFF1E2430),
        topLeft = Offset(potholePos.x - 22f * density, potholePos.y - 12f * density),
        size = Size(44f * density, 24f * density)
    )

    // 2. Jagged asphalt edge border
    drawOval(
        color = Color(0xFF0D1117),
        topLeft = Offset(potholePos.x - 20f * density, potholePos.y - 10f * density),
        size = Size(40f * density, 20f * density),
        style = Stroke(width = 2f * density)
    )

    // 3. Amber warning ring
    drawOval(
        color = StatusAmber.copy(alpha = if (isDisturbanceActive) 0.90f else 0.50f),
        topLeft = Offset(potholePos.x - 25f * density, potholePos.y - 14f * density),
        size = Size(50f * density, 28f * density),
        style = Stroke(width = 1.6f * density)
    )

    // 4. Amber radar ripple if disturbance is currently active
    if (isDisturbanceActive) {
        drawOval(
            color = StatusAmber.copy(alpha = 0.35f),
            topLeft = Offset(potholePos.x - 36f * density * pulse, potholePos.y - 20f * density * pulse),
            size = Size(72f * density * pulse, 40f * density * pulse),
            style = Stroke(width = 2.5f * density)
        )
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Pothole Shock Gating Shield Aura
// ─────────────────────────────────────────────────────────────────────────────
private fun DrawScope.drawPotholeActiveShield(puckPos: Offset, pulse: Float) {
    drawCircle(
        color = Color(0xFF06B6D4).copy(alpha = 0.40f),
        center = puckPos,
        radius = 42f * density * pulse,
        style = Stroke(width = 3f * density)
    )

    drawCircle(
        color = StatusAmber.copy(alpha = 0.25f),
        center = puckPos,
        radius = 52f * density * pulse,
        style = Stroke(width = 2f * density)
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Raw INS Drift Trajectory vs SAARTHI Road Constraint (Clean, Uncluttered)
// ─────────────────────────────────────────────────────────────────────────────
private fun DrawScope.drawGhostTrajectory(actualPos: Offset, ghostPos: Offset) {
    // 1. Red Beacon for raw unconstrained IMU
    drawCircle(
        color = GhostRed.copy(alpha = 0.28f),
        center = ghostPos,
        radius = 16f * density
    )
    drawCircle(
        color = Color(0xFFEF4444),
        center = ghostPos,
        radius = 8f * density
    )
    drawCircle(
        color = Color.White,
        center = ghostPos,
        radius = 3.5f * density
    )

    // 2. Connecting dashed line showing lateral drift separation
    drawLine(
        color = GhostLine,
        start = actualPos,
        end = ghostPos,
        strokeWidth = 2.2f * density,
        pathEffect = PathEffect.dashPathEffect(floatArrayOf(8f, 6f))
    )
}

// ─────────────────────────────────────────────────────────────────────────────
// Navigation Puck
// ─────────────────────────────────────────────────────────────────────────────
private fun DrawScope.drawNavigationPuck(pos: Offset, headingDeg: Float, auraScale: Float) {
    val r = 22f * density

    // 1. Soft glowing outer aura
    drawCircle(
        color = PuckAura,
        center = pos,
        radius = r * 1.8f * auraScale
    )

    // 2. Drop shadow
    drawCircle(
        color = PuckShadow,
        center = Offset(pos.x + 1.5f, pos.y + 3f),
        radius = r + 4f
    )

    // 3. Crisp white circular border
    drawCircle(
        color = PuckWhite,
        center = pos,
        radius = r + 4f
    )

    // 4. Vibrant electric blue inner circle
    drawCircle(
        color = PuckBlue,
        center = pos,
        radius = r
    )

    // 5. White navigation delta arrow
    rotate(degrees = headingDeg, pivot = pos) {
        val arrow = Path().apply {
            val tip = Offset(pos.x, pos.y - r * 0.65f)
            val baseL = Offset(pos.x - r * 0.44f, pos.y + r * 0.48f)
            val baseM = Offset(pos.x, pos.y + r * 0.18f)
            val baseR = Offset(pos.x + r * 0.44f, pos.y + r * 0.48f)

            moveTo(tip.x, tip.y)
            lineTo(baseR.x, baseR.y)
            lineTo(baseM.x, baseM.y)
            lineTo(baseL.x, baseL.y)
            close()
        }
        drawPath(arrow, color = PuckWhite)
    }
}

private fun lerp(a: Float, b: Float, t: Float) = a + (b - a) * t
