package com.example.sarathi.ui.splash

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.example.sarathi.theme.PuckBlue
import com.example.sarathi.theme.StatusGreen

@Composable
fun SplashScreen(
    onSplashFinished: () -> Unit
) {
    val scale = remember { Animatable(0.7f) }
    val alpha = remember { Animatable(0.0f) }

    val infiniteTransition = rememberInfiniteTransition(label = "splashPulse")
    val pulseScale by infiniteTransition.animateFloat(
        initialValue = 0.95f,
        targetValue = 1.08f,
        animationSpec = infiniteRepeatable(
            animation = tween(1400, easing = FastOutSlowInEasing),
            repeatMode = RepeatMode.Reverse
        ),
        label = "pulse"
    )

    LaunchedEffect(Unit) {
        scale.animateTo(
            targetValue = 1.0f,
            animationSpec = tween(durationMillis = 800, easing = FastOutSlowInEasing)
        )
        alpha.animateTo(
            targetValue = 1.0f,
            animationSpec = tween(durationMillis = 600, easing = LinearEasing)
        )
        kotlinx.coroutines.delay(2200L)
        onSplashFinished()
    }

    Box(
        modifier = Modifier
            .fillMaxSize()
            .background(
                Brush.verticalGradient(
                    colors = listOf(
                        Color(0xFF0F172A),
                        Color(0xFF1E293B),
                        Color(0xFF0F172A)
                    )
                )
            )
            .clickable { onSplashFinished() },
        contentAlignment = Alignment.Center
    ) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
            modifier = Modifier
                .scale(scale.value)
                .alpha(alpha.value)
                .padding(24.dp)
        ) {
            // Glowing Nav Delta Icon
            Box(
                modifier = Modifier
                    .size(110.dp)
                    .scale(pulseScale),
                contentAlignment = Alignment.Center
            ) {
                // Outer glowing rings
                Canvas(modifier = Modifier.fillMaxSize()) {
                    val center = Offset(size.width / 2f, size.height / 2f)
                    drawCircle(
                        color = PuckBlue.copy(alpha = 0.18f),
                        radius = size.width * 0.48f
                    )
                    drawCircle(
                        color = PuckBlue.copy(alpha = 0.35f),
                        radius = size.width * 0.40f,
                        style = Stroke(width = 2.dp.toPx())
                    )
                }

                // Core Delta Emblem
                Box(
                    modifier = Modifier
                        .size(72.dp)
                        .clip(CircleShape)
                        .background(
                            Brush.radialGradient(
                                colors = listOf(PuckBlue, Color(0xFF1D4ED8))
                            )
                        ),
                    contentAlignment = Alignment.Center
                ) {
                    Canvas(modifier = Modifier.size(36.dp)) {
                        val w = size.width
                        val h = size.height
                        val delta = Path().apply {
                            moveTo(w * 0.5f, h * 0.12f)
                            lineTo(w * 0.88f, h * 0.85f)
                            lineTo(w * 0.5f, h * 0.65f)
                            lineTo(w * 0.12f, h * 0.85f)
                            close()
                        }
                        drawPath(delta, color = Color.White)
                    }
                }
            }

            Spacer(modifier = Modifier.height(28.dp))

            // App Name
            Text(
                text = "SAARTHI",
                color = Color.White,
                fontSize = 34.sp,
                fontWeight = FontWeight.ExtraBold,
                letterSpacing = 4.sp
            )

            Spacer(modifier = Modifier.height(8.dp))

            // 3-word subtitle: PRECISION DEAD RECKONING
            Text(
                text = "PRECISION DEAD RECKONING",
                color = Color(0xFF38BDF8),
                fontSize = 13.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 2.5.sp
            )

            Spacer(modifier = Modifier.height(6.dp))

            Text(
                text = "ISRO NavIC  •  SAC Ahmedabad",
                color = Color(0xFF94A3B8),
                fontSize = 11.sp,
                letterSpacing = 1.sp
            )

            Spacer(modifier = Modifier.height(48.dp))

            // Loading / Sensor Ready Status Indicator
            Box(
                modifier = Modifier
                    .clip(CircleShape)
                    .background(Color(0xFF1E293B))
                    .padding(horizontal = 16.dp, vertical = 8.dp)
            ) {
                Text(
                    text = "● INITIALIZING KINEMATIC ENGINE",
                    color = StatusGreen,
                    fontSize = 10.sp,
                    fontWeight = FontWeight.SemiBold,
                    letterSpacing = 1.sp
                )
            }
        }
    }
}
