package com.example.sarathi.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val SarathiLightColorScheme = lightColorScheme(
    primary = NavRouteBlue,
    onPrimary = SurfaceCardLight,
    primaryContainer = StatusGnssBgLight,
    onPrimaryContainer = StatusGnssGreen,
    secondary = StatusGnssGreen,
    onSecondary = SurfaceCardLight,
    tertiary = StatusOutageAmber,
    onTertiary = SurfaceCardLight,
    background = MapTerrainBg,
    onBackground = TextPrimaryDark,
    surface = SurfaceCardLight,
    onSurface = TextPrimaryDark,
    surfaceVariant = SurfaceCardSubtle,
    onSurfaceVariant = TextSecondaryDark,
    outline = BorderLight
)

@Composable
fun SarathiTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = SarathiLightColorScheme,
        typography = Typography,
        content = content
    )
}
