package com.example.sarathi.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val SarathiColorScheme = darkColorScheme(
    primary = AccentCyan,
    onPrimary = BgBase,
    primaryContainer = AccentCyanSubtle,
    onPrimaryContainer = AccentCyan,
    secondary = StatusGnss,
    onSecondary = BgBase,
    tertiary = StatusOutage,
    onTertiary = BgBase,
    background = BgBase,
    onBackground = TextPrimary,
    surface = BgSurface,
    onSurface = TextPrimary,
    surfaceVariant = BgCard,
    onSurfaceVariant = TextSecondary,
    outline = BorderSubtle
)

@Composable
fun SarathiTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = SarathiColorScheme,
        typography = Typography,
        content = content
    )
}
