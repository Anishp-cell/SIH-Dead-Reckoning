package com.example.sarathi.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.darkColorScheme
import androidx.compose.runtime.Composable

private val SarathiDarkColorScheme = darkColorScheme(
    primary = PuckBlue,
    onPrimary = TextPrimary,
    secondary = StatusGreen,
    onSecondary = TextPrimary,
    tertiary = StatusAmber,
    onTertiary = TextPrimary,
    background = MapBg,
    onBackground = TextPrimary,
    surface = SurfaceNav,
    onSurface = TextPrimary,
    surfaceVariant = ChipBg,
    onSurfaceVariant = TextChip,
    outline = ChipBorder
)

@Composable
fun SarathiTheme(
    content: @Composable () -> Unit
) {
    MaterialTheme(
        colorScheme = SarathiDarkColorScheme,
        typography = Typography,
        content = content
    )
}
