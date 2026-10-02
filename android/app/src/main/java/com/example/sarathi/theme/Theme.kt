package com.example.sarathi.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val SarathiColorScheme = lightColorScheme(
    primary           = PuckBlue,
    onPrimary         = PuckWhite,
    secondary         = StatusGreen,
    onSecondary       = PuckWhite,
    tertiary          = StatusAmber,
    background        = HomeBg,
    onBackground      = HomeText,
    surface           = SheetBg,
    onSurface         = SheetText,
    surfaceVariant    = ChipBarBg,
    onSurfaceVariant  = PuckWhite,
    outline           = HomeBorder
)

@Composable
fun SarathiTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = SarathiColorScheme,
        typography  = Typography,
        content     = content
    )
}
