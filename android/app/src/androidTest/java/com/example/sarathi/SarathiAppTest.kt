package com.example.sarathi

import androidx.activity.ComponentActivity
import androidx.compose.ui.test.assertIsDisplayed
import androidx.compose.ui.test.junit4.createAndroidComposeRule
import androidx.compose.ui.test.onNodeWithText
import androidx.compose.ui.test.performClick
import org.junit.Rule
import org.junit.Test

class SarathiAppTest {

    @get:Rule
    val composeTestRule = createAndroidComposeRule<ComponentActivity>()

    @Test
    fun app_startsOnHomeScreen_andNavigatesBetweenTabs() {
        composeTestRule.setContent {
            com.example.sarathi.ui.SarathiApp()
        }

        // Check header and vehicle selection on Home
        composeTestRule.onNodeWithText("SARATHI").assertIsDisplayed()
        composeTestRule.onNodeWithText("START NAVIGATION").assertIsDisplayed()

        // Switch to Navigation Tab
        composeTestRule.onNodeWithText("NAVIGATION").performClick()
        composeTestRule.onNodeWithText("RESTART").assertIsDisplayed()

        // Switch to Audit Tab
        composeTestRule.onNodeWithText("AUDIT").performClick()
        composeTestRule.onNodeWithText("PERFORMANCE AUDIT").assertIsDisplayed()
        composeTestRule.onNodeWithText("PASSED").assertIsDisplayed()
    }
}
