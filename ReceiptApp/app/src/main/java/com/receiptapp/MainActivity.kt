package com.receiptapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import com.receiptapp.ui.AppScreen
import com.receiptapp.ui.CaptureViewModel
import com.receiptapp.ui.ReceiptCaptureScreen
import com.receiptapp.ui.ReceiptReviewScreen
import com.receiptapp.ui.SettingsScreen

class MainActivity : ComponentActivity() {
    private val viewModel: CaptureViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            MaterialTheme {
                ReceiptApp(viewModel)
            }
        }
    }
}

@Composable
private fun ReceiptApp(viewModel: CaptureViewModel) {
    val state by viewModel.uiState.collectAsState()
    var screen by remember { mutableStateOf(AppScreen.CAPTURE) }
    val context = LocalContext.current

    when (screen) {
        AppScreen.CAPTURE -> ReceiptCaptureScreen(
            state = state,
            onCapturedFile = {
                viewModel.processCameraFile(it)
                screen = AppScreen.REVIEW
            },
            onCaptureError = viewModel::showError,
            onGalleryUri = {
                viewModel.processGalleryUri(it)
                screen = AppScreen.REVIEW
            },
            onSettings = { screen = AppScreen.SETTINGS },
        )

        AppScreen.REVIEW -> ReceiptReviewScreen(
            state = state,
            onShare = {
                viewModel.createShareIntent()?.let(context::startActivity)
            },
            onUpload = { viewModel.uploadToServer() },
            onShowJson = viewModel::setShowJson,
            onRetake = {
                viewModel.reset()
                screen = AppScreen.CAPTURE
            },
        )

        AppScreen.SETTINGS -> SettingsScreen(
            state = state,
            onServerUrlChanged = viewModel::setServerUrl,
            onScriptChanged = viewModel::setOcrScript,
            onModeChanged = viewModel::setInferenceMode,
            onBack = { screen = AppScreen.CAPTURE },
        )
    }
}
