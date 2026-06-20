package com.receiptapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import com.receiptapp.ui.AppScreen
import com.receiptapp.ui.CaptureViewModel
import com.receiptapp.ui.ReceiptDashboardScreen
import com.receiptapp.ui.ReceiptCaptureScreen
import com.receiptapp.ui.ReceiptHistoryScreen
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
    var screen by remember { mutableStateOf(AppScreen.DASHBOARD) }
    val context = LocalContext.current

    Scaffold(
        bottomBar = {
            if (screen != AppScreen.SETTINGS) {
                NavigationBar {
                    listOf(
                        AppScreen.DASHBOARD to "Dashboard",
                        AppScreen.SCAN to "Scan",
                        AppScreen.HISTORY to "Receipts",
                    ).forEach { (target, label) ->
                        NavigationBarItem(
                            selected = screen == target,
                            onClick = {
                                if (target == AppScreen.HISTORY || target == AppScreen.DASHBOARD) {
                                    viewModel.refreshHistory()
                                }
                                screen = target
                            },
                            icon = { Text(label.take(1)) },
                            label = { Text(label) },
                        )
                    }
                }
            }
        },
    ) { padding ->
        when (screen) {
            AppScreen.DASHBOARD -> ReceiptDashboardScreen(
                state = state,
                contentPadding = padding,
                onPeriodChanged = viewModel::setPeriodFilter,
            )

            AppScreen.SCAN -> {
                if (state.record == null) {
                    ReceiptCaptureScreen(
                        state = state,
                        contentPadding = padding,
                        onCapturedFile = {
                            viewModel.processCameraFile(it)
                        },
                        onCaptureError = viewModel::showError,
                        onGalleryUri = {
                            viewModel.processGalleryUri(it)
                        },
                        onSettings = { screen = AppScreen.SETTINGS },
                    )
                } else {
                    ReceiptReviewScreen(
                        state = state,
                        contentPadding = padding,
                        onShare = {
                            viewModel.createShareIntent()?.let(context::startActivity)
                        },
                        onUpload = { viewModel.uploadToServer() },
                        onRunOnDevice = { viewModel.runOnDeviceInference() },
                        onShowJson = viewModel::setShowJson,
                        onRetake = {
                            viewModel.reset()
                        },
                    )
                }
            }

            AppScreen.HISTORY -> ReceiptHistoryScreen(
                state = state,
                contentPadding = padding,
                onPeriodChanged = viewModel::setPeriodFilter,
                onRefresh = viewModel::refreshHistory,
                onDeleteReceipt = viewModel::deleteReceipt,
                onDeleteAllReceipts = viewModel::deleteAllReceipts,
            )

            AppScreen.SETTINGS -> SettingsScreen(
                state = state,
                onServerUrlChanged = viewModel::setServerUrl,
                onScriptChanged = viewModel::setOcrScript,
                onModeChanged = viewModel::setInferenceMode,
                onBack = { screen = AppScreen.SCAN },
            )
        }
    }
}
