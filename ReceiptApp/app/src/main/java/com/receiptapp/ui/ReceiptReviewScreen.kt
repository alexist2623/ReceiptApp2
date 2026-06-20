package com.receiptapp.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.receiptapp.BuildConfig
import com.receiptapp.util.JsonUtils

@Composable
fun ReceiptReviewScreen(
    state: CaptureUiState,
    contentPadding: PaddingValues = PaddingValues(0.dp),
    onShare: () -> Unit,
    onUpload: () -> Unit,
    onRunOnDevice: () -> Unit,
    onShowJson: (Boolean) -> Unit,
    onRetake: () -> Unit,
) {
    val record = state.record
    if (record == null) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(contentPadding)
                .padding(16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp),
        ) {
            Text("OCR Review", style = MaterialTheme.typography.titleLarge)
            state.statusMessage?.let { Text(it, color = MaterialTheme.colorScheme.primary) }
            state.errorMessage?.let { Text(it, color = MaterialTheme.colorScheme.error) }
            OutlinedButton(onClick = onRetake, enabled = !state.isBusy) { Text("Back to camera") }
        }
        return
    }
    val debugValidation = record.ocrPayload.debug?.coordinateValidation
    val validationOk = (debugValidation == null || debugValidation == "OK") &&
        record.ocrPayload.image_width == record.ocrPayload.image.width &&
        record.ocrPayload.image_height == record.ocrPayload.image.height
    val actualWidth = record.ocrPayload.debug?.canonicalImageActualWidth ?: record.ocrPayload.image.width
    val actualHeight = record.ocrPayload.debug?.canonicalImageActualHeight ?: record.ocrPayload.image.height
    val showInferenceOverlay = remember(record.captureId) { mutableStateOf(false) }
    val overlayMode = remember(record.captureId) { mutableStateOf(ReceiptOverlayMode.ITEM) }
    state.serverResponse?.let { response ->
        if (showInferenceOverlay.value) {
            ReceiptInferenceOverlayDialog(
                imageFile = record.imageFile,
                words = record.ocrPayload.words,
                response = response,
                initialMode = overlayMode.value,
                onDismiss = { showInferenceOverlay.value = false },
            )
        }
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .verticalScroll(rememberScrollState())
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Text("OCR Review", style = MaterialTheme.typography.titleLarge)
        OcrOverlayComposable(
            imageFile = record.imageFile,
            words = record.ocrPayload.words,
            showText = false,
        )
        Text("capture_id: ${record.captureId}")
        Text("actual JPG size: $actualWidth x $actualHeight")
        Text("OCR JSON top-level size: ${record.ocrPayload.image_width} x ${record.ocrPayload.image_height}")
        Text("OCR JSON image.width/height: ${record.ocrPayload.image.width} x ${record.ocrPayload.image.height}")
        Text("word count: ${record.ocrPayload.words.size}")
        Text(
            if (validationOk) {
                "validation status: OK"
            } else {
                "validation status: MISMATCH - Coordinate mismatch. Do not use this export for training."
            },
            color = if (validationOk) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.error,
        )
        Text("app version: ${record.ocrPayload.app.versionName} (${record.ocrPayload.app.versionCode})")
        Text("build: ${if (BuildConfig.DEBUG) "debug" else "release"} ${BuildConfig.GIT_COMMIT_SHA}")
        Text("OCR script: ${record.ocrPayload.ocr.script}")
        Text("coordinate space: ${record.ocrPayload.image.coordinateSpace}")
        Text("JSON: ${record.ocrJsonFile.absolutePath}")
        Text("First words: ${record.ocrPayload.words.take(20).joinToString { it.text }}")
        state.statusMessage?.let { Text(it, color = MaterialTheme.colorScheme.primary) }
        state.errorMessage?.let { Text(it, color = MaterialTheme.colorScheme.error) }

        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            Button(onClick = onShare, enabled = !state.isBusy) { Text("Share ZIP") }
            Button(onClick = onUpload, enabled = !state.isBusy) { Text("Upload") }
        }
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
            Button(onClick = onRunOnDevice, enabled = !state.isBusy) { Text("Run INT8") }
            OutlinedButton(onClick = { onShowJson(true) }) { Text("View JSON") }
            OutlinedButton(onClick = onRetake) { Text("Retake") }
        }

        state.serverResponse?.let { response ->
            Spacer(Modifier.height(8.dp))
            Text("Inference result: ${response.status}", style = MaterialTheme.typography.titleMedium)
            response.storeName?.let { Text("Store: ${it.text}") }
            response.items.forEach { item ->
                Text("${item.itemIndex}. ${item.menuName?.text.orEmpty()} / ${item.price?.text.orEmpty()}")
            }
            if (response.taxes.isNotEmpty()) {
                response.taxes.forEachIndexed { index, tax ->
                    val value = listOfNotNull(tax.rate?.text, tax.price?.text).joinToString(" / ")
                    Text("Tax ${index + 1}: ${tax.name?.text ?: "Tax"} ${value.ifBlank { "-" }}")
                }
            } else {
                response.tax?.get("price")?.let { Text("Tax: ${it.text}") }
            }
            response.total?.get("price")?.let { Text("Total: ${it.text}") }
            Text("Inference overlay", style = MaterialTheme.typography.titleMedium)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                Button(onClick = {
                    overlayMode.value = ReceiptOverlayMode.ITEM
                    showInferenceOverlay.value = true
                }) { Text("ITEM overlay") }
                Button(onClick = {
                    overlayMode.value = ReceiptOverlayMode.TAX
                    showInferenceOverlay.value = true
                }) { Text("TAX overlay") }
            }
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
                Button(onClick = {
                    overlayMode.value = ReceiptOverlayMode.LABELS
                    showInferenceOverlay.value = true
                }) { Text("Labels overlay") }
                OutlinedButton(onClick = {
                    overlayMode.value = ReceiptOverlayMode.ALL
                    showInferenceOverlay.value = true
                }) { Text("All") }
            }
        }
    }

    if (state.showJson) {
        AlertDialog(
            onDismissRequest = { onShowJson(false) },
            confirmButton = {
                TextButton(onClick = { onShowJson(false) }) { Text("Close") }
            },
            title = { Text("OCR JSON") },
            text = {
                Text(
                    JsonUtils.encodePretty(record.ocrPayload).take(6000),
                    modifier = Modifier.verticalScroll(rememberScrollState()),
                )
            },
        )
    }
}
