package com.receiptapp.ui

import com.receiptapp.inference.InferenceMode
import com.receiptapp.inference.ReceiptInferenceResponse
import com.receiptapp.ocr.OcrScript
import com.receiptapp.receipt.ReceiptCaptureRecord

data class CaptureUiState(
    val isBusy: Boolean = false,
    val statusMessage: String? = null,
    val errorMessage: String? = null,
    val record: ReceiptCaptureRecord? = null,
    val serverResponse: ReceiptInferenceResponse? = null,
    val serverUrl: String = "",
    val ocrScript: OcrScript = OcrScript.LATIN,
    val inferenceMode: InferenceMode = InferenceMode.OCR_ONLY,
    val showJson: Boolean = false,
)

enum class AppScreen {
    CAPTURE,
    REVIEW,
    SETTINGS,
}
