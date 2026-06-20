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
    val history: List<ReceiptHistoryEntry> = emptyList(),
    val periodFilter: PeriodFilter = PeriodFilter.MONTH,
    val serverUrl: String = "",
    val ocrScript: OcrScript = OcrScript.LATIN,
    val inferenceMode: InferenceMode = InferenceMode.ON_DEVICE_INT8,
    val showJson: Boolean = false,
)

enum class AppScreen {
    DASHBOARD,
    SCAN,
    HISTORY,
    SETTINGS,
}

enum class PeriodFilter(val label: String) {
    DAY("Day"),
    WEEK("Week"),
    MONTH("Month"),
}

data class ReceiptHistoryEntry(
    val record: ReceiptCaptureRecord,
    val inference: ReceiptInferenceResponse? = null,
)
