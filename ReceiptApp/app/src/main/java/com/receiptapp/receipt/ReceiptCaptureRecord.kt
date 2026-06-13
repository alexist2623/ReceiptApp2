package com.receiptapp.receipt

import com.receiptapp.ocr.ReceiptOcrPayload
import java.io.File

data class ReceiptCaptureRecord(
    val captureId: String,
    val imageFile: File,
    val ocrJsonFile: File,
    val ocrPayload: ReceiptOcrPayload,
    val serverResultFile: File? = null,
)
