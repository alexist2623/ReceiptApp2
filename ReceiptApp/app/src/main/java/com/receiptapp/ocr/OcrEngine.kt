package com.receiptapp.ocr

import java.io.File

interface OcrEngine {
    suspend fun recognize(
        captureId: String,
        canonicalImageFile: File,
        script: OcrScript,
    ): ReceiptOcrPayload
}
