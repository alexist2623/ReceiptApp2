package com.receiptapp.receipt

import android.content.Context
import android.util.Log
import com.receiptapp.capture.CanonicalImageResult
import com.receiptapp.ocr.ReceiptOcrPayload
import com.receiptapp.util.JsonUtils
import java.io.File

class ReceiptRepository(
    context: Context,
    private val fileStore: ReceiptFileStore = ReceiptFileStore(context),
) {
    fun saveOcrPayload(
        imageResult: CanonicalImageResult,
        payload: ReceiptOcrPayload,
    ): ReceiptCaptureRecord {
        val validation = ReceiptExportValidator.validateImageAndPayload(imageResult.imageFile, payload)
        require(validation.ok) {
            "saveOcrPayload(${imageResult.captureId}) failed:\n" + validation.errors.joinToString(separator = "\n")
        }
        Log.i(
            "ReceiptOCR",
            "OCR coordinate validation OK: image=${validation.imageWidth}x${validation.imageHeight} " +
                "json=${validation.jsonImageWidth}x${validation.jsonImageHeight} " +
                "capture=${imageResult.captureId}",
        )
        val jsonFile = fileStore.ocrJsonFile(imageResult.captureId)
        jsonFile.writeText(JsonUtils.encodePretty(payload), Charsets.UTF_8)
        Log.i("ReceiptOCR", "Saved OCR JSON: ${jsonFile.absolutePath}")
        return ReceiptCaptureRecord(
            captureId = imageResult.captureId,
            imageFile = imageResult.imageFile,
            ocrJsonFile = jsonFile,
            ocrPayload = payload,
        )
    }

    fun saveServerResult(captureId: String, responseBody: String): File {
        val file = fileStore.serverResultFile(captureId)
        file.writeText(responseBody, Charsets.UTF_8)
        Log.i("ReceiptUpload", "Saved server result: ${file.absolutePath}")
        return file
    }
}
