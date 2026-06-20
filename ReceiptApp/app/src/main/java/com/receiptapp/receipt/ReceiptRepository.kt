package com.receiptapp.receipt

import android.content.Context
import android.util.Log
import com.receiptapp.capture.CanonicalImageResult
import com.receiptapp.inference.ReceiptInferenceResponse
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

    fun saveInferenceResult(captureId: String, response: ReceiptInferenceResponse): File {
        return saveServerResult(captureId, JsonUtils.encodePretty(response))
    }

    fun listCapturedReceipts(): List<ReceiptCaptureRecord> {
        return fileStore.receiptsRootDir()
            .listFiles()
            ?.filter { it.isDirectory }
            ?.mapNotNull { dir ->
                val captureId = dir.name
                val imageFile = fileStore.imageFile(captureId)
                val ocrJsonFile = fileStore.ocrJsonFile(captureId)
                if (!imageFile.exists() || !ocrJsonFile.exists()) {
                    return@mapNotNull null
                }
                val payload = runCatching {
                    JsonUtils.decode<ReceiptOcrPayload>(ocrJsonFile.readText(Charsets.UTF_8))
                }.getOrNull() ?: return@mapNotNull null
                ReceiptCaptureRecord(
                    captureId = captureId,
                    imageFile = imageFile,
                    ocrJsonFile = ocrJsonFile,
                    ocrPayload = payload,
                    serverResultFile = fileStore.serverResultFile(captureId).takeIf { it.exists() },
                )
            }
            ?.sortedByDescending { it.captureId }
            .orEmpty()
    }

    fun loadInferenceResult(record: ReceiptCaptureRecord): ReceiptInferenceResponse? {
        val file = record.serverResultFile?.takeIf { it.exists() } ?: fileStore.serverResultFile(record.captureId).takeIf { it.exists() }
        return file?.let {
            runCatching { JsonUtils.decode<ReceiptInferenceResponse>(it.readText(Charsets.UTF_8)) }.getOrNull()
        }
    }

    fun deleteReceipt(captureId: String): Boolean {
        val dir = fileStore.existingReceiptDir(captureId)
        if (!dir.exists()) return false
        require(dir.parentFile?.canonicalFile == fileStore.receiptsRootDir().canonicalFile) {
            "Refusing to delete outside receipts root: ${dir.absolutePath}"
        }
        return dir.deleteRecursively()
    }

    fun deleteAllReceipts(): Int {
        val root = fileStore.receiptsRootDir().canonicalFile
        return root
            .listFiles()
            ?.filter { it.isDirectory && it.parentFile?.canonicalFile == root }
            ?.count { it.deleteRecursively() }
            ?: 0
    }
}
