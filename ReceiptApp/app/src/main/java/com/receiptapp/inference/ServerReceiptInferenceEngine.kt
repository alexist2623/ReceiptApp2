package com.receiptapp.inference

import com.receiptapp.network.ReceiptUploadClient
import com.receiptapp.receipt.ReceiptCaptureRecord
import com.receiptapp.receipt.ReceiptRepository
import com.receiptapp.util.JsonUtils

class ServerReceiptInferenceEngine(
    private val uploadClient: ReceiptUploadClient,
    private val repository: ReceiptRepository,
) : ReceiptInferenceEngine {
    override suspend fun infer(input: ReceiptInferenceInput): ReceiptInferenceResult {
        val jsonFile = input.ocrJsonFile
            ?: return ReceiptInferenceResult.Failure("Server inference requires a saved OCR JSON file.")
        val record = ReceiptCaptureRecord(
            captureId = input.captureId,
            imageFile = input.imageFile,
            ocrJsonFile = jsonFile,
            ocrPayload = input.ocrPayload,
        )
        return infer(record)
    }

    suspend fun infer(record: ReceiptCaptureRecord): ReceiptInferenceResult {
        val upload = uploadClient.upload(record)
        if (!upload.success || upload.responseBody.isNullOrBlank()) {
            return ReceiptInferenceResult.Failure(upload.errorMessage ?: "Upload failed.")
        }
        repository.saveServerResult(record.captureId, upload.responseBody)
        val parsed = runCatching {
            JsonUtils.decode<ReceiptInferenceResponse>(upload.responseBody)
        }.getOrElse {
            return ReceiptInferenceResult.Failure("Server returned non-inference JSON.", it)
        }
        return ReceiptInferenceResult.Success(parsed)
    }
}
