package com.receiptapp.inference

import com.receiptapp.ocr.ReceiptOcrPayload
import java.io.File
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.JsonElement

data class ReceiptInferenceInput(
    val captureId: String,
    val imageFile: File,
    val ocrPayload: ReceiptOcrPayload,
    val ocrJsonFile: File? = null,
)

sealed class ReceiptInferenceResult {
    data class Success(val response: ReceiptInferenceResponse) : ReceiptInferenceResult()
    data class Failure(val message: String, val throwable: Throwable? = null) : ReceiptInferenceResult()
}

@Serializable
data class ReceiptInferenceResponse(
    val schemaVersion: String = "receipt_inference_v1",
    val captureId: String,
    val status: String,
    val items: List<ReceiptItemDto> = emptyList(),
    val subtotal: Map<String, FieldValueDto>? = null,
    val total: Map<String, FieldValueDto>? = null,
    val warnings: List<String>? = null,
    val debug: Map<String, JsonElement>? = null,
)

@Serializable
data class ReceiptItemDto(
    val itemIndex: Int,
    val menuName: FieldValueDto? = null,
    val price: FieldValueDto? = null,
    val count: FieldValueDto? = null,
    val unitPrice: FieldValueDto? = null,
    val relGProb: Float? = null,
    val linkMargin: Float? = null,
    val linkStatus: String? = null,
)

@Serializable
data class FieldValueDto(
    val text: String,
    val confidence: Float? = null,
    val box: List<Int>? = null,
    val wordIndices: List<Int>? = null,
)
