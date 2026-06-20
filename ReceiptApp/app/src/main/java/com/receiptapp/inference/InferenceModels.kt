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
    val storeName: FieldValueDto? = null,
    val items: List<ReceiptItemDto> = emptyList(),
    val subtotal: Map<String, FieldValueDto>? = null,
    val tax: Map<String, FieldValueDto>? = null,
    val taxes: List<ReceiptAmountDto> = emptyList(),
    val total: Map<String, FieldValueDto>? = null,
    val wordLabels: List<WordLabelDto> = emptyList(),
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
data class ReceiptAmountDto(
    val amountIndex: Int,
    val type: String,
    val name: FieldValueDto? = null,
    val price: FieldValueDto? = null,
    val rate: FieldValueDto? = null,
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

@Serializable
data class WordLabelDto(
    val wordIndex: Int,
    val text: String,
    val label: String,
    val confidence: Float? = null,
    val box: List<Int>? = null,
    val wordId: String? = null,
    val lineId: String? = null,
)
