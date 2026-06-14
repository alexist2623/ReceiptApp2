package com.receiptapp.ocr

import kotlinx.serialization.Serializable

@Serializable
data class ReceiptOcrPayload(
    val schemaVersion: String = "receipt_ocr_v1",
    val captureId: String,
    val createdAtUtc: String,
    val device: DeviceInfoDto,
    val app: AppInfoDto,
    val image: ImageInfoDto,
    val ocr: OcrInfoDto,
    val blocks: List<OcrBlockDto>,
    val lines: List<OcrLineDto>,
    val words: List<OcrWordDto>,
    val image_width: Int,
    val image_height: Int,
    val debug: OcrDebugInfoDto? = null,
)

@Serializable
data class DeviceInfoDto(
    val manufacturer: String,
    val model: String,
    val androidVersion: String,
    val sdkInt: Int,
)

@Serializable
data class AppInfoDto(
    val packageName: String,
    val versionName: String,
    val versionCode: Long,
)

@Serializable
data class ImageInfoDto(
    val fileName: String,
    val width: Int,
    val height: Int,
    val mimeType: String,
    val coordinateSpace: String = "saved_canonical_image_pixels",
    val exifOrientationApplied: Boolean,
    val rotationDegreesApplied: Int,
    val sha256: String? = null,
    val originalWidth: Int? = null,
    val originalHeight: Int? = null,
)

@Serializable
data class OcrInfoDto(
    val engine: String = "mlkit_text_recognition_v2",
    val script: String,
    val source: String = "on_device_ocr",
    val confidenceAvailable: Boolean,
    val note: String? = null,
)

@Serializable
data class OcrDebugInfoDto(
    val canonicalImageActualWidth: Int,
    val canonicalImageActualHeight: Int,
    val ocrInputBitmapWidth: Int,
    val ocrInputBitmapHeight: Int,
    val savedImagePathHint: String? = null,
    val coordinateValidation: String,
    val appBuildType: String? = null,
    val gitCommitHint: String? = null,
    val buildTimeUtc: String? = null,
)

@Serializable
data class OcrBlockDto(
    val blockId: String,
    val text: String,
    val box: List<Int>,
    val cornerPoints: List<List<Int>>? = null,
    val lineIds: List<String>,
)

@Serializable
data class OcrLineDto(
    val lineId: String,
    val blockId: String,
    val text: String,
    val box: List<Int>,
    val cornerPoints: List<List<Int>>? = null,
    val wordIds: List<String>,
)

@Serializable
data class OcrWordDto(
    val wordId: String,
    val blockId: String,
    val lineId: String,
    val wordIndexInLine: Int,
    val globalWordIndex: Int,
    val text: String,
    val box: List<Int>,
    val cornerPoints: List<List<Int>>? = null,
    val confidence: Float? = null,
    val recognizedLanguage: String? = null,
)

data class RecognizedTextSnapshot(
    val blocks: List<RecognizedBlock>,
)

data class RecognizedBlock(
    val text: String,
    val box: List<Int>?,
    val cornerPoints: List<List<Int>>?,
    val lines: List<RecognizedLine>,
)

data class RecognizedLine(
    val text: String,
    val box: List<Int>?,
    val cornerPoints: List<List<Int>>?,
    val words: List<RecognizedWord>,
)

data class RecognizedWord(
    val text: String,
    val box: List<Int>?,
    val cornerPoints: List<List<Int>>?,
    val confidence: Float? = null,
    val recognizedLanguage: String? = null,
)
