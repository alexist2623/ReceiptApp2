package com.receiptapp.receipt

import com.receiptapp.ocr.ReceiptOcrPayload
import com.receiptapp.util.ImageSizeUtils
import java.io.File
import kotlinx.serialization.Serializable

@Serializable
data class ExportValidationResult(
    val ok: Boolean,
    val imageWidth: Int? = null,
    val imageHeight: Int? = null,
    val jsonImageWidth: Int? = null,
    val jsonImageHeight: Int? = null,
    val errors: List<String> = emptyList(),
    val warnings: List<String> = emptyList(),
    val outOfBoundsWordCount: Int = 0,
)

@Serializable
data class ExportValidationSummary(
    val captureId: String,
    val imageWidth: Int?,
    val imageHeight: Int?,
    val jsonImageWidth: Int?,
    val jsonImageHeight: Int?,
    val ok: Boolean,
    val errors: List<String>,
    val warnings: List<String>,
    val outOfBoundsWordCount: Int,
)

object ReceiptExportValidator {
    fun validateRecordForExport(record: ReceiptCaptureRecord): ExportValidationResult {
        val preflightErrors = buildList {
            if (!record.imageFile.exists()) add("Image file missing: ${record.imageFile.absolutePath}")
            if (!record.ocrJsonFile.exists()) add("OCR JSON file missing: ${record.ocrJsonFile.absolutePath}")
        }
        if (preflightErrors.isNotEmpty()) {
            return ExportValidationResult(ok = false, errors = preflightErrors)
        }
        return validateImageAndPayload(record.imageFile, record.ocrPayload)
    }

    fun validateImageAndPayload(imageFile: File, payload: ReceiptOcrPayload): ExportValidationResult {
        return runCatching {
            val imageSize = ImageSizeUtils.requirePositiveImageSize(imageFile)
            validateSizes(
                actualWidth = imageSize.width,
                actualHeight = imageSize.height,
                payload = payload,
                imageFile = imageFile,
            )
        }.getOrElse { throwable ->
            ExportValidationResult(
                ok = false,
                errors = listOf(throwable.message ?: throwable::class.java.simpleName),
            )
        }
    }

    fun validateSizes(
        actualWidth: Int,
        actualHeight: Int,
        payload: ReceiptOcrPayload,
        imageFile: File? = null,
    ): ExportValidationResult {
        val errors = mutableListOf<String>()
        val warnings = mutableListOf<String>()
        val pathHint = imageFile?.absolutePath?.let { " file=$it" }.orEmpty()

        if (payload.image_width != actualWidth || payload.image_height != actualHeight) {
            errors += "Export coordinate mismatch: image file is ${actualWidth}x${actualHeight} " +
                "but OCR JSON top-level size is ${payload.image_width}x${payload.image_height}.$pathHint"
        }
        if (payload.image.width != actualWidth || payload.image.height != actualHeight) {
            errors += "Export coordinate mismatch: image file is ${actualWidth}x${actualHeight} " +
                "but OCR JSON image.width/height is ${payload.image.width}x${payload.image.height}.$pathHint"
        }
        if (payload.image_width != payload.image.width || payload.image_height != payload.image.height) {
            errors += "OCR JSON self mismatch: top-level=${payload.image_width}x${payload.image_height}, " +
                "image=${payload.image.width}x${payload.image.height}."
        }

        val outOfBounds = payload.words.count { word ->
            val box = word.box
            box.size != 4 ||
                box[2] <= box[0] ||
                box[3] <= box[1] ||
                box[0] < 0 ||
                box[1] < 0 ||
                box[2] >= actualWidth ||
                box[3] >= actualHeight
        }
        if (outOfBounds > 0) {
            errors += "$outOfBounds OCR word boxes are outside the actual image bounds ${actualWidth}x${actualHeight}."
        }
        if (payload.words.isEmpty()) {
            warnings += "OCR JSON has no words."
        }

        return ExportValidationResult(
            ok = errors.isEmpty(),
            imageWidth = actualWidth,
            imageHeight = actualHeight,
            jsonImageWidth = payload.image_width,
            jsonImageHeight = payload.image_height,
            errors = errors,
            warnings = warnings,
            outOfBoundsWordCount = outOfBounds,
        )
    }

    fun requireValidForExport(record: ReceiptCaptureRecord): ExportValidationResult {
        val result = validateRecordForExport(record)
        require(result.ok) { result.errors.joinToString(separator = "\n") }
        return result
    }

    fun toSummary(captureId: String, result: ExportValidationResult): ExportValidationSummary {
        return ExportValidationSummary(
            captureId = captureId,
            imageWidth = result.imageWidth,
            imageHeight = result.imageHeight,
            jsonImageWidth = result.jsonImageWidth,
            jsonImageHeight = result.jsonImageHeight,
            ok = result.ok,
            errors = result.errors,
            warnings = result.warnings,
            outOfBoundsWordCount = result.outOfBoundsWordCount,
        )
    }
}
