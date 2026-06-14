package com.receiptapp.util

import com.receiptapp.ocr.ReceiptOcrPayload
import java.io.File

data class ImageDimensions(
    val width: Int,
    val height: Int,
)

object ImageDimensionValidator {
    fun readImageSize(imageFile: File): ImageDimensions {
        val size = ImageSizeUtils.requirePositiveImageSize(imageFile)
        return ImageDimensions(size.width, size.height)
    }

    fun validateImageInfoMatchesFile(
        imageFile: File,
        expectedWidth: Int,
        expectedHeight: Int,
        context: String,
    ) {
        val actual = readImageSize(imageFile)
        require(actual.width == expectedWidth && actual.height == expectedHeight) {
            "$context coordinate mismatch: image file is ${actual.width}x${actual.height} " +
                "but metadata is ${expectedWidth}x${expectedHeight}. " +
                "Do not export or train with this pair."
        }
    }

    fun validatePayloadMatchesImageFile(
        imageFile: File,
        payload: ReceiptOcrPayload,
        context: String,
    ) {
        val actual = readImageSize(imageFile)
        val topW = payload.image_width
        val topH = payload.image_height
        val nestedW = payload.image.width
        val nestedH = payload.image.height

        require(topW == nestedW && topH == nestedH) {
            "$context OCR JSON self mismatch: top-level=${topW}x${topH}, image=${nestedW}x${nestedH}"
        }

        require(actual.width == topW && actual.height == topH) {
            if (context.startsWith("zipExport")) {
                "Export coordinate mismatch: image is ${actual.width}x${actual.height} " +
                    "but OCR JSON is ${topW}x${topH}. " +
                    "Re-run OCR using the saved canonical image or delete stale capture."
            } else {
                "$context coordinate mismatch: image file is ${actual.width}x${actual.height} " +
                    "but OCR JSON is ${topW}x${topH}. " +
                    "Saved canonical JPG and OCR boxes must share the same coordinate space."
            }
        }
    }
}
