package com.receiptapp.receipt

import com.receiptapp.ocr.AppInfoDto
import com.receiptapp.ocr.DeviceInfoDto
import com.receiptapp.ocr.ImageInfoDto
import com.receiptapp.ocr.OcrInfoDto
import com.receiptapp.ocr.OcrWordDto
import com.receiptapp.ocr.ReceiptOcrPayload
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test

class ReceiptExportValidatorTest {
    @Test
    fun matchingImageAndPayloadSizePasses() {
        val result = ReceiptExportValidator.validateSizes(
            actualWidth = 1536,
            actualHeight = 2048,
            payload = payload(width = 1536, height = 2048),
        )

        assertTrue(result.ok)
    }

    @Test
    fun topLevelSizeMismatchFails() {
        val result = ReceiptExportValidator.validateSizes(
            actualWidth = 1536,
            actualHeight = 2048,
            payload = payload(width = 3000, height = 4000),
        )

        assertFalse(result.ok)
        assertTrue(result.errors.any { it.contains("top-level size") })
    }

    @Test
    fun nestedImageSizeMismatchFails() {
        val result = ReceiptExportValidator.validateSizes(
            actualWidth = 1536,
            actualHeight = 2048,
            payload = payload(width = 1536, height = 2048, nestedWidth = 3000, nestedHeight = 4000),
        )

        assertFalse(result.ok)
        assertTrue(result.errors.any { it.contains("image.width/height") })
    }

    @Test
    fun outOfBoundsWordBoxFails() {
        val result = ReceiptExportValidator.validateSizes(
            actualWidth = 1536,
            actualHeight = 2048,
            payload = payload(width = 1536, height = 2048, box = listOf(10, 20, 1600, 80)),
        )

        assertFalse(result.ok)
        assertTrue(result.outOfBoundsWordCount == 1)
    }

    private fun payload(
        width: Int,
        height: Int,
        nestedWidth: Int = width,
        nestedHeight: Int = height,
        box: List<Int> = listOf(10, 20, 120, 80),
    ): ReceiptOcrPayload {
        return ReceiptOcrPayload(
            captureId = "capture_test",
            createdAtUtc = "2026-06-13T00:00:00Z",
            device = DeviceInfoDto("maker", "model", "16", 35),
            app = AppInfoDto("com.receiptapp", "0.1.0", 1),
            image = ImageInfoDto(
                fileName = "capture_test.jpg",
                width = nestedWidth,
                height = nestedHeight,
                mimeType = "image/jpeg",
                exifOrientationApplied = true,
                rotationDegreesApplied = 0,
            ),
            ocr = OcrInfoDto(script = "latin", confidenceAvailable = false),
            blocks = emptyList(),
            lines = emptyList(),
            words = listOf(
                OcrWordDto(
                    wordId = "w_000000",
                    blockId = "b_000000",
                    lineId = "l_000000",
                    wordIndexInLine = 0,
                    globalWordIndex = 0,
                    text = "coffee",
                    box = box,
                ),
            ),
            image_width = width,
            image_height = height,
        )
    }
}
