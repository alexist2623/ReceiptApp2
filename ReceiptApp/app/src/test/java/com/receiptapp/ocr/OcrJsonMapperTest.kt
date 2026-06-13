package com.receiptapp.ocr

import com.receiptapp.util.JsonUtils
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class OcrJsonMapperTest {
    @Test
    fun mapsWordsWithPixelBoxesAndTopLevelPythonFields() {
        val payload = OcrJsonMapper.toPayload(
            captureId = "20260613T143012Z_ab12cd34",
            createdAtUtc = "2026-06-13T14:30:12Z",
            device = DeviceInfoDto("Google", "Pixel", "16", 35),
            app = AppInfoDto("com.receiptapp", "0.1.0", 1),
            image = ImageInfoDto(
                fileName = "20260613T143012Z_ab12cd34.jpg",
                width = 800,
                height = 1200,
                mimeType = "image/jpeg",
                exifOrientationApplied = true,
                rotationDegreesApplied = 90,
            ),
            script = OcrScript.LATIN,
            snapshot = RecognizedTextSnapshot(
                blocks = listOf(
                    RecognizedBlock(
                        text = "americano 4000",
                        box = null,
                        cornerPoints = null,
                        lines = listOf(
                            RecognizedLine(
                                text = "americano 4000",
                                box = null,
                                cornerPoints = null,
                                words = listOf(
                                    RecognizedWord("americano", listOf(120, 340, 310, 385), null),
                                    RecognizedWord("4000", listOf(600, 340, 700, 385), null),
                                ),
                            ),
                        ),
                    ),
                ),
            ),
        )

        assertEquals(800, payload.image_width)
        assertEquals(1200, payload.image_height)
        assertEquals(2, payload.words.size)
        assertEquals("americano", payload.words[0].text)
        assertEquals(listOf(120, 340, 310, 385), payload.words[0].box)

        val json = JsonUtils.encodePretty(payload)
        assertTrue(json.contains("\"image_width\""))
        assertTrue(json.contains("\"image_height\""))
        assertTrue(json.contains("\"words\""))
        assertTrue(json.contains("\"box\""))
    }

    @Test
    fun clampsInvalidOutsideBoxToImageBounds() {
        val box = OcrJsonMapper.clampBox(listOf(-50, 10, 900, 1300), 800, 1200)
        assertEquals(listOf(0, 10, 799, 1199), box)
    }
}
