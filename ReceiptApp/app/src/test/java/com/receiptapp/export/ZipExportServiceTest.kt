package com.receiptapp.export

import java.io.File
import java.util.zip.ZipFile
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test

class ZipExportServiceTest {
    @Test
    fun zipIncludesImageAndJson() {
        val dir = createTempDir(prefix = "receipt_zip_test")
        val image = File(dir, "capture.jpg").apply { writeText("image") }
        val json = File(dir, "capture_ocr.json").apply { writeText("{}") }
        val zip = File(dir, "capture_receipt_ocr.zip")

        ZipExportService.zipFiles(
            zipFile = zip,
            files = listOf(image to image.name, json to json.name),
        )

        assertTrue(zip.exists())
        ZipFile(zip).use { zipFile ->
            val entries = zipFile.entries().asSequence().map { it.name }.toList()
            assertEquals(listOf("capture.jpg", "capture_ocr.json"), entries)
        }
    }
}
