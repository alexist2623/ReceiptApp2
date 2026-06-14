package com.receiptapp.export

import com.receiptapp.receipt.ReceiptExportValidator
import com.receiptapp.receipt.ReceiptCaptureRecord
import com.receiptapp.receipt.ReceiptFileStore
import com.receiptapp.util.JsonUtils
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class ZipExportService(
    private val fileStore: ReceiptFileStore,
) : ReceiptExportService {
    override fun export(record: ReceiptCaptureRecord): File {
        val validation = ReceiptExportValidator.requireValidForExport(record)
        val zipFile = fileStore.zipFile(record.captureId)
        val validationJson = JsonUtils.encodePretty(
            ReceiptExportValidator.toSummary(record.captureId, validation),
        )
        val files = buildList {
            add(record.imageFile to "${record.captureId}.jpg")
            add(record.ocrJsonFile to "${record.captureId}_ocr.json")
            record.serverResultFile?.takeIf { it.exists() }?.let {
                add(it to "${record.captureId}_server_result.json")
            }
        }
        zipFiles(
            zipFile = zipFile,
            files = files,
            textEntries = listOf("${record.captureId}_export_validation.json" to validationJson),
        )
        return zipFile
    }

    companion object {
        fun zipFiles(
            zipFile: File,
            files: List<Pair<File, String>>,
            textEntries: List<Pair<String, String>> = emptyList(),
        ) {
            ZipOutputStream(FileOutputStream(zipFile)).use { zip ->
                files.forEach { (file, entryName) ->
                    addFile(zip, file, entryName)
                }
                textEntries.forEach { (entryName, text) ->
                    addText(zip, entryName, text)
                }
            }
        }

        private fun addFile(zip: ZipOutputStream, file: File, entryName: String) {
            zip.putNextEntry(ZipEntry(entryName))
            FileInputStream(file).use { input ->
                input.copyTo(zip)
            }
            zip.closeEntry()
        }

        private fun addText(zip: ZipOutputStream, entryName: String, text: String) {
            zip.putNextEntry(ZipEntry(entryName))
            zip.write(text.toByteArray(Charsets.UTF_8))
            zip.closeEntry()
        }
    }
}
