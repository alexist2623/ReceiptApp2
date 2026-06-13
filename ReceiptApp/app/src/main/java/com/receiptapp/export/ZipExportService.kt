package com.receiptapp.export

import com.receiptapp.receipt.ReceiptCaptureRecord
import com.receiptapp.receipt.ReceiptFileStore
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

class ZipExportService(
    private val fileStore: ReceiptFileStore,
) : ReceiptExportService {
    override fun export(record: ReceiptCaptureRecord): File {
        val zipFile = fileStore.zipFile(record.captureId)
        val files = buildList {
            add(record.imageFile to record.imageFile.name)
            add(record.ocrJsonFile to record.ocrJsonFile.name)
            record.serverResultFile?.takeIf { it.exists() }?.let { add(it to it.name) }
        }
        zipFiles(zipFile, files)
        return zipFile
    }

    companion object {
        fun zipFiles(zipFile: File, files: List<Pair<File, String>>) {
            ZipOutputStream(FileOutputStream(zipFile)).use { zip ->
                files.forEach { (file, entryName) ->
                    addFile(zip, file, entryName)
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
    }
}
