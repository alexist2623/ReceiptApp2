package com.receiptapp.receipt

import android.content.Context
import java.io.File

class ReceiptFileStore(
    context: Context,
) {
    private val rootDir: File = File(context.filesDir, "receipts")

    fun receiptsRootDir(): File = rootDir.apply { mkdirs() }

    fun existingReceiptDir(captureId: String): File = File(rootDir, captureId)

    fun receiptDir(captureId: String): File = File(rootDir, captureId).apply { mkdirs() }

    fun imageFile(captureId: String): File = File(receiptDir(captureId), "$captureId.jpg")

    fun ocrJsonFile(captureId: String): File = File(receiptDir(captureId), "${captureId}_ocr.json")

    fun serverResultFile(captureId: String): File = File(receiptDir(captureId), "${captureId}_server_result.json")

    fun zipFile(captureId: String): File = File(receiptDir(captureId), "${captureId}_receipt_ocr.zip")
}
