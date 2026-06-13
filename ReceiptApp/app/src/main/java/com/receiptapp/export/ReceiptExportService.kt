package com.receiptapp.export

import com.receiptapp.receipt.ReceiptCaptureRecord
import java.io.File

interface ReceiptExportService {
    fun export(record: ReceiptCaptureRecord): File
}
