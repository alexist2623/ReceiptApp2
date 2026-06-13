package com.receiptapp.export

import android.content.Context
import android.content.Intent
import androidx.core.content.FileProvider
import java.io.File

class ShareReceiptIntentFactory(
    private val context: Context,
) {
    fun createShareZipIntent(zipFile: File): Intent {
        val uri = FileProvider.getUriForFile(
            context,
            "${context.packageName}.fileprovider",
            zipFile,
        )
        return Intent(Intent.ACTION_SEND).apply {
            type = "application/zip"
            putExtra(Intent.EXTRA_STREAM, uri)
            addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
        }
    }
}
