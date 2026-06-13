package com.receiptapp.network

data class ReceiptUploadResult(
    val captureId: String,
    val success: Boolean,
    val responseCode: Int?,
    val responseBody: String?,
    val errorMessage: String?,
)
