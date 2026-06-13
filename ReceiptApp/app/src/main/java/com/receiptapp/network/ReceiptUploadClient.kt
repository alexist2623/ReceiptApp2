package com.receiptapp.network

import android.util.Log
import com.receiptapp.receipt.ReceiptCaptureRecord
import java.util.concurrent.TimeUnit
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.MultipartBody
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.asRequestBody
import okhttp3.RequestBody.Companion.toRequestBody

class ReceiptUploadClient(
    private val serverConfig: ServerConfig,
    private val client: OkHttpClient = OkHttpClient.Builder()
        .connectTimeout(30, TimeUnit.SECONDS)
        .readTimeout(90, TimeUnit.SECONDS)
        .writeTimeout(90, TimeUnit.SECONDS)
        .build(),
) {
    suspend fun upload(record: ReceiptCaptureRecord): ReceiptUploadResult = withContext(Dispatchers.IO) {
        val baseUrl = serverConfig.baseUrl
        if (baseUrl.isBlank()) {
            return@withContext ReceiptUploadResult(
                captureId = record.captureId,
                success = false,
                responseCode = null,
                responseBody = null,
                errorMessage = "Server URL is empty. Use ZIP share instead.",
            )
        }

        val url = "${baseUrl.trimEnd('/')}/api/receipt/ocr"
        Log.i("ReceiptUpload", "Uploading receipt OCR to $url")
        val body = MultipartBody.Builder()
            .setType(MultipartBody.FORM)
            .addFormDataPart("capture_id", record.captureId)
            .addFormDataPart(
                "image",
                record.imageFile.name,
                record.imageFile.asRequestBody("image/jpeg".toMediaType()),
            )
            .addFormDataPart(
                "ocr_json",
                record.ocrJsonFile.name,
                record.ocrJsonFile.asRequestBody("application/json".toMediaType()),
            )
            .build()
        val request = Request.Builder()
            .url(url)
            .post(body)
            .build()

        runCatching {
            client.newCall(request).execute().use { response ->
                val responseText = response.body?.string()
                Log.i("ReceiptUpload", "Upload response code=${response.code}")
                ReceiptUploadResult(
                    captureId = record.captureId,
                    success = response.isSuccessful,
                    responseCode = response.code,
                    responseBody = responseText,
                    errorMessage = if (response.isSuccessful) null else response.message,
                )
            }
        }.getOrElse { throwable ->
            Log.e("ReceiptUpload", "Upload failed", throwable)
            ReceiptUploadResult(
                captureId = record.captureId,
                success = false,
                responseCode = null,
                responseBody = null,
                errorMessage = throwable.message ?: throwable::class.java.simpleName,
            )
        }
    }
}
