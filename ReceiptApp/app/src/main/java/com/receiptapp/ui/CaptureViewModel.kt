package com.receiptapp.ui

import android.app.Application
import android.content.Intent
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.receiptapp.capture.CanonicalImageWriter
import com.receiptapp.export.ShareReceiptIntentFactory
import com.receiptapp.export.ZipExportService
import com.receiptapp.inference.InferenceMode
import com.receiptapp.inference.OnDeviceReceiptInferenceEngine
import com.receiptapp.inference.ReceiptInferenceInput
import com.receiptapp.inference.ReceiptInferenceResult
import com.receiptapp.inference.ServerReceiptInferenceEngine
import com.receiptapp.network.ReceiptUploadClient
import com.receiptapp.network.ServerConfig
import com.receiptapp.ocr.MlKitOcrEngine
import com.receiptapp.ocr.OcrScript
import com.receiptapp.receipt.ReceiptFileStore
import com.receiptapp.receipt.ReceiptExportValidator
import com.receiptapp.receipt.ReceiptCaptureRecord
import com.receiptapp.receipt.ReceiptRepository
import java.io.File
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class CaptureViewModel(
    application: Application,
) : AndroidViewModel(application) {
    private val appContext = application.applicationContext
    private val fileStore = ReceiptFileStore(appContext)
    private val repository = ReceiptRepository(appContext, fileStore)
    private val canonicalImageWriter = CanonicalImageWriter(appContext, fileStore)
    private val serverConfig = ServerConfig(appContext)
    private val uploadClient = ReceiptUploadClient(serverConfig)
    private val serverInferenceEngine = ServerReceiptInferenceEngine(uploadClient, repository)
    private val onDeviceInferenceEngine = OnDeviceReceiptInferenceEngine(appContext)
    private val zipExportService = ZipExportService(fileStore)
    private val shareIntentFactory = ShareReceiptIntentFactory(appContext)

    private val _uiState = MutableStateFlow(CaptureUiState(serverUrl = serverConfig.baseUrl))
    val uiState: StateFlow<CaptureUiState> = _uiState.asStateFlow()

    init {
        refreshHistory()
    }

    fun setOcrScript(script: OcrScript) {
        _uiState.value = _uiState.value.copy(ocrScript = script)
    }

    fun setInferenceMode(mode: InferenceMode) {
        _uiState.value = _uiState.value.copy(inferenceMode = mode)
    }

    fun setPeriodFilter(period: PeriodFilter) {
        _uiState.value = _uiState.value.copy(periodFilter = period)
    }

    fun setServerUrl(value: String) {
        val normalized = ServerConfig.normalizeBaseUrl(value)
        serverConfig.baseUrl = normalized
        _uiState.value = _uiState.value.copy(serverUrl = normalized)
    }

    fun setShowJson(show: Boolean) {
        _uiState.value = _uiState.value.copy(showJson = show)
    }

    fun showError(message: String) {
        _uiState.value = _uiState.value.copy(
            isBusy = false,
            statusMessage = null,
            errorMessage = message,
        )
    }

    fun reset() {
        _uiState.value = _uiState.value.copy(
            record = null,
            serverResponse = null,
            statusMessage = null,
            errorMessage = null,
            showJson = false,
        )
    }

    fun refreshHistory() {
        viewModelScope.launch {
            val history = withContext(Dispatchers.IO) {
                repository.listCapturedReceipts().map { record ->
                    ReceiptHistoryEntry(record = record, inference = repository.loadInferenceResult(record))
                }
            }
            _uiState.value = _uiState.value.copy(history = history)
        }
    }

    fun deleteReceipt(captureId: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isBusy = true, errorMessage = null, statusMessage = "Deleting receipt...")
            val deleted = withContext(Dispatchers.IO) { repository.deleteReceipt(captureId) }
            val currentDeleted = _uiState.value.record?.captureId == captureId
            val history = withContext(Dispatchers.IO) {
                repository.listCapturedReceipts().map { record ->
                    ReceiptHistoryEntry(record = record, inference = repository.loadInferenceResult(record))
                }
            }
            _uiState.value = _uiState.value.copy(
                isBusy = false,
                record = if (currentDeleted) null else _uiState.value.record,
                serverResponse = if (currentDeleted) null else _uiState.value.serverResponse,
                history = history,
                statusMessage = if (deleted) "Receipt deleted." else "Receipt was already removed.",
                errorMessage = null,
            )
        }
    }

    fun deleteAllReceipts() {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isBusy = true, errorMessage = null, statusMessage = "Deleting all receipts...")
            val deletedCount = withContext(Dispatchers.IO) { repository.deleteAllReceipts() }
            _uiState.value = _uiState.value.copy(
                isBusy = false,
                record = null,
                serverResponse = null,
                history = emptyList(),
                statusMessage = "Deleted $deletedCount receipt(s).",
                errorMessage = null,
            )
        }
    }

    fun processCameraFile(file: File) {
        viewModelScope.launch {
            processImage {
                canonicalImageWriter.writeFromFile(file)
            }
        }
    }

    fun processGalleryUri(uri: Uri) {
        viewModelScope.launch {
            processImage {
                canonicalImageWriter.writeFromUri(uri)
            }
        }
    }

    private suspend fun processImage(createCanonical: () -> com.receiptapp.capture.CanonicalImageResult) {
        val script = _uiState.value.ocrScript
        _uiState.value = _uiState.value.copy(
            isBusy = true,
            statusMessage = "OCR running...",
            errorMessage = null,
            serverResponse = null,
        )
        try {
            val record = withContext(Dispatchers.IO) {
                val imageResult = createCanonical()
                val ocrEngine = MlKitOcrEngine(appContext) { _, _ -> imageResult.imageInfo }
                val payload = ocrEngine.recognize(
                    captureId = imageResult.captureId,
                    canonicalImageFile = imageResult.imageFile,
                    script = script,
                )
                repository.saveOcrPayload(imageResult, payload)
            }
            _uiState.value = _uiState.value.copy(
                isBusy = false,
                statusMessage = "OCR complete: ${record.ocrPayload.words.size} words, " +
                    "image=${record.ocrPayload.image.width}x${record.ocrPayload.image.height}, " +
                    "json=${record.ocrPayload.image_width}x${record.ocrPayload.image_height}",
                record = record,
            )
            Log.i("ReceiptOCR", "Capture folder: ${record.imageFile.parentFile?.absolutePath}")
            refreshHistory()
            if (_uiState.value.inferenceMode == InferenceMode.ON_DEVICE_INT8) {
                runOnDeviceInference(record)
            }
        } catch (throwable: Throwable) {
            Log.e("ReceiptOCR", "OCR failed", throwable)
            _uiState.value = _uiState.value.copy(
                isBusy = false,
                errorMessage = throwable.message ?: throwable::class.java.simpleName,
                statusMessage = null,
            )
        }
    }

    fun createShareIntent(): Intent? {
        val record = _uiState.value.record ?: return null
        return runCatching {
            val zip = zipExportService.export(record)
            Intent.createChooser(
                shareIntentFactory.createShareZipIntent(zip),
                "Share receipt OCR ZIP",
            )
        }.getOrElse { throwable ->
            _uiState.value = _uiState.value.copy(
                isBusy = false,
                errorMessage = throwable.message ?: throwable::class.java.simpleName,
                statusMessage = null,
            )
            null
        }
    }

    fun uploadToServer() {
        val record = _uiState.value.record ?: return
        viewModelScope.launch {
            val validation = ReceiptExportValidator.validateRecordForExport(record)
            if (!validation.ok) {
                _uiState.value = _uiState.value.copy(
                    isBusy = false,
                    statusMessage = null,
                    errorMessage = validation.errors.joinToString(separator = "\n"),
                )
                return@launch
            }
            _uiState.value = _uiState.value.copy(isBusy = true, statusMessage = "Uploading...", errorMessage = null)
            val result = serverInferenceEngine.infer(record)
            when (result) {
                is ReceiptInferenceResult.Success -> {
                    _uiState.value = _uiState.value.copy(
                        isBusy = false,
                        statusMessage = "Upload complete",
                        serverResponse = result.response,
                    )
                }
                is ReceiptInferenceResult.Failure -> {
                    _uiState.value = _uiState.value.copy(
                        isBusy = false,
                        errorMessage = result.message,
                        statusMessage = "Upload failed. Share ZIP is still available.",
                    )
                }
            }
        }
    }

    fun runOnDeviceInference(record: ReceiptCaptureRecord? = _uiState.value.record) {
        record ?: return
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isBusy = true, statusMessage = "Running on-device INT8 model...", errorMessage = null)
            val result = withContext(Dispatchers.IO) {
                onDeviceInferenceEngine.infer(
                    ReceiptInferenceInput(
                        captureId = record.captureId,
                        imageFile = record.imageFile,
                        ocrPayload = record.ocrPayload,
                        ocrJsonFile = record.ocrJsonFile,
                    ),
                )
            }
            when (result) {
                is ReceiptInferenceResult.Success -> {
                    val resultFile = withContext(Dispatchers.IO) {
                        repository.saveInferenceResult(record.captureId, result.response)
                    }
                    val updatedRecord = record.copy(serverResultFile = resultFile)
                    _uiState.value = _uiState.value.copy(
                        isBusy = false,
                        statusMessage = "On-device INT8 inference complete",
                        record = updatedRecord,
                        serverResponse = result.response,
                    )
                    refreshHistory()
                }
                is ReceiptInferenceResult.Failure -> {
                    _uiState.value = _uiState.value.copy(
                        isBusy = false,
                        statusMessage = "OCR complete. On-device inference is not ready.",
                        errorMessage = result.message,
                    )
                    refreshHistory()
                }
            }
        }
    }
}
