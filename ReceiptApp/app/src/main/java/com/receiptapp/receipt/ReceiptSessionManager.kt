package com.receiptapp.receipt

import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow

class ReceiptSessionManager {
    private val _currentRecord = MutableStateFlow<ReceiptCaptureRecord?>(null)
    val currentRecord: StateFlow<ReceiptCaptureRecord?> = _currentRecord

    fun setCurrent(record: ReceiptCaptureRecord) {
        _currentRecord.value = record
    }

    fun clear() {
        _currentRecord.value = null
    }
}
