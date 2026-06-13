package com.receiptapp.ocr

enum class OcrScript(val wireValue: String) {
    LATIN("latin"),
    KOREAN("korean");

    companion object {
        fun fromWireValue(value: String?): OcrScript {
            return entries.firstOrNull { it.wireValue == value?.lowercase() } ?: LATIN
        }
    }
}
