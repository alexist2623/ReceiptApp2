package com.receiptapp.inference

interface ReceiptInferenceEngine {
    suspend fun infer(input: ReceiptInferenceInput): ReceiptInferenceResult
}
