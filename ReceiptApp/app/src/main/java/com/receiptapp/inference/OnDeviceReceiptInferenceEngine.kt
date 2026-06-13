package com.receiptapp.inference

class OnDeviceReceiptInferenceEngine : ReceiptInferenceEngine {
    override suspend fun infer(input: ReceiptInferenceInput): ReceiptInferenceResult {
        return ReceiptInferenceResult.Failure(
            message = "On-device LayoutLMv3/span-relg inference is not implemented yet.",
            throwable = null,
        )
        /*
         * TODO:
         * - Run LayoutLMv3 token classification with ONNX Runtime Mobile.
         * - Bundle tokenizer vocab/merges/config and image processor config.
         * - Normalize OCR boxes to 0..1000.
         * - Recreate LayoutLMv3 image preprocessing on Android.
         * - Run span-level rel-g via ONNX or a lightweight Kotlin implementation.
         */
    }
}
