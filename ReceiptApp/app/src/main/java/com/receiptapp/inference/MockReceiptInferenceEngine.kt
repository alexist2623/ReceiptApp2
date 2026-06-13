package com.receiptapp.inference

class MockReceiptInferenceEngine : ReceiptInferenceEngine {
    override suspend fun infer(input: ReceiptInferenceInput): ReceiptInferenceResult {
        return ReceiptInferenceResult.Success(
            ReceiptInferenceResponse(
                captureId = input.captureId,
                status = "mock",
                items = listOf(
                    ReceiptItemDto(
                        itemIndex = 0,
                        menuName = FieldValueDto("mock item"),
                        price = FieldValueDto("0"),
                        linkStatus = "mock",
                    ),
                ),
                warnings = listOf("This is a UI-only mock response."),
            ),
        )
    }
}
