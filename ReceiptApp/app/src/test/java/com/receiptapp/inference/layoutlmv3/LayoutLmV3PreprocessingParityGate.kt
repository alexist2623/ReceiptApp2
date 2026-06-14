package com.receiptapp.inference.layoutlmv3

import kotlin.math.abs

data class TensorDiff(
    val exact: Boolean,
    val maxAbsDiff: Double,
    val meanAbsDiff: Double,
    val mismatchCount: Int,
)

data class PreprocessingParityReport(
    val inputIds: TensorDiff,
    val attentionMask: TensorDiff,
    val bbox: TensorDiff,
    val pixelValues: TensorDiff,
) {
    val exactTokenInputs: Boolean
        get() = inputIds.exact && attentionMask.exact && bbox.exact
}

object LayoutLmV3PreprocessingParityGate {
    fun compare(expected: LayoutLmV3TensorInputs, actual: LayoutLmV3TensorInputs): PreprocessingParityReport {
        return PreprocessingParityReport(
            inputIds = compareLongs(expected.inputIds, actual.inputIds),
            attentionMask = compareLongs(expected.attentionMask, actual.attentionMask),
            bbox = compareLongs(expected.bbox, actual.bbox),
            pixelValues = compareFloats(expected.pixelValues, actual.pixelValues),
        )
    }

    private fun compareLongs(expected: LongArray, actual: LongArray): TensorDiff {
        require(expected.size == actual.size) { "Tensor size mismatch: expected=${expected.size} actual=${actual.size}" }
        var mismatch = 0
        var maxDiff = 0.0
        var sumDiff = 0.0
        for (idx in expected.indices) {
            val diff = abs(expected[idx] - actual[idx]).toDouble()
            if (diff != 0.0) mismatch += 1
            if (diff > maxDiff) maxDiff = diff
            sumDiff += diff
        }
        return TensorDiff(
            exact = mismatch == 0,
            maxAbsDiff = maxDiff,
            meanAbsDiff = if (expected.isEmpty()) 0.0 else sumDiff / expected.size.toDouble(),
            mismatchCount = mismatch,
        )
    }

    private fun compareFloats(expected: FloatArray, actual: FloatArray): TensorDiff {
        require(expected.size == actual.size) { "Tensor size mismatch: expected=${expected.size} actual=${actual.size}" }
        var mismatch = 0
        var maxDiff = 0.0
        var sumDiff = 0.0
        for (idx in expected.indices) {
            val diff = abs(expected[idx] - actual[idx]).toDouble()
            if (diff != 0.0) mismatch += 1
            if (diff > maxDiff) maxDiff = diff
            sumDiff += diff
        }
        return TensorDiff(
            exact = mismatch == 0,
            maxAbsDiff = maxDiff,
            meanAbsDiff = if (expected.isEmpty()) 0.0 else sumDiff / expected.size.toDouble(),
            mismatchCount = mismatch,
        )
    }
}
