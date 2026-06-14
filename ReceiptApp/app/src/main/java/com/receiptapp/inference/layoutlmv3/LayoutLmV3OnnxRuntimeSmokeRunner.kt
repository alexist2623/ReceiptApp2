package com.receiptapp.inference.layoutlmv3

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.file.Path

data class LayoutLmV3TensorInputs(
    val inputIds: LongArray,
    val attentionMask: LongArray,
    val bbox: LongArray,
    val pixelValues: FloatArray,
    val inputIdsShape: LongArray = longArrayOf(1, 512),
    val attentionMaskShape: LongArray = longArrayOf(1, 512),
    val bboxShape: LongArray = longArrayOf(1, 512, 4),
    val pixelValuesShape: LongArray = longArrayOf(1, 3, 224, 224),
) {
    fun validate() {
        require(inputIds.size == inputIdsShape.product()) { "input_ids size mismatch" }
        require(attentionMask.size == attentionMaskShape.product()) { "attention_mask size mismatch" }
        require(bbox.size == bboxShape.product()) { "bbox size mismatch" }
        require(pixelValues.size == pixelValuesShape.product()) { "pixel_values size mismatch" }
    }
}

data class LayoutLmV3OnnxOutputs(
    val logits: Array<Array<FloatArray>>,
    val lastHiddenState: Array<Array<FloatArray>>,
) {
    val logitsShape: List<Int>
        get() = listOf(logits.size, logits.firstOrNull()?.size ?: 0, logits.firstOrNull()?.firstOrNull()?.size ?: 0)

    val lastHiddenStateShape: List<Int>
        get() = listOf(
            lastHiddenState.size,
            lastHiddenState.firstOrNull()?.size ?: 0,
            lastHiddenState.firstOrNull()?.firstOrNull()?.size ?: 0,
        )
}

data class ExpectedWordLabel(
    val wordIdx: Int,
    val firstTokenIdx: Int,
    val labelId: Int,
    val label: String,
)

data class WordLabelAgreement(
    val checked: Int,
    val matched: Int,
    val agreement: Double,
    val mismatches: List<String>,
)

object LayoutLmV3OnnxRuntimeSmokeRunner {
    fun run(modelPath: Path, inputs: LayoutLmV3TensorInputs): LayoutLmV3OnnxOutputs {
        inputs.validate()
        val env = OrtEnvironment.getEnvironment()
        env.createSession(modelPath.toString()).use { session ->
            val inputIds = OnnxTensor.createTensor(env, LongBuffer.wrap(inputs.inputIds), inputs.inputIdsShape)
            val attentionMask = OnnxTensor.createTensor(env, LongBuffer.wrap(inputs.attentionMask), inputs.attentionMaskShape)
            val bbox = OnnxTensor.createTensor(env, LongBuffer.wrap(inputs.bbox), inputs.bboxShape)
            val pixelValues = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputs.pixelValues), inputs.pixelValuesShape)
            inputIds.use {
                attentionMask.use {
                    bbox.use {
                        pixelValues.use {
                            val feed = mapOf(
                                "input_ids" to inputIds,
                                "attention_mask" to attentionMask,
                                "bbox" to bbox,
                                "pixel_values" to pixelValues,
                            )
                            session.run(feed).use { result ->
                                val logits = result.get("logits").orElseThrow { IllegalStateException("Missing logits output") }
                                    .value as Array<Array<FloatArray>>
                                val hidden = result.get("last_hidden_state")
                                    .orElseThrow { IllegalStateException("Missing last_hidden_state output") }
                                    .value as Array<Array<FloatArray>>
                                return LayoutLmV3OnnxOutputs(logits = logits, lastHiddenState = hidden)
                            }
                        }
                    }
                }
            }
        }
    }

    fun compareExpectedWordLabels(
        logits: Array<Array<FloatArray>>,
        expected: List<ExpectedWordLabel>,
    ): WordLabelAgreement {
        val sampleLogits = logits.firstOrNull() ?: return WordLabelAgreement(0, 0, 0.0, listOf("empty logits"))
        var matched = 0
        val mismatches = mutableListOf<String>()
        for (row in expected) {
            if (row.firstTokenIdx !in sampleLogits.indices) {
                mismatches += "word=${row.wordIdx} token=${row.firstTokenIdx} outside logits"
                continue
            }
            val predicted = argmax(sampleLogits[row.firstTokenIdx])
            if (predicted == row.labelId) {
                matched += 1
            } else {
                mismatches += "word=${row.wordIdx} token=${row.firstTokenIdx} expected=${row.labelId}/${row.label} predicted=$predicted"
            }
        }
        val checked = expected.size
        return WordLabelAgreement(
            checked = checked,
            matched = matched,
            agreement = if (checked == 0) 0.0 else matched.toDouble() / checked.toDouble(),
            mismatches = mismatches,
        )
    }

    private fun argmax(values: FloatArray): Int {
        var bestIndex = 0
        var bestValue = Float.NEGATIVE_INFINITY
        for (idx in values.indices) {
            if (values[idx] > bestValue) {
                bestValue = values[idx]
                bestIndex = idx
            }
        }
        return bestIndex
    }
}

private fun LongArray.product(): Int = fold(1L) { acc, value -> acc * value }.toInt()
