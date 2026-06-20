package com.receiptapp.inference.relg

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import java.nio.FloatBuffer
import java.nio.LongBuffer
import java.nio.file.Path

data class SpanRelGInputs(
    val nodeHidden: FloatArray,
    val nodeFieldIds: LongArray,
    val nodeKindIds: LongArray,
    val nodeBoxes: FloatArray,
    val nodeMask: FloatArray,
    val candidatePairs: LongArray,
    val nodeCount: Int,
    val pairCount: Int,
    val hiddenDim: Int = 768,
)

data class SpanRelGOutputs(
    val logits: FloatArray,
    val probs: FloatArray,
)

object SpanRelGOnnxRunner {
    fun run(modelPath: Path, inputs: SpanRelGInputs): SpanRelGOutputs {
        if (inputs.pairCount == 0) return SpanRelGOutputs(FloatArray(0), FloatArray(0))
        require(inputs.nodeHidden.size == inputs.nodeCount * inputs.hiddenDim) { "node_hidden size mismatch" }
        require(inputs.nodeFieldIds.size == inputs.nodeCount) { "node_field_ids size mismatch" }
        require(inputs.nodeKindIds.size == inputs.nodeCount) { "node_kind_ids size mismatch" }
        require(inputs.nodeBoxes.size == inputs.nodeCount * 4) { "node_boxes size mismatch" }
        require(inputs.nodeMask.size == inputs.nodeCount) { "node_mask size mismatch" }
        require(inputs.candidatePairs.size == inputs.pairCount * 3) { "candidate_pairs size mismatch" }

        val env = OrtEnvironment.getEnvironment()
        env.createSession(modelPath.toString()).use { session ->
            val nodeHidden = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(inputs.nodeHidden),
                longArrayOf(1, inputs.nodeCount.toLong(), inputs.hiddenDim.toLong()),
            )
            val nodeFieldIds = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(inputs.nodeFieldIds),
                longArrayOf(1, inputs.nodeCount.toLong()),
            )
            val nodeKindIds = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(inputs.nodeKindIds),
                longArrayOf(1, inputs.nodeCount.toLong()),
            )
            val nodeBoxes = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(inputs.nodeBoxes),
                longArrayOf(1, inputs.nodeCount.toLong(), 4),
            )
            val nodeMask = OnnxTensor.createTensor(
                env,
                FloatBuffer.wrap(inputs.nodeMask),
                longArrayOf(1, inputs.nodeCount.toLong()),
            )
            val candidatePairs = OnnxTensor.createTensor(
                env,
                LongBuffer.wrap(inputs.candidatePairs),
                longArrayOf(inputs.pairCount.toLong(), 3),
            )
            nodeHidden.use {
                nodeFieldIds.use {
                    nodeKindIds.use {
                        nodeBoxes.use {
                            nodeMask.use {
                                candidatePairs.use {
                                    val feed = mapOf(
                                        "node_hidden" to nodeHidden,
                                        "node_field_ids" to nodeFieldIds,
                                        "node_kind_ids" to nodeKindIds,
                                        "node_boxes" to nodeBoxes,
                                        "node_mask" to nodeMask,
                                        "candidate_pairs" to candidatePairs,
                                    )
                                    session.run(feed).use { result ->
                                        val logits = flattenFloat(result.get("logits").orElseThrow().value)
                                        val probs = flattenFloat(result.get("probs").orElseThrow().value)
                                        return SpanRelGOutputs(logits = logits, probs = probs)
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    private fun flattenFloat(value: Any?): FloatArray {
        return when (value) {
            is FloatArray -> value
            is Array<*> -> value.flatMap { flattenFloat(it).asIterable() }.toFloatArray()
            else -> error("Unsupported ONNX float output type: ${value?.javaClass?.name}")
        }
    }
}
