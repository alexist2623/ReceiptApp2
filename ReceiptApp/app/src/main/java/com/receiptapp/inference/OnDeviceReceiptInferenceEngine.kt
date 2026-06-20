package com.receiptapp.inference

import android.content.Context
import com.receiptapp.inference.layoutlmv3.LayoutLmV3Labels
import com.receiptapp.inference.layoutlmv3.LayoutLmV3ModelFileResolver
import com.receiptapp.inference.layoutlmv3.LayoutLmV3OnnxRuntimeSmokeRunner
import com.receiptapp.inference.layoutlmv3.LayoutLmV3Preprocessor
import com.receiptapp.inference.layoutlmv3.LayoutLmV3Tokenizer
import com.receiptapp.inference.relg.RelGWordPrediction
import com.receiptapp.inference.relg.SpanRelGPipeline
import com.receiptapp.inference.relg.SpanRelGSchema
import com.receiptapp.ocr.OcrWordDto

class OnDeviceReceiptInferenceEngine(
    private val context: Context,
) : ReceiptInferenceEngine {
    override suspend fun infer(input: ReceiptInferenceInput): ReceiptInferenceResult {
        val files = LayoutLmV3ModelFileResolver(context).resolve()
        if (!files.ready) {
            return ReceiptInferenceResult.Failure(
                "On-device INT8 model files not found. Put model.onnx, tokenizer.json, labels.json under " +
                    "app files/models/layoutlmv3-item-policy-int8 or app/src/main/assets_dev/layoutlmv3-item-policy-int8.",
            )
        }
        if (!files.relGReady) {
            return ReceiptInferenceResult.Failure(
                "On-device span rel-g files not found. Put span_relg.onnx and span_relg_schema.json under " +
                    "app files/models/layoutlmv3-item-policy-int8 or app/src/main/assets_dev/layoutlmv3-item-policy-int8.",
            )
        }
        return runCatching {
            val labels = LayoutLmV3Labels.fromFile(files.labelsJson)
            val relGSchema = SpanRelGSchema.fromFile(files.relGSchemaJson)
            val tokenizer = LayoutLmV3Tokenizer.fromFile(files.tokenizerJson)
            val words = input.ocrPayload.words.filter { it.text.isNotBlank() && it.box.size == 4 }
            val width = input.ocrPayload.image.width
            val height = input.ocrPayload.image.height
            val normalizedBoxes = words.map { LayoutLmV3Preprocessor.normalizeBox(it.box, width, height) }
            val encoding = tokenizer.encodeToInputs(
                words = words.map { it.text },
                boxes = normalizedBoxes,
                pixelValues = LayoutLmV3Preprocessor.pixelValues(input.imageFile),
            )
            val outputs = LayoutLmV3OnnxRuntimeSmokeRunner.run(files.model.toPath(), encoding.inputs)
            val predictions = recoverWordPredictions(
                words = words,
                normalizedBoxes = normalizedBoxes,
                logits = outputs.logits,
                firstTokenForWord = encoding.firstTokenForWord,
                labels = labels,
            )
            ReceiptInferenceResult.Success(
                SpanRelGPipeline.buildResponse(
                    captureId = input.captureId,
                    predictions = predictions,
                    firstTokenForWord = encoding.firstTokenForWord,
                    lastHiddenState = outputs.lastHiddenState,
                    relGModel = files.relGModel,
                    schema = relGSchema,
                ),
            )
        }.getOrElse { throwable ->
            ReceiptInferenceResult.Failure("On-device INT8 inference failed: ${throwable.message}", throwable)
        }
    }

    private fun recoverWordPredictions(
        words: List<OcrWordDto>,
        normalizedBoxes: List<List<Int>>,
        logits: Array<Array<FloatArray>>,
        firstTokenForWord: IntArray,
        labels: LayoutLmV3Labels,
    ): List<RelGWordPrediction> {
        val tokenLogits = logits.firstOrNull().orEmpty()
        return words.mapIndexed { index, word ->
            val tokenIndex = firstTokenForWord.getOrElse(index) { -1 }
            val scores = tokenLogits.getOrNull(tokenIndex)
            if (scores == null) {
                RelGWordPrediction(index, word, "O", 0f, normalizedBoxes[index])
            } else {
                val best = scores.indices.maxBy { scores[it] }
                RelGWordPrediction(index, word, labels.label(best), softmaxConfidence(scores, best), normalizedBoxes[index])
            }
        }
    }

    private fun softmaxConfidence(scores: FloatArray, bestIndex: Int): Float {
        val max = scores.maxOrNull() ?: return 0f
        var sum = 0.0
        scores.forEach { sum += kotlin.math.exp((it - max).toDouble()) }
        return (kotlin.math.exp((scores[bestIndex] - max).toDouble()) / sum).toFloat()
    }
}
