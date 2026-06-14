package com.receiptapp.inference.layoutlmv3

import java.nio.file.Files
import java.nio.file.Path
import java.nio.file.Paths
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.int
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

data class LayoutLmV3ParityFixture(
    val root: Path,
    val sampleDir: Path,
    val modelPath: Path,
) {
    val sampleId: String = sampleDir.fileName.toString()

    fun loadInputs(): LayoutLmV3TensorInputs {
        val inputIds = NpyReader.read(sampleDir.resolve("input_ids.npy"))
        val attentionMask = NpyReader.read(sampleDir.resolve("attention_mask.npy"))
        val bbox = NpyReader.read(sampleDir.resolve("bbox.npy"))
        val pixelValues = NpyReader.read(sampleDir.resolve("pixel_values.npy"))
        return LayoutLmV3TensorInputs(
            inputIds = inputIds.requireLongs(),
            attentionMask = attentionMask.requireLongs(),
            bbox = bbox.requireLongs(),
            pixelValues = pixelValues.requireFloats(),
            inputIdsShape = inputIds.shape,
            attentionMaskShape = attentionMask.shape,
            bboxShape = bbox.shape,
            pixelValuesShape = pixelValues.shape,
        )
    }

    fun loadExpectedWordLabels(): List<ExpectedWordLabel> {
        val json = Json.parseToJsonElement(sampleDir.resolve("expected_word_labels.json").toFile().readText())
        return json.jsonArray.mapNotNull { element ->
            val obj = element.jsonObject
            val tokenIdx = obj["first_token_idx"]?.jsonPrimitive?.int ?: return@mapNotNull null
            val labelId = obj["label_id"]?.jsonPrimitive?.int ?: return@mapNotNull null
            ExpectedWordLabel(
                wordIdx = obj["word_idx"]?.jsonPrimitive?.int ?: -1,
                firstTokenIdx = tokenIdx,
                labelId = labelId,
                label = obj["label"]?.jsonPrimitive?.content ?: "",
            )
        }
    }

    companion object {
        fun locate(): LayoutLmV3ParityFixture? {
            val fixtureRoot = resolveFirstExistingDirectory(
                System.getProperty("layoutlmv3.fixtureDir") ?: System.getenv("LAYOUTLMV3_FIXTURE_DIR"),
                listOf(
                    "../fixtures/layoutlmv3_cord_int8_android",
                    "../../fixtures/layoutlmv3_cord_int8_android",
                    "fixtures/layoutlmv3_cord_int8_android",
                ),
            ) ?: return null
            val modelPath = resolveFirstExistingFile(
                System.getProperty("layoutlmv3.modelPath") ?: System.getenv("LAYOUTLMV3_MODEL_PATH"),
                listOf(
                    "../models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx",
                    "../../models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx",
                    "models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx",
                ),
            ) ?: return null
            if (!Files.isDirectory(fixtureRoot) || !Files.isRegularFile(modelPath)) {
                return null
            }
            val sampleDir = Files.list(fixtureRoot).use { stream ->
                stream.filter { Files.isDirectory(it) }
                    .sorted()
                    .findFirst()
                    .orElse(null)
            } ?: return null
            return LayoutLmV3ParityFixture(root = fixtureRoot, sampleDir = sampleDir, modelPath = modelPath)
        }

        private fun resolveFirstExistingDirectory(explicit: String?, defaults: List<String>): Path? {
            return sequenceOf(explicit).filterNotNull().plus(defaults.asSequence())
                .map { Paths.get(it).toAbsolutePath().normalize() }
                .firstOrNull { Files.isDirectory(it) }
        }

        private fun resolveFirstExistingFile(explicit: String?, defaults: List<String>): Path? {
            return sequenceOf(explicit).filterNotNull().plus(defaults.asSequence())
                .map { Paths.get(it).toAbsolutePath().normalize() }
                .firstOrNull { Files.isRegularFile(it) }
        }
    }
}
