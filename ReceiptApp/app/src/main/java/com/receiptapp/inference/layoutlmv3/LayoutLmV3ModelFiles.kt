package com.receiptapp.inference.layoutlmv3

import android.content.Context
import java.io.File

data class LayoutLmV3ModelFiles(
    val model: File,
    val tokenizerJson: File,
    val labelsJson: File,
    val relGModel: File,
    val relGSchemaJson: File,
) {
    val ready: Boolean
        get() = model.exists() && tokenizerJson.exists() && labelsJson.exists()

    val relGReady: Boolean
        get() = relGModel.exists() && relGSchemaJson.exists()
}

class LayoutLmV3ModelFileResolver(
    private val context: Context,
) {
    private val modelDir: File = File(context.filesDir, "models/layoutlmv3-item-policy-int8")
    private val assetRoot = "layoutlmv3-item-policy-int8"

    fun resolve(): LayoutLmV3ModelFiles {
        modelDir.mkdirs()
        copyAssetIfMissingOrChanged("model.onnx")
        copyAssetIfMissingOrChanged("tokenizer.json")
        copyAssetIfMissingOrChanged("labels.json")
        copyAssetIfMissingOrChanged("span_relg.onnx")
        copyAssetIfMissingOrChanged("span_relg.onnx.data")
        copyAssetIfMissingOrChanged("span_relg_schema.json")
        return LayoutLmV3ModelFiles(
            model = File(modelDir, "model.onnx"),
            tokenizerJson = File(modelDir, "tokenizer.json"),
            labelsJson = File(modelDir, "labels.json"),
            relGModel = File(modelDir, "span_relg.onnx"),
            relGSchemaJson = File(modelDir, "span_relg_schema.json"),
        )
    }

    private fun copyAssetIfMissingOrChanged(name: String) {
        val target = File(modelDir, name)
        val assetPath = "$assetRoot/$name"
        val assetLength = assetLength(assetPath) ?: if (target.exists()) return else null
        if (target.exists() && assetLength != null && target.length() == assetLength) return
        runCatching {
            context.assets.open(assetPath).use { input ->
                target.outputStream().use { output -> input.copyTo(output) }
            }
        }.onFailure {
            target.delete()
        }
    }

    private fun assetLength(assetPath: String): Long? {
        return runCatching {
            context.assets.openFd(assetPath).use { descriptor -> descriptor.length }
        }.getOrNull()
    }
}
