package com.receiptapp.inference.layoutlmv3

import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

class LayoutLmV3Labels private constructor(
    private val idToLabel: Map<Int, String>,
) {
    val size: Int = idToLabel.size

    fun label(id: Int): String = idToLabel[id] ?: "O"

    companion object {
        fun fromFile(file: File): LayoutLmV3Labels {
            val root = Json.parseToJsonElement(file.readText(Charsets.UTF_8)).jsonObject
            val id2label = root["id2label"]?.jsonObject
                ?: error("labels.json missing id2label")
            return LayoutLmV3Labels(
                id2label.mapKeys { it.key.toInt() }.mapValues { it.value.jsonPrimitive.content },
            )
        }
    }
}
