package com.receiptapp.inference.layoutlmv3

import java.io.File
import java.nio.charset.StandardCharsets
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

data class LayoutLmV3Encoding(
    val inputs: LayoutLmV3TensorInputs,
    val firstTokenForWord: IntArray,
)

class LayoutLmV3Tokenizer private constructor(
    private val vocab: Map<String, Int>,
    merges: List<Pair<String, String>>,
    private val maxLength: Int = 512,
) {
    private val ranks: Map<Pair<String, String>, Int> = merges.withIndex().associate { it.value to it.index }
    private val cache = mutableMapOf<String, List<String>>()
    private val byteEncoder = bytesToUnicode()
    private val clsId = vocab.getValue("<s>")
    private val sepId = vocab.getValue("</s>")
    private val padId = vocab.getValue("<pad>")
    private val unkId = vocab.getValue("<unk>")

    fun encodeToInputs(words: List<String>, boxes: List<List<Int>>, pixelValues: FloatArray): LayoutLmV3Encoding {
        val tokenIds = mutableListOf<Int>()
        val wordIds = mutableListOf<Int?>()
        val tokenBoxes = mutableListOf<List<Int>>()
        tokenIds += clsId
        wordIds += null
        tokenBoxes += ZERO_BOX
        for (wordIndex in words.indices) {
            for (piece in tokenizeWord(words[wordIndex])) {
                if (tokenIds.size >= maxLength - 1) break
                tokenIds += vocab[piece] ?: unkId
                wordIds += wordIndex
                tokenBoxes += boxes[wordIndex]
            }
            if (tokenIds.size >= maxLength - 1) break
        }
        tokenIds += sepId
        wordIds += null
        tokenBoxes += ZERO_BOX
        val inputIds = LongArray(maxLength) { padId.toLong() }
        val attention = LongArray(maxLength)
        val bbox = LongArray(maxLength * 4)
        val firstTokenForWord = IntArray(words.size) { -1 }
        for (idx in tokenIds.indices) {
            inputIds[idx] = tokenIds[idx].toLong()
            attention[idx] = 1L
            val box = tokenBoxes[idx]
            for (axis in 0 until 4) {
                bbox[idx * 4 + axis] = box[axis].toLong()
            }
            val wordId = wordIds[idx]
            if (wordId != null && wordId in firstTokenForWord.indices && firstTokenForWord[wordId] < 0) {
                firstTokenForWord[wordId] = idx
            }
        }
        return LayoutLmV3Encoding(
            inputs = LayoutLmV3TensorInputs(
                inputIds = inputIds,
                attentionMask = attention,
                bbox = bbox,
                pixelValues = pixelValues,
            ),
            firstTokenForWord = firstTokenForWord,
        )
    }

    private fun tokenizeWord(word: String): List<String> {
        val encoded = byteEncode(" $word")
        return bpe(encoded)
    }

    private fun byteEncode(value: String): String {
        val bytes = value.toByteArray(StandardCharsets.UTF_8)
        return buildString {
            bytes.forEach { append(byteEncoder[it.toInt() and 0xFF]) }
        }
    }

    private fun bpe(token: String): List<String> {
        cache[token]?.let { return it }
        if (token.isEmpty()) return emptyList()
        var word = token.map { it.toString() }
        if (word.size == 1) return word
        while (true) {
            var bestRank = Int.MAX_VALUE
            var bestPair: Pair<String, String>? = null
            for (idx in 0 until word.lastIndex) {
                val pair = word[idx] to word[idx + 1]
                val rank = ranks[pair] ?: continue
                if (rank < bestRank) {
                    bestRank = rank
                    bestPair = pair
                }
            }
            val pair = bestPair ?: break
            val merged = mutableListOf<String>()
            var idx = 0
            while (idx < word.size) {
                if (idx < word.lastIndex && word[idx] == pair.first && word[idx + 1] == pair.second) {
                    merged += pair.first + pair.second
                    idx += 2
                } else {
                    merged += word[idx]
                    idx += 1
                }
            }
            word = merged
            if (word.size == 1) break
        }
        cache[token] = word
        return word
    }

    companion object {
        private val ZERO_BOX = listOf(0, 0, 0, 0)

        fun fromFile(file: File, maxLength: Int = 512): LayoutLmV3Tokenizer {
            val root = Json.parseToJsonElement(file.readText(Charsets.UTF_8)).jsonObject
            val model = root["model"]!!.jsonObject
            val vocab = model["vocab"]!!.jsonObject.mapValues { it.value.jsonPrimitive.content.toInt() }
            val merges = model["merges"]!!.jsonArray.map { element ->
                if (element is JsonArray) {
                    element[0].jsonPrimitive.content to element[1].jsonPrimitive.content
                } else {
                    val parts = element.jsonPrimitive.content.split(" ")
                    parts[0] to parts[1]
                }
            }
            return LayoutLmV3Tokenizer(vocab = vocab, merges = merges, maxLength = maxLength)
        }

        private fun bytesToUnicode(): Map<Int, Char> {
            val bs = mutableListOf<Int>()
            bs += 33..126
            bs += 161..172
            bs += 174..255
            val cs = bs.toMutableList()
            var n = 0
            for (b in 0..255) {
                if (b !in bs) {
                    bs += b
                    cs += 256 + n
                    n += 1
                }
            }
            return bs.zip(cs.map { it.toChar() }).toMap()
        }
    }
}
