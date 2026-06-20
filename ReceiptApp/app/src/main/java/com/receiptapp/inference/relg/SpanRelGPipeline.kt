package com.receiptapp.inference.relg

import com.receiptapp.inference.FieldValueDto
import com.receiptapp.inference.ReceiptAmountDto
import com.receiptapp.inference.ReceiptInferenceResponse
import com.receiptapp.inference.ReceiptItemDto
import com.receiptapp.inference.WordLabelDto
import com.receiptapp.ocr.OcrWordDto
import kotlinx.serialization.json.JsonPrimitive
import java.io.File

data class RelGWordPrediction(
    val index: Int,
    val word: OcrWordDto,
    val label: String,
    val confidence: Float,
    val normalizedBox: List<Int>,
)

data class RelGSpan(
    val spanId: Int,
    val field: String,
    val words: List<RelGWordPrediction>,
) {
    val firstWordIndex: Int = words.first().index
    val wordIndices: List<Int> = words.map { it.index }
    val confidence: Float = words.map { it.confidence }.average().toFloat()
    val text: String = words.joinToString(" ") { it.word.text }
    val box: List<Int> = unionBoxes(words.map { it.word.box })
    val normalizedBox: List<Int> = unionBoxes(words.map { it.normalizedBox })

    fun toFieldValue(): FieldValueDto = FieldValueDto(
        text = text,
        confidence = confidence,
        box = box,
        wordIndices = wordIndices,
    )
}

data class RelGNode(
    val nodeId: Int,
    val nodeKind: String,
    val field: String,
    val text: String,
    val wordIndices: List<Int>,
    val box: List<Int>,
    val normalizedBox: List<Int>,
    val hidden: FloatArray,
    val spanId: Int?,
    val confidence: Float,
)

data class RelGEdge(
    val pairIndex: Int,
    val headNodeId: Int,
    val depNodeId: Int,
    val headField: String,
    val depField: String,
    val headText: String,
    val depText: String,
    val prob: Float,
    val linkMargin: Float? = null,
)

object SpanRelGPipeline {
    private const val THRESHOLD = 0.5f
    private const val HIDDEN_DIM = 768

    fun buildResponse(
        captureId: String,
        predictions: List<RelGWordPrediction>,
        firstTokenForWord: IntArray,
        lastHiddenState: Array<Array<FloatArray>>,
        relGModel: File,
        schema: SpanRelGSchema,
    ): ReceiptInferenceResponse {
        val spans = predictionsToSpans(predictions)
        val nodes = buildNodes(spans, predictions, firstTokenForWord, lastHiddenState, schema)
        val candidates = buildCandidateEdges(nodes, schema)
        val probs = runRelG(relGModel, schema, nodes, candidates)
        val selectedEdges = selectEdges(candidates, probs)
        val items = decodeItems(nodes, selectedEdges)
        val taxes = decodeAmounts(nodes, selectedEdges, "TAX")
        return ReceiptInferenceResponse(
            captureId = captureId,
            status = "on_device_int8_relg",
            storeName = firstSpanValue(nodes, "STORE_NAME"),
            items = items,
            subtotal = amountMap(nodes, "SUBTOTAL"),
            tax = taxes.firstOrNull()?.toLegacyMap() ?: amountMap(nodes, "TAX"),
            taxes = taxes,
            total = amountMap(nodes, "TOTAL"),
            wordLabels = predictions.map { prediction ->
                WordLabelDto(
                    wordIndex = prediction.index,
                    text = prediction.word.text,
                    label = prediction.label,
                    confidence = prediction.confidence,
                    box = prediction.word.box,
                    wordId = prediction.word.wordId,
                    lineId = prediction.word.lineId,
                )
            },
            warnings = buildList {
                add("On-device LayoutLMv3 INT8 inference ran locally.")
                add("Item-price, tax, subtotal, and total grouping used span-level rel-g ONNX; no line-distance heuristic was used.")
                if (candidates.isEmpty()) add("No rel-g candidate pairs were produced from predicted spans.")
                if (selectedEdges.isEmpty()) add("No rel-g edges passed threshold $THRESHOLD.")
            },
            debug = mapOf(
                "num_words" to JsonPrimitive(predictions.size),
                "num_spans" to JsonPrimitive(spans.size),
                "num_nodes" to JsonPrimitive(nodes.size),
                "num_candidate_edges" to JsonPrimitive(candidates.size),
                "num_selected_edges" to JsonPrimitive(selectedEdges.size),
                "num_tax_relations" to JsonPrimitive(taxes.size),
                "decoder" to JsonPrimitive("span_relg_onnx"),
                "relg_threshold" to JsonPrimitive(THRESHOLD),
            ),
        )
    }

    private fun predictionsToSpans(predictions: List<RelGWordPrediction>): List<RelGSpan> {
        val spans = mutableListOf<RelGSpan>()
        var currentField: String? = null
        var current = mutableListOf<RelGWordPrediction>()
        fun flush() {
            val field = currentField
            if (field != null && current.isNotEmpty()) {
                spans += RelGSpan(spans.size, field, current.toList())
            }
            currentField = null
            current = mutableListOf()
        }
        for (prediction in predictions) {
            val canonical = SpanRelGSchema.canonicalField(prediction.label)
            if (canonical == "O") {
                flush()
                continue
            }
            val isBegin = prediction.label.startsWith("B-")
            if (isBegin || canonical != currentField) {
                flush()
                currentField = canonical
            }
            current += prediction
        }
        flush()
        return spans
    }

    private fun buildNodes(
        spans: List<RelGSpan>,
        predictions: List<RelGWordPrediction>,
        firstTokenForWord: IntArray,
        lastHiddenState: Array<Array<FloatArray>>,
        schema: SpanRelGSchema,
    ): List<RelGNode> {
        val sampleHidden = lastHiddenState.firstOrNull().orEmpty()
        val nodes = mutableListOf<RelGNode>()
        for (span in spans) {
            if (schema.fieldId(span.field) == null) continue
            val hidden = hiddenForWord(span.firstWordIndex, firstTokenForWord, sampleHidden) ?: continue
            nodes += RelGNode(
                nodeId = nodes.size,
                nodeKind = "SPAN",
                field = span.field,
                text = span.text,
                wordIndices = span.wordIndices,
                box = span.box,
                normalizedBox = span.normalizedBox,
                hidden = hidden,
                spanId = span.spanId,
                confidence = span.confidence,
            )
        }
        for (prediction in predictions) {
            val hidden = hiddenForWord(prediction.index, firstTokenForWord, sampleHidden) ?: continue
            nodes += RelGNode(
                nodeId = nodes.size,
                nodeKind = "TOKEN",
                field = "CONTEXT_TOKEN",
                text = prediction.word.text,
                wordIndices = listOf(prediction.index),
                box = prediction.word.box,
                normalizedBox = prediction.normalizedBox,
                hidden = hidden,
                spanId = null,
                confidence = prediction.confidence,
            )
        }
        return nodes
    }

    private fun hiddenForWord(wordIndex: Int, firstTokenForWord: IntArray, sampleHidden: Array<out FloatArray>): FloatArray? {
        val tokenIndex = firstTokenForWord.getOrElse(wordIndex) { -1 }
        if (tokenIndex < 0 || tokenIndex >= sampleHidden.size) return null
        val hidden = sampleHidden[tokenIndex]
        if (hidden.size != HIDDEN_DIM) return null
        return hidden
    }

    private fun buildCandidateEdges(nodes: List<RelGNode>, schema: SpanRelGSchema): List<RelGEdge> {
        val heads = nodes.filter { it.nodeKind == "SPAN" && schema.isHead(it.field) }
        val deps = nodes.filter { it.nodeKind == "SPAN" && schema.isCandidateDependent(it.field) }
        val edges = mutableListOf<RelGEdge>()
        for (head in heads) {
            for (dep in deps) {
                if (head.nodeId == dep.nodeId) continue
                edges += RelGEdge(
                    pairIndex = edges.size,
                    headNodeId = head.nodeId,
                    depNodeId = dep.nodeId,
                    headField = head.field,
                    depField = dep.field,
                    headText = head.text,
                    depText = dep.text,
                    prob = 0f,
                )
            }
        }
        return edges
    }

    private fun runRelG(
        relGModel: File,
        schema: SpanRelGSchema,
        nodes: List<RelGNode>,
        candidates: List<RelGEdge>,
    ): FloatArray {
        if (nodes.isEmpty() || candidates.isEmpty()) return FloatArray(0)
        val nodeHidden = FloatArray(nodes.size * HIDDEN_DIM)
        val fieldIds = LongArray(nodes.size)
        val kindIds = LongArray(nodes.size)
        val nodeBoxes = FloatArray(nodes.size * 4)
        val nodeMask = FloatArray(nodes.size) { 1f }
        nodes.forEachIndexed { nodeIndex, node ->
            node.hidden.copyInto(nodeHidden, nodeIndex * HIDDEN_DIM)
            fieldIds[nodeIndex] = schema.fieldId(node.field)?.toLong()
                ?: error("Field ${node.field} is not present in rel-g schema.")
            kindIds[nodeIndex] = schema.kindId(node.nodeKind).toLong()
            val unit = node.normalizedBox.map { (it.toFloat() / 1000f).coerceIn(0f, 1f) }
            for (i in 0 until 4) nodeBoxes[nodeIndex * 4 + i] = unit[i]
        }
        val candidatePairs = LongArray(candidates.size * 3)
        candidates.forEachIndexed { idx, edge ->
            candidatePairs[idx * 3] = 0L
            candidatePairs[idx * 3 + 1] = edge.headNodeId.toLong()
            candidatePairs[idx * 3 + 2] = edge.depNodeId.toLong()
        }
        val outputs = SpanRelGOnnxRunner.run(
            relGModel.toPath(),
            SpanRelGInputs(
                nodeHidden = nodeHidden,
                nodeFieldIds = fieldIds,
                nodeKindIds = kindIds,
                nodeBoxes = nodeBoxes,
                nodeMask = nodeMask,
                candidatePairs = candidatePairs,
                nodeCount = nodes.size,
                pairCount = candidates.size,
            ),
        )
        return outputs.probs
    }

    private fun selectEdges(candidates: List<RelGEdge>, probs: FloatArray): List<RelGEdge> {
        val selected = candidates.mapIndexedNotNull { idx, edge ->
            val prob = probs.getOrNull(idx) ?: 0f
            if (prob >= THRESHOLD) edge.copy(prob = prob) else null
        }
        return selected
            .groupBy { it.depNodeId }
            .values
            .map { edges ->
                val sorted = edges.sortedByDescending { it.prob }
                val best = sorted.first()
                best.copy(linkMargin = sorted.getOrNull(1)?.let { best.prob - it.prob })
            }
            .sortedBy { it.pairIndex }
    }

    private fun decodeItems(nodes: List<RelGNode>, selectedEdges: List<RelGEdge>): List<ReceiptItemDto> {
        val edgesByHead = selectedEdges.groupBy { it.headNodeId }
        val spanNodes = nodes.filter { it.nodeKind == "SPAN" }
        return spanNodes.filter { it.field == "ITEM_NAME" }.mapIndexed { itemIndex, itemNode ->
            var price: FieldValueDto? = null
            var count: FieldValueDto? = null
            var unitPrice: FieldValueDto? = null
            var bestProb: Float? = null
            var bestMargin: Float? = null
            for (edge in edgesByHead[itemNode.nodeId].orEmpty().sortedByDescending { it.prob }) {
                val dep = nodes.getOrNull(edge.depNodeId) ?: continue
                when (dep.field) {
                    "ITEM_PRICE" -> if (price == null) {
                        price = dep.toFieldValue()
                        bestProb = edge.prob
                        bestMargin = edge.linkMargin
                    }
                    "ITEM_QTY" -> if (count == null) count = dep.toFieldValue()
                    "ITEM_UNIT_PRICE" -> if (unitPrice == null) unitPrice = dep.toFieldValue()
                }
            }
            ReceiptItemDto(
                itemIndex = itemIndex,
                menuName = itemNode.toFieldValue(),
                price = price,
                count = count,
                unitPrice = unitPrice,
                relGProb = bestProb,
                linkMargin = bestMargin,
                linkStatus = if (price == null) "no_rel_g_edge" else "rel_g_selected",
            )
        }
    }

    private fun decodeAmounts(nodes: List<RelGNode>, selectedEdges: List<RelGEdge>, prefix: String): List<ReceiptAmountDto> {
        val headField = "${prefix}_NAME"
        val priceField = "${prefix}_PRICE"
        val rateField = "${prefix}_RATE"
        val edgesByHead = selectedEdges.groupBy { it.headNodeId }
        val heads = nodes.filter { it.nodeKind == "SPAN" && it.field == headField }
        val linked = heads.mapIndexedNotNull { index, head ->
            var price: FieldValueDto? = null
            var rate: FieldValueDto? = null
            var bestProb: Float? = null
            var bestMargin: Float? = null
            for (edge in edgesByHead[head.nodeId].orEmpty().sortedByDescending { it.prob }) {
                val dep = nodes.getOrNull(edge.depNodeId) ?: continue
                when (dep.field) {
                    priceField -> if (price == null) {
                        price = dep.toFieldValue()
                        bestProb = edge.prob
                        bestMargin = edge.linkMargin
                    }
                    rateField -> if (rate == null) {
                        rate = dep.toFieldValue()
                        if (bestProb == null) bestProb = edge.prob
                        if (bestMargin == null) bestMargin = edge.linkMargin
                    }
                }
            }
            if (price == null && rate == null) {
                null
            } else {
                ReceiptAmountDto(
                    amountIndex = index,
                    type = prefix.lowercase(),
                    name = head.toFieldValue(),
                    price = price,
                    rate = rate,
                    relGProb = bestProb,
                    linkMargin = bestMargin,
                    linkStatus = "rel_g_selected",
                )
            }
        }
        val linkedWordIndices = linked.flatMap { amount ->
            buildList {
                amount.name?.wordIndices?.let { addAll(it) }
                amount.price?.wordIndices?.let { addAll(it) }
                amount.rate?.wordIndices?.let { addAll(it) }
            }
        }.toSet()
        val unlinkedPrices = nodes
            .filter { it.nodeKind == "SPAN" && it.field == priceField && it.wordIndices.none(linkedWordIndices::contains) }
            .mapIndexed { index, node ->
                ReceiptAmountDto(
                    amountIndex = linked.size + index,
                    type = prefix.lowercase(),
                    price = node.toFieldValue(),
                    linkStatus = "unlinked_${prefix.lowercase()}_span",
                )
            }
        return linked + unlinkedPrices
    }

    private fun firstSpanValue(nodes: List<RelGNode>, field: String): FieldValueDto? {
        return nodes.firstOrNull { it.nodeKind == "SPAN" && it.field == field }?.toFieldValue()
    }

    private fun amountMap(nodes: List<RelGNode>, prefix: String): Map<String, FieldValueDto>? {
        val name = firstSpanValue(nodes, "${prefix}_NAME")
        val price = firstSpanValue(nodes, "${prefix}_PRICE")
        val rate = firstSpanValue(nodes, "${prefix}_RATE")
        if (name == null && price == null && rate == null) return null
        return buildMap {
            name?.let { put("name", it) }
            price?.let { put("price", it) }
            rate?.let { put("rate", it) }
        }
    }
}

private fun ReceiptAmountDto.toLegacyMap(): Map<String, FieldValueDto> = buildMap {
    name?.let { put("name", it) }
    price?.let { put("price", it) }
    rate?.let { put("rate", it) }
}

private fun RelGNode.toFieldValue(): FieldValueDto = FieldValueDto(
    text = text,
    confidence = confidence,
    box = box,
    wordIndices = wordIndices,
)

private fun unionBoxes(boxes: List<List<Int>>): List<Int> {
    return listOf(
        boxes.minOf { it[0] },
        boxes.minOf { it[1] },
        boxes.maxOf { it[2] },
        boxes.maxOf { it[3] },
    )
}
