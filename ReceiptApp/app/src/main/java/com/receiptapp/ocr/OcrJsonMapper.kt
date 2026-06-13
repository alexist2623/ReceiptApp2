package com.receiptapp.ocr

import android.util.Log

object OcrJsonMapper {
    private const val TAG = "ReceiptOCR"

    fun toPayload(
        captureId: String,
        createdAtUtc: String,
        device: DeviceInfoDto,
        app: AppInfoDto,
        image: ImageInfoDto,
        script: OcrScript,
        snapshot: RecognizedTextSnapshot,
    ): ReceiptOcrPayload {
        val blocks = mutableListOf<OcrBlockDto>()
        val lines = mutableListOf<OcrLineDto>()
        val words = mutableListOf<OcrWordDto>()

        snapshot.blocks.forEachIndexed { blockIndex, block ->
            val blockId = "b_${blockIndex.toString().padStart(6, '0')}"
            val lineIds = mutableListOf<String>()

            block.lines.forEachIndexed { lineIndex, line ->
                val lineId = "l_${lines.size.toString().padStart(6, '0')}"
                val wordIds = mutableListOf<String>()

                line.words.forEachIndexed { wordIndexInLine, word ->
                    val text = word.text.trim()
                    if (text.isEmpty()) {
                        return@forEachIndexed
                    }
                    val rawBox = word.box ?: boxFromCornerPoints(word.cornerPoints)
                    val box = clampBox(rawBox, image.width, image.height)
                    if (box == null) {
                        Log.w(TAG, "Skipping OCR word without valid box: $text")
                        return@forEachIndexed
                    }
                    val wordId = "w_${words.size.toString().padStart(6, '0')}"
                    wordIds += wordId
                    words += OcrWordDto(
                        wordId = wordId,
                        blockId = blockId,
                        lineId = lineId,
                        wordIndexInLine = wordIndexInLine,
                        globalWordIndex = words.size,
                        text = text,
                        box = box,
                        cornerPoints = clampCornerPoints(word.cornerPoints, image.width, image.height),
                        confidence = word.confidence,
                        recognizedLanguage = word.recognizedLanguage,
                    )
                }

                if (wordIds.isNotEmpty()) {
                    lineIds += lineId
                    lines += OcrLineDto(
                        lineId = lineId,
                        blockId = blockId,
                        text = line.text,
                        box = clampBox(line.box ?: unionBoxes(words.filter { it.lineId == lineId }.map { it.box }), image.width, image.height)
                            ?: listOf(0, 0, 1, 1),
                        cornerPoints = clampCornerPoints(line.cornerPoints, image.width, image.height),
                        wordIds = wordIds,
                    )
                }
            }

            if (lineIds.isNotEmpty()) {
                blocks += OcrBlockDto(
                    blockId = blockId,
                    text = block.text,
                    box = clampBox(block.box ?: unionBoxes(lines.filter { it.blockId == blockId }.map { it.box }), image.width, image.height)
                        ?: listOf(0, 0, 1, 1),
                    cornerPoints = clampCornerPoints(block.cornerPoints, image.width, image.height),
                    lineIds = lineIds,
                )
            }
        }

        return ReceiptOcrPayload(
            captureId = captureId,
            createdAtUtc = createdAtUtc,
            device = device,
            app = app,
            image = image,
            ocr = OcrInfoDto(script = script.wireValue, confidenceAvailable = false),
            blocks = blocks,
            lines = lines,
            words = words,
            image_width = image.width,
            image_height = image.height,
        )
    }

    fun clampBox(box: List<Int>?, width: Int, height: Int): List<Int>? {
        if (box == null || box.size != 4 || width <= 0 || height <= 0) return null
        var left = box[0].coerceIn(0, width - 1)
        var top = box[1].coerceIn(0, height - 1)
        var right = box[2].coerceIn(0, width - 1)
        var bottom = box[3].coerceIn(0, height - 1)
        if (right < left) {
            val previousLeft = left
            left = right
            right = previousLeft
        }
        if (bottom < top) {
            val previousTop = top
            top = bottom
            bottom = previousTop
        }
        if (right <= left || bottom <= top) return null
        return listOf(left, top, right, bottom)
    }

    fun boxFromCornerPoints(points: List<List<Int>>?): List<Int>? {
        if (points.isNullOrEmpty()) return null
        val xs = points.mapNotNull { it.getOrNull(0) }
        val ys = points.mapNotNull { it.getOrNull(1) }
        if (xs.isEmpty() || ys.isEmpty()) return null
        return listOf(xs.min(), ys.min(), xs.max(), ys.max())
    }

    fun unionBoxes(boxes: List<List<Int>>): List<Int>? {
        if (boxes.isEmpty()) return null
        return listOf(
            boxes.minOf { it[0] },
            boxes.minOf { it[1] },
            boxes.maxOf { it[2] },
            boxes.maxOf { it[3] },
        )
    }

    private fun clampCornerPoints(points: List<List<Int>>?, width: Int, height: Int): List<List<Int>>? {
        if (points.isNullOrEmpty()) return null
        return points.mapNotNull { point ->
            if (point.size < 2) null else listOf(point[0].coerceIn(0, width - 1), point[1].coerceIn(0, height - 1))
        }.takeIf { it.isNotEmpty() }
    }
}
