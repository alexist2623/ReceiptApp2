package com.receiptapp.ui

import android.graphics.BitmapFactory
import android.graphics.Paint
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectTransformGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.compose.ui.window.Dialog
import androidx.compose.ui.window.DialogProperties
import com.receiptapp.inference.FieldValueDto
import com.receiptapp.inference.ReceiptAmountDto
import com.receiptapp.inference.ReceiptInferenceResponse
import com.receiptapp.inference.WordLabelDto
import com.receiptapp.ocr.OcrWordDto
import java.io.File
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

enum class ReceiptOverlayMode(val label: String) {
    ITEM("ITEM"),
    TAX("TAX"),
    SUMMARY("Summary"),
    LABELS("Labels"),
    ALL("All"),
}

@Composable
fun ReceiptInferenceOverlayDialog(
    imageFile: File,
    words: List<OcrWordDto>,
    response: ReceiptInferenceResponse?,
    initialMode: ReceiptOverlayMode = ReceiptOverlayMode.ITEM,
    onDismiss: () -> Unit,
) {
    var mode by remember { mutableStateOf(initialMode) }
    var zoom by remember { mutableFloatStateOf(1f) }
    var pan by remember { mutableStateOf(Offset.Zero) }

    Dialog(
        onDismissRequest = onDismiss,
        properties = DialogProperties(usePlatformDefaultWidth = false),
    ) {
        Surface(modifier = Modifier.fillMaxSize(), color = Color.Black) {
            Column(modifier = Modifier.fillMaxSize()) {
                Row(
                    modifier = Modifier
                        .fillMaxWidth()
                        .background(Color(0xFF111111))
                        .padding(10.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    ReceiptOverlayMode.entries.forEach { target ->
                        FilterChip(
                            selected = mode == target,
                            onClick = { mode = target },
                            label = { Text(target.label) },
                        )
                    }
                    Button(
                        onClick = {
                            zoom = 1f
                            pan = Offset.Zero
                        },
                    ) { Text("Reset") }
                    Button(onClick = onDismiss) { Text("Close") }
                }
                Box(modifier = Modifier.weight(1f)) {
                    ReceiptInferenceOverlayCanvas(
                        imageFile = imageFile,
                        words = words,
                        response = response,
                        mode = mode,
                        zoom = zoom,
                        pan = pan,
                        onTransform = { zoomChange, panChange ->
                            zoom = (zoom * zoomChange).coerceIn(1f, 8f)
                            pan += panChange
                        },
                    )
                }
            }
        }
    }
}

@Composable
private fun ReceiptInferenceOverlayCanvas(
    imageFile: File,
    words: List<OcrWordDto>,
    response: ReceiptInferenceResponse?,
    mode: ReceiptOverlayMode,
    zoom: Float,
    pan: Offset,
    onTransform: (Float, Offset) -> Unit,
) {
    val bitmap = remember(imageFile.absolutePath) {
        BitmapFactory.decodeFile(imageFile.absolutePath)?.asImageBitmap()
    }
    if (bitmap == null) {
        Text("Image could not be loaded.", color = MaterialTheme.colorScheme.error)
        return
    }

    Canvas(
        modifier = Modifier
            .fillMaxSize()
            .background(Color.Black)
            .pointerInput(Unit) {
                detectTransformGestures { _, gesturePan, gestureZoom, _ ->
                    onTransform(gestureZoom, gesturePan)
                }
            },
    ) {
        val baseScale = min(size.width / bitmap.width, size.height / bitmap.height)
        val scale = baseScale * zoom
        val drawWidth = bitmap.width * scale
        val drawHeight = bitmap.height * scale
        val left = (size.width - drawWidth) / 2f + pan.x
        val top = (size.height - drawHeight) / 2f + pan.y

        drawImage(
            image = bitmap,
            dstOffset = androidx.compose.ui.unit.IntOffset(left.toInt(), top.toInt()),
            dstSize = androidx.compose.ui.unit.IntSize(drawWidth.toInt(), drawHeight.toInt()),
        )

        words.forEach { word ->
            drawBox(word.box, left, top, scale, Color(0x338A8A8A), strokeWidth = 1.5f)
        }

        if (response == null) return@Canvas

        if (mode == ReceiptOverlayMode.ITEM || mode == ReceiptOverlayMode.ALL) {
            response.items.forEach { item ->
                val name = item.menuName
                val price = item.price
                name?.let { drawField(it, left, top, scale, Color(0xFF22C55E), "ITEM") }
                price?.let { drawField(it, left, top, scale, Color(0xFF3B82F6), "PRICE") }
                if (name?.box != null && price?.box != null) {
                    drawArrow(name.box, price.box, left, top, scale, Color(0xFF22C55E), item.relGProb?.let { "%.2f".format(it) })
                }
            }
        }

        if (mode == ReceiptOverlayMode.TAX || mode == ReceiptOverlayMode.ALL) {
            response.taxes.forEachIndexed { index, tax ->
                drawAmount(tax, left, top, scale, Color(0xFFF97316), "TAX ${index + 1}")
            }
            if (response.taxes.isEmpty()) {
                response.tax?.let { drawLegacyAmount(it, left, top, scale, Color(0xFFF97316), "TAX") }
            }
        }

        if (mode == ReceiptOverlayMode.SUMMARY || mode == ReceiptOverlayMode.ALL) {
            response.subtotal?.let { drawLegacyAmount(it, left, top, scale, Color(0xFFA855F7), "SUBTOTAL") }
            response.total?.let { drawLegacyAmount(it, left, top, scale, Color(0xFFEF4444), "TOTAL") }
        }

        if (mode == ReceiptOverlayMode.LABELS) {
            if (response.wordLabels.isNotEmpty()) {
                response.wordLabels.forEach { wordLabel ->
                    drawWordLabel(wordLabel, left, top, scale)
                }
            } else {
                val fallbackCount = drawGroupedLabelFallback(response, left, top, scale)
                if (fallbackCount == 0) {
                    drawOverlayMessage("No saved labels. Run INT8 again.", Color(0xFFFFD166))
                }
            }
        }
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawGroupedLabelFallback(
    response: ReceiptInferenceResponse,
    left: Float,
    top: Float,
    scale: Float,
): Int {
    var count = 0
    fun draw(value: FieldValueDto?, label: String) {
        if (value?.box == null) return
        drawLabeledField(value, left, top, scale, label)
        count += 1
    }
    draw(response.storeName, "STORE_NAME")
    response.items.forEach { item ->
        draw(item.menuName, "ITEM_NAME")
        draw(item.price, "ITEM_PRICE")
        draw(item.count, "ITEM_QTY")
        draw(item.unitPrice, "ITEM_UNIT_PRICE")
    }
    response.taxes.forEach { tax ->
        draw(tax.name, "TAX_NAME")
        draw(tax.price, "TAX_PRICE")
        draw(tax.rate, "TAX_RATE")
    }
    if (response.taxes.isEmpty()) {
        response.tax?.let { tax ->
            draw(tax["name"], "TAX_NAME")
            draw(tax["price"], "TAX_PRICE")
            draw(tax["rate"], "TAX_RATE")
        }
    }
    response.subtotal?.let { subtotal ->
        draw(subtotal["name"], "SUBTOTAL_NAME")
        draw(subtotal["price"], "SUBTOTAL_PRICE")
        draw(subtotal["rate"], "SUBTOTAL_RATE")
    }
    response.total?.let { total ->
        draw(total["name"], "TOTAL_NAME")
        draw(total["price"], "TOTAL_PRICE")
        draw(total["rate"], "TOTAL_RATE")
    }
    return count
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawLabeledField(
    value: FieldValueDto,
    left: Float,
    top: Float,
    scale: Float,
    label: String,
) {
    val box = value.box ?: return
    val color = wordLabelColor(label)
    drawBox(box, left, top, scale, color, strokeWidth = 3.25f)
    drawCompactLabel(
        box = box,
        left = left,
        top = top,
        scale = scale,
        color = color,
        text = "$label ${value.text.take(14)}",
    )
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawWordLabel(
    wordLabel: WordLabelDto,
    left: Float,
    top: Float,
    scale: Float,
) {
    val box = wordLabel.box ?: return
    val label = displayLabel(wordLabel.label)
    val isOutside = label == "O"
    val color = if (isOutside) Color(0x668A8A8A) else wordLabelColor(label)
    drawBox(
        box = box,
        left = left,
        top = top,
        scale = scale,
        color = color,
        strokeWidth = if (isOutside) 1.25f else 3.25f,
    )
    if (!isOutside) {
        val confidence = wordLabel.confidence?.let { " ${"%.2f".format(it)}" }.orEmpty()
        drawCompactLabel(
            box = box,
            left = left,
            top = top,
            scale = scale,
            color = color,
            text = "$label ${wordLabel.text.take(14)}$confidence",
        )
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawOverlayMessage(
    text: String,
    color: Color,
) {
    val paint = Paint().apply {
        this.color = color.toArgbInt()
        textSize = 32f
        isAntiAlias = true
        isFakeBoldText = true
        setShadowLayer(5f, 1f, 1f, android.graphics.Color.BLACK)
    }
    drawContext.canvas.nativeCanvas.drawText(text, 24f, 48f, paint)
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawAmount(
    amount: ReceiptAmountDto,
    left: Float,
    top: Float,
    scale: Float,
    color: Color,
    label: String,
) {
    amount.name?.let { drawField(it, left, top, scale, color, label) }
    amount.price?.let { drawField(it, left, top, scale, color, "$label PRICE") }
    amount.rate?.let { drawField(it, left, top, scale, color, "$label RATE") }
    val target = amount.price ?: amount.rate
    if (amount.name?.box != null && target?.box != null) {
        drawArrow(amount.name.box, target.box, left, top, scale, color, amount.relGProb?.let { "%.2f".format(it) })
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawLegacyAmount(
    amount: Map<String, FieldValueDto>,
    left: Float,
    top: Float,
    scale: Float,
    color: Color,
    label: String,
) {
    val name = amount["name"]
    val price = amount["price"]
    val rate = amount["rate"]
    name?.let { drawField(it, left, top, scale, color, label) }
    price?.let { drawField(it, left, top, scale, color, "$label PRICE") }
    rate?.let { drawField(it, left, top, scale, color, "$label RATE") }
    val target = price ?: rate
    if (name?.box != null && target?.box != null) {
        drawArrow(name.box, target.box, left, top, scale, color, null)
    }
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawField(
    value: FieldValueDto,
    left: Float,
    top: Float,
    scale: Float,
    color: Color,
    label: String,
) {
    val box = value.box ?: return
    drawBox(box, left, top, scale, color, strokeWidth = 4f)
    drawLabel(box, left, top, scale, color, "$label ${value.text.take(20)}")
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawBox(
    box: List<Int>,
    left: Float,
    top: Float,
    scale: Float,
    color: Color,
    strokeWidth: Float,
) {
    if (box.size != 4) return
    val x = left + box[0] * scale
    val y = top + box[1] * scale
    val w = (box[2] - box[0]).coerceAtLeast(1) * scale
    val h = (box[3] - box[1]).coerceAtLeast(1) * scale
    drawRect(
        color = color,
        topLeft = Offset(x, y),
        size = Size(w, h),
        style = Stroke(width = strokeWidth),
    )
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawLabel(
    box: List<Int>,
    left: Float,
    top: Float,
    scale: Float,
    color: Color,
    text: String,
) {
    if (box.size != 4) return
    val x = left + box[0] * scale
    val y = top + box[1] * scale
    val paint = Paint().apply {
        this.color = color.toArgbInt()
        textSize = (22f * scale.coerceIn(0.8f, 2.2f)).coerceIn(18f, 34f)
        isAntiAlias = true
        isFakeBoldText = true
        setShadowLayer(4f, 1f, 1f, android.graphics.Color.BLACK)
    }
    drawContext.canvas.nativeCanvas.drawText(text, x, (y - 5f).coerceAtLeast(20f), paint)
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawCompactLabel(
    box: List<Int>,
    left: Float,
    top: Float,
    scale: Float,
    color: Color,
    text: String,
) {
    if (box.size != 4) return
    val x = left + box[0] * scale
    val y = top + box[1] * scale
    val paint = Paint().apply {
        this.color = color.toArgbInt()
        textSize = (17f * scale.coerceIn(0.8f, 1.8f)).coerceIn(15f, 26f)
        isAntiAlias = true
        isFakeBoldText = true
        setShadowLayer(4f, 1f, 1f, android.graphics.Color.BLACK)
    }
    drawContext.canvas.nativeCanvas.drawText(text, x, (y - 4f).coerceAtLeast(18f), paint)
}

private fun androidx.compose.ui.graphics.drawscope.DrawScope.drawArrow(
    fromBox: List<Int>,
    toBox: List<Int>,
    left: Float,
    top: Float,
    scale: Float,
    color: Color,
    label: String?,
) {
    if (fromBox.size != 4 || toBox.size != 4) return
    val start = center(fromBox, left, top, scale)
    val end = center(toBox, left, top, scale)
    val stroke = (4f * scale.coerceIn(0.8f, 1.8f)).coerceIn(3f, 7f)
    drawLine(color = color, start = start, end = end, strokeWidth = stroke)
    val angle = atan2((end.y - start.y).toDouble(), (end.x - start.x).toDouble())
    val arrow = 20f * scale.coerceIn(0.8f, 1.7f)
    val a1 = Offset(
        x = end.x - arrow * cos(angle - 0.45).toFloat(),
        y = end.y - arrow * sin(angle - 0.45).toFloat(),
    )
    val a2 = Offset(
        x = end.x - arrow * cos(angle + 0.45).toFloat(),
        y = end.y - arrow * sin(angle + 0.45).toFloat(),
    )
    drawLine(color = color, start = end, end = a1, strokeWidth = stroke)
    drawLine(color = color, start = end, end = a2, strokeWidth = stroke)
    label?.let {
        val paint = Paint().apply {
            this.color = color.toArgbInt()
            textSize = 28f
            isAntiAlias = true
            isFakeBoldText = true
            setShadowLayer(4f, 1f, 1f, android.graphics.Color.BLACK)
        }
        drawContext.canvas.nativeCanvas.drawText(it, (start.x + end.x) / 2f, (start.y + end.y) / 2f, paint)
    }
}

private fun center(box: List<Int>, left: Float, top: Float, scale: Float): Offset {
    return Offset(
        x = left + ((box[0] + box[2]) / 2f) * scale,
        y = top + ((box[1] + box[3]) / 2f) * scale,
    )
}

private fun displayLabel(label: String): String {
    return label
        .removePrefix("B-")
        .removePrefix("I-")
        .ifBlank { "O" }
}

private fun wordLabelColor(label: String): Color {
    return when {
        label == "O" -> Color(0x668A8A8A)
        label.startsWith("ITEM_NAME") -> Color(0xFF22C55E)
        label.startsWith("ITEM_PRICE") -> Color(0xFF3B82F6)
        label.startsWith("ITEM_QTY") -> Color(0xFF14B8A6)
        label.startsWith("ITEM_UNIT_PRICE") -> Color(0xFF06B6D4)
        label.startsWith("TAX") -> Color(0xFFF97316)
        label.startsWith("SUBTOTAL") -> Color(0xFFA855F7)
        label.startsWith("TOTAL") -> Color(0xFFEF4444)
        label.startsWith("STORE") -> Color(0xFF38BDF8)
        label.startsWith("PAYMENT") -> Color(0xFF6366F1)
        else -> {
            val hash = label.hashCode()
            val red = 96 + ((hash ushr 16) and 0x5F)
            val green = 96 + ((hash ushr 8) and 0x5F)
            val blue = 96 + (hash and 0x5F)
            Color(red = red / 255f, green = green / 255f, blue = blue / 255f)
        }
    }
}

private fun Color.toArgbInt(): Int {
    return android.graphics.Color.argb(
        (alpha * 255).toInt(),
        (red * 255).toInt(),
        (green * 255).toInt(),
        (blue * 255).toInt(),
    )
}
