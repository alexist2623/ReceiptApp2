package com.receiptapp.ui

import android.graphics.BitmapFactory
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.unit.dp
import com.receiptapp.ocr.OcrWordDto
import java.io.File
import kotlin.math.min

@Composable
fun OcrOverlayComposable(
    imageFile: File,
    words: List<OcrWordDto>,
    modifier: Modifier = Modifier,
    showText: Boolean = false,
) {
    val bitmap = remember(imageFile.absolutePath) {
        BitmapFactory.decodeFile(imageFile.absolutePath)?.asImageBitmap()
    }
    if (bitmap == null) return

    Canvas(
        modifier = modifier
            .fillMaxWidth()
            .heightIn(min = 320.dp, max = 620.dp),
    ) {
        val scale = min(size.width / bitmap.width, size.height / bitmap.height)
        val drawWidth = bitmap.width * scale
        val drawHeight = bitmap.height * scale
        val left = (size.width - drawWidth) / 2f
        val top = (size.height - drawHeight) / 2f
        drawImage(
            image = bitmap,
            dstOffset = androidx.compose.ui.unit.IntOffset(left.toInt(), top.toInt()),
            dstSize = androidx.compose.ui.unit.IntSize(drawWidth.toInt(), drawHeight.toInt()),
        )
        words.forEach { word ->
            val box = word.box
            if (box.size == 4) {
                val x = left + box[0] * scale
                val y = top + box[1] * scale
                val w = (box[2] - box[0]) * scale
                val h = (box[3] - box[1]) * scale
                drawRect(
                    color = Color(0xFF22C55E),
                    topLeft = Offset(x, y),
                    size = Size(w, h),
                    style = Stroke(width = 2f),
                )
                if (showText) {
                    drawContext.canvas.nativeCanvas.drawText(
                        word.text.take(18),
                        x,
                        (y - 4f).coerceAtLeast(12f),
                        android.graphics.Paint().apply {
                            color = android.graphics.Color.rgb(20, 120, 60)
                            textSize = 22f
                            isAntiAlias = true
                        },
                    )
                }
            }
        }
    }
}
