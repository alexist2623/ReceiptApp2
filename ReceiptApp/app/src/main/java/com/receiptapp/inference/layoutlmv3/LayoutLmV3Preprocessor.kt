package com.receiptapp.inference.layoutlmv3

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import java.io.File
import kotlin.math.max
import kotlin.math.min

object LayoutLmV3Preprocessor {
    const val IMAGE_SIZE = 224

    fun normalizeBox(box: List<Int>, width: Int, height: Int): List<Int> {
        if (box.size != 4 || width <= 0 || height <= 0) return listOf(0, 0, 0, 0)
        val x0 = box[0].coerceIn(0, max(0, width - 1))
        val y0 = box[1].coerceIn(0, max(0, height - 1))
        val x1 = box[2].coerceIn(0, max(0, width - 1))
        val y1 = box[3].coerceIn(0, max(0, height - 1))
        return listOf(
            (1000L * min(x0, x1) / width).toInt().coerceIn(0, 1000),
            (1000L * min(y0, y1) / height).toInt().coerceIn(0, 1000),
            (1000L * max(x0, x1) / width).toInt().coerceIn(0, 1000),
            (1000L * max(y0, y1) / height).toInt().coerceIn(0, 1000),
        )
    }

    fun pixelValues(imageFile: File): FloatArray {
        val bitmap = BitmapFactory.decodeFile(imageFile.absolutePath)
            ?: error("Could not decode image: ${imageFile.absolutePath}")
        val resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)
        val values = FloatArray(3 * IMAGE_SIZE * IMAGE_SIZE)
        for (y in 0 until IMAGE_SIZE) {
            for (x in 0 until IMAGE_SIZE) {
                val color = resized.getPixel(x, y)
                val offset = y * IMAGE_SIZE + x
                values[offset] = normalizeChannel(Color.red(color))
                values[IMAGE_SIZE * IMAGE_SIZE + offset] = normalizeChannel(Color.green(color))
                values[2 * IMAGE_SIZE * IMAGE_SIZE + offset] = normalizeChannel(Color.blue(color))
            }
        }
        if (resized !== bitmap) resized.recycle()
        bitmap.recycle()
        return values
    }

    private fun normalizeChannel(value: Int): Float {
        return (value / 255.0f - 0.5f) / 0.5f
    }
}
