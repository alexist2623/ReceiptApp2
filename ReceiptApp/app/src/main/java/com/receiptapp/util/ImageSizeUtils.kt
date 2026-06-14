package com.receiptapp.util

import android.graphics.BitmapFactory
import java.io.File

data class ImageSize(
    val width: Int,
    val height: Int,
)

object ImageSizeUtils {
    fun readImageSize(file: File): ImageSize {
        require(file.exists()) { "Image file not found: ${file.absolutePath}" }
        val options = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.absolutePath, options)
        return ImageSize(options.outWidth, options.outHeight)
    }

    fun requirePositiveImageSize(file: File): ImageSize {
        val size = readImageSize(file)
        require(size.width > 0 && size.height > 0) {
            "Could not decode positive image bounds: ${file.absolutePath}, size=${size.width}x${size.height}"
        }
        return size
    }
}
