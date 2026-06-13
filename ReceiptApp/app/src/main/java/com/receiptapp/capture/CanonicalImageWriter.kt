package com.receiptapp.capture

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.util.Log
import com.receiptapp.ocr.ImageInfoDto
import com.receiptapp.receipt.ReceiptFileStore
import com.receiptapp.util.TimeUtils
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest

data class CanonicalImageResult(
    val captureId: String,
    val imageFile: File,
    val imageInfo: ImageInfoDto,
)

class CanonicalImageWriter(
    private val context: Context,
    private val fileStore: ReceiptFileStore = ReceiptFileStore(context),
) {
    fun writeFromFile(sourceFile: File, captureId: String = TimeUtils.newCaptureId()): CanonicalImageResult {
        val originalBounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(sourceFile.absolutePath, originalBounds)
        val originalWidth = originalBounds.outWidth.takeIf { it > 0 }
        val originalHeight = originalBounds.outHeight.takeIf { it > 0 }
        val rawBitmap = BitmapFactory.decodeFile(sourceFile.absolutePath)
            ?: error("Could not decode captured image: ${sourceFile.absolutePath}")
        val rotationDegrees = ImageRotationUtils.exifRotationDegrees(sourceFile)
        val canonicalBitmap = ImageRotationUtils.rotateIfNeeded(rawBitmap, rotationDegrees)
        return writeBitmap(canonicalBitmap, captureId, rotationDegrees, originalWidth, originalHeight)
    }

    fun writeFromUri(uri: Uri, captureId: String = TimeUtils.newCaptureId()): CanonicalImageResult {
        val bytes = context.contentResolver.openInputStream(uri)?.use { it.readBytes() }
            ?: error("Could not read image URI: $uri")
        val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
            ?: error("Could not decode image URI: $uri")
        // Content URI EXIF handling is intentionally conservative; gallery imports should be checked with overlay.
        return writeBitmap(bitmap, captureId, 0, bitmap.width, bitmap.height)
    }

    private fun writeBitmap(
        bitmap: Bitmap,
        captureId: String,
        rotationDegreesApplied: Int,
        originalWidth: Int?,
        originalHeight: Int?,
    ): CanonicalImageResult {
        val imageFile = fileStore.imageFile(captureId)
        imageFile.parentFile?.mkdirs()
        FileOutputStream(imageFile).use { output ->
            bitmap.compress(Bitmap.CompressFormat.JPEG, 95, output)
        }
        Log.i("ReceiptOCR", "Saved canonical image: ${imageFile.absolutePath}")
        return CanonicalImageResult(
            captureId = captureId,
            imageFile = imageFile,
            imageInfo = ImageInfoDto(
                fileName = imageFile.name,
                width = bitmap.width,
                height = bitmap.height,
                mimeType = "image/jpeg",
                exifOrientationApplied = true,
                rotationDegreesApplied = rotationDegreesApplied,
                sha256 = sha256(imageFile),
                originalWidth = originalWidth,
                originalHeight = originalHeight,
            ),
        )
    }

    private fun sha256(file: File): String {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val read = input.read(buffer)
                if (read <= 0) break
                digest.update(buffer, 0, read)
            }
        }
        return digest.digest().joinToString("") { "%02x".format(it) }
    }
}
