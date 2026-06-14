package com.receiptapp.ocr

import android.content.Context
import android.graphics.BitmapFactory
import android.os.Build
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.korean.KoreanTextRecognizerOptions
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import com.receiptapp.BuildConfig
import com.receiptapp.util.ImageDimensionValidator
import com.receiptapp.util.TimeUtils
import java.io.File
import kotlinx.coroutines.tasks.await

class MlKitOcrEngine(
    private val context: Context,
    private val imageInfoProvider: (String, File) -> ImageInfoDto,
) : OcrEngine {
    override suspend fun recognize(
        captureId: String,
        canonicalImageFile: File,
        script: OcrScript,
    ): ReceiptOcrPayload {
        val recognizer = when (script) {
            OcrScript.LATIN -> TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS)
            OcrScript.KOREAN -> TextRecognition.getClient(KoreanTextRecognizerOptions.Builder().build())
        }
        val bitmap = BitmapFactory.decodeFile(canonicalImageFile.absolutePath)
            ?: error("Could not decode canonical image for OCR: ${canonicalImageFile.absolutePath}")
        val imageInfo = imageInfoProvider(captureId, canonicalImageFile)
        val actualSize = ImageDimensionValidator.readImageSize(canonicalImageFile)
        ImageDimensionValidator.validateImageInfoMatchesFile(
            imageFile = canonicalImageFile,
            expectedWidth = imageInfo.width,
            expectedHeight = imageInfo.height,
            context = "ML Kit OCR input",
        )
        require(bitmap.width == imageInfo.width && bitmap.height == imageInfo.height) {
            "ML Kit OCR bitmap mismatch: decoded bitmap is ${bitmap.width}x${bitmap.height} " +
                "but image metadata is ${imageInfo.width}x${imageInfo.height}"
        }
        val image = InputImage.fromBitmap(bitmap, 0)
        val result = recognizer.process(image).await()
        return OcrJsonMapper.toPayload(
            captureId = captureId,
            createdAtUtc = TimeUtils.nowUtcIso(),
            device = deviceInfo(),
            app = appInfo(),
            image = imageInfo,
            script = script,
            snapshot = result.toSnapshot(),
            debug = OcrDebugInfoDto(
                canonicalImageActualWidth = actualSize.width,
                canonicalImageActualHeight = actualSize.height,
                ocrInputBitmapWidth = bitmap.width,
                ocrInputBitmapHeight = bitmap.height,
                savedImagePathHint = canonicalImageFile.absolutePath,
                coordinateValidation = "OK",
                appBuildType = if (BuildConfig.DEBUG) "debug" else "release",
                gitCommitHint = BuildConfig.GIT_COMMIT_SHA,
                buildTimeUtc = BuildConfig.BUILD_TIME_UTC,
            ),
        )
    }

    private fun Text.toSnapshot(): RecognizedTextSnapshot {
        return RecognizedTextSnapshot(
            blocks = textBlocks.map { block ->
                RecognizedBlock(
                    text = block.text,
                    box = block.boundingBox?.toBox(),
                    cornerPoints = block.cornerPoints?.toCornerPoints(),
                    lines = block.lines.map { line ->
                        RecognizedLine(
                            text = line.text,
                            box = line.boundingBox?.toBox(),
                            cornerPoints = line.cornerPoints?.toCornerPoints(),
                            words = line.elements.map { element ->
                                RecognizedWord(
                                    text = element.text,
                                    box = element.boundingBox?.toBox(),
                                    cornerPoints = element.cornerPoints?.toCornerPoints(),
                                    confidence = null,
                                    recognizedLanguage = null,
                                )
                            },
                        )
                    },
                )
            },
        )
    }

    private fun android.graphics.Rect.toBox(): List<Int> = listOf(left, top, right, bottom)

    private fun Array<android.graphics.Point>.toCornerPoints(): List<List<Int>> {
        return map { listOf(it.x, it.y) }
    }

    private fun deviceInfo(): DeviceInfoDto {
        return DeviceInfoDto(
            manufacturer = Build.MANUFACTURER.orEmpty(),
            model = Build.MODEL.orEmpty(),
            androidVersion = Build.VERSION.RELEASE.orEmpty(),
            sdkInt = Build.VERSION.SDK_INT,
        )
    }

    private fun appInfo(): AppInfoDto {
        val packageInfo = context.packageManager.getPackageInfo(context.packageName, 0)
        val versionCode = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.P) {
            packageInfo.longVersionCode
        } else {
            @Suppress("DEPRECATION")
            packageInfo.versionCode.toLong()
        }
        return AppInfoDto(
            packageName = context.packageName,
            versionName = packageInfo.versionName ?: "0",
            versionCode = versionCode,
        )
    }
}
