package com.receiptapp.inference.layoutlmv3

import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets
import java.nio.file.Files
import java.nio.file.Path

data class NpyArray(
    val descr: String,
    val shape: LongArray,
    val longData: LongArray? = null,
    val floatData: FloatArray? = null,
) {
    fun requireLongs(): LongArray = requireNotNull(longData) { "NPY array $descr is not int64" }
    fun requireFloats(): FloatArray = requireNotNull(floatData) { "NPY array $descr is not float32" }
}

object NpyReader {
    fun read(path: Path): NpyArray {
        val bytes = Files.readAllBytes(path)
        require(bytes.size > 10) { "Invalid NPY file: $path" }
        val magic = String(bytes.copyOfRange(0, 6), StandardCharsets.ISO_8859_1)
        require(magic == "\u0093NUMPY") { "Invalid NPY magic for $path" }
        val major = bytes[6].toInt()
        val headerLength: Int
        val headerStart: Int
        if (major == 1) {
            headerLength = ByteBuffer.wrap(bytes, 8, 2).order(ByteOrder.LITTLE_ENDIAN).short.toInt() and 0xffff
            headerStart = 10
        } else {
            headerLength = ByteBuffer.wrap(bytes, 8, 4).order(ByteOrder.LITTLE_ENDIAN).int
            headerStart = 12
        }
        val header = String(bytes.copyOfRange(headerStart, headerStart + headerLength), StandardCharsets.ISO_8859_1)
        val descr = Regex("'descr'\\s*:\\s*'([^']+)'").find(header)?.groupValues?.get(1)
            ?: error("Missing descr in $path header=$header")
        val fortran = Regex("'fortran_order'\\s*:\\s*(True|False)").find(header)?.groupValues?.get(1)
            ?: error("Missing fortran_order in $path")
        require(fortran == "False") { "Fortran-order NPY is not supported: $path" }
        val shapeText = Regex("'shape'\\s*:\\s*\\(([^)]*)\\)").find(header)?.groupValues?.get(1)
            ?: error("Missing shape in $path header=$header")
        val shape = shapeText.split(",")
            .mapNotNull { it.trim().takeIf(String::isNotEmpty)?.toLong() }
            .toLongArray()
        val dataOffset = headerStart + headerLength
        val buffer = ByteBuffer.wrap(bytes, dataOffset, bytes.size - dataOffset).order(ByteOrder.LITTLE_ENDIAN)
        val elementCount = shape.fold(1L) { acc, value -> acc * value }.toInt()
        return when (descr) {
            "<i8", "|i8" -> {
                val data = LongArray(elementCount)
                for (idx in data.indices) data[idx] = buffer.long
                NpyArray(descr = descr, shape = shape, longData = data)
            }
            "<f4", "|f4" -> {
                val data = FloatArray(elementCount)
                for (idx in data.indices) data[idx] = buffer.float
                NpyArray(descr = descr, shape = shape, floatData = data)
            }
            else -> error("Unsupported NPY dtype $descr in $path")
        }
    }
}
