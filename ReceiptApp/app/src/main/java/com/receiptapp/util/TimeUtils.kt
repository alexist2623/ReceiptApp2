package com.receiptapp.util

import java.time.Instant
import java.time.ZoneOffset
import java.time.format.DateTimeFormatter
import java.util.UUID

object TimeUtils {
    private val captureFormatter: DateTimeFormatter =
        DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'").withZone(ZoneOffset.UTC)

    fun nowUtcIso(): String = Instant.now().toString()

    fun newCaptureId(): String {
        val timestamp = captureFormatter.format(Instant.now())
        val suffix = UUID.randomUUID().toString().replace("-", "").take(8)
        return "${timestamp}_$suffix"
    }
}
