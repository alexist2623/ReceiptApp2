package com.receiptapp.ui

import com.receiptapp.inference.FieldValueDto
import com.receiptapp.inference.ReceiptInferenceResponse
import java.time.Instant
import java.time.LocalDate
import java.time.ZoneId
import java.time.format.DateTimeFormatter

data class CategorySpend(
    val name: String,
    val amount: Double,
)

object ReceiptAnalytics {
    private val captureIdFormatter = DateTimeFormatter.ofPattern("yyyyMMdd'T'HHmmss'Z'")

    fun filtered(entries: List<ReceiptHistoryEntry>, period: PeriodFilter, now: Instant = Instant.now()): List<ReceiptHistoryEntry> {
        val zone = ZoneId.systemDefault()
        val today = LocalDate.ofInstant(now, zone)
        return entries.filter { entry ->
            val date = captureDate(entry.record.captureId, zone) ?: return@filter true
            when (period) {
                PeriodFilter.DAY -> date == today
                PeriodFilter.WEEK -> !date.isBefore(today.minusDays(6))
                PeriodFilter.MONTH -> date.year == today.year && date.month == today.month
            }
        }
    }

    fun captureDate(captureId: String, zone: ZoneId = ZoneId.systemDefault()): LocalDate? {
        return runCatching {
            val timestamp = captureId.substringBefore("_")
            LocalDate.ofInstant(Instant.from(captureIdFormatter.parse(timestamp)), zone)
        }.getOrNull()
    }

    fun totalAmount(response: ReceiptInferenceResponse?): Double {
        if (response == null) return 0.0
        val explicitTotal = response.total?.get("price")?.amount()
        if (explicitTotal != null && explicitTotal > 0.0) return explicitTotal
        return response.items.sumOf { it.price.amount() }
    }

    fun categorySpend(entries: List<ReceiptHistoryEntry>): List<CategorySpend> {
        val total = entries.sumOf { totalAmount(it.inference) }
        return if (total > 0.0) {
            listOf(CategorySpend("Uncategorized", total))
        } else {
            listOf(CategorySpend("Uncategorized", entries.sumOf { it.record.ocrPayload.words.size }.toDouble()))
        }
    }

    fun storeName(entry: ReceiptHistoryEntry): String {
        return entry.inference?.storeName?.text
            ?.takeIf { it.isNotBlank() }
            ?: entry.record.ocrPayload.lines.firstOrNull()?.text
            ?: entry.record.captureId
    }

    fun FieldValueDto?.amount(): Double {
        val text = this?.text ?: return 0.0
        val cleaned = text.replace(Regex("[^0-9.\\-]"), "")
        return cleaned.toDoubleOrNull() ?: 0.0
    }
}
