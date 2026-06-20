package com.receiptapp.ui

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.unit.dp
import kotlin.math.min

@Composable
fun ReceiptDashboardScreen(
    state: CaptureUiState,
    contentPadding: PaddingValues,
    onPeriodChanged: (PeriodFilter) -> Unit,
) {
    val entries = ReceiptAnalytics.filtered(state.history, state.periodFilter)
    val categories = ReceiptAnalytics.categorySpend(entries)
    val total = categories.sumOf { it.amount }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .verticalScroll(rememberScrollState())
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(18.dp),
    ) {
        Text("Spending Overview", style = MaterialTheme.typography.headlineSmall)
        PeriodSelector(selected = state.periodFilter, onSelected = onPeriodChanged)
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            SpendingPieChart(
                categories = categories,
                modifier = Modifier.size(180.dp),
            )
            Column(verticalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.weight(1f)) {
                Text("Placeholder categories", style = MaterialTheme.typography.titleMedium)
                categories.forEach { category ->
                    Text("${category.name}: ${formatMoney(category.amount)}")
                }
                Text("Receipts: ${entries.size}")
                Text("Total: ${formatMoney(total)}", style = MaterialTheme.typography.titleLarge)
            }
        }
        Surface(color = MaterialTheme.colorScheme.surfaceVariant, modifier = Modifier.fillMaxWidth()) {
            Column(modifier = Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
                Text("Category model placeholder", style = MaterialTheme.typography.titleMedium)
                Text("Item category classification is not implemented yet. All recognized spending is grouped as Uncategorized.")
            }
        }
        Spacer(Modifier.height(24.dp))
    }
}

@Composable
private fun SpendingPieChart(categories: List<CategorySpend>, modifier: Modifier = Modifier) {
    val palette = listOf(Color(0xFF2563EB), Color(0xFF16A34A), Color(0xFFF59E0B), Color(0xFFDB2777))
    val total = categories.sumOf { it.amount }.takeIf { it > 0.0 } ?: 1.0
    Canvas(modifier = modifier) {
        val diameter = min(size.width, size.height)
        val topLeft = Offset((size.width - diameter) / 2f, (size.height - diameter) / 2f)
        var start = -90f
        categories.forEachIndexed { index, category ->
            val sweep = (category.amount / total * 360.0).toFloat().coerceAtLeast(0.5f)
            drawArc(
                color = palette[index % palette.size],
                startAngle = start,
                sweepAngle = sweep,
                useCenter = false,
                topLeft = topLeft,
                size = Size(diameter, diameter),
                style = Stroke(width = diameter * 0.22f),
            )
            start += sweep
        }
    }
}

fun formatMoney(value: Double): String = "$" + "%,.2f".format(value)
