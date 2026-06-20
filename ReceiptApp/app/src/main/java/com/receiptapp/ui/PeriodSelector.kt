package com.receiptapp.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.unit.dp

@Composable
fun PeriodSelector(
    selected: PeriodFilter,
    onSelected: (PeriodFilter) -> Unit,
) {
    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
        PeriodFilter.entries.forEach { period ->
            FilterChip(
                selected = selected == period,
                onClick = { onSelected(period) },
                label = { Text(period.label) },
            )
        }
    }
}
