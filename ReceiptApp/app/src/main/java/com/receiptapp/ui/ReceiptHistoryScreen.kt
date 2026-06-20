package com.receiptapp.ui

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.Divider
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp

@Composable
fun ReceiptHistoryScreen(
    state: CaptureUiState,
    contentPadding: PaddingValues,
    onPeriodChanged: (PeriodFilter) -> Unit,
    onRefresh: () -> Unit,
    onDeleteReceipt: (String) -> Unit,
    onDeleteAllReceipts: () -> Unit,
) {
    val entries = ReceiptAnalytics.filtered(state.history, state.periodFilter)
    val expanded = remember { mutableStateMapOf<String, Boolean>() }
    val confirmDeleteAll = remember { mutableStateOf(false) }
    if (confirmDeleteAll.value) {
        AlertDialog(
            onDismissRequest = { confirmDeleteAll.value = false },
            title = { Text("Delete all receipts?") },
            text = { Text("This removes every saved receipt image, OCR JSON, ZIP, and inference result from this device.") },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmDeleteAll.value = false
                        onDeleteAllReceipts()
                    },
                ) { Text("Delete all", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = {
                TextButton(onClick = { confirmDeleteAll.value = false }) { Text("Cancel") }
            },
        )
    }
    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(contentPadding)
            .padding(20.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp),
    ) {
        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
            Text("Receipts", style = MaterialTheme.typography.headlineSmall)
            Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                OutlinedButton(onClick = { confirmDeleteAll.value = true }, enabled = state.history.isNotEmpty() && !state.isBusy) {
                    Text("Delete all")
                }
                Button(onClick = onRefresh, enabled = !state.isBusy) { Text("Refresh") }
            }
        }
        PeriodSelector(selected = state.periodFilter, onSelected = onPeriodChanged)
        LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            items(entries, key = { it.record.captureId }) { entry ->
                ReceiptHistoryRow(
                    entry = entry,
                    expanded = expanded[entry.record.captureId] == true,
                    onToggle = { expanded[entry.record.captureId] = expanded[entry.record.captureId] != true },
                    onDelete = onDeleteReceipt,
                )
            }
        }
    }
}

@Composable
private fun ReceiptHistoryRow(
    entry: ReceiptHistoryEntry,
    expanded: Boolean,
    onToggle: () -> Unit,
    onDelete: (String) -> Unit,
) {
    val response = entry.inference
    val showOverlay = remember(entry.record.captureId) { mutableStateOf(false) }
    val overlayMode = remember(entry.record.captureId) { mutableStateOf(ReceiptOverlayMode.ITEM) }
    val confirmDelete = remember(entry.record.captureId) { mutableStateOf(false) }
    if (confirmDelete.value) {
        AlertDialog(
            onDismissRequest = { confirmDelete.value = false },
            title = { Text("Delete this receipt?") },
            text = { Text("${ReceiptAnalytics.storeName(entry)}\n${entry.record.captureId}") },
            confirmButton = {
                TextButton(
                    onClick = {
                        confirmDelete.value = false
                        onDelete(entry.record.captureId)
                    },
                ) { Text("Delete", color = MaterialTheme.colorScheme.error) }
            },
            dismissButton = {
                TextButton(onClick = { confirmDelete.value = false }) { Text("Cancel") }
            },
        )
    }
    if (showOverlay.value) {
        ReceiptInferenceOverlayDialog(
            imageFile = entry.record.imageFile,
            words = entry.record.ocrPayload.words,
            response = response,
            initialMode = overlayMode.value,
            onDismiss = { showOverlay.value = false },
        )
    }
    Surface(
        tonalElevation = 1.dp,
        modifier = Modifier
            .fillMaxWidth()
            .clickable(onClick = onToggle),
    ) {
        Column(modifier = Modifier.padding(14.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) {
            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                Text(ReceiptAnalytics.storeName(entry), style = MaterialTheme.typography.titleMedium)
                Text(formatMoney(ReceiptAnalytics.totalAmount(response)))
            }
            Text(entry.record.captureId, style = MaterialTheme.typography.bodySmall)
            if (expanded) {
                Divider()
                if (response == null) {
                    Text("No inference result saved yet.")
                    Text("OCR words: ${entry.record.ocrPayload.words.size}")
                    OutlinedButton(onClick = { confirmDelete.value = true }) {
                        Text("Delete receipt", color = MaterialTheme.colorScheme.error)
                    }
                } else {
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = {
                            overlayMode.value = ReceiptOverlayMode.ITEM
                            showOverlay.value = true
                        }) {
                            Text("ITEM overlay")
                        }
                        Button(onClick = {
                            overlayMode.value = ReceiptOverlayMode.TAX
                            showOverlay.value = true
                        }) {
                            Text("TAX overlay")
                        }
                        Button(onClick = {
                            overlayMode.value = ReceiptOverlayMode.LABELS
                            showOverlay.value = true
                        }) {
                            Text("Labels")
                        }
                    }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = { confirmDelete.value = true }) {
                            Text("Delete")
                        }
                    }
                    response.items.forEach { item ->
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(item.menuName?.text.orEmpty().ifBlank { "(item)" }, modifier = Modifier.weight(1f))
                            Text(item.price?.text.orEmpty().ifBlank { "-" })
                        }
                    }
                    if (response.taxes.isNotEmpty()) {
                        response.taxes.forEachIndexed { index, tax ->
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                val name = tax.name?.text ?: "Tax ${index + 1}"
                                val value = listOfNotNull(tax.rate?.text, tax.price?.text).joinToString(" / ").ifBlank { "-" }
                                Text(name, modifier = Modifier.weight(1f))
                                Text(value)
                            }
                        }
                    } else {
                        response.tax?.get("price")?.let {
                            Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                                Text(response.tax["name"]?.text ?: "Tax")
                                Text(it.text)
                            }
                        }
                    }
                    response.total?.get("price")?.let {
                        Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(response.total["name"]?.text ?: "Total", style = MaterialTheme.typography.titleMedium)
                            Text(it.text, style = MaterialTheme.typography.titleMedium)
                        }
                    }
                    response.warnings?.forEach { warning ->
                        Text(warning, color = MaterialTheme.colorScheme.tertiary, style = MaterialTheme.typography.bodySmall)
                    }
                }
            }
        }
    }
}
