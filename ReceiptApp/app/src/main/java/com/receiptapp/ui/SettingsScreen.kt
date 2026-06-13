package com.receiptapp.ui

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExposedDropdownMenuBox
import androidx.compose.material3.ExposedDropdownMenuDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.receiptapp.inference.InferenceMode
import com.receiptapp.ocr.OcrScript

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(
    state: CaptureUiState,
    onServerUrlChanged: (String) -> Unit,
    onScriptChanged: (OcrScript) -> Unit,
    onModeChanged: (InferenceMode) -> Unit,
    onBack: () -> Unit,
) {
    var scriptExpanded by remember { mutableStateOf(false) }
    var modeExpanded by remember { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp),
        verticalArrangement = Arrangement.spacedBy(16.dp),
    ) {
        Text("Settings", style = MaterialTheme.typography.titleLarge)
        OutlinedTextField(
            value = state.serverUrl,
            onValueChange = onServerUrlChanged,
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Server base URL") },
            placeholder = { Text("http://192.168.0.10:8000") },
            singleLine = true,
        )

        ExposedDropdownMenuBox(
            expanded = scriptExpanded,
            onExpandedChange = { scriptExpanded = !scriptExpanded },
        ) {
            OutlinedTextField(
                value = state.ocrScript.name,
                onValueChange = {},
                readOnly = true,
                label = { Text("OCR script") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(scriptExpanded) },
                modifier = Modifier.menuAnchor().fillMaxWidth(),
            )
            ExposedDropdownMenu(expanded = scriptExpanded, onDismissRequest = { scriptExpanded = false }) {
                OcrScript.entries.forEach { script ->
                    DropdownMenuItem(
                        text = { Text(script.name) },
                        onClick = {
                            onScriptChanged(script)
                            scriptExpanded = false
                        },
                    )
                }
            }
        }

        ExposedDropdownMenuBox(
            expanded = modeExpanded,
            onExpandedChange = { modeExpanded = !modeExpanded },
        ) {
            OutlinedTextField(
                value = state.inferenceMode.name,
                onValueChange = {},
                readOnly = true,
                label = { Text("Inference mode") },
                trailingIcon = { ExposedDropdownMenuDefaults.TrailingIcon(modeExpanded) },
                modifier = Modifier.menuAnchor().fillMaxWidth(),
            )
            ExposedDropdownMenu(expanded = modeExpanded, onDismissRequest = { modeExpanded = false }) {
                InferenceMode.entries.forEach { mode ->
                    DropdownMenuItem(
                        text = { Text(mode.name) },
                        onClick = {
                            onModeChanged(mode)
                            modeExpanded = false
                        },
                    )
                }
            }
        }

        Row {
            Button(onClick = onBack) { Text("Back") }
        }
    }
}
