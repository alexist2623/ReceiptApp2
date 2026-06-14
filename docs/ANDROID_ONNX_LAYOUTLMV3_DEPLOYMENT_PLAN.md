# Android ONNX LayoutLMv3 Deployment Plan

This document prepares a later Android integration for the CORD-only quantized
LayoutLMv3 pipeline. It does not integrate ONNX Runtime into the Android app yet.

## Current Model Baseline

- PyTorch checkpoint: `models/layoutlmv3-cord-full/best`
- ONNX FP32 model: `models/layoutlmv3-cord-onnx/fp32/model.onnx`
- ONNX dynamic INT8 model: `models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx`
- Labels: `models/layoutlmv3-cord-onnx/int8_dynamic/labels.json`
- Processor/tokenizer files: saved beside the ONNX export/quantization output

Use the dynamic INT8 model first because it stayed within the CORD-only quality
budget during validation.

## Runtime Gate Order

The Android MVP must be gated in this order:

1. **Phase A: precomputed tensor ONNX smoke**
   - Python exports ONNX-ready tensors with `AutoProcessor(apply_ocr=false)`.
   - Android/JVM test loads `input_ids`, `attention_mask`, `bbox`, `pixel_values`.
   - ONNX Runtime runs the selected INT8 model.
   - The test compares word-level top labels against Python expected labels.
2. **Phase B: preprocessing parity**
   - Android preprocessing receives the fixture `words.json` and `boxes.json`.
   - Android-generated tensors are compared with Python fixture tensors.
   - `input_ids`, `attention_mask`, and `bbox` must match exactly.
   - `pixel_values` reports max/mean absolute diff.
3. **Phase C: OCR to label overlay MVP**
   - Only after Phase A/B pass, connect ML Kit OCR output to LayoutLMv3 logits.
   - Recover word-level labels and draw label overlay.
4. **Phase D: rel-g integration later**
   - Do not add rel-g to Android until LayoutLMv3 preprocessing and label overlay are stable.

The current repo includes the Phase A ONNX smoke runner and a Phase B parity
comparator gate. The real Android tokenizer/image preprocessing implementation
is intentionally still separate from OCR and not wired into the app flow yet.

## Python Fixture Generation

Generate ignored fixture artifacts from WSL:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate receipt-ml

python scripts/quantization/export_android_layoutlmv3_parity_fixtures.py \
  --checkpoint_for_processor models/layoutlmv3-cord-full/best \
  --onnx_model models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --out_dir fixtures/layoutlmv3_cord_int8_android \
  --splits validation,test \
  --samples_per_split 2 \
  --onnx_provider cpu \
  --local_files_only \
  --overwrite
```

Each fixture sample contains:

```text
input_ids.npy
attention_mask.npy
bbox.npy
pixel_values.npy
logits.npy
last_hidden_state.npy
expected_word_labels.json
words.json
boxes.json
normalized_boxes.json
metadata.json
```

`fixtures/` is gitignored.

## Model Artifact Manifest

Generate the ignored Android model manifest:

```bash
python scripts/quantization/create_layoutlmv3_android_model_manifest.py \
  --model models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx \
  --checkpoint_for_processor models/layoutlmv3-cord-full/best \
  --out artifacts/layoutlmv3_cord_int8_dynamic_manifest.json
```

The manifest records model hash, tokenizer hashes, labels hash, input/output
names, dtypes, shapes, `max_length=512`, image input size, and bbox policy.

`artifacts/` is gitignored.

## ONNX Inputs

The exported LayoutLMv3 ONNX graph expects these inputs:

| Input | Shape | Type | Notes |
| --- | --- | --- | --- |
| `input_ids` | `[1, 512]` | int64 | Token ids from the same LayoutLMv3 tokenizer |
| `attention_mask` | `[1, 512]` | int64 | 1 for real tokens, 0 for padding |
| `bbox` | `[1, 512, 4]` | int64 | LayoutLMv3 0-1000 normalized token boxes |
| `pixel_values` | `[1, 3, 224, 224]` | float32 | LayoutLMv3 image processor output |

The model outputs:

| Output | Shape | Notes |
| --- | --- | --- |
| `logits` | `[1, 512, num_labels]` | Token classification logits |
| `last_hidden_state` | usually `[1, 709, 768]` | Text tokens plus visual tokens; use text token positions for rel-g spans |

## Android Inputs Available From The App

The app should provide:

- Canonical receipt image used for OCR and model input.
- OCR words in reading order.
- OCR word boxes in pixel coordinates.
- Image width and height after EXIF orientation handling.
- Optional export zip containing the image, OCR JSON, and debug metadata.

LayoutLMv3 does not run OCR. The app must pass words and boxes from ML Kit or
another OCR engine with `apply_ocr=false` semantics.

## Android Components To Implement Later

1. Hugging Face-compatible tokenizer.
2. Word to subword mapping.
3. Word box to token box alignment.
4. Pixel box to LayoutLMv3 0-1000 bbox normalization.
5. LayoutLMv3 image preprocessing to `[1, 3, 224, 224]` float32.
6. ONNX Runtime session for `model.onnx`.
7. Logits decode to token labels.
8. First-subword word-level prediction recovery.
9. BIO span merge.
10. Optional `last_hidden_state` extraction for span-level rel-g.

## Python vs Android Parity Test

Before shipping Android inference, run the same 10 receipts through Python and
Android preprocessing and compare:

- `input_ids`
- `attention_mask`
- `bbox`
- `pixel_values` min, max, mean, std
- tokenizer output strings
- word ids / word-to-token alignment
- final word-level labels

The Android path should not be considered valid until these match closely.

## Android Test Commands

The Windows user profile path contains non-ASCII characters on this machine.
If Gradle test workers cannot read their classpath argfile, set an ASCII
`GRADLE_USER_HOME`:

```powershell
$env:JAVA_HOME='C:\Program Files\Android\Android Studio\jbr'
$env:Path="$env:JAVA_HOME\bin;$env:Path"
$env:GRADLE_USER_HOME='C:\JeonghyunPark\.gradle-codex'
cd ReceiptApp
.\gradlew.bat test
.\gradlew.bat assembleDebug
```

The ONNX smoke test defaults to:

```text
../models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx
../fixtures/layoutlmv3_cord_int8_android
```

Override with:

```powershell
$env:LAYOUTLMV3_MODEL_PATH='C:\path\to\model.onnx'
$env:LAYOUTLMV3_FIXTURE_DIR='C:\path\to\fixtures\layoutlmv3_cord_int8_android'
```

If the host JVM cannot initialize the ONNX Runtime native DLL, the host unit
test is skipped with an assumption message. This keeps normal JVM tests usable
while still allowing the same test to run on hosts where ONNX Runtime Java loads
correctly. The Android APK still includes `onnxruntime-android`; a later
device/instrumented test should be used as the final Phase A runtime gate.

## Candidate Android Dependencies

- ONNX Runtime Android for initial integration.
- ONNX Runtime Mobile or a reduced operator build later if APK size is too high.
- A tokenizer implementation that can reproduce the LayoutLMv3 tokenizer output.

## Suggested Asset Layout

```text
ReceiptApp/app/src/main/assets/models/layoutlmv3_cord_int8_dynamic/
  model.onnx
  labels.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
  merges.txt
  preprocessor_config.json
  export_config.json
```

`export_config.json` should record:

```json
{
  "model_variant": "layoutlmv3-cord-int8-dynamic",
  "max_length": 512,
  "bbox_scale": 1000,
  "apply_ocr": false,
  "inputs": ["input_ids", "attention_mask", "bbox", "pixel_values"],
  "outputs": ["logits", "last_hidden_state"]
}
```

## First Android MVP

The first mobile milestone should be:

1. OCR JSON plus receipt image.
2. Android preprocessing.
3. ONNX LayoutLMv3 logits inference.
4. Word-level label recovery.
5. Label overlay on the receipt image.

Keep rel-g on PC/server until the LayoutLMv3 Android preprocessing parity passes.

## Next Android Milestone

After the MVP works:

1. Merge BIO labels into spans.
2. Extract text-token hidden states from `last_hidden_state`.
3. Pool span hidden features.
4. Run span-level rel-g.
5. Produce grouped item JSON.

## Risks

- Tokenizer mismatch between Python and Android.
- LayoutLMv3 image preprocessing mismatch.
- EXIF orientation and OCR coordinate mismatch.
- APK size from ONNX model and tokenizer assets.
- Runtime memory use from `last_hidden_state`.
- Inference latency on mid-range devices.
- Extra memory if rel-g is also moved onto mobile.

## Required Gates Before Android Integration

- ONNX FP32 parity passes.
- Dynamic INT8 token F1 drop is <= 2%.
- Rel-g hidden impact drop is <= 3%.
- Python/Android preprocessing parity passes on at least 10 receipts.
- OCR boxes visually align with the canonical image used for inference.

## What Was Not Done In This Step

- No user-facing Android inference flow was changed.
- No OCR-to-model pipeline was connected.
- No model file was copied into Android assets.
- No user labeled data was used.
- No rel-g Android integration was added.
