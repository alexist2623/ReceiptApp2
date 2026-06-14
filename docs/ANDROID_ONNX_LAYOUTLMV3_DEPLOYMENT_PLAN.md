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

- No Android app code was changed.
- No Gradle dependency was added.
- No model file was copied into Android assets.
- No user labeled data was used.
