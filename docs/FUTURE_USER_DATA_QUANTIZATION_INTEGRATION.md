# Future User Data Quantization Integration

This document explains how to reuse the CORD-only quantization pipeline after
enough high-quality user labeled receipts are available.

Nothing in this document was executed during the CORD-only quantization work.
The current quantized baseline uses only CORD data and the checkpoint:

```text
models/layoutlmv3-cord-full/best
```

## When To Start

Do not start user or mixed quantization until all of these are true:

- At least 50-100 hand-labeled user receipts are available.
- User validation and test sets are fixed and will not be augmented with repeat/copy data.
- Temp, scratch, or failed export directories are excluded.
- Image size and OCR box coordinates are validated.
- Labels use the current canonical schema.
- Relation annotations exist for rel-g positive examples.
- User-only evaluation scripts are ready.

## Suggested User Data Layout

```text
user_data/
  train/
    <capture_id>_receipt_ocr/
      <capture_id>.jpg
      <capture_id>_ocr.json
      <capture_id>_labeled_v2_1.json
  validation/
    <capture_id>_receipt_ocr/
      <capture_id>.jpg
      <capture_id>_ocr.json
      <capture_id>_labeled_v2_1.json
  test/
    <capture_id>_receipt_ocr/
      <capture_id>.jpg
      <capture_id>_ocr.json
      <capture_id>_labeled_v2_1.json
```

Keep `validation/` and `test/` stable. Do not add repeated/augmented copies to
those splits.

## Validation Before Training

Run label and coordinate validation before any fine-tuning:

```bash
python scripts/validate_user_labels_v2.py ...
python scripts/validate_receipt_export_coordinates.py ...
```

Samples should be rejected from training if:

- Image dimensions do not match the label JSON and cannot be safely scaled.
- OCR boxes are missing or zero-area.
- BIO labels are malformed.
- Relation ids point to missing words/spans.
- A rel-g training sample has no positive relation when it should.

## Mixed LayoutLMv3 Training

Future mixed fine-tuning should start from the preserved pre-user baseline, not
from an overfit or already mixed checkpoint.

Recommended starting checkpoint:

```text
models/layoutlmv3-cord-full/best
```

The future mixed training script should explicitly record:

- CORD sample count.
- User sample count.
- CORD/user sampling ratio.
- Whether user samples are oversampled.
- Frozen or unfrozen backbone policy.
- Validation split names and sizes.

## Reusing The Quantization Pipeline

After a mixed checkpoint exists, rerun the same pipeline with only the checkpoint
path changed:

1. PyTorch FP32 eval on CORD and user-only validation/test.
2. ONNX FP32 export.
3. PyTorch vs ONNX FP32 parity.
4. Dynamic INT8 quantization.
5. ONNX INT8 token eval.
6. Rel-g hidden impact eval.
7. Optional static/selective quantization if the INT8 drop is too large.

Example:

```bash
python scripts/quantization/export_layoutlmv3_to_onnx.py \
  --checkpoint models/layoutlmv3-mixed-user-cord/best \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --out_dir models/layoutlmv3-mixed-user-cord-onnx/fp32 \
  --local_files_only \
  --overwrite
```

Then run:

```bash
python scripts/quantization/compare_pytorch_onnx_layoutlmv3.py ...
python scripts/quantization/quantize_layoutlmv3_onnx_dynamic.py ...
python scripts/quantization/eval_layoutlmv3_cord_onnx.py ...
python scripts/quantization/build_span_relg_dataset_from_onnx.py ...
python scripts/quantization/evaluate_relg_with_quantized_layout_hidden.py ...
```

## Required Future Metrics

Track both CORD and user-only quality:

- CORD token F1.
- CORD key field F1.
- User token F1.
- User `ITEM_NAME` F1.
- User `ITEM_PRICE` F1.
- User `TOTAL_PRICE` F1.
- User item-price pair F1.
- hard negative false positive count.
- total/subtotal false positive count.
- dependent collision count.
- latency and model size.

## Comparison Table

The future report should compare:

| Variant | CORD F1 | User F1 | User item-price F1 | Size MB | Latency ms | Notes |
| --- | --- | --- | --- | --- | --- | --- |
| CORD-only PyTorch FP32 | | | | | | Baseline |
| CORD-only INT8 | | | | | | Current quantized baseline |
| Mixed PyTorch FP32 | | | | | | Future |
| Mixed INT8 | | | | | | Future |

## Guardrails

- Do not mix Temp directories into training.
- Do not use unlabeled OCR exports as supervised labels.
- Do not pick checkpoints using the test split.
- Do not replace rel-g with y-coordinate heuristic-only grouping.
- Do not directly predict `group_id` classes.
- Do not commit generated `models/`, `outputs/`, or `processed_data/` artifacts.

## Current Status

This step is documentation only. No user data was used, no mixed checkpoint was
created, and no model was fine-tuned.
