# Test Exported Receipt ZIP

## Purpose

This check verifies that an Android app OCR export, after being unzipped, can enter the computer ML pipeline.

It checks:

1. image/OCR JSON pair discovery,
2. OCR JSON schema compatibility,
3. OCR box coordinates on the saved image,
4. optional LayoutLMv3 word-level prediction,
5. optional span-level rel-g grouped item output.

No OCR, training, or fine-tuning is performed in this step.

## Prerequisites

- The Android ZIP has already been extracted.
- WSL conda env `receipt-ml` is used.
- `models/layoutlmv3-cord-full/best/` exists.
- `models/span-relg-context/best/` exists.
- `processed_data/cord_bio/labels.json` exists.

Environment:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate receipt-ml
which python
pip install -r requirements-ml.txt
```

`which python` should be:

```text
/home/alexist/miniconda3/envs/receipt-ml/bin/python
```

## Supported Unzipped Folder Shapes

Shape A:

```text
ZIP_EXTRACT_DIR/
  <capture_id>.jpg
  <capture_id>_ocr.json
  <capture_id>_export_validation.json
  <capture_id>_server_result.json
```

This is the current Android ZIP structure.

Shape B:

```text
ZIP_EXTRACT_DIR/
  <capture_id>_receipt_ocr/
    <capture_id>.jpg
    <capture_id>_ocr.json
    <capture_id>_server_result.json
```

Shape C:

```text
ZIP_EXTRACT_DIR/
  images/
    <capture_id>.jpg
  ocr_json/
    <capture_id>.json
```

Shape D:

```text
ZIP_EXTRACT_DIR/
  <capture_id>/
    images/
      <capture_id>.jpg
    ocr_json/
      <capture_id>.json
```

Shape E:

```text
ZIP_EXTRACT_DIR/
  <capture_id>.jpg
  <capture_id>.json
```

The script also treats `<capture_id>_ocr.json` as matching `<capture_id>.jpg`.

## Full Pipeline Test

Replace `path/to/unzipped_receipt_export` with the actual extracted folder:

```bash
python scripts/test_exported_receipt_zip.py \
  --input_dir path/to/unzipped_receipt_export \
  --layoutlm_checkpoint models/layoutlmv3-cord-full/best \
  --relg_checkpoint models/span-relg-context/best \
  --labels processed_data/cord_bio/labels.json \
  --out_dir outputs/exported_receipt_zip_test \
  --device auto \
  --local_files_only \
  --debug
```

## Schema And OCR Overlay Only

Before model inference or labeling, run the coordinate validator:

```bash
python scripts/validate_receipt_export_coordinates.py \
  --input_dir path/to/unzipped_receipt_export \
  --strict \
  --out_json outputs/coordinate_validation_summary.json
```

```bash
python scripts/test_exported_receipt_zip.py \
  --input_dir path/to/unzipped_receipt_export \
  --out_dir outputs/exported_receipt_zip_test_schema_only \
  --skip_model_inference \
  --debug
```

## Outputs

```text
outputs/exported_receipt_zip_test/
  ocr_overlay/
    <capture_id>_ocr_overlay.png
  staged_user_receipts/
    images/
    ocr_json/
  layoutlmv3_predictions/
  grouped/
  grouped_overlays/
  staged_pairs.json
  test_summary.json
  test_summary.csv
  index.html
```

## What To Check

1. OCR overlay boxes align with the text in the image.
2. JSON `image_width` / `image_height` match the actual saved image size.
3. `words[].box` is `[left, top, right, bottom]` in saved image pixels.
4. LayoutLMv3 overlay labels such as `ITEM_NAME`/`ITEM_PRICE` look plausible. Legacy `MENU_NM`/`MENU_PRICE` labels are still accepted as aliases.
5. Grouped rel-g JSON connects item names and prices correctly.

## Troubleshooting

- If boxes are rotated by 90 degrees, check Android canonical image and EXIF handling.
- If boxes have a scale mismatch, the image used for OCR and the exported image are different.
- If `words` is empty, app OCR failed or the JSON schema is wrong.
- If image size mismatches, the JSON was likely generated for another image.
- If grouped output is empty, inspect OCR quality and LayoutLMv3 span prediction first.
