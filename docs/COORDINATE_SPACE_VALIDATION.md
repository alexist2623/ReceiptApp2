# Coordinate Space Validation

## Problem

A common failure case is:

```text
actual image size: 1536x2048
JSON coordinate size: 3000x4000
```

The overlay then looks shifted or scaled because the boxes were drawn in a
different coordinate space from the image.

## Required Invariant

Android export and Python training data must satisfy:

```text
saved JPG size == OCR input bitmap size == OCR JSON image_width/image_height == box coordinate space
```

If any part is different, do not fine-tune with that image/JSON pair.

## Android Enforcement

The app validates coordinates at three points:

1. `CanonicalImageWriter` re-reads the saved JPG bounds and stores those actual
   dimensions in `ImageInfoDto`.
2. `ReceiptRepository.saveOcrPayload` uses `ReceiptExportValidator` to block JSON
   save if the payload size does not match the saved image.
3. `CaptureViewModel.uploadToServer` uses `ReceiptExportValidator` to block
   upload if the pair is invalid.
4. `ZipExportService.export` uses `ReceiptExportValidator` to block ZIP creation
   if the image and OCR JSON do not match, and writes
   `<capture_id>_export_validation.json` into the ZIP.

## Validate An Export

```bash
python scripts/validate_receipt_export_coordinates.py \
  --input_dir path/to/unzipped_or_label_bundle \
  --strict \
  --out_json outputs/coordinate_validation_summary.json
```

Strict mode exits non-zero when an image/JSON coordinate mismatch is found.

## Overlay Modes

Strict mode is the default and should be used before training:

```bash
python scripts/overlay_labeled_receipt_json.py \
  --image path/to/capture.jpg \
  --label_json path/to/capture_labeled_v2_1.json \
  --coordinate_mode strict
```

Auto-scale is for visualization only:

```bash
python scripts/overlay_labeled_receipt_json.py \
  --image path/to/capture.jpg \
  --label_json path/to/capture_labeled_v2_1.json \
  --coordinate_mode auto-scale
```

If a legacy label JSON has no size fields but boxes are known to be in actual
image pixels:

```bash
python scripts/overlay_labeled_receipt_json.py \
  --image path/to/capture.jpg \
  --label_json path/to/capture_labeled_v2_1.json \
  --coordinate_mode assume-image
```

## Rescale A Label JSON Copy

Use this only when you intentionally want to create a corrected copy. The
original file is not overwritten.

```bash
python scripts/rescale_labeled_receipt_json.py \
  --image path/to/capture.jpg \
  --label_json path/to/capture_labeled_v2_1.json \
  --out path/to/capture_labeled_v2_1_scaled.json
```

Fine-tuning should use only files that pass strict validation after any manual
correction.
