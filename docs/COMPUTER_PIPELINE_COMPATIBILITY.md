# Computer Pipeline Compatibility

The Android app exports a ZIP that contains at least:

- `<capture_id>.jpg`
- `<capture_id>_ocr.json`

The recommended computer-side unpacked structure is:

```text
user_receipts/
  images/
    <capture_id>.jpg
  ocr_json/
    <capture_id>.json
```

If the ZIP is unpacked with different filenames, copy or rename the image and JSON into this structure before running the Python pipeline.

## Word-Level Prediction

```bash
python scripts/batch_infer_user_ocr_json.py \
  --input_dir user_receipts \
  --checkpoint models/layoutlmv3-cord-full/best \
  --labels processed_data/cord_bio/labels.json \
  --out_dir outputs/user_ocr_batch_inference \
  --device auto \
  --debug
```

## Span Rel-G Grouping

```bash
python scripts/infer_user_span_relg.py \
  --prediction_json outputs/user_ocr_batch_inference/predictions/<capture_id>_prediction.json \
  --layoutlm_checkpoint models/layoutlmv3-cord-full/best \
  --relg_checkpoint models/span-relg-context/best \
  --out_json outputs/user_ocr_batch_inference/<capture_id>_grouped_relg.json \
  --device auto \
  --local_files_only \
  --debug
```

## Critical Compatibility Rule

The exported image and OCR JSON must use the same coordinate space. If the app sends a rotated image but OCR boxes from a pre-rotation bitmap, the LayoutLMv3 and rel-g outputs will be wrong.

Always inspect OCR box overlays before trusting model output.
