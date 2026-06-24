# ReceiptApp2
LayoutLMv3 based

## PaddleOCR word-box pipeline

The PaddleOCR return-word-box path was verified in the WSL `receipt-ml` environment with:

- `paddleocr==3.7.0`
- `paddlepaddle==3.0.0`
- OCR model pair: `PP-OCRv5_mobile_det` + `PP-OCRv5_mobile_rec`

Use `scripts/paddleocr3_only_return_word_box_ocr.py` when PaddleOCR itself must produce the word boxes. This path reads PaddleOCR 3 result fields `res.text_word` and `res.text_word_boxes` directly; it does not use custom line-box projection or whitespace-based splitting.

Example:

```bash
python scripts/paddleocr3_only_return_word_box_ocr.py \
  --image path/to/receipt.jpg \
  --out_dir outputs/paddleocr3_only_return_word_box
```

To feed that OCR JSON into the current LayoutLMv3 + rel-g pipeline:

```bash
python scripts/run_paddleocr_current_model_pipeline.py \
  --image path/to/receipt.jpg \
  --ocr_json outputs/paddleocr3_only_return_word_box/<receipt>/<receipt>_paddleocr3_only_return_word_box_ocr.json \
  --device auto \
  --local_files_only
```

## Docs

- [Receipt schema v2](docs/RECEIPT_SCHEMA_V2.md)
- [Receipt OCR JSON schema](docs/RECEIPT_OCR_JSON_SCHEMA.md)
- [Android OCR app](docs/ANDROID_OCR_APP.md)
