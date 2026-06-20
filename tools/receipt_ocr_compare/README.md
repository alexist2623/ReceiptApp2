# Receipt OCR Token Overlay Comparison

This tool compares OCR recognizers on the same receipt token crops. It is designed for numeric-token inspection: prices, totals, dates, approval numbers, masked card numbers, signs, decimal points, thousands separators, and currency symbols.

All outputs, downloaded source archives, extracted vendor code, and model files are kept under `tools/receipt_ocr_compare/`. The tool does not create submodules and must not contain nested `.git` directories.

## Install

```bash
python -m pip install -r tools/receipt_ocr_compare/requirements.txt
```

Optional real-model dependencies:

```bash
python -m pip install paddleocr
```

Install PaddlePaddle following the PaddleOCR/PaddlePaddle platform instructions for CPU or GPU.

## UI

```bash
streamlit run tools/receipt_ocr_compare/app.py
```

The UI supports image upload, batch directory processing, model selection, recognition-only and end-to-end mode selection, detector selection, crop padding, CPU/GPU selection, numeric-only views, confidence and box display toggles, mismatch-only overlays, ground-truth upload, and result downloads.

## CLI

```bash
python -m tools.receipt_ocr_compare.cli compare \
  --input ./sample_receipts \
  --models svtrv2_b,paddleocr,existing \
  --mode recognition \
  --model-dir tools/receipt_ocr_compare/models \
  --vendor-dir tools/receipt_ocr_compare/vendor \
  --output tools/receipt_ocr_compare/outputs/run_001
```

With ground truth:

```bash
python -m tools.receipt_ocr_compare.cli compare \
  --input tools/receipt_ocr_compare/sample_data/synthetic_receipt.png \
  --models svtrv2_b,paddleocr,existing \
  --mode recognition \
  --detector ground_truth \
  --ground-truth tools/receipt_ocr_compare/sample_data/synthetic_receipt_gt.jsonl \
  --model-dir tools/receipt_ocr_compare/models \
  --vendor-dir tools/receipt_ocr_compare/vendor \
  --output tools/receipt_ocr_compare/outputs/run_001
```

Without ground truth, the tool writes overlays and CSV/JSON files but marks the run as:

```text
Visual comparison only: no ground truth provided.
```

It does not rank models or report accuracy without ground truth.

## Source Downloads

```bash
python tools/receipt_ocr_compare/scripts/download_sources.py \
  --sources openocr,paddleocr \
  --vendor-dir tools/receipt_ocr_compare/vendor
```

The script downloads GitHub archive zips, extracts them under `vendor/`, removes disallowed Git metadata, and writes `vendor/source_manifest.json`.

## Model Downloads

```bash
python tools/receipt_ocr_compare/scripts/download_models.py \
  --models svtrv2_b,paddleocr \
  --model-dir tools/receipt_ocr_compare/models \
  --svtrv2-url <direct-svtrv2-b-checkpoint-url>
```

PaddleOCR model files are discovered from the Hugging Face model repositories and downloaded into:

- `tools/receipt_ocr_compare/models/paddleocr/det/`
- `tools/receipt_ocr_compare/models/paddleocr/rec/`

SVTRv2-B requires a direct checkpoint URL because the OpenOCR project documents the SVTRv2-B checkpoint through external model links. The script writes `models/model_manifest.json` with source URL, repository, revision, local path, checksum, license, download time, and framework version. Downloads use `.partial` files and rename after completion. Existing files are reused only when the recorded checksum matches.

Model checkpoint files stay inside `tools/receipt_ocr_compare/models/`, but this tool's `.gitignore` excludes large checkpoint extensions by default. Commit model binaries only if the repository policy explicitly requires it.

## Verification

```bash
python tools/receipt_ocr_compare/scripts/verify_models.py \
  --model-dir tools/receipt_ocr_compare/models \
  --vendor-dir tools/receipt_ocr_compare/vendor
```

Smoke test without large models:

```bash
python tools/receipt_ocr_compare/scripts/smoke_test.py
```

Unit tests:

```bash
python -m pytest tools/receipt_ocr_compare/tests
```

Nested Git check:

```bash
find . -mindepth 2 -type d -name .git -print
find . -name .gitmodules -print
```

## Ground Truth JSONL

Each line:

```json
{"image":"receipt_001.jpg","tokens":[{"id":"token_001","bbox":[10,20,100,45],"text":"12.30","type":"number"}]}
```

Primary metrics use `raw_text`. Optional correction values such as `O -> 0` are stored separately as `corrected_text_optional` and are not used for primary accuracy.

## Adapters

- `svtrv2_b`: checks for vendored OpenOCR source and a local checkpoint under `models/svtrv2_b/`. It does not silently fall back if unavailable.
- `paddleocr`: uses local PaddleOCR detector/recognizer model directories under `models/paddleocr/` and does not auto-download package-managed models.
- `existing`: detects the Android ML Kit OCR source in the project, but reports unavailable unless `RECEIPT_EXISTING_OCR_CMD` or a config command provides a Python/subprocess runner.
