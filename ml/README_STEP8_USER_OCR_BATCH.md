# Step 8: User OCR Batch Inference

## Purpose

여러 사용자 영수증 이미지와 OCR JSON pair를 batch로 처리해서 fine-tuned LayoutLMv3 prediction JSON, overlay PNG, OCR debug JSON, summary, HTML gallery를 생성한다.

이번 단계는 7단계의 단일 사용자 OCR JSON inference를 여러 샘플에 적용하는 단계다. 학습, CORD 평가, OCR 실행, menu-price grouping은 하지 않는다.

## Prerequisites

- `models/layoutlmv3-cord-full/best/` checkpoint가 있어야 한다.
- `processed_data/cord_bio/labels.json`이 있어야 한다.
- WSL conda `ml` 또는 ML용 conda 환경에서 실행해야 한다.
- 사용자 영수증 이미지와 OCR JSON은 git에 commit하지 않는다.

## Recommended Input Layout

`user_receipts/`는 `.gitignore`에 포함되어 있다.

```text
user_receipts/
  receipt_001.jpg
  receipt_001_ocr.json
  receipt_002.png
  receipt_002.json
```

이미지 stem과 OCR JSON stem이 같거나, OCR JSON이 `_ocr.json`, `.ocr.json` 같은 suffix를 가지면 자동으로 pair로 매칭된다.

별도 폴더도 사용할 수 있다.

```text
user_receipts/
  images/
    receipt_001.jpg
  ocr/
    receipt_001_ocr.json
```

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ml
which python
pip install -r requirements-ml.txt
```

## Dry Run

```bash
python scripts/batch_infer_user_ocr_json.py \
  --input_dir user_receipts \
  --checkpoint models/layoutlmv3-cord-full/best \
  --labels processed_data/cord_bio/labels.json \
  --dry_run
```

## Batch Inference

```bash
python scripts/batch_infer_user_ocr_json.py \
  --input_dir user_receipts \
  --checkpoint models/layoutlmv3-cord-full/best \
  --labels processed_data/cord_bio/labels.json \
  --out_dir outputs/user_ocr_batch_inference \
  --device auto \
  --debug
```

## Separate Image/OCR Directories

```bash
python scripts/batch_infer_user_ocr_json.py \
  --image_dir user_receipts/images \
  --ocr_dir user_receipts/ocr \
  --checkpoint models/layoutlmv3-cord-full/best \
  --labels processed_data/cord_bio/labels.json \
  --out_dir outputs/user_ocr_batch_inference \
  --device auto
```

## CSV Pair List

CSV columns must include `image` and `ocr_json`. Optional `id` is supported.

```csv
id,image,ocr_json
receipt_001,user_receipts/images/receipt_001.jpg,user_receipts/ocr/receipt_001_ocr.json
```

```bash
python scripts/batch_infer_user_ocr_json.py \
  --pairs_csv user_receipts/pairs.csv \
  --checkpoint models/layoutlmv3-cord-full/best \
  --labels processed_data/cord_bio/labels.json \
  --out_dir outputs/user_ocr_batch_inference
```

## Results

```text
outputs/user_ocr_batch_inference/
  predictions/
    <id>_prediction.json
  overlays/
    <id>_overlay.png
  debug/
    <id>_ocr_debug.json
  summary.json
  summary.csv
  gallery.html
  run_config.json
```

## Checks

1. OCR boxes align with the receipt image.
2. Predicted labels look plausible for `MENU_NM`, `MENU_PRICE`, `TOTAL_TOTAL_PRICE`, etc.
3. `O` label collapse is not happening.
4. OCR JSON image size matches the actual image size.
5. Low confidence or repeated confusions are visible in `summary.csv` and `gallery.html`.

## Common Options

- `--assume_boxes_normalized`: OCR JSON boxes are already 0-1000 normalized coordinates.
- `--box_format xywh`: OCR boxes are `[x, y, width, height]`.
- `--hide_o`: hide `O` predictions in overlays.
- `--draw_conf_threshold 0.5`: hide labels below a confidence threshold.
- `--limit 5`: process only the first 5 matched pairs.

## Next Step

9단계에서는 word-level prediction을 menu-price grouped JSON으로 변환한다.
