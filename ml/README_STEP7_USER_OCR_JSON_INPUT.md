# Step 7: User OCR JSON Input

## Purpose

사용자 OCR JSON(`words` + `boxes`)을 fine-tuned LayoutLMv3 checkpoint에 넣어 word-level prediction을 수행한다.

## Prerequisites

- `models/layoutlmv3-cord-full/best/` checkpoint가 있어야 한다.
- `processed_data/cord_bio/labels.json`이 있어야 한다.
- WSL conda `ml` 또는 ML용 conda 환경에서 실행해야 한다.

LayoutLMv3는 OCR을 포함하지 않는다. OCR은 외부 앱이나 서비스에서 수행되어야 하며, 이번 단계에서는 반드시 `apply_ocr=False`를 사용한다.

## Recommended OCR JSON Schema

```json
{
  "image_width": 800,
  "image_height": 1200,
  "words": [
    {"text": "volcano", "box": [80, 120, 210, 155]},
    {"text": "iced", "box": [220, 120, 300, 155]},
    {"text": "coffee", "box": [80, 160, 220, 195]},
    {"text": "4,000", "box": [580, 120, 690, 155]}
  ]
}
```

Supported input variants include `tokens`, nested `lines[].words`, ML Kit-like `textBlocks[].lines[].elements`, and a simple top-level list of word objects.

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ml
which python
pip install -r requirements-ml.txt
```

## Create Sample OCR JSON

```bash
python scripts/make_sample_ocr_json.py --out sample_data/sample_user_ocr.json
```

## Run Inference

```bash
python scripts/infer_user_ocr_json.py \
  --image path/to/receipt.jpg \
  --ocr_json path/to/ocr.json \
  --checkpoint models/layoutlmv3-cord-full/best \
  --labels processed_data/cord_bio/labels.json \
  --device auto \
  --debug
```

## Results

- `outputs/user_ocr_inference/<image_name>_prediction.json`
- `outputs/user_ocr_inference/<image_name>_overlay.png`
- `outputs/user_ocr_inference/<image_name>_ocr_debug.json`
- `outputs/user_ocr_inference/run_config.json`

## Checks

1. OCR box가 이미지 위에 맞는가
2. predicted label이 `MENU_NM`, `MENU_PRICE`, `TOTAL_TOTAL_PRICE` 등으로 나오는가
3. `O` label만 나오는 collapse가 없는가
4. image size와 OCR JSON size가 같은가
5. 이미지 회전/EXIF 때문에 box가 어긋나지 않는가

## Common Issues

1. OCR JSON box가 normalized인데 pixel로 해석함  
   해결: `--assume_boxes_normalized`
2. box가 `xywh`인데 `xyxy`로 해석함  
   해결: `--box_format xywh`
3. 이미지가 EXIF 회전되어 box가 어긋남  
   해결: OCR에 사용한 이미지와 inference image를 동일하게 저장
4. OCR word 단위가 CORD와 달라 prediction 품질이 낮음
5. 실제 사용자 영수증 도메인이 CORD와 달라 성능이 낮음

## Next Step

8단계에서는 실제 사용자 영수증 이미지와 앱 OCR JSON을 넣어서 prediction overlay를 검증한다. 9단계에서는 word-level prediction을 menu-price grouped JSON으로 변환한다.
