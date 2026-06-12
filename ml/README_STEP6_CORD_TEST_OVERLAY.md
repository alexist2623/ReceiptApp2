# Step 6: CORD Test Prediction Overlay

## Purpose

Fine-tuned LayoutLMv3 checkpoint로 CORD test split prediction을 수행하고, 원본 이미지 위에 GT/pred label overlay를 저장한다.

## Prerequisites

- `../receipt_training_data2/`에 CORD-v2 원본 dataset이 있어야 한다.
- `processed_data/cord_bio/test.jsonl`이 있어야 한다.
- `models/layoutlmv3-cord-full/best/` checkpoint가 있어야 한다.
- WSL conda `ml` 또는 ML용 conda 환경에서 실행해야 한다.

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ml
which python
pip install -r requirements-ml.txt
```

## Single Sample

```bash
python scripts/infer_cord_test_overlay.py --bio_dir processed_data/cord_bio --raw_data_dir ../receipt_training_data2 --checkpoint models/layoutlmv3-cord-full/best --split test --index 0 --device auto --debug
```

## Multiple Samples

```bash
python scripts/infer_cord_test_overlay.py --bio_dir processed_data/cord_bio --raw_data_dir ../receipt_training_data2 --checkpoint models/layoutlmv3-cord-full/best --split test --start_index 0 --num_samples 20 --device auto
```

## Full Test Evaluation

```bash
python scripts/infer_cord_test_overlay.py --bio_dir processed_data/cord_bio --raw_data_dir ../receipt_training_data2 --checkpoint models/layoutlmv3-cord-full/best --split test --all --save_overlay_limit 30 --device auto
```

## Results

- `outputs/cord_test_pred_overlay/test/`
- `outputs/cord_test_pred_overlay/metrics_summary.json`
- `outputs/cord_test_pred_overlay/seqeval_report.txt`
- `outputs/cord_test_pred_overlay/confusion_top.json`
- `outputs/cord_test_pred_overlay/run_config.json`

## Checks

1. box가 실제 word 위치와 맞는가
2. GT label과 pred label이 잘 표시되는가
3. `MENU_NM`, `MENU_PRICE`, `TOTAL_TOTAL_PRICE` 같은 주요 label이 맞는가
4. total price와 menu price를 혼동하지 않는가
5. O label만 예측하는 collapse가 없는가

## Next Step

7단계에서는 사용자 OCR JSON 입력을 지원해서 CORD GT OCR이 아닌 실제 앱 OCR 결과를 모델에 넣는다.
