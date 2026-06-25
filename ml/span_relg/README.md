# Span-Level Rel-G Parser

## Purpose

This module assumes LayoutLMv3 has already found field spans, then applies a SPADE-style rel-g idea at span level to connect `MENU_NM` spans to dependent spans such as `MENU_PRICE`, `MENU_CNT`, and `MENU_UNITPRICE`.

## Difference From Full SPADE

- `rel-s` is not implemented.
- Field span extraction is handled by LayoutLMv3 token classification.
- `rel-g` is implemented as span-level relation prediction.

## Context Encoding

Each receipt graph contains field span nodes and optional context token nodes. A spatial self-attention encoder uses relative box features so nearby OCR context can influence span representations.

## Training Target

CORD-v2 `group_id`, `sub_group_id`, and `row_id` are used to build same-group relation labels. The IDs themselves are never trained as classes; only pair labels are trained.

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ml
which python
pip install -r requirements-ml.txt
```

## Debug Cache

```bash
python scripts/build_span_relg_dataset.py \
  --raw_data_dir ../receipt_training_data2 \
  --checkpoint models/layoutlmv3-cord-full/best \
  --out_dir processed_data/span_relg_debug \
  --max_samples 5 \
  --device auto \
  --local_files_only \
  --overwrite \
  --debug
```

## Debug Train

```bash
python scripts/train_span_relg.py \
  --dataset_dir processed_data/span_relg_debug \
  --output_dir models/span-relg-context-debug \
  --epochs 30 \
  --batch_size 2 \
  --device auto \
  --overwrite_output_dir \
  --debug
```

## Full Cache

```bash
python scripts/build_span_relg_dataset.py \
  --raw_data_dir ../receipt_training_data2 \
  --checkpoint models/layoutlmv3-cord-full/best \
  --out_dir processed_data/span_relg \
  --device auto \
  --local_files_only \
  --overwrite
```

## Full Train

```bash
python scripts/train_span_relg.py \
  --dataset_dir processed_data/span_relg \
  --output_dir models/span-relg-context \
  --epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --device auto \
  --overwrite_output_dir
```

## Continue Training From Previous Checkpoint

```bash
python scripts/train_span_relg.py \
  --dataset_dir processed_data/span_relg \
  --resume_from_checkpoint models/span-relg-context/last \
  --output_dir models/span-relg-context-f1-continued-200 \
  --epochs 200 \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --best_metric menu_price_pair_f1 \
  --device auto \
  --overwrite_output_dir
```

The saved `best/` checkpoint is selected by validation `MENU_PRICE` pair F1 by default. The training curve is saved at `training_curve.png`.

## Test Eval

```bash
python scripts/eval_span_relg.py \
  --dataset_dir processed_data/span_relg \
  --checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --split test \
  --threshold 0.84 \
  --out_dir outputs/span_relg_eval \
  --device auto \
  --debug
```

## Visualize

```bash
python scripts/visualize_span_relg.py \
  --raw_data_dir ../receipt_training_data2 \
  --dataset_dir processed_data/span_relg \
  --checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --split test \
  --index 0 \
  --out_dir outputs/span_relg_overlay \
  --device auto \
  --debug
```

## User Prediction JSON Inference

```bash
python scripts/infer_user_span_relg.py \
  --prediction_json outputs/user_ocr_inference/receipt_001_prediction.json \
  --layoutlm_checkpoint models/layoutlmv3-cord-full/best \
  --relg_checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --out_json outputs/user_ocr_inference/receipt_001_grouped_relg.json \
  --device auto \
  --local_files_only \
  --debug
```
