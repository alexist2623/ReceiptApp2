# Span Rel-G Test Evaluation And Overlay

## Purpose

Evaluate a trained span-level rel-g parser on the CORD-v2 test split and generate graph overlays for manual inspection.

The main checks are:

- MENU_NM -> MENU_PRICE pair precision/recall/F1
- hard negative false positives, especially TOTAL_* or SUBTOTAL_* prices attached to menu names
- dependent collisions where the same dependent span is attached to multiple heads
- whether count/unit price/price edges visually match the receipt image

## Prerequisites

- `processed_data/span_relg/` exists.
- `models/span-relg-context/best/model.pt` exists.
- `../receipt_training_data2/` exists for image overlays.
- Run in WSL with the `receipt-ml` conda environment.

This repo's span-relg cache may not be a single `test.pt` file. The evaluation scripts support both direct split files and the current `manifest.json` + per-sample `.pt` structure. Checkpoints also support both `config.json` and `model_config.json`.

Field vocabulary is resolved from the first available source:

1. `processed_data/span_relg/field_vocab.json`
2. `models/span-relg-context/best/field_vocab.json`
3. `processed_data/span_relg/schema.json`
4. `models/span-relg-context/best/schema.json`
5. `processed_data/span_relg/manifest.json`
6. `models/span-relg-context/best/config.json` or `model_config.json`

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate receipt-ml
which python
```

Expected Python:

```text
/home/alexist/miniconda3/envs/receipt-ml/bin/python
```

## Path Debug

Run this before evaluation when cache/checkpoint layout changes:

```bash
python scripts/debug_span_relg_paths.py \
  --dataset_dir processed_data/span_relg \
  --checkpoint models/span-relg-context/best \
  --split test
```

It prints the resolved split cache path, sample count estimate, config path, field vocab source, and whether `model.pt` exists.

## Test Evaluation

```bash
python scripts/eval_span_relg.py \
  --dataset_dir processed_data/span_relg \
  --checkpoint models/span-relg-context/best \
  --split test \
  --threshold 0.8 \
  --out_dir outputs/span_relg_eval \
  --device auto \
  --debug
```

Key outputs:

- `outputs/span_relg_eval/metrics_summary.json`
- `outputs/span_relg_eval/edge_predictions.jsonl`
- `outputs/span_relg_eval/item_predictions.jsonl`
- `outputs/span_relg_eval/hard_negative_errors.json`
- `outputs/span_relg_eval/collision_errors.json`
- `outputs/span_relg_eval/no_price_items.json`

## Threshold Sweep

```bash
python scripts/eval_span_relg.py \
  --dataset_dir processed_data/span_relg \
  --checkpoint models/span-relg-context/best \
  --split test \
  --out_dir outputs/span_relg_eval_sweep \
  --device auto \
  --sweep_thresholds \
  --debug
```

Output:

- `outputs/span_relg_eval_sweep/threshold_sweep.json`

The best threshold is selected by MENU_NM -> MENU_PRICE pair F1.

## Single Overlay

```bash
python scripts/visualize_span_relg.py \
  --raw_data_dir ../receipt_training_data2 \
  --dataset_dir processed_data/span_relg \
  --checkpoint models/span-relg-context/best \
  --split test \
  --index 0 \
  --threshold 0.8 \
  --out_dir outputs/span_relg_overlay \
  --device auto \
  --debug
```

Outputs:

- `outputs/span_relg_overlay/test_000000_relg.png`
- `outputs/span_relg_overlay/test_000000_relg_debug.json`

## What To Inspect

1. Menu names connect to the correct menu price.
2. Total/subtotal prices are not connected as menu prices.
3. A price span is not reused by multiple menu names unless that is truly correct.
4. Count, unit price, and price edges attach to the same menu item.
5. Missed gold edges and wrong predicted edges are visible in the debug JSON.
