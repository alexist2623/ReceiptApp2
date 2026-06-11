# Step 1: CORD-v2 GT Overlay

## Purpose

This step checks whether CORD-v2 ground-truth boxes are aligned correctly on top of the source receipt image.

## Prerequisite

`../receipt_training_data2/` must contain a CORD-v2 dataset that can be loaded with `datasets.load_from_disk`.

## Install

```bash
pip install -r requirements-ml.txt
```

## Run

```bash
python scripts/overlay_cord_gt.py --data_dir ../receipt_training_data2 --split train --index 0
python scripts/overlay_cord_gt.py --data_dir ../receipt_training_data2 --split validation --index 0
python scripts/overlay_cord_gt.py --data_dir ../receipt_training_data2 --split test --index 0
python scripts/overlay_cord_gt.py --data_dir ../receipt_training_data2 --split train --num_samples 5
```

## Output

Overlay images and debug JSON files are written to:

```text
outputs/cord_gt_overlay/<split>/
```

## What To Check

1. Word boxes align with the actual OCR word positions.
2. Line boxes wrap each full line correctly.
3. Categories match regions such as `menu`, `sub_total`, and `total`.
4. There are no image rotation or coordinate scale problems.

## Next Step

Only after this overlay is correct, move on to a LayoutLMv3 forward smoke test and BIO label conversion.
