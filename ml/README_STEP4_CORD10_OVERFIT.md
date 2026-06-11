# Step 4: CORD10 LayoutLMv3 Overfit

## Purpose

Run a LayoutLMv3 token classification overfit sanity check on 10 CORD BIO samples.

## Prerequisite

- `../receipt_training_data2/` must contain the original CORD-v2 dataset.
- `processed_data/cord_bio/` must contain the Step 3 BIO conversion output.
- Run this step in the WSL conda ML environment.

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ml
which python
pip install -r requirements-ml.txt
```

## Run

```bash
python scripts/overfit_layoutlmv3_cord10.py --bio_dir processed_data/cord_bio --raw_data_dir ../receipt_training_data2 --split train --num_samples 10 --output_dir models/layoutlmv3-cord10-overfit --epochs 50 --device auto --debug
```

## Use Local LayoutLMv3 Base

```bash
python scripts/overfit_layoutlmv3_cord10.py --bio_dir processed_data/cord_bio --raw_data_dir ../receipt_training_data2 --model_name_or_path models/layoutlmv3-base --local_files_only --num_samples 10 --epochs 50
```

## Results

```text
models/layoutlmv3-cord10-overfit/
outputs/cord10_overfit_overlay/
```

## What To Check

1. Loss decreases over epochs.
2. Token accuracy increases.
3. Seqeval F1 increases.
4. Prediction overlays mostly match GT labels.

## If It Fails

Suspect BIO conversion, word/subword label alignment, bbox normalization, `apply_ocr` configuration, or `labels.json` id mapping.

## Next Step

If this sanity check passes, move on to Step 5: full CORD fine-tuning.
