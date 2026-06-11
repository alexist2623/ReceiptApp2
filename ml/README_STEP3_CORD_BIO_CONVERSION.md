# Step 3: CORD-v2 BIO Conversion

## Purpose

Convert CORD-v2 `valid_line` categories into word-level BIO labels.

## Prerequisite

- `../receipt_training_data2/` must contain CORD-v2 in a form that can be loaded with `datasets.load_from_disk`.
- Run this step in the WSL conda ML environment.

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ml
which python
pip install -r requirements-ml.txt
```

## Inspect

```bash
python scripts/cord_to_bio.py --data_dir ../receipt_training_data2 --inspect_only
```

## Convert

```bash
python scripts/cord_to_bio.py --data_dir ../receipt_training_data2 --out_dir processed_data/cord_bio --overwrite
```

## Debug Subset

```bash
python scripts/cord_to_bio.py --data_dir ../receipt_training_data2 --out_dir processed_data/cord_bio_debug --max_samples 5 --overwrite
```

## Output Files

```text
processed_data/cord_bio/train.jsonl
processed_data/cord_bio/validation.jsonl
processed_data/cord_bio/test.jsonl
processed_data/cord_bio/labels.json
processed_data/cord_bio/category_stats.json
processed_data/cord_bio/summary.json
```

## Next Step

Step 4 reads 10 samples from `processed_data/cord_bio_debug` or `processed_data/cord_bio` and runs a LayoutLMv3 token classification overfit test.
