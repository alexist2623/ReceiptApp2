# Step 2: LayoutLMv3 Forward Smoke Test

## Purpose

This step verifies that a CORD-v2 receipt image plus CORD GT OCR words and boxes can pass through the `microsoft/layoutlmv3-base` backbone without errors.

## Prerequisite

`../receipt_training_data2/` must contain CORD-v2 in a form that can be loaded with `datasets.load_from_disk`.

## Environment

Use WSL and the WSL conda ML environment. Do not use Windows Python for this step.

LayoutLMv3 is not an OCR model. OCR words and boxes must be provided externally. In this step, CORD GT words and boxes are used as the OCR results.

This step does not do fine-tuning, BIO conversion, or label prediction.

## Install

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ml
pip install -r requirements-ml.txt
```

## Run

```bash
python scripts/layoutlmv3_forward_smoke.py --data_dir ../receipt_training_data2 --split train --index 0
python scripts/layoutlmv3_forward_smoke.py --data_dir ../receipt_training_data2 --split validation --index 0
python scripts/layoutlmv3_forward_smoke.py --data_dir ../receipt_training_data2 --split test --index 0
```

## Save Local Model

```bash
python scripts/layoutlmv3_forward_smoke.py --data_dir ../receipt_training_data2 --split train --index 0 --save_local_model
```

## Offline/Local Run

```bash
python scripts/layoutlmv3_forward_smoke.py --data_dir ../receipt_training_data2 --split train --index 0 --model_name_or_path models/layoutlmv3-base --local_files_only
```

## Output

Debug JSON files are written to:

```text
outputs/layoutlmv3_smoke/<split>/
```

## Next Step

If this smoke test passes, move on to Step 3: converting CORD annotations to BIO labels.
