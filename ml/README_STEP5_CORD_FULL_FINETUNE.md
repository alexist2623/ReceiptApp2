# Step 5: CORD Full Fine-Tuning

## Purpose

전체 CORD-v2 train split으로 LayoutLMv3 token classification fine-tuning을 수행하고, validation F1이 가장 높은 best checkpoint를 저장한다.

## Prerequisites

- `../receipt_training_data2/`에 CORD-v2 원본 dataset이 있어야 한다.
- `processed_data/cord_bio/`에 3단계 BIO 변환 결과가 있어야 한다.
- 4단계 overfit sanity check가 통과되어야 한다.
- WSL conda `ml` 또는 ML용 conda 환경에서 실행해야 한다.
- GPU 사용을 권장한다. CUDA가 없으면 전체 학습은 실행하지 않는다.

## Environment

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate ml
which python
nvidia-smi
pip install -r requirements-ml.txt
```

## Debug Dry-Run

```bash
python scripts/train_layoutlmv3_cord_full.py --bio_dir processed_data/cord_bio --raw_data_dir ../receipt_training_data2 --model_name_or_path models/layoutlmv3-base --local_files_only --output_dir models/layoutlmv3-cord-full-debug --max_train_samples 16 --max_eval_samples 16 --epochs 1 --batch_size 1 --device auto --debug --overwrite_output_dir
```

## Full Training

```bash
python scripts/train_layoutlmv3_cord_full.py --bio_dir processed_data/cord_bio --raw_data_dir ../receipt_training_data2 --model_name_or_path models/layoutlmv3-base --local_files_only --output_dir models/layoutlmv3-cord-full --epochs 10 --batch_size 2 --gradient_accumulation_steps 4 --learning_rate 5e-5 --device auto --fp16 --overwrite_output_dir
```

## Remote Model Fallback

```bash
python scripts/train_layoutlmv3_cord_full.py --bio_dir processed_data/cord_bio --raw_data_dir ../receipt_training_data2 --model_name_or_path microsoft/layoutlmv3-base --output_dir models/layoutlmv3-cord-full --epochs 10 --batch_size 2 --gradient_accumulation_steps 4 --learning_rate 5e-5 --device auto --fp16 --overwrite_output_dir
```

## Results

- `models/layoutlmv3-cord-full/best/`
- `models/layoutlmv3-cord-full/last/`
- `models/layoutlmv3-cord-full/training_history.json`
- `models/layoutlmv3-cord-full/best_metrics.json`
- `outputs/layoutlmv3_cord_full_eval/validation_preview_epoch_best.jsonl`
- `outputs/layoutlmv3_cord_full_eval/validation_preview_last.jsonl`

## Checks

1. train loss가 감소하는가
2. validation F1이 epoch마다 개선되는가
3. best checkpoint가 저장되는가
4. O label만 예측하는 collapse가 발생하지 않는가
5. validation prediction preview에서 주요 label이 나오는가

## Failure Checklist

1. BIO label 변환 오류
2. word/subword label alignment 오류
3. bbox normalization 오류
4. batch에서 labels가 대부분 `-100`이 되는 문제
5. learning rate 문제
6. GPU memory 문제

## Next Step

6단계에서는 `models/layoutlmv3-cord-full/best/` checkpoint로 CORD test prediction overlay를 만든다.
