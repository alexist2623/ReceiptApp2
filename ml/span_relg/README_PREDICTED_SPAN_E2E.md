# Predicted Span Rel-G End-To-End Evaluation

## Purpose

Evaluate the span-level rel-g parser with predicted spans instead of oracle CORD spans.

This is needed because the app will not know CORD GT spans at runtime. The runtime pipeline has to:

1. Use external OCR words and boxes.
2. Run LayoutLMv3 token classification with `apply_ocr=False`.
3. Merge word-level BIO predictions into spans.
4. Feed predicted spans and LayoutLMv3 hidden states into the span-level rel-g parser.
5. Decode menu item relations and compare them against CORD gold groups for evaluation only.

No training, OCR, or Android code changes are performed in this step.

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

## 1. Oracle Eval Check

```bash
python scripts/debug_span_relg_paths.py \
  --dataset_dir processed_data/span_relg \
  --checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --split test

python scripts/eval_span_relg.py \
  --dataset_dir processed_data/span_relg \
  --checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --split test \
  --threshold 0.84 \
  --out_dir outputs/span_relg_eval \
  --device auto \
  --debug

python scripts/eval_span_relg.py \
  --dataset_dir processed_data/span_relg \
  --checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --split test \
  --out_dir outputs/span_relg_eval_sweep \
  --device auto \
  --sweep_thresholds \
  --debug

python scripts/write_span_relg_oracle_report.py
```

Oracle outputs:

- `outputs/span_relg_eval/metrics_summary.json`
- `outputs/span_relg_eval_sweep/threshold_sweep.json`
- `outputs/span_relg_eval/oracle_span_report.md`
- `outputs/span_relg_overlay/*.png`

## 2. Predicted-Span Debug Run

```bash
python scripts/eval_predicted_span_relg_e2e.py \
  --raw_data_dir ../receipt_training_data2 \
  --layoutlm_checkpoint models/layoutlmv3-cord-full/best \
  --relg_checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --span_relg_dataset_dir processed_data/span_relg \
  --split test \
  --threshold 0.84 \
  --max_samples 10 \
  --out_dir outputs/predicted_span_relg_e2e_debug \
  --device auto \
  --local_files_only \
  --debug
```

## 3. Full Predicted-Span Eval

```bash
python scripts/eval_predicted_span_relg_e2e.py \
  --raw_data_dir ../receipt_training_data2 \
  --layoutlm_checkpoint models/layoutlmv3-cord-full/best \
  --relg_checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --span_relg_dataset_dir processed_data/span_relg \
  --split test \
  --threshold 0.84 \
  --out_dir outputs/predicted_span_relg_e2e \
  --device auto \
  --local_files_only \
  --debug
```

## 4. Predicted-Span Threshold Sweep

```bash
python scripts/eval_predicted_span_relg_e2e.py \
  --raw_data_dir ../receipt_training_data2 \
  --layoutlm_checkpoint models/layoutlmv3-cord-full/best \
  --relg_checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --span_relg_dataset_dir processed_data/span_relg \
  --split test \
  --out_dir outputs/predicted_span_relg_e2e_sweep \
  --device auto \
  --local_files_only \
  --sweep_thresholds \
  --debug
```

## Results

- `outputs/predicted_span_relg_e2e/metrics_summary.json`
- `outputs/predicted_span_relg_e2e/oracle_vs_predicted_report.md`
- `outputs/predicted_span_relg_e2e/index.html`
- `outputs/predicted_span_relg_e2e/overlays/*.png`
- `outputs/predicted_span_relg_e2e/predicted_spans.jsonl`
- `outputs/predicted_span_relg_e2e/edge_predictions.jsonl`
- `outputs/predicted_span_relg_e2e/item_predictions.jsonl`

## Interpretation

- Oracle F1 high but predicted F1 low: span extraction is the likely bottleneck.
- Oracle F1 and predicted F1 both low: rel-g parser or target generation needs work.
- Many total/subtotal false positives: hard negatives need improvement.
- Many collisions: tune threshold or decoding collision avoidance.
