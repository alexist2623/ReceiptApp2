# Custom Domain Training Flow

This repo now separates three concerns:

1. Convert public/custom labels into the canonical `receipt_labels_v2` BIO schema.
2. Fine-tune LayoutLMv3 from a stable pre-custom checkpoint.
3. Rebuild span-level rel-g cache from the new LayoutLMv3 checkpoint and evaluate predicted-span e2e output.

Generated artifacts under `models/`, `outputs/`, and `processed_data/` are intentionally not committed.

## Canonical Label Schema

Use `schemas/receipt_labels_v2.json` and `ml/receipt_schema.py` as the source of truth.

Supported public/custom labels include item, summary, tax, total, document, and payment fields such as:

- `ITEM_NAME`, `ITEM_QTY`, `ITEM_UNIT_PRICE`, `ITEM_PRICE`
- `SUBTOTAL_NAME`, `SUBTOTAL_PRICE`
- `TAX_NAME`, `TAX_PRICE`, `TAX_RATE`
- `TOTAL_NAME`, `TOTAL_PRICE`
- `PAYMENT_METHOD`, `PAYMENT_INFO`

Ambiguous public labels such as WildReceipt `Others` are emitted as `IGNORE`. Training masks `IGNORE` to `-100`, so these words still provide image/text context but do not contribute label loss.

## 1. Convert WildReceipt

```bash
conda activate receipt-ml

python scripts/convert_wildreceipt_to_receipt_v2_bio.py \
  --wildreceipt_root /mnt/c/JeonghyunPark/Workspace/receipt_training_data3/wildreceipt/wildreceipt \
  --out_dir processed_data/wildreceipt_bio \
  --ignore_ambiguous_others
```

Smoke conversion:

```bash
python scripts/convert_wildreceipt_to_receipt_v2_bio.py \
  --wildreceipt_root /mnt/c/JeonghyunPark/Workspace/receipt_training_data3/wildreceipt/wildreceipt \
  --out_dir processed_data/wildreceipt_bio_smoke \
  --max_samples 5 \
  --debug

python scripts/validate_receipt_v2_labels.py --input processed_data/wildreceipt_bio_smoke/train.jsonl
python scripts/validate_receipt_v2_labels.py --input processed_data/wildreceipt_bio_smoke/validation.jsonl
python scripts/validate_receipt_v2_labels.py --input processed_data/wildreceipt_bio_smoke/test.jsonl
```

## 2. Mixed LayoutLMv3 Fine-Tuning

Start from the stable pre-custom checkpoint, not from an experimental overfit checkpoint.

Recommended starting point:

```bash
python scripts/train_mixed_layoutlmv3_public_user.py \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --wildreceipt_bio_dir processed_data/wildreceipt_bio \
  --user_input_dir /mnt/c/path/to/custom/receipt_labels \
  --model_name_or_path models/layoutlmv3-cord-full/best \
  --local_files_only \
  --output_dir models/layoutlmv3-mixed-public-user \
  --sources cord,wild,user \
  --cord_repeat 1 \
  --wild_repeat 1 \
  --user_repeat 10 \
  --user_validation_count 2 \
  --user_test_count 3 \
  --epochs 100 \
  --batch_size 2 \
  --gradient_accumulation_steps 1 \
  --learning_rate 5e-5 \
  --device auto \
  --fp16 \
  --overwrite_output_dir
```

The trainer writes:

- `models/layoutlmv3-mixed-public-user/best/`
- `models/layoutlmv3-mixed-public-user/last/`
- `models/layoutlmv3-mixed-public-user/training_history.json`
- `models/layoutlmv3-mixed-public-user/training_curve.png`
- `models/layoutlmv3-mixed-public-user/best_metrics.json`
- `models/layoutlmv3-mixed-public-user/split_manifest.json`
- `models/layoutlmv3-mixed-public-user/field_metrics_validation.json`
- `models/layoutlmv3-mixed-public-user/field_metrics_custom_validation.json`
- `models/layoutlmv3-mixed-public-user/field_metrics_custom_test.json` when a custom test split exists
- source-specific metrics in `training_history.json`

## 3. BIO Boundary Repair

Optional repair fixes safe same-line fragments like:

```text
B-ITEM_NAME B-ITEM_NAME B-ITEM_NAME -> B-ITEM_NAME I-ITEM_NAME I-ITEM_NAME
```

It avoids summary/payment heads and unsafe price merges.

Smoke check:

```bash
python scripts/smoke_bio_repair.py
```

## 4. Rebuild Span Rel-G Cache

After a new LayoutLMv3 checkpoint is selected, rebuild the rel-g cache with frozen LayoutLMv3 hidden states.

```bash
python scripts/build_user_span_relg_dataset.py \
  --input_dir /mnt/c/path/to/custom/receipt_labels \
  --layout_checkpoint models/layoutlmv3-mixed-public-user/best \
  --out_dir processed_data/user_span_relg_mixed_public_user \
  --include_context_tokens all \
  --span_pooling first \
  --split_manifest path/to/split_manifest.json \
  --repair_bio_boundaries \
  --device auto \
  --local_files_only \
  --overwrite
```

`--split_manifest` is optional. If omitted, the script creates a deterministic train/validation split.

Supported split manifest shapes:

```json
{"splits": {"train": ["capture_a"], "validation": ["capture_b"]}}
```

```json
{"splits": {"train": [{"id": "capture_a"}], "validation": [{"id": "capture_b"}]}}
```

JSONL entries with `id` and `split` are also supported.

## 5. Rel-G Evaluation

Oracle/cache-based evaluation:

```bash
python scripts/eval_span_relg.py \
  --dataset_dir processed_data/user_span_relg_mixed_public_user \
  --checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --split validation \
  --threshold 0.84 \
  --out_dir outputs/span_relg_eval_custom \
  --device auto \
  --debug
```

Predicted-span custom e2e evaluation:

```bash
python scripts/eval_custom_layoutlmv3_span_relg_e2e.py \
  --input_dir /mnt/c/path/to/custom/receipt_labels \
  --split_manifest models/layoutlmv3-mixed-public-user/split_manifest.json \
  --split test \
  --layoutlm_checkpoint models/layoutlmv3-mixed-public-user/best \
  --relg_checkpoint models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best \
  --span_relg_dataset_dir processed_data/user_span_relg_mixed_public_user \
  --threshold 0.84 \
  --out_dir outputs/custom_predicted_span_relg_e2e \
  --device auto \
  --local_files_only \
  --sweep_thresholds \
  --debug
```

This script uses hand labels only as gold references. Input spans come from LayoutLMv3 predicted BIO labels, then rel-g predicts edges.

## 6. What To Compare

Use these metrics to decide the next work:

- LayoutLMv3 source-specific F1: CORD vs WildReceipt vs user.
- Field metrics for `ITEM_NAME`, `ITEM_PRICE`, `TAX_PRICE`, `TOTAL_PRICE`, `SUBTOTAL_PRICE`.
- Oracle rel-g `menu_price_pair_f1`.
- Predicted-span rel-g `menu_price_pair_f1`.
- `hard_negative_false_positive_count`.
- `total_subtotal_false_positive_count`.
- `dependent_collision_count`.

If oracle rel-g is good but predicted-span e2e is bad, improve LayoutLMv3 span extraction. If both are bad, improve rel-g features/model or relation labels.
