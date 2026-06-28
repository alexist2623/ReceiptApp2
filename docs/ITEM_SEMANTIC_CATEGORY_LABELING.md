# Item Semantic Category Labeling

This repo keeps semantic item categories separate from receipt BIO fields.

BIO labels such as `B-ITEM_NAME`, `B-ITEM_PRICE`, `B-TAX_PRICE`, and
`B-TOTAL_PRICE` must not be replaced by product categories. Semantic product
categories are attached as metadata for a later item-category classifier.

## Category Set

- `FOOD`
- `DRINK`
- `GROCERY`
- `SNACK`
- `ALCOHOL`
- `HOUSEHOLD`
- `PERSONAL_CARE`
- `HEALTH`
- `CLOTHING`
- `ELECTRONICS`
- `TRANSPORT`
- `SERVICE`
- `ENTERTAINMENT`
- `TAX_FEE`
- `OTHER`

The canonical taxonomy is stored in:

```bash
schemas/item_semantic_categories.json
```

Manual corrections reviewed from image contact sheets are stored in:

```bash
schemas/item_semantic_category_manual_overrides.json
```

## Labeling Command

```bash
python scripts/label_item_semantic_categories.py
```

The script loads `schemas/item_semantic_category_manual_overrides.json` by
default and applies those reviewed labels before keyword rules.

Default inputs:

```text
C:\Users\박정현\OneDrive\APK_Receipt2
..\receipt_training_data3\wildreceipt\wildreceipt_custom_structure
processed_data/custom_rotated_receipt_v2_bio
processed_data/wildreceipt_custom_structure_rotated_receipt_v2_bio
processed_data/cord_bio
```

`Temp` folders are skipped by default.

## Output Fields

For labeled receipt JSON files, the script adds:

```json
{
  "item_semantic_category_taxonomy": ["FOOD", "DRINK", "..."],
  "item_semantic_category_labeling": {
    "schema_version": "item_semantic_categories_v1",
    "method": "keyword_rules_v1"
  },
  "item_category_annotations": [
    {
      "item_name_text": "Latte",
      "item_name_word_indices": [10],
      "category": "DRINK",
      "confidence": 0.88,
      "rule": "LATTE"
    }
  ]
}
```

For BIO JSONL records, the script adds:

```json
{
  "semantic_item_categories": [null, "DRINK", "DRINK", null],
  "item_category_annotations": [...]
}
```

The script also annotates `ITEM_NAME` words with `semantic_item_category`.

## Validation

Run:

```bash
python -m py_compile scripts/label_item_semantic_categories.py
python scripts/label_item_semantic_categories.py --dry_run
```

The summary is written to:

```text
outputs/item_category_labeling/item_category_labeling_summary.json
```

Backups are written under:

```text
outputs/item_category_labeling_backups/
```

## Manual Review Sheets

Generate image crops for direct category review:

```bash
python scripts/build_item_category_review_sheet.py \
  --max_custom 200 \
  --max_wild 120 \
  --wild_mode frequent
```

Outputs:

```text
outputs/item_category_manual_review/custom_review_sheet.png
outputs/item_category_manual_review/wild_frequent_review_sheet.png
outputs/item_category_manual_review/custom_review_manifest.json
outputs/item_category_manual_review/wild_frequent_review_manifest.json
```

After adding corrections to
`schemas/item_semantic_category_manual_overrides.json`, rerun:

```bash
python scripts/label_item_semantic_categories.py
```

## LayoutLMv3 Category Head

Train a category classifier head on frozen LayoutLMv3 hidden states:

```bash
python scripts/train_layoutlmv3_item_category_head.py \
  --local_files_only \
  --device auto \
  --rebuild_cache
```

The script defaults are set to the best verified category-head run:

- `layoutlm_checkpoint`: `models/layoutlmv3-base`
- hidden layers: `0,4,8,12`
- pooling: `attention_stats`
- `word_token_mode`: `all`
- `min_validation_per_label`: `25`
- `seed`: `13`
- `learning_rate`: `5e-4`
- `weight_decay`: `5e-2`
- `dropout`: `0.4`
- `class_weight_power`: `0.9`
- `label_smoothing`: `0.02`
- `exclude_other`: enabled by default

This freezes LayoutLMv3 and trains only the item-category head over `ITEM_NAME`
span hidden states. The best verified run reached validation macro F1
`0.9180981060488538` at epoch `39`:

```text
outputs/item_category_layoutlmv3_base_head_valcov25_layers04812_attnstats_reg_ls_seed13_180ep/metrics_summary.json
```

## Caveat

`keyword_rules_v1` is an initial deterministic label pass. `OTHER` and
low-confidence categories should be manually reviewed before treating them as
high-quality ground truth.
