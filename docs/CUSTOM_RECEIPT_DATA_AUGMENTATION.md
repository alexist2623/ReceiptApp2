# Custom Receipt Data Augmentation

This pipeline creates augmented copies of hand-labeled receipt image/OCR folders
without modifying the original data. It is intended for custom receipt
fine-tuning data that already contains word-level BIO labels and relation
annotations.

## Input Layout

The augmenter scans an input directory for folders named `*_receipt_ocr` and
expects this structure:

```text
<input_dir>/
  <capture_id>_receipt_ocr/
    <capture_id>.jpg
    <capture_id>_ocr.json
    <capture_id>_labeled_v2_1.json
```

Any path containing the configured `--exclude_dir_name` is skipped. The default
is `Temp`.

## What Is Preserved

- OCR token count and token order.
- BIO labels.
- Word boxes and corner points stay aligned to the augmented image.
- `spans` are rebuilt from the augmented words and unchanged labels.
- Relation arrays are preserved and refreshed:
  - `relations`
  - `item_relations`
  - `summary_relations`
  - `payment_relations`
  - `rel_g_edges`
  - `relation_ids_as_head`
  - `relation_ids_as_tail`
- `rel_g_edges` remain item-level positive edges only. Summary/payment
  relations are not added to `rel_g_edges`.

## What Is Augmented

Geometry:

- Small rotation.
- Small translation.
- Small scale.
- Output image size stays identical to the source image.
- If any word box becomes invalid after geometry, the whole variant is skipped.

Image degradation:

- Mild blur.
- Mild brightness/contrast changes.
- Mild Gaussian noise.
- JPEG recompression.

Text noise:

- OCR-like substitutions such as `O`/`0`, `S`/`5`, `B`/`8`, `m`/`rn`.
- Very conservative deletion/swap/casing perturbations.
- Price-like fields keep deletion and swap conservative.

Numeric noise:

- Numeric fields can be perturbed while preserving token count and boxes.
- Arithmetic consistency across receipt totals is not guaranteed.
- Each augmented JSON records `"arithmetic_consistency": "not_guaranteed"`.

Masking:

- Low-probability token masking with `****` by default.

## Generate Augmented Data

```bash
python scripts/augment_labeled_receipt_dataset.py \
  --input_dir /mnt/c/Users/<user>/OneDrive/APK_Receipt2 \
  --output_dir /mnt/c/Users/<user>/OneDrive/APK_Receipt2_Augmented \
  --exclude_dir_name Temp \
  --variants_per_sample 10 \
  --seed 42 \
  --geometry_prob 0.8 \
  --text_noise_prob 0.4 \
  --numeric_noise_prob 0.3 \
  --mask_prob 0.2 \
  --image_noise_prob 0.5 \
  --make_overlays \
  --overwrite
```

By default, augmentation is treated as train-only. If a split manifest is
provided, only parent samples assigned to `train` are augmented unless
`--apply_to_splits all` is used.

```bash
python scripts/augment_labeled_receipt_dataset.py \
  --input_dir /mnt/c/Users/<user>/OneDrive/APK_Receipt2 \
  --output_dir /mnt/c/Users/<user>/OneDrive/APK_Receipt2_Augmented \
  --split_manifest path/to/split_manifest.json \
  --apply_to_splits train \
  --parent_split_policy no_leakage \
  --variants_per_sample 5 \
  --overwrite
```

`numeric_mode=receipt_consistent` is intentionally unsupported in this tool.
Use `price_like`, `independent`, or `none`.

## Outputs

The output directory contains:

```text
<output_dir>/
  augmentation_manifest.json
  augmentation_summary.json
  all_labeled_v2_1_augmented.jsonl
  <aug_capture_id>_receipt_ocr/
    <aug_capture_id>.jpg
    <aug_capture_id>_ocr.json
    <aug_capture_id>_labeled_v2_1.json
    <aug_capture_id>_labeled_v2_1_overlay.png      # with --make_overlays
    <aug_capture_id>_relations_overlay.png         # with --make_overlays
```

Generated data should not be committed to git.

## Validate Augmented Data

```bash
python scripts/validate_augmented_receipt_dataset.py \
  --input_dir /mnt/c/Users/<user>/OneDrive/APK_Receipt2_Augmented \
  --label_schema schemas/receipt_labels_v2.json \
  --strict \
  --out outputs/augmentation_validation_summary.json
```

Validation checks:

- Image size matches labeled JSON and OCR JSON size.
- `words`, `labels`, and boxes are valid.
- BIO sequences are valid.
- Relation word indices are in range.
- Relation text matches the current word text.
- `rel_g_edges` use `ITEM_NAME` as head and item-level dependent fields only.
- `all_labeled_v2_1_augmented.jsonl` is parseable and has the expected count.

## Smoke Test

Use a small variant count first:

```bash
python scripts/augment_labeled_receipt_dataset.py \
  --input_dir /mnt/c/Users/<user>/OneDrive/APK_Receipt2 \
  --output_dir /mnt/c/Users/<user>/OneDrive/APK_Receipt2_Augmented_Smoke \
  --exclude_dir_name Temp \
  --variants_per_sample 2 \
  --max_samples 1 \
  --seed 42 \
  --make_overlays \
  --overwrite \
  --debug

python scripts/validate_augmented_receipt_dataset.py \
  --input_dir /mnt/c/Users/<user>/OneDrive/APK_Receipt2_Augmented_Smoke \
  --label_schema schemas/receipt_labels_v2.json \
  --strict \
  --out outputs/augmentation_validation_summary.json
```

If the validator reports relation text mismatches, inspect whether the source
labels contain relation text that was manually edited away from the indexed
words. The augmenter refreshes relation text from word indices by design.
