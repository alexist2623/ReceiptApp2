# Overlay Labeled Relations

## Purpose

`scripts/overlay_labeled_relations.py` draws hand-labeled relation edges on top
of a receipt image. It is separate from
`scripts/overlay_labeled_receipt_json.py`, which only draws BIO word labels.

The script reads relation fields such as:

- `relations`
- `item_relations`
- `summary_relations`
- `payment_relations`
- `rel_g_edges`

## Single Receipt

```bash
python scripts/overlay_labeled_relations.py \
  --image data/receipts/20260613T173633Z_3718dbbb_receipt_ocr/20260613T173633Z_3718dbbb.jpg \
  --label_json data/receipts/20260613T173633Z_3718dbbb_receipt_ocr/20260613T173633Z_3718dbbb_labeled_v2_1.json \
  --relation_source all \
  --coordinate_mode strict \
  --out outputs/relation_overlay/20260613T173633Z_3718dbbb_relations_overlay.png \
  --summary_out outputs/relation_overlay/20260613T173633Z_3718dbbb_relations_summary.json
```

`--relation_source all` uses `relations` when present. If `relations` is absent,
it falls back to `item_relations + summary_relations + payment_relations`; if
those are absent too, it falls back to `rel_g_edges`.

Other source options:

```bash
--relation_source relations
--relation_source item_relations
--relation_source summary_relations
--relation_source payment_relations
--relation_source rel_g_edges
```

## Coordinate Modes

Strict mode is the default:

```bash
--coordinate_mode strict
```

It fails if the actual image size differs from the labeled JSON
`image_width/image_height`.

For visual diagnosis only:

```bash
--coordinate_mode auto-scale
```

Auto-scale draws boxes after scaling from the JSON coordinate space to the
actual image size. Do not treat auto-scale as training validation.

## Relation Text

Relation text such as `ITEM_NAME -> ITEM_PRICE` is hidden by default so dense
rel-g overlays stay readable. Colors distinguish relation and field types. Add
`--show_relation_labels` only when you need the text labels for debugging.

## Batch

```bash
python scripts/batch_overlay_labeled_relations.py \
  --input_dir path/to/receipt_label_v2_1_corrected_1536x2048_with_relations \
  --out_dir outputs/relation_overlay \
  --relation_source all \
  --coordinate_mode strict
```

The batch script finds `*_receipt_ocr` folders and expects:

```text
<capture_id>_receipt_ocr/
  <capture_id>.jpg
  <capture_id>_labeled_v2_1.json
```

It writes per-sample overlays, per-sample summaries, and
`batch_relations_summary.json`.

## What To Check

1. `ITEM_NAME -> ITEM_PRICE` arrows connect the correct product and price.
2. `SUBTOTAL_NAME -> SUBTOTAL_PRICE`, `TAX_NAME -> TAX_PRICE`, and
   `TOTAL_NAME -> TOTAL_PRICE` are correct.
3. Payment fields such as `PAYMENT_METHOD -> PAYMENT_CARD` or
   `PAYMENT_METHOD -> CARD_PRICE` are plausible.
4. `skipped_relation_count` is zero or the skipped reasons are expected.
