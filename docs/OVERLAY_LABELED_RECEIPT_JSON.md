# Overlay Labeled Receipt JSON

## Purpose

Use this tool to visually inspect manual receipt labels before fine-tuning. It
draws BIO labels from a labeled receipt JSON on top of the original receipt
image and writes a label summary JSON.

This step does not run fine-tuning, model inference, OCR, or Tesseract.

## Supported Label JSON Formats

Format A:

```json
{
  "image_width": 3000,
  "image_height": 4000,
  "words": [
    {
      "word_idx": 0,
      "text": "WINNERS",
      "box": [871, 155, 2138, 356],
      "label": "B-STORE_NAME"
    }
  ]
}
```

Format B:

```json
{
  "image_width": 3000,
  "image_height": 4000,
  "words": [
    {"word_idx": 0, "text": "WINNERS", "box": [871, 155, 2138, 356]}
  ],
  "labels": ["B-STORE_NAME"]
}
```

If `words[].label` exists, it is preferred. Otherwise the top-level `labels`
array is used.

## Run

Default mode is strict coordinate validation. The actual image size must match
`image_width` / `image_height` in the labeled JSON.

```bash
python scripts/overlay_labeled_receipt_json.py \
  --image path/to/20260613T072738Z_37a06228.jpg \
  --label_json path/to/20260613T072738Z_37a06228_labeled_v2.json \
  --out outputs/labeled_overlay/20260613T072738Z_37a06228_labeled_overlay.png \
  --summary_out outputs/labeled_overlay/20260613T072738Z_37a06228_label_summary.json \
  --show_text \
  --draw_legend \
  --debug
```

If the label JSON was created in a different coordinate space, strict mode
fails with an error. For visual diagnosis only, use auto-scale:

```bash
python scripts/overlay_labeled_receipt_json.py \
  --image path/to/capture.jpg \
  --label_json path/to/capture_labeled_v2_1.json \
  --coordinate_mode auto-scale \
  --out outputs/labeled_overlay/capture_scaled_overlay.png \
  --summary_out outputs/labeled_overlay/capture_scaled_summary.json
```

Auto-scale changes only the overlay drawing coordinates. It is not proof that
the training pair is coordinate-consistent. To create a separate corrected copy
for later manual review:

```bash
python scripts/overlay_labeled_receipt_json.py \
  --image path/to/capture.jpg \
  --label_json path/to/capture_labeled_v2_1.json \
  --coordinate_mode auto-scale \
  --write_scaled_label_json path/to/capture_labeled_v2_1_scaled.json
```

If a legacy label JSON has no `image_width` / `image_height`, use
`--coordinate_mode assume-image` only when you know boxes are already in actual
image pixels.

## Outputs

- `outputs/labeled_overlay/<capture_id>_labeled_overlay.png`
- `outputs/labeled_overlay/<capture_id>_label_summary.json`

## Check

1. Box positions match the words.
2. `WINNERS` is labeled `STORE_NAME`.
3. `SKINCARE & MAK` is labeled `ITEM_NAME`.
4. `$16.99`, `$9.99`, `$12.99` are labeled `ITEM_PRICE`.
5. `$39.97` is labeled `SUBTOTAL_PRICE`.
6. `$44.77` is separated appropriately as `TOTAL_PRICE` or `CARD_PRICE`.
7. Tax amount is labeled `TAX_PRICE`.

Raw OCR tokens should remain unmerged. Split prices are validated through BIO
spans, not by merging OCR words.

Fine-tuning inputs must pass strict coordinate validation. Do not train on an
auto-scaled overlay unless you intentionally wrote and reviewed a scaled label
JSON copy.
