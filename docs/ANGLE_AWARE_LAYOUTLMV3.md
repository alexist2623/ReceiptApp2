# Angle-Aware LayoutLMv3 Pipeline

This repo's default LayoutLMv3 path remains axis-aligned and backward compatible.
The angle-aware path is optional and adds a token-aligned `angle_features` tensor
beside the existing LayoutLMv3 inputs.

## What Changes

Standard LayoutLMv3 inputs:

- `input_ids`
- `attention_mask`
- `bbox`
- `pixel_values`
- `labels`

Angle-aware inputs add:

- `angle_features`: `[batch, max_length, dim]`

The `bbox` tensor still uses axis-aligned 0..1000 boxes. The angle feature is
derived from OCR `cornerPoints`, `quad`, `vertices`, or explicit `angleDeg`.
If no angle information exists, the angle feature row is all zeros.

Supported feature modes:

- `sincos_scalar`: backward-compatible 9-dim feature vector
- `angle_quad`: 18-dim feature vector with angle, relative quad, area, and presence flags
- `relative_quad`, `sincos`, `raw_scalar`, `none`: lightweight alternatives

The recommended new mixed training path uses `angle_quad`.

## Feature Modules

- `ml/angle_geometry.py`
  - parses quadrilaterals and OCR corner points
  - computes word text angle
  - creates configurable word feature vectors
  - rotates images and quadrilateral boxes for WildReceipt augmentation
  - aligns word features to token positions

- `ml/layoutlmv3_angle_inputs.py`
  - wraps Hugging Face processor calls
  - appends token-level `angle_features`
  - preserves existing ignored-label behavior

- `ml/layoutlmv3_angle_model.py`
  - defines `AngleAwareLayoutLMv3ForTokenClassification`
  - loads normal LayoutLMv3 token-classification checkpoints
  - initializes only the small angle projection when the source checkpoint has no angle weights

## OCR JSON

The recommended OCR word schema is:

```json
{
  "text": "americano",
  "box": [120, 340, 310, 385],
  "cornerPoints": [[120, 342], [309, 340], [310, 382], [121, 385]],
  "angleDeg": -0.6
}
```

`box` is still required. `cornerPoints` and `angleDeg` are optional.

## Smoke Test

```bash
python scripts/smoke_angle_aware_layoutlmv3.py \
  --image path/to/receipt.jpg \
  --label_json path/to/receipt_labeled_v2_1.json \
  --label_schema schemas/receipt_labels_v2.json \
  --model_name_or_path models/layoutlmv3-cord-full/best \
  --local_files_only \
  --out_dir outputs/smoke_angle_aware_layoutlmv3 \
  --steps 2 \
  --device cuda \
  --debug
```

If the sample has no 4-point geometry but you want to test the tensor path:

```bash
python scripts/smoke_angle_aware_layoutlmv3.py \
  --image path/to/receipt.jpg \
  --label_json path/to/receipt_labeled_v2_1.json \
  --model_name_or_path models/layoutlmv3-cord-full/best \
  --local_files_only \
  --synthetic_angle_deg 5 \
  --steps 1
```

## Mixed Fine-Tuning

Use the angle-specific script rather than replacing the existing mixed trainer:

```bash
python scripts/train_mixed_layoutlmv3_angle_public_user.py \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --wildreceipt_bio_dir processed_data/wildreceipt_rotated_receipt_v2_bio \
  --user_input_dir path/to/user_labels \
  --model_name_or_path models/layoutlmv3-cord-full/best \
  --local_files_only \
  --output_dir models/layoutlmv3-angle-mixed-public-user \
  --sources cord,wild,user \
  --angle_encoding_mode angle_quad \
  --angle_feature_dim 18 \
  --epochs 100 \
  --batch_size 2 \
  --device cuda \
  --fp16 \
  --overwrite_output_dir
```

CORD records generally do not carry 4-point word geometry in the intermediate
BIO JSONL, so those samples use zero angle features unless the dataset builder
is extended to preserve raw CORD quads.

## Span Rel-G Feature Cache

`scripts/build_span_relg_dataset.py` and
`scripts/build_user_span_relg_dataset.py` support:

```bash
--use_angle_features auto|true|false
--disable_angle_features
```

`auto` uses angle features only when the LayoutLMv3 checkpoint is angle-aware.
Standard checkpoints continue to work without any angle tensor.

## Predicted-Span E2E Evaluation

```bash
python scripts/eval_predicted_span_relg_e2e.py \
  --raw_data_dir ../receipt_training_data2 \
  --layoutlm_checkpoint models/layoutlmv3-angle-mixed-cord-user-non-temp/best \
  --relg_checkpoint models/span-relg-context/best \
  --span_relg_dataset_dir processed_data/span_relg \
  --split test \
  --threshold 0.5 \
  --out_dir outputs/predicted_span_relg_e2e_angle \
  --device auto \
  --local_files_only \
  --use_angle_features auto
```

## Compatibility Notes

- Existing non-angle scripts remain valid.
- Existing checkpoints can initialize the angle-aware model.
- Missing angle data never fails the pipeline.
- This change does not claim an accuracy improvement by itself; it only makes
  tilted OCR geometry available to the model.
