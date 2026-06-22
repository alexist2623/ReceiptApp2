# WildReceipt Rotated Angle-Aware Pipeline

This pipeline adds synthetic rotation geometry to WildReceipt, then trains an
angle-aware LayoutLMv3 token classifier and rebuilds span-level rel-g features
from that checkpoint. Generated datasets, checkpoints, and outputs must stay out
of git.

## 1. Convert WildReceipt With Rotation

WildReceipt does not provide per-token angle or quad labels. The converter
rotates the full receipt image and applies the same affine transform to every
OCR word box. Each output record keeps:

- `boxes`: rotated axis-aligned pixel bbox
- `normalized_boxes`: 0..1000 LayoutLMv3 bbox
- `word_payloads[].quad`: rotated 4-point quadrilateral
- `word_payloads[].angle_deg`: angle estimated from the rotated quad
- `rotation_deg`: whole-page synthetic rotation
- `source_image_id`: original WildReceipt receipt id

```bash
python scripts/convert_wildreceipt_rotated_to_receipt_v2_bio.py \
  --wildreceipt_root /path/to/wildreceipt \
  --out_dir processed_data/wildreceipt_rotated_receipt_v2_bio \
  --rotation_degrees -10,-5,0,5,10
```

Validate before training:

```bash
python scripts/validate_receipt_v2_bio_jsonl.py \
  --bio_dir processed_data/wildreceipt_rotated_receipt_v2_bio \
  --require_angle
```

## 2. Angle-Aware LayoutLMv3 Fine-Tuning

CORD records remain usable and receive zero angle features. Rotated WildReceipt
and custom labeled JSON can provide quad/angle features through `word_payloads`.

```bash
python scripts/train_mixed_layoutlmv3_angle_public_user.py \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --wildreceipt_bio_dir processed_data/wildreceipt_rotated_receipt_v2_bio \
  --user_input_dir /path/to/custom_receipts \
  --model_name_or_path models/layoutlmv3-cord-full/best \
  --output_dir models/layoutlmv3-angle-mixed-public-user \
  --sources cord,wild,user \
  --angle_encoding_mode angle_quad \
  --angle_feature_dim 18 \
  --angle_hidden_size 128 \
  --angle_fusion add \
  --epochs 100 \
  --batch_size 2 \
  --device auto \
  --fp16 \
  --overwrite_output_dir
```

The best checkpoint stores `angle_model_config.json` so downstream feature
cache scripts can automatically align the correct angle feature dimension.

## 3. Rebuild Span Rel-G Cache

Use the newly trained angle-aware LayoutLMv3 checkpoint as the frozen feature
extractor. Existing no-angle checkpoints still work.

```bash
python scripts/build_user_span_relg_dataset.py \
  --input_dir /path/to/custom_receipts \
  --layout_checkpoint models/layoutlmv3-angle-mixed-public-user/best \
  --out_dir processed_data/user_span_relg_angle_mixed \
  --use_angle_features auto \
  --include_context_tokens all \
  --overwrite
```

For CORD oracle-span cache:

```bash
python scripts/build_span_relg_dataset.py \
  --raw_data_dir ../receipt_training_data2 \
  --checkpoint models/layoutlmv3-angle-mixed-public-user/best \
  --out_dir processed_data/span_relg_angle_mixed \
  --use_angle_features auto \
  --overwrite
```

## 4. Train Span Rel-G

Rel-g still trains on cached hidden features and binary relation labels. It does
not predict `group_id` as a class.

```bash
python scripts/train_span_relg.py \
  --dataset_dir processed_data/user_span_relg_angle_mixed \
  --output_dir models/span-relg-angle-mixed \
  --epochs 300 \
  --device auto \
  --overwrite_output_dir
```

## 5. Custom End-to-End Evaluation

This evaluates the real pipeline: LayoutLMv3 predicted BIO spans, then span
rel-g grouping. It does not use GT spans for rel-g inference.

```bash
python scripts/eval_custom_angle_pipeline_e2e.py \
  --input_dir /path/to/custom_receipts \
  --layoutlm_checkpoint models/layoutlmv3-angle-mixed-public-user/best \
  --relg_checkpoint models/span-relg-angle-mixed/best \
  --span_relg_dataset_dir processed_data/user_span_relg_angle_mixed \
  --out_dir outputs/custom_angle_pipeline_e2e \
  --device auto \
  --local_files_only \
  --sweep_thresholds \
  --debug
```

## 6. Smoke Checks

```bash
python scripts/smoke_wildreceipt_rotation_augmentation.py
python scripts/smoke_angle_layoutlmv3_train.py --angle_encoding_mode angle_quad
python -m py_compile scripts/train_mixed_layoutlmv3_angle_public_user.py
python -m py_compile scripts/eval_custom_angle_pipeline_e2e.py
```

## Notes

- LayoutLMv3 still receives axis-aligned `bbox`; angle/quad enters only through
  the optional `angle_features` tensor in angle-aware checkpoints.
- Missing angle data is valid. The pipeline emits zeros for no-angle records.
- `angle_quad` currently uses 18 dimensions: token angle, relative angle,
  relative quad offsets, quad area, and presence flags.
- `bucket_embedding` and `spade_like` are reserved modes and intentionally raise
  `NotImplementedError`.
