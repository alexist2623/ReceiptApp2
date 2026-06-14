# CORD-only LayoutLMv3 Quantization Pipeline

This pipeline keeps the baseline fixed to the CORD-only checkpoint:

```text
models/layoutlmv3-cord-full/best
```

It intentionally does not use user labeled data, mixed checkpoints, Android integration, or any fine-tuning.

## Stages

1. PyTorch FP32 CORD validation/test baseline
2. ONNX FP32 export with `logits` and `last_hidden_state`
3. PyTorch FP32 vs ONNX FP32 parity check
4. ONNX dynamic INT8 quantization
5. INT8 token/latency evaluation and rel-g hidden-state impact
6. Optional static INT8 PTQ script

Generated artifacts are written under `models/layoutlmv3-cord-onnx/`, `outputs/quantization/`, and `processed_data/span_relg_cord_onnx_*`; these are not intended for git commits.

## Stage 6 Static PTQ

Static INT8 PTQ is implemented as an optional fallback:

```bash
python scripts/quantization/quantize_layoutlmv3_onnx_static.py \
  --input_onnx models/layoutlmv3-cord-onnx/fp32/model.onnx \
  --checkpoint_for_processor models/layoutlmv3-cord-full/best \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --calibration_split train \
  --calibration_samples 100 \
  --out_dir models/layoutlmv3-cord-onnx/int8_static \
  --calibration_method minmax \
  --quant_format qdq \
  --activation_type qint8 \
  --weight_type qint8 \
  --per_channel \
  --local_files_only \
  --overwrite \
  --debug
```

Use only CORD calibration samples. Do not mix user labeled receipts into this calibration run.

The dynamic INT8 run is preferred when it stays within the quality budget:

- token F1 drop <= 2%
- item price F1 drop <= 3%
- rel-g item-price pair F1 drop <= 3%

For the current CORD-only validation run, dynamic INT8 stayed well within that budget, so static PTQ is available but not required.
