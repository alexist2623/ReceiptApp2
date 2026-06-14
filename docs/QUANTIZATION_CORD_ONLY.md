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
