# CORD-only LayoutLMv3 Quantization Final Report

This report summarizes the CORD-only quantization pipeline implemented in stages
1-10. No user labeled receipts, mixed checkpoints, fine-tuning, rel-g training,
or Android app integration were used.

## Environment

- WSL conda env: `receipt-ml`
- Python: `/home/alexist/miniconda3/envs/receipt-ml/bin/python`
- Torch: `2.11.0+cu128`
- CUDA: available
- GPU: NVIDIA GeForce RTX 5070

## Baseline And Artifacts

- PyTorch checkpoint: `models/layoutlmv3-cord-full/best`
- ONNX FP32: `models/layoutlmv3-cord-onnx/fp32/model.onnx`
- ONNX dynamic INT8: `models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx`
- Optional static PTQ script: `scripts/quantization/quantize_layoutlmv3_onnx_static.py`
- Optional selective quantization script: `scripts/quantization/run_selective_quantization_experiments.py`

Generated models and metrics live under `models/`, `outputs/`, and
`processed_data/`. They are intentionally not committed.

## Validation Metrics

| Variant | Samples | Token F1 | ITEM_NAME F1 | ITEM_PRICE F1 | TOTAL_PRICE F1 | Size MB | Avg latency ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| PyTorch FP32 | 100 | 0.960912 | 0.981132 | 0.993763 | 0.992941 | n/a | n/a |
| ONNX FP32 | 100 | 0.960912 | 0.981132 | 0.993763 | 0.992941 | 480.75 | 331.42 |
| ONNX dynamic INT8 | 100 | 0.959317 | 0.978339 | 0.993763 | 1.000000 | 121.50 | 203.15 |

Dynamic INT8 token F1 drop vs PyTorch FP32 validation:

```text
0.960912 - 0.959317 = 0.001596
```

This is well below the 2% token F1 drop budget.

## ONNX FP32 Parity

Parity was checked on 50 validation samples:

- PyTorch F1: 0.971382
- ONNX F1: 0.971382
- F1 drop: 0.000000
- token argmax agreement: 1.000000
- word argmax agreement: 1.000000
- hidden mean absolute difference: 0.000000624
- hidden max absolute difference: 0.000331

ONNX FP32 parity passed.

## Dynamic INT8 Size And Latency

- FP32 ONNX size: 480.75 MB
- Dynamic INT8 ONNX size: 121.50 MB
- Size reduction: 74.73%
- ONNX FP32 avg latency: 331.42 ms
- Dynamic INT8 avg latency: 203.15 ms

Dynamic INT8 improved both size and CPU latency while keeping token quality
within budget.

## Rel-g Hidden Impact

The INT8 ONNX `last_hidden_state` was used to rebuild a CORD-only span rel-g
validation cache and evaluate the existing rel-g checkpoint.

| Metric | Value |
| --- | ---: |
| Samples | 100 |
| Candidate pairs | 2685 |
| Positive pairs | 580 |
| Negative pairs | 2105 |
| Edge precision | 0.848160 |
| Edge recall | 0.953448 |
| Edge F1 | 0.897727 |
| MENU_NM -> MENU_PRICE precision | 0.884615 |
| MENU_NM -> MENU_PRICE recall | 0.978723 |
| MENU_NM -> MENU_PRICE F1 | 0.929293 |
| Hard negative false positives | 2 |
| Total/subtotal false positives | 2 |
| Dependent collisions | 94 |

The rel-g impact is acceptable for a CORD-only quantization smoke path. Final
mobile selection should still be based on end-to-end item-price F1, not model
size alone.

## Selective Quantization Variants

All selective dynamic variants produced ONNX models and passed smoke tests.

| Variant | Size MB | Reduction | Smoke |
| --- | ---: | ---: | --- |
| `matmul_gemm_only` | 236.26 | 50.86% | pass |
| `matmul_only` | 236.26 | 50.86% | pass |
| `gemm_only` | 480.63 | 0.03% | pass |
| `exclude_layernorm_embedding` | 377.67 | 21.44% | pass |
| `exclude_classifier` | 236.26 | 50.86% | pass |
| `safest_dynamic` | 377.67 | 21.44% | pass |

Recommended evaluation order if dynamic INT8 ever regresses:

1. `safest_dynamic`
2. `matmul_gemm_only`
3. `exclude_classifier`
4. `exclude_layernorm_embedding`
5. `matmul_only`
6. `gemm_only`

## Decision

Use `models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx` as the current
CORD-only quantized candidate.

Reasons:

- ONNX FP32 parity passed.
- Dynamic INT8 token F1 drop is about 0.16 percentage points.
- Size reduction is about 74.73%.
- Average ONNX CPU latency improved from about 331 ms to about 203 ms.
- Rel-g hidden impact remains usable for validation smoke testing.

## Android Readiness

Android integration is not done yet. Before adding ONNX Runtime to the app,
complete the preprocessing parity checklist in:

```text
docs/ANDROID_ONNX_LAYOUTLMV3_DEPLOYMENT_PLAN.md
```

The most important gate is exact or near-exact parity for:

- tokenizer output,
- word-to-token alignment,
- bbox normalization,
- LayoutLMv3 image preprocessing,
- word-level label recovery.

## Future User Data

Future user/mixed checkpoint integration is documented in:

```text
docs/FUTURE_USER_DATA_QUANTIZATION_INTEGRATION.md
```

Do not swap in user data until there is a stable user validation/test set and
coordinate/label/relation validation passes.

## Reproduction Commands

Run in WSL:

```bash
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate receipt-ml
which python
```

Core checks:

```bash
python scripts/quantization/eval_layoutlmv3_cord_pytorch.py --checkpoint models/layoutlmv3-cord-full/best --cord_bio_dir processed_data/cord_bio --cord_raw_data_dir ../receipt_training_data2 --split validation --max_samples 100 --device cuda --local_files_only --out_dir outputs/quantization/cord_baseline_pytorch_fp32
python scripts/quantization/export_layoutlmv3_to_onnx.py --checkpoint models/layoutlmv3-cord-full/best --cord_bio_dir processed_data/cord_bio --cord_raw_data_dir ../receipt_training_data2 --split validation --sample_index 0 --out_dir models/layoutlmv3-cord-onnx/fp32 --opset 17 --device cpu --local_files_only --overwrite
python scripts/quantization/compare_pytorch_onnx_layoutlmv3.py --pytorch_checkpoint models/layoutlmv3-cord-full/best --onnx_model models/layoutlmv3-cord-onnx/fp32/model.onnx --cord_bio_dir processed_data/cord_bio --cord_raw_data_dir ../receipt_training_data2 --split validation --max_samples 50 --device cuda --onnx_provider cpu --local_files_only --out_dir outputs/quantization/cord_onnx_fp32
python scripts/quantization/quantize_layoutlmv3_onnx_dynamic.py --input_onnx models/layoutlmv3-cord-onnx/fp32/model.onnx --checkpoint_for_processor models/layoutlmv3-cord-full/best --cord_bio_dir processed_data/cord_bio --cord_raw_data_dir ../receipt_training_data2 --out_dir models/layoutlmv3-cord-onnx/int8_dynamic --weight_type qint8 --per_channel --local_files_only --overwrite
python scripts/quantization/eval_layoutlmv3_cord_onnx.py --onnx_model models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx --checkpoint_for_processor models/layoutlmv3-cord-full/best --cord_bio_dir processed_data/cord_bio --cord_raw_data_dir ../receipt_training_data2 --split validation --max_samples 100 --out_dir outputs/quantization/cord_onnx_int8_dynamic --local_files_only
python scripts/quantization/summarize_quantization_results.py --out_dir outputs/quantization/reports --split validation
```

## Git Policy

Commit source code and documentation only. Do not commit generated models,
metrics, caches, or output overlays.
