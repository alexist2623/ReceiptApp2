# 05. INT8 평가 및 rel-g hidden impact 측정

Codex에게 아래 지시를 그대로 붙여넣어라.

```text
ONNX FP32와 ONNX INT8 dynamic 모델을 CORD-only 기준으로 평가하고, last_hidden_state 변화가 rel-g 성능에 미치는 영향을 측정하는 script들을 작성해라.

목표:
- PyTorch FP32 baseline, ONNX FP32, ONNX INT8 dynamic을 같은 CORD validation/test subset에서 비교한다.
- token classification 성능뿐 아니라 rel-g가 사용할 hidden state 영향도 측정한다.
- 사용자 데이터는 사용하지 않는다.

새 파일:
1. scripts/quantization/eval_layoutlmv3_cord_onnx.py
2. scripts/quantization/build_span_relg_dataset_from_onnx.py
3. scripts/quantization/evaluate_relg_with_quantized_layout_hidden.py
4. scripts/quantization/summarize_quantization_results.py

================================================================================
A. eval_layoutlmv3_cord_onnx.py
================================================================================

입력:
--onnx_model models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx
--checkpoint_for_processor models/layoutlmv3-cord-full/best
--cord_bio_dir processed_data/cord_bio
--cord_raw_data_dir ../receipt_training_data2
--split validation
--max_samples 100

기능:
- ONNX Runtime으로 logits/last_hidden_state를 얻는다.
- logits로 token prediction 평가.
- PyTorch baseline evaluator와 같은 metrics를 출력.
- latency 측정:
  - warmup 5
  - repeat per sample 1 또는 3
  - avg, p50, p95
- model size 저장.

출력:
outputs/quantization/cord_onnx_int8_dynamic/
├─ metrics_validation.json
├─ predictions_validation.jsonl
├─ latency_validation.json
└─ field_metrics_validation.json

================================================================================
B. build_span_relg_dataset_from_onnx.py
================================================================================

목표:
- 기존 `scripts/build_span_relg_dataset.py`는 PyTorch LayoutLMv3 checkpoint로 word hidden을 계산한다.
- 이번 script는 ONNX model의 `last_hidden_state`를 사용해서 동일한 span rel-g cache를 만든다.
- CORD-only raw data와 CORD gold spans를 사용한다.
- 사용자 데이터는 사용하지 않는다.

입력:
--raw_data_dir ../receipt_training_data2
--cord_bio_dir processed_data/cord_bio
--onnx_model models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx
--checkpoint_for_processor models/layoutlmv3-cord-full/best
--out_dir processed_data/span_relg_cord_onnx_int8_dynamic
--split validation
--max_samples 100

구현:
- 기존 `scripts/build_span_relg_dataset.py`의 make_gold_spans_from_cord / build_cache_sample 흐름 재사용.
- `compute_word_hidden_onnx()`를 새로 구현.
- ONNX output `last_hidden_state`에서 first subword token hidden을 word_hidden으로 모은다.
- `encoding.word_ids(batch_index=0)`는 processor output에서 얻는다.
- sample cache format은 기존 PyTorch cache와 동일해야 한다:
  - node_hidden
  - node_field_ids
  - node_kind_ids
  - node_boxes
  - candidate_pairs
  - pair_labels
  - pair_meta
  - pair_fields

출력:
processed_data/span_relg_cord_onnx_int8_dynamic/
├─ schema.json
├─ manifest.json
├─ summary.json
└─ validation/*.pt

================================================================================
C. evaluate_relg_with_quantized_layout_hidden.py
================================================================================

목표:
- quantized LayoutLMv3 hidden으로 만든 cache를 기존 CORD rel-g checkpoint에 넣어 성능을 본다.
- rel-g checkpoint는 기본 `models/span-relg-context/best`.
- rel-g 모델 자체는 양자화하지 않는다.
- 이 평가는 LayoutLMv3 hidden quantization이 rel-g 성능에 주는 영향만 보는 것이다.

입력:
--dataset_dir processed_data/span_relg_cord_onnx_int8_dynamic
--relg_checkpoint models/span-relg-context/best
--split validation
--threshold 0.5

기능:
- existing `scripts/train_span_relg.py`의 evaluate 로직 또는 `scripts/visualize_span_relg.py`의 model load/run_one 로직을 재사용.
- metrics:
  - edge precision/recall/F1
  - item_price_pair precision/recall/F1
  - menu_price_pair alias
  - hard_negative_false_positive_count
  - total_subtotal_false_positive_count
  - dependent_collision_count
- threshold sweep:
  - 0.1~0.9 step 0.05
  - best threshold by item_price_pair_f1 저장

출력:
outputs/quantization/cord_relg_hidden_impact/
├─ relg_metrics_onnx_int8_validation.json
├─ threshold_sweep_onnx_int8_validation.json
└─ decoded_preview_onnx_int8_validation.jsonl

================================================================================
D. summarize_quantization_results.py
================================================================================

목표:
여러 결과를 표 하나로 합친다.

입력:
- PyTorch FP32 baseline metrics
- ONNX FP32 metrics
- ONNX INT8 dynamic metrics
- rel-g metrics

출력:
outputs/quantization/reports/
├─ quantization_summary.json
├─ quantization_summary.md
└─ quantization_summary.csv

표 컬럼:
- model_variant
- model_path
- model_size_mb
- split
- num_samples
- token_seqeval_f1
- ITEM_NAME_f1
- ITEM_PRICE_f1
- TOTAL_PRICE_f1
- avg_latency_ms
- p95_latency_ms
- relg_item_price_pair_f1
- relg_hard_negative_fp
- notes

================================================================================
실행 명령
================================================================================

1. ONNX FP32 평가:
python scripts/quantization/eval_layoutlmv3_cord_onnx.py \
  --onnx_model models/layoutlmv3-cord-onnx/fp32/model.onnx \
  --checkpoint_for_processor models/layoutlmv3-cord-full/best \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --split validation \
  --max_samples 100 \
  --out_dir outputs/quantization/cord_onnx_fp32 \
  --local_files_only \
  --debug

2. ONNX INT8 dynamic 평가:
python scripts/quantization/eval_layoutlmv3_cord_onnx.py \
  --onnx_model models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx \
  --checkpoint_for_processor models/layoutlmv3-cord-full/best \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --split validation \
  --max_samples 100 \
  --out_dir outputs/quantization/cord_onnx_int8_dynamic \
  --local_files_only \
  --debug

3. INT8 hidden 기반 rel-g cache:
python scripts/quantization/build_span_relg_dataset_from_onnx.py \
  --raw_data_dir ../receipt_training_data2 \
  --onnx_model models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx \
  --checkpoint_for_processor models/layoutlmv3-cord-full/best \
  --out_dir processed_data/span_relg_cord_onnx_int8_dynamic \
  --split validation \
  --max_samples 100 \
  --device cpu \
  --local_files_only \
  --overwrite \
  --debug

4. rel-g 평가:
python scripts/quantization/evaluate_relg_with_quantized_layout_hidden.py \
  --dataset_dir processed_data/span_relg_cord_onnx_int8_dynamic \
  --relg_checkpoint models/span-relg-context/best \
  --split validation \
  --threshold 0.5 \
  --device cuda \
  --out_dir outputs/quantization/cord_relg_hidden_impact \
  --debug

5. summary:
python scripts/quantization/summarize_quantization_results.py \
  --out_dir outputs/quantization/reports \
  --debug

6. syntax check:
python -m py_compile scripts/quantization/eval_layoutlmv3_cord_onnx.py
python -m py_compile scripts/quantization/build_span_relg_dataset_from_onnx.py
python -m py_compile scripts/quantization/evaluate_relg_with_quantized_layout_hidden.py
python -m py_compile scripts/quantization/summarize_quantization_results.py

최종 보고:
- PyTorch FP32 token F1
- ONNX FP32 token F1
- ONNX INT8 token F1
- INT8 F1 drop
- size reduction
- avg/p95 latency
- rel-g item_price_pair_f1 drop
- hard negative FP 변화
- quantized model을 Android 후보로 볼 수 있는지
```

## 핵심 판단

INT8 logits 성능이 좋아도, `last_hidden_state`가 rel-g에 악영향을 주면 전체 item-price 성능은 떨어질 수 있다. 그래서 이 단계는 반드시 수행해야 한다.
