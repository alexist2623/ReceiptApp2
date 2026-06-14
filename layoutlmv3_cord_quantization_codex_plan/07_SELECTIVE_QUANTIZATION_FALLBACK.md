# 07. Selective Quantization Fallback

Codex에게 아래 지시를 그대로 붙여넣어라. 이 단계는 INT8 정확도 손실이 클 때만 진행한다.

```text
ONNX INT8 양자화에서 정확도 손실이 크면 selective quantization fallback을 구현해라.

목표:
- 전체 모델을 무작정 INT8로 바꾸지 않고, 민감한 op/layer를 제외한 quantized 모델을 여러 개 생성한다.
- CORD-only 기준으로 정확도/속도/용량 trade-off를 비교한다.
- 사용자 데이터는 사용하지 않는다.

새 파일:
scripts/quantization/run_selective_quantization_experiments.py

입력:
--input_onnx models/layoutlmv3-cord-onnx/fp32/model.onnx
--out_root models/layoutlmv3-cord-onnx/selective

실험 variant:
1. matmul_gemm_only
   - op_types_to_quantize=["MatMul", "Gemm"]

2. matmul_only
   - op_types_to_quantize=["MatMul"]

3. gemm_only
   - op_types_to_quantize=["Gemm"]

4. exclude_layernorm_embedding
   - LayerNormalization, Gather, Embedding 관련 node 이름 제외
   - node name 패턴 기반 제외

5. exclude_classifier
   - classifier head 관련 node 제외
   - node 이름에서 classifier, logits, score가 들어가면 제외 후보

6. safest_dynamic
   - MatMul/Gemm only + per_channel=True + QInt8

구현 요구:
- onnx graph를 inspect해서 node names/op types를 list로 저장한다.
- 각 variant별 quant_config.json 저장.
- 각 variant별 model.onnx 저장.
- size report 저장.
- smoke test 수행.

출력:
models/layoutlmv3-cord-onnx/selective/
├─ matmul_gemm_only/model.onnx
├─ matmul_only/model.onnx
├─ gemm_only/model.onnx
├─ exclude_classifier/model.onnx
└─ selective_summary.json

추가:
- 각 variant는 05단계 evaluator로 평가할 수 있게 path를 summary에 기록한다.
- 자동 평가까지 연결 가능하면 `--run_eval` 옵션을 추가한다.

CLI:
python scripts/quantization/run_selective_quantization_experiments.py \
  --input_onnx models/layoutlmv3-cord-onnx/fp32/model.onnx \
  --checkpoint_for_processor models/layoutlmv3-cord-full/best \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --out_root models/layoutlmv3-cord-onnx/selective \
  --local_files_only \
  --overwrite \
  --debug

syntax check:
python -m py_compile scripts/quantization/run_selective_quantization_experiments.py

최종 보고:
- variant 목록
- 각 model size
- smoke test 여부
- 추천 평가 순서
```

## 판단 기준

정확도 손실이 크면 전체 INT8보다 selective INT8이 더 낫다. 최종 선택은 model size가 아니라 end-to-end item_price_pair_f1 기준이다.
