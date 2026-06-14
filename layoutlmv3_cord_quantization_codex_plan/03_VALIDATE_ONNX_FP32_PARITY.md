# 03. PyTorch FP32 vs ONNX FP32 Parity 검증 구현

Codex에게 아래 지시를 그대로 붙여넣어라.

```text
PyTorch FP32 LayoutLMv3와 ONNX FP32 LayoutLMv3 결과가 같은지 검증하는 script를 작성해라.

목표:
- export된 ONNX FP32가 PyTorch checkpoint와 사실상 같은 출력을 내는지 검증한다.
- logits argmax label agreement와 last_hidden_state 차이를 측정한다.
- ONNX FP32가 PyTorch와 안 맞으면 INT8 quantization을 진행하지 않는다.
- 사용자 데이터는 사용하지 않는다.

새 파일:
scripts/quantization/compare_pytorch_onnx_layoutlmv3.py

입력:
--pytorch_checkpoint models/layoutlmv3-cord-full/best
--onnx_model models/layoutlmv3-cord-onnx/fp32/model.onnx
--cord_bio_dir processed_data/cord_bio
--cord_raw_data_dir ../receipt_training_data2
--split validation
--max_samples 50

출력:
outputs/quantization/cord_onnx_fp32/
├─ parity_report_validation.json
├─ mismatch_examples_validation.jsonl
└─ per_sample_diff_validation.jsonl

구현 요구:

1. PyTorch model load:
- AutoModelForTokenClassification
- output_hidden_states=True
- eval mode

2. ONNX Runtime session:
- providers:
  - CPUExecutionProvider 기본
  - CUDAExecutionProvider는 optional
- provider 정보 report에 저장

3. 동일한 processor를 사용한다.
- checkpoint processor 사용
- apply_ocr=False
- same words/boxes/image
- max_length=512

4. 각 sample에 대해:
- PyTorch logits, last_hidden_state
- ONNX logits, last_hidden_state
- logits max_abs_diff
- logits mean_abs_diff
- hidden max_abs_diff
- hidden mean_abs_diff
- argmax label agreement
- non-pad token argmax agreement
- gold label이 있는 token만 agreement
- word-level label agreement
- sample seqeval contribution용 gold/pred 저장

5. label 비교:
- PyTorch predicted label sequence
- ONNX predicted label sequence
- label mismatch top examples 저장

6. 통과 기준:
기본 warning threshold:
- overall token argmax agreement >= 0.99
- word-level argmax agreement >= 0.99
- ONNX FP32 seqeval F1 drop <= 0.001 absolute
- hidden mean_abs_diff가 지나치게 크지 않을 것

엄격히 fail하지는 말고 `passed: true/false`를 report에 저장한다.
passed=false면 다음 단계에서 INT8 진행하지 말라는 메시지를 출력한다.

7. report:
{
  "pytorch_checkpoint": "...",
  "onnx_model": "...",
  "split": "validation",
  "max_samples": 50,
  "num_samples": ...,
  "num_tokens": ...,
  "pytorch_seqeval_f1": ...,
  "onnx_seqeval_f1": ...,
  "f1_drop": ...,
  "overall_token_argmax_agreement": ...,
  "word_level_argmax_agreement": ...,
  "logits_max_abs_diff": ...,
  "logits_mean_abs_diff": ...,
  "hidden_max_abs_diff": ...,
  "hidden_mean_abs_diff": ...,
  "passed": true/false
}

8. CLI:
python scripts/quantization/compare_pytorch_onnx_layoutlmv3.py \
  --pytorch_checkpoint models/layoutlmv3-cord-full/best \
  --onnx_model models/layoutlmv3-cord-onnx/fp32/model.onnx \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --split validation \
  --max_samples 50 \
  --max_length 512 \
  --device cuda \
  --onnx_provider cpu \
  --local_files_only \
  --out_dir outputs/quantization/cord_onnx_fp32 \
  --debug

9. syntax check:
python -m py_compile scripts/quantization/compare_pytorch_onnx_layoutlmv3.py

10. 최종 보고:
- PyTorch F1
- ONNX F1
- F1 drop
- argmax agreement
- hidden diff
- passed 여부
```

## 의도

INT8 모델 평가 전에 ONNX FP32 자체가 맞는지 먼저 확인해야 한다. 여기서 틀리면 quantization 문제가 아니라 export/preprocessing 문제가 된다.
