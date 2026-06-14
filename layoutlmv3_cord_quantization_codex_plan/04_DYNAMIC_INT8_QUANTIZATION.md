# 04. ONNX Dynamic INT8 Quantization 구현

Codex에게 아래 지시를 그대로 붙여넣어라.

```text
ONNX FP32 LayoutLMv3 모델에 dynamic INT8 quantization을 적용하는 script를 작성해라.

목표:
- `models/layoutlmv3-cord-onnx/fp32/model.onnx`를 dynamic INT8 모델로 변환한다.
- 처음에는 transformer에 일반적으로 안전한 dynamic quantization을 사용한다.
- 사용자 데이터는 사용하지 않는다.
- quantization 후 ONNX Runtime smoke test를 수행한다.

새 파일:
scripts/quantization/quantize_layoutlmv3_onnx_dynamic.py

입력:
--input_onnx models/layoutlmv3-cord-onnx/fp32/model.onnx
--out_dir models/layoutlmv3-cord-onnx/int8_dynamic

출력:
models/layoutlmv3-cord-onnx/int8_dynamic/
├─ model.onnx
├─ quant_config.json
├─ smoke_test_report.json
└─ model_size_report.json

구현 요구:

1. onnxruntime.quantization import:
from onnxruntime.quantization import quantize_dynamic, QuantType

2. quantize_dynamic 호출:
- weight_type=QuantType.QInt8
- per_channel=True 옵션을 지원하되, 설치된 onnxruntime 버전에서 지원하지 않으면 fallback
- op_types_to_quantize argument optional:
  기본 None
  fallback으로 ["MatMul", "Gemm"] 지원

3. CLI 옵션:
--weight_type qint8|quint8
--per_channel
--op_types MatMul,Gemm
--disable_per_channel
--force

4. 모델 크기 비교:
- input ONNX size MB
- output ONNX size MB
- size reduction percentage

5. smoke test:
- CORD validation sample 하나로 input 생성
- ONNX Runtime session으로 inference
- logits shape와 last_hidden_state shape 확인
- NaN/Inf 검사
- logits output 존재 확인
- last_hidden_state output 존재 확인

6. quant_config.json:
{
  "input_onnx": "...",
  "output_onnx": "...",
  "method": "dynamic",
  "weight_type": "QInt8",
  "per_channel": true,
  "op_types_to_quantize": null,
  "user_data_used": false,
  "source": "CORD-only",
  "input_size_mb": ...,
  "output_size_mb": ...,
  "size_reduction_percent": ...
}

7. CLI:
python scripts/quantization/quantize_layoutlmv3_onnx_dynamic.py \
  --input_onnx models/layoutlmv3-cord-onnx/fp32/model.onnx \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --checkpoint_for_processor models/layoutlmv3-cord-full/best \
  --out_dir models/layoutlmv3-cord-onnx/int8_dynamic \
  --weight_type qint8 \
  --per_channel \
  --local_files_only \
  --overwrite \
  --debug

8. syntax check:
python -m py_compile scripts/quantization/quantize_layoutlmv3_onnx_dynamic.py

9. 최종 보고:
- input size
- output size
- reduction
- quantized op types
- smoke test output shapes
- any warnings/errors
```

## 주의

Dynamic INT8은 첫 단계다. 여기서 정확도 손실이 작고 크기/속도 이득이 있으면 Android MVP 후보가 된다.
