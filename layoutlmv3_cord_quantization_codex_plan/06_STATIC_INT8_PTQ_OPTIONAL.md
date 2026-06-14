# 06. Static INT8 PTQ Optional 단계

Codex에게 아래 지시를 그대로 붙여넣어라. 이 단계는 Dynamic INT8 결과가 충분하지 않을 때만 진행한다.

```text
CORD-only calibration data로 static INT8 PTQ를 시도할 수 있게 구현해라.

중요:
- 이 단계는 optional이다.
- dynamic INT8이 충분히 좋으면 static INT8은 건너뛴다.
- 사용자 데이터는 사용하지 않는다.
- calibration은 CORD train/validation subset만 사용한다.

새 파일:
scripts/quantization/quantize_layoutlmv3_onnx_static.py

입력:
--input_onnx models/layoutlmv3-cord-onnx/fp32/model.onnx
--checkpoint_for_processor models/layoutlmv3-cord-full/best
--cord_bio_dir processed_data/cord_bio
--cord_raw_data_dir ../receipt_training_data2
--calibration_split train
--calibration_samples 100
--out_dir models/layoutlmv3-cord-onnx/int8_static

구현 요구:

1. CalibrationDataReader 구현:
class LayoutLMv3CordCalibrationReader(CalibrationDataReader):
    - CORD BIO samples를 읽는다.
    - processor로 input_ids, attention_mask, bbox, pixel_values를 만든다.
    - batch_size는 1부터 시작한다.
    - get_next()는 numpy arrays dict 반환.

2. quantize_static 사용:
from onnxruntime.quantization import quantize_static, CalibrationMethod, QuantFormat, QuantType

기본:
- quant_format=QuantFormat.QDQ
- activation_type=QuantType.QInt8 또는 QUInt8 실험 옵션
- weight_type=QuantType.QInt8
- calibrate_method=CalibrationMethod.MinMax
- per_channel=True 가능하면 적용

3. CLI 옵션:
--calibration_method minmax|entropy|percentile
--activation_type qint8|quint8
--weight_type qint8|quint8
--per_channel
--op_types MatMul,Gemm
--nodes_to_exclude optional file

4. static 모델 저장:
models/layoutlmv3-cord-onnx/int8_static/
├─ model.onnx
├─ quant_config.json
└─ calibration_summary.json

5. smoke test:
- validation sample 1개 inference
- logits / last_hidden_state output 존재 확인
- NaN/Inf 체크

6. 이후 반드시 05단계 evaluator로 평가한다.
- token F1
- rel-g hidden impact
- latency
- model size

7. 정확도 손실이 dynamic INT8보다 크면 static INT8은 폐기한다.
폐기 기준:
- token F1 drop > 2%
- item_price_pair_f1 drop > 3%
- ITEM_PRICE F1 drop > 3%

CLI:
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

syntax check:
python -m py_compile scripts/quantization/quantize_layoutlmv3_onnx_static.py
```

## 판단 기준

Static INT8은 activation scale을 calibration set으로 고정하기 때문에, calibration set이 부족하면 오히려 정확도가 떨어질 수 있다. 지금은 사용자 데이터가 부족하므로 CORD-only static PTQ는 참고 실험으로만 둔다.
