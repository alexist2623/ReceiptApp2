# 09. Future User Data Integration — 사용자 데이터가 충분해진 뒤

Codex에게 아래 지시를 그대로 붙여넣어라. 이 단계는 지금 수행하지 않는다.

```text
이 문서는 나중에 사용자 labeled data가 50~100장 이상 모였을 때, CORD-only quantization pipeline을 user/mixed checkpoint에 재사용하는 방법을 정리하는 것이다.

지금은 실행하지 마라.

나중 단계 목표:
1. user-only validation/test set을 고정한다.
2. mixed LayoutLMv3 checkpoint를 만든다.
3. 같은 ONNX export / INT8 quantization / evaluation pipeline을 재사용한다.
4. CORD-only quantized model과 mixed quantized model을 비교한다.

나중에 필요한 데이터 구조:
user_data/
├─ train/
├─ validation/
└─ test/

각 sample:
<id>_receipt_ocr/
├─ <id>.jpg
├─ <id>_ocr.json
├─ <id>_labeled_v2_1.json
└─ <id>_labeled_v2_1.jsonl

절대 하지 말 것:
- validation/test set에 user_repeat 적용하지 마라.
- Temp 폴더와 실제 학습 폴더를 섞지 마라.
- coordinate mismatch가 있는 sample을 학습하지 마라.
- relation 없는 sample을 rel-g positive 학습에 억지로 넣지 마라.

나중 실행 순서:
1. user label validation:
python scripts/validate_user_labels_v2.py ...

2. coordinate validation:
python scripts/validate_receipt_export_coordinates.py ...

3. mixed LayoutLMv3 fine-tune:
train_mixed_layoutlmv3_user_cord.py

4. mixed checkpoint export:
export_layoutlmv3_to_onnx.py \
  --checkpoint models/layoutlmv3-mixed-.../best

5. ONNX FP32 parity:
compare_pytorch_onnx_layoutlmv3.py

6. INT8 dynamic:
quantize_layoutlmv3_onnx_dynamic.py

7. user-only eval:
eval_layoutlmv3_user_onnx.py를 별도 구현하거나 CORD evaluator를 user evaluator로 확장한다.

8. rel-g hidden impact:
user rel-g cache도 ONNX hidden으로 재생성해서 비교한다.

필수 metric:
- CORD token F1
- user token F1
- user ITEM_NAME F1
- user ITEM_PRICE F1
- user TOTAL_PRICE F1
- user item_price_pair_f1
- hard negative false positive count

결론:
CORD-only quantization pipeline을 먼저 안정화한 뒤, user data가 충분해졌을 때 checkpoint만 바꿔서 같은 pipeline을 재사용한다.
```
