# 08. Android Deployment Prep — 나중 단계

Codex에게 아래 지시를 그대로 붙여넣어라. 이 단계에서는 Android 앱에 실제 모델을 넣지 않고, 준비 문서와 체크리스트만 만든다.

```text
CORD-only ONNX/INT8 LayoutLMv3를 나중에 Android 앱에 넣기 위한 준비 문서를 작성해라.

이번 작업에서 Android 앱에 ONNX Runtime을 실제 통합하지 마라.
이번 작업은 PC/WSL에서 quantized ONNX 모델을 만들고 평가하는 것이 목표다.

새 문서:
docs/ANDROID_ONNX_LAYOUTLMV3_DEPLOYMENT_PLAN.md

문서에 포함할 내용:

1. Android에서 필요한 입력
- input_ids [1,512] int64
- attention_mask [1,512] int64
- bbox [1,512,4] int64
- pixel_values [1,3,224,224] float32

2. Android에서 이미 있는 것
- OCR words
- OCR boxes
- saved canonical image
- app export zip
- ML Kit OCR

3. Android에서 새로 구현해야 할 것
- Hugging Face tokenizer와 동일한 tokenizer
- word -> subword mapping
- bbox alignment
- LayoutLMv3 image preprocessing
- ONNX Runtime session
- logits decode
- last_hidden_state extraction
- BIO span merge
- rel-g inference

4. Python과 Android preprocessing parity 검증
같은 receipt 10개에 대해 다음을 비교:
- input_ids
- attention_mask
- bbox
- pixel_values mean/std/min/max
- token strings
- word_ids mapping

5. Android dependency 후보
- ONNX Runtime Android
- ONNX Runtime Mobile
- 추후 reduced operator build 고려

6. assets 구조 후보
ReceiptApp/app/src/main/assets/models/layoutlmv3_cord_int8_dynamic/
├─ model.onnx
├─ labels.json
├─ tokenizer files
└─ export_config.json

7. 처음 Android MVP 목표
- OCR JSON -> ONNX LayoutLMv3 logits -> word label overlay
- rel-g는 아직 PC/server 유지 가능

8. 두 번째 Android 목표
- hidden_state -> span merge -> rel-g -> grouped JSON

9. 리스크
- tokenizer 불일치
- pixel_values preprocessing 불일치
- APK 크기
- RAM 사용량
- inference latency
- hidden_state output size
- rel-g까지 모바일에서 돌릴 경우 추가 메모리

10. Android 통합 전 필수 조건
- ONNX FP32 parity pass
- ONNX INT8 token F1 drop <= 2%
- rel-g hidden impact drop <= 3%
- Android preprocessing parity pass

최종 보고:
- 문서 경로
- Android 통합은 아직 수행하지 않았음을 명시
```
