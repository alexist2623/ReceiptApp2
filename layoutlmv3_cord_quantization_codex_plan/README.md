# LayoutLMv3 CORD-only 양자화 구현 계획 — Codex 지시서 모음

이 ZIP은 `ReceiptApp2` 현재 상태에서 **사용자 데이터 없이 CORD 데이터만으로 LayoutLMv3 경량화/양자화 pipeline을 구현**하도록 Codex에게 줄 상세 지시서를 모은 것입니다.

현재 목표는 사용자 데이터가 부족한 상태에서 사용자 데이터까지 섞어 성능을 판단하는 것이 아니라, 다음을 먼저 고정하는 것입니다.

```text
CORD-only LayoutLMv3 checkpoint
→ ONNX FP32 export
→ PyTorch FP32 vs ONNX FP32 parity 검증
→ ONNX Dynamic INT8
→ CORD-only token F1 / latency / model size 평가
→ rel-g가 필요로 하는 last_hidden_state까지 비교
→ static INT8은 optional 실험으로 준비
→ 나중에 user checkpoint가 생기면 같은 export/quant/eval pipeline 재사용
```

## 사용 기준

기본 checkpoint는 CORD-only 모델입니다.

```text
models/layoutlmv3-cord-full/best
```

사용자 데이터는 이번 작업에서 사용하지 않습니다.

```text
금지:
- /mnt/c/.../APK_Receipt2 같은 user labeled data 사용 금지
- mixed CORD+user checkpoint 사용 금지
- user relation JSON 사용 금지
- user rel-g cache 사용 금지
```

## 파일 구성

```text
README.md
00_GLOBAL_RULES.md
01_BASELINE_CORD_ONLY_EVAL.md
02_EXPORT_LAYOUTLMV3_ONNX_FP32.md
03_VALIDATE_ONNX_FP32_PARITY.md
04_DYNAMIC_INT8_QUANTIZATION.md
05_EVALUATE_INT8_AND_RELG_IMPACT.md
06_STATIC_INT8_PTQ_OPTIONAL.md
07_SELECTIVE_QUANTIZATION_FALLBACK.md
08_ANDROID_DEPLOYMENT_PREP_LATER.md
09_FUTURE_USER_DATA_INTEGRATION.md
10_FULL_CODEX_PROMPT.md
```

## 권장 실행 순서

1. `00_GLOBAL_RULES.md`
2. `01_BASELINE_CORD_ONLY_EVAL.md`
3. `02_EXPORT_LAYOUTLMV3_ONNX_FP32.md`
4. `03_VALIDATE_ONNX_FP32_PARITY.md`
5. `04_DYNAMIC_INT8_QUANTIZATION.md`
6. `05_EVALUATE_INT8_AND_RELG_IMPACT.md`
7. `06_STATIC_INT8_PTQ_OPTIONAL.md`는 dynamic INT8 결과가 충분하지 않을 때만 진행
8. `07_SELECTIVE_QUANTIZATION_FALLBACK.md`는 정확도 손실이 클 때 진행
9. Android 배포는 `08_ANDROID_DEPLOYMENT_PREP_LATER.md` 단계에서 준비만 한다
10. 사용자 데이터가 충분해진 뒤 `09_FUTURE_USER_DATA_INTEGRATION.md` 기준으로 재사용한다

## 결과물 목표

최종적으로 repo 안에 다음 결과가 생기면 됩니다.

```text
models/layoutlmv3-cord-onnx/
├─ fp32/
│  ├─ model.onnx
│  ├─ export_config.json
│  └─ parity_report.json
├─ int8_dynamic/
│  ├─ model.onnx
│  ├─ quant_config.json
│  └─ eval_report.json
└─ int8_static/                  # optional
   ├─ model.onnx
   ├─ quant_config.json
   └─ eval_report.json

outputs/quantization/
├─ cord_baseline_pytorch_fp32/
├─ cord_onnx_fp32/
├─ cord_onnx_int8_dynamic/
├─ cord_relg_hidden_impact/
└─ reports/
```

## 성공 기준

처음 성공 기준은 엄격하게 잡지 말고 다음 정도로 둡니다.

```text
ONNX FP32:
- PyTorch FP32와 argmax label agreement >= 99%
- logits max_abs_diff가 과도하지 않음
- CORD token F1이 PyTorch baseline과 사실상 동일

ONNX INT8 dynamic:
- token F1 drop <= 1~2 percentage point
- ITEM_NAME / ITEM_PRICE F1 drop <= 2 percentage point
- hidden_state 기반 rel-g item_price_pair_f1 drop <= 2~3 percentage point
- model size 감소 확인
- CPU latency 개선 또는 Android 배포 가능성 확인
```

## 참고

공식 ONNX Runtime 문서는 dynamic/static quantization과 `quantize_dynamic`, `quantize_static` API를 설명하고, transformer 기반 모델에는 dynamic quantization을 일반적으로 권장한다고 설명합니다. 이 문서들은 Codex 작업 중 버전별 API 확인용으로 참고하세요.

```text
ONNX Runtime quantization:
https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html

ONNX Runtime mobile:
https://onnxruntime.ai/docs/tutorials/mobile/

Hugging Face Optimum ONNX Runtime quantization:
https://huggingface.co/docs/optimum-onnx/onnxruntime/usage_guides/quantization
```
