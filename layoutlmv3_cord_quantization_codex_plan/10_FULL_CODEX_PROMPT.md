# 10. Full Codex Prompt — CORD-only LayoutLMv3 Quantization Pipeline 전체 구현 지시

아래 전체를 Codex에 한 번에 붙여넣으면 된다. 너무 길면 01~09 파일을 단계별로 나눠서 수행한다.

```text
현재 repo ReceiptApp2에서 CORD-only LayoutLMv3 양자화 pipeline을 구현해라.

현재 목적:
- 사용자 데이터가 부족하므로 이번에는 사용자 데이터 없이 CORD-only로만 진행한다.
- `models/layoutlmv3-cord-full/best`를 기준으로 PyTorch FP32 baseline을 평가한다.
- ONNX FP32 export를 수행한다.
- PyTorch FP32 vs ONNX FP32 parity를 검증한다.
- ONNX dynamic INT8 quantization을 수행한다.
- CORD-only token F1, field F1, latency, model size를 비교한다.
- rel-g가 사용하는 last_hidden_state까지 비교해서 item_price_pair_f1 영향도 본다.
- static INT8은 optional로 준비만 한다.
- Android 통합은 지금 하지 않고 문서/체크리스트만 만든다.

절대 하지 말 것:
- user labeled data 사용 금지
- /mnt/c/.../APK_Receipt2 사용 금지
- mixed checkpoint 사용 금지
- models/layoutlmv3-mixed-* 사용 금지
- models/span-relg-mixed-* 사용 금지
- fine-tuning 실행 금지
- rel-g training 실행 금지
- Android 앱에 ONNX Runtime 통합 금지
- 모델과 output 파일 git commit 금지

환경:
WSL bash에서 실행.
conda env는 receipt-ml.
which python은 `/home/alexist/miniconda3/envs/receipt-ml/bin/python`이어야 한다.

먼저 실행:
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate receipt-ml
which python
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

필요 패키지:
- torch
- transformers
- datasets
- pillow
- numpy
- seqeval
- onnx
- onnxruntime
- onnxruntime-tools optional
- pandas optional
- matplotlib optional

패키지가 없으면 requirements-ml.txt에 추가하고 pip install -r requirements-ml.txt 하되, 기존 환경을 무리하게 깨지 마라.

생성할 파일:
1. scripts/quantization/eval_layoutlmv3_cord_pytorch.py
2. scripts/quantization/export_layoutlmv3_to_onnx.py
3. scripts/quantization/compare_pytorch_onnx_layoutlmv3.py
4. scripts/quantization/quantize_layoutlmv3_onnx_dynamic.py
5. scripts/quantization/eval_layoutlmv3_cord_onnx.py
6. scripts/quantization/build_span_relg_dataset_from_onnx.py
7. scripts/quantization/evaluate_relg_with_quantized_layout_hidden.py
8. scripts/quantization/summarize_quantization_results.py
9. scripts/quantization/quantize_layoutlmv3_onnx_static.py optional
10. scripts/quantization/run_selective_quantization_experiments.py optional
11. docs/QUANTIZATION_CORD_ONLY.md
12. docs/ANDROID_ONNX_LAYOUTLMV3_DEPLOYMENT_PLAN.md

Step 1. CORD PyTorch FP32 baseline 평가
- checkpoint: models/layoutlmv3-cord-full/best
- split: validation
- max_samples: 100
- output: outputs/quantization/cord_baseline_pytorch_fp32
- metrics:
  - seqeval precision/recall/F1
  - token accuracy
  - field별 F1
  - ITEM_NAME, ITEM_PRICE, TOTAL_PRICE, SUBTOTAL_PRICE, TAX_PRICE
  - confusion top 50
- CORD old label은 canonicalize_label로 ITEM_* schema에 맞춰 평가.

Step 2. ONNX FP32 export
- wrapper는 logits와 last_hidden_state를 출력해야 한다.
- input:
  input_ids, attention_mask, bbox, pixel_values
- output:
  logits, last_hidden_state
- dynamic batch only, seq length는 512 고정
- opset 17 기본
- output: models/layoutlmv3-cord-onnx/fp32/model.onnx
- onnx.checker와 onnxruntime smoke test 수행.

Step 3. PyTorch vs ONNX FP32 parity 검증
- validation 50 sample
- logits diff, hidden diff, argmax agreement, F1 drop 측정
- passed=false이면 INT8으로 가지 말고 보고.

Step 4. ONNX dynamic INT8
- input: models/layoutlmv3-cord-onnx/fp32/model.onnx
- output: models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx
- quantize_dynamic 사용
- QInt8, per_channel 우선
- model size report 저장
- smoke test 수행.

Step 5. ONNX INT8 평가
- CORD validation 100 sample
- token F1 / field F1 / latency / p95 / model size 측정
- outputs/quantization/cord_onnx_int8_dynamic에 저장.

Step 6. rel-g hidden impact
- ONNX INT8의 last_hidden_state로 CORD span rel-g cache 생성.
- 기존 rel-g checkpoint models/span-relg-context/best를 사용해 평가.
- rel-g 모델은 양자화하지 않는다.
- item_price_pair_f1 drop을 본다.
- threshold sweep 0.1~0.9 step 0.05도 저장한다.

Step 7. 결과 summary
- PyTorch FP32 / ONNX FP32 / ONNX INT8 dynamic을 한 표로 비교.
- outputs/quantization/reports/quantization_summary.md 생성.

Step 8. static INT8 optional
- dynamic 결과가 충분하지 않을 때만 quantize_static 구현.
- calibration은 CORD train subset만 사용.
- user data 사용 금지.

Step 9. selective quantization optional
- 정확도 drop이 크면 MatMul/Gemm only, classifier 제외, LayerNorm 제외 variant 생성.

Step 10. Android deployment prep 문서
- 실제 Android 통합은 하지 말고 문서만 작성.
- ONNX Runtime Android에 넣으려면 tokenizer/pixel_values/bbox parity가 필수라고 명시.

실행할 명령:
python -m py_compile scripts/quantization/eval_layoutlmv3_cord_pytorch.py
python -m py_compile scripts/quantization/export_layoutlmv3_to_onnx.py
python -m py_compile scripts/quantization/compare_pytorch_onnx_layoutlmv3.py
python -m py_compile scripts/quantization/quantize_layoutlmv3_onnx_dynamic.py
python -m py_compile scripts/quantization/eval_layoutlmv3_cord_onnx.py
python -m py_compile scripts/quantization/build_span_relg_dataset_from_onnx.py
python -m py_compile scripts/quantization/evaluate_relg_with_quantized_layout_hidden.py
python -m py_compile scripts/quantization/summarize_quantization_results.py

최종 보고:
- 생성/수정 파일 목록
- PyTorch baseline F1
- ONNX FP32 F1
- ONNX FP32 parity pass 여부
- INT8 model size reduction
- INT8 token F1 drop
- INT8 latency
- INT8 hidden 기반 rel-g item_price_pair_f1
- Android 후보로 볼 수 있는지
- 실패한 단계와 전체 traceback
```
