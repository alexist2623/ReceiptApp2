# 00. Global Rules — Codex 공통 지시

아래 내용을 Codex 작업 시작 전에 그대로 붙여넣어라.

```text
현재 repo는 ReceiptApp2이다. Android app과 Python ML pipeline이 함께 있다.

이번 작업의 목표는 사용자 데이터가 부족한 상태에서 mixed/user fine-tuning을 계속하는 것이 아니라, CORD-only LayoutLMv3 checkpoint를 기준으로 양자화 pipeline을 구현하는 것이다.

기본 모델:
models/layoutlmv3-cord-full/best

기본 CORD BIO:
processed_data/cord_bio

기본 CORD raw data:
../receipt_training_data2

기본 rel-g checkpoint:
models/span-relg-context/best

이번 작업에서 금지:
- user labeled data 사용 금지
- /mnt/c/Users/.../APK_Receipt2 사용 금지
- user_repeat 사용 금지
- mixed CORD+user checkpoint 사용 금지
- models/layoutlmv3-mixed-* 사용 금지
- models/span-relg-mixed-* 사용 금지
- fine-tuning 실행 금지
- rel-g training 실행 금지
- Android 앱 inference 통합 구현 금지
- 모델 파일을 git commit 금지
- outputs/, processed_data generated cache, models generated output commit 금지

이번 작업에서 해야 하는 것:
- CORD-only baseline 평가
- CORD-only LayoutLMv3 ONNX FP32 export
- PyTorch FP32 vs ONNX FP32 parity 검증
- ONNX dynamic INT8 quantization
- CORD-only token F1 / field F1 / latency / model size 평가
- last_hidden_state 출력이 rel-g에 줄 영향을 측정
- static INT8 PTQ는 optional 준비
- Android 배포는 나중에 하도록 문서와 산출물만 준비

모든 Python 작업은 WSL bash에서 수행한다.

conda env:
receipt-ml

환경 확인:
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || true
conda activate receipt-ml

which python
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

which python이 /home/alexist/miniconda3/envs/receipt-ml/bin/python이 아니면 중단하고 보고한다.

먼저 git 상태 확인:
git status --short
git branch --show-current
git rev-parse HEAD

새 파일은 scripts/quantization/ 아래에 만들고, 문서는 docs/QUANTIZATION_CORD_ONLY.md에 작성한다.

실패하면 숨기지 말고 전체 traceback과 원인을 보고한다.
```

## 의도

이 단계에서 가장 중요한 것은 **기준을 CORD-only로 고정**하는 것이다.

사용자 데이터가 섞이면 quantization 손실인지, 사용자 도메인 adaptation 문제인지, 라벨 품질 문제인지 구분이 안 된다. 따라서 먼저 CORD-only에서 export/quantization/evaluation이 정확히 되는지 확인한다.
