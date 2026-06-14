# 01. CORD-only PyTorch FP32 Baseline 평가 구현

Codex에게 아래 지시를 그대로 붙여넣어라.

```text
CORD-only LayoutLMv3 PyTorch FP32 baseline 평가 스크립트를 구현해라.

목표:
- checkpoint `models/layoutlmv3-cord-full/best`를 기준 baseline으로 고정한다.
- CORD BIO validation/test subset에서 token classification 성능을 측정한다.
- 이후 ONNX FP32/INT8과 비교할 수 있도록 동일한 evaluator를 만든다.
- 사용자 데이터는 절대 사용하지 않는다.

입력:
--checkpoint models/layoutlmv3-cord-full/best
--cord_bio_dir processed_data/cord_bio
--cord_raw_data_dir ../receipt_training_data2
--split validation 또는 test
--max_samples 100 또는 None
--out_dir outputs/quantization/cord_baseline_pytorch_fp32

새 파일:
scripts/quantization/eval_layoutlmv3_cord_pytorch.py

기능:
1. CORD BIO JSONL을 읽는다.
   - processed_data/cord_bio/<split>.jsonl
   - record에는 words, labels, normalized_boxes, boxes, index, split 등이 있을 수 있다.
   - 기존 `scripts/train_mixed_layoutlmv3_user_cord.py`의 `load_cord_records`, `MixedReceiptDataset` 일부 로직을 참고하되 user data 로직은 넣지 마라.

2. CORD raw image를 읽는다.
   - `datasets.load_from_disk(args.cord_raw_data_dir)`
   - split/index로 image를 찾는다.

3. processor는 반드시 apply_ocr=False.
   - AutoProcessor.from_pretrained(checkpoint, apply_ocr=False, local_files_only=True)

4. model:
   - AutoModelForTokenClassification.from_pretrained(checkpoint, local_files_only=True)
   - eval mode
   - output_hidden_states=True도 옵션으로 가능하게 한다.

5. 입력:
   - image
   - words
   - boxes=normalized_boxes
   - word_labels=label_ids
   - max_length=512
   - padding=max_length
   - truncation=True

6. 평가 지표:
   - seqeval precision/recall/F1
   - token accuracy
   - field별 precision/recall/F1
     최소:
       ITEM_NAME 또는 MENU_NM
       ITEM_PRICE 또는 MENU_PRICE
       TOTAL_PRICE 또는 TOTAL_TOTAL_PRICE
       SUBTOTAL_PRICE
       TAX_PRICE
   - label distribution
   - confusion count top 50

7. CORD old label을 canonical schema로 평가한다.
   - `ml.receipt_schema.canonicalize_label` 사용
   - MENU_NM -> ITEM_NAME
   - MENU_PRICE -> ITEM_PRICE
   - TOTAL_TOTAL_PRICE -> TOTAL_PRICE
   - SUBTOTAL_TAX_PRICE -> TAX_PRICE

8. 결과 저장:
outputs/quantization/cord_baseline_pytorch_fp32/
├─ metrics_<split>.json
├─ predictions_<split>.jsonl
├─ label_distribution_<split>.json
└─ confusion_top_<split>.json

9. predictions JSONL record 예:
{
  "id": "...",
  "source": "cord",
  "split": "validation",
  "index": 0,
  "words": [...],
  "gold_labels": [...],
  "pred_labels": [...],
  "canonical_gold_labels": [...],
  "canonical_pred_labels": [...],
  "confidences": [...],
  "image_size": [w, h]
}

10. CLI:
python scripts/quantization/eval_layoutlmv3_cord_pytorch.py \
  --checkpoint models/layoutlmv3-cord-full/best \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --split validation \
  --max_samples 100 \
  --device cuda \
  --local_files_only \
  --out_dir outputs/quantization/cord_baseline_pytorch_fp32 \
  --debug

11. test split도 돌릴 수 있게 한다.

12. syntax check:
python -m py_compile scripts/quantization/eval_layoutlmv3_cord_pytorch.py

13. 최종 보고:
- 실행 command
- checkpoint
- split
- max_samples
- num_samples
- num_tokens
- seqeval F1
- ITEM_NAME F1
- ITEM_PRICE F1
- TOTAL_PRICE F1
- output files
```

## 성공 기준

이 단계에서 baseline metric을 저장해야 한다. 이후 ONNX FP32/INT8은 이 baseline과 비교한다.

중요한 점은 현재 목표가 “성능을 올리는 것”이 아니라, **양자화 전 기준선을 고정하는 것**이다.
