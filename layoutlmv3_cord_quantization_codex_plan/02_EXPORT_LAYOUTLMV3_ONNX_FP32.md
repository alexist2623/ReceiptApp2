# 02. LayoutLMv3 CORD-only ONNX FP32 Export 구현

Codex에게 아래 지시를 그대로 붙여넣어라.

```text
CORD-only LayoutLMv3 checkpoint를 ONNX FP32로 export하는 스크립트를 구현해라.

목표:
- `models/layoutlmv3-cord-full/best`를 ONNX FP32로 export한다.
- rel-g가 LayoutLMv3 hidden state를 쓰므로, ONNX output은 logits뿐 아니라 last_hidden_state도 포함해야 한다.
- 사용자 데이터는 사용하지 않는다.
- ONNX export 성공 후 onnx.checker와 onnxruntime inference smoke test를 수행한다.

새 파일:
scripts/quantization/export_layoutlmv3_to_onnx.py

출력 디렉토리:
models/layoutlmv3-cord-onnx/fp32

출력 파일:
models/layoutlmv3-cord-onnx/fp32/
├─ model.onnx
├─ export_config.json
├─ labels.json
├─ processor_config.json 등 processor files
└─ smoke_test_report.json

구현 요구:

1. 모델 로드
- AutoModelForTokenClassification.from_pretrained(checkpoint, local_files_only=True)
- AutoProcessor.from_pretrained(checkpoint, apply_ocr=False, local_files_only=True)

2. wrapper class 작성:
class LayoutLMv3TokenAndHiddenWrapper(torch.nn.Module):
    def __init__(self, model):
        ...
    def forward(self, input_ids, attention_mask, bbox, pixel_values):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            bbox=bbox,
            pixel_values=pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.logits, outputs.hidden_states[-1]

3. dummy input은 실제 CORD sample에서 만든다.
- processed_data/cord_bio/validation.jsonl 첫 sample
- ../receipt_training_data2 validation image
- processor로 실제 input 생성
- batch_size=1
- max_length=512

4. input names:
- input_ids
- attention_mask
- bbox
- pixel_values

5. output names:
- logits
- last_hidden_state

6. dynamic axes:
처음은 안전하게 sequence length 512 고정으로 export해도 된다.
단 batch dimension은 dynamic으로 둔다.
dynamic_axes:
{
  "input_ids": {0: "batch"},
  "attention_mask": {0: "batch"},
  "bbox": {0: "batch"},
  "pixel_values": {0: "batch"},
  "logits": {0: "batch"},
  "last_hidden_state": {0: "batch"}
}

7. opset:
- 기본 opset 17
- 실패하면 18로 재시도할 수 있게 argument `--opset` 추가

8. torch.onnx.export 설정:
- do_constant_folding=True
- input_names/output_names 지정
- dynamic_axes 지정

9. export 후 검증:
- onnx.checker.check_model
- onnxruntime.InferenceSession으로 1 sample inference
- output shape 확인:
  logits: [1, 512, num_labels]
  last_hidden_state: [1, 512, hidden_dim]

10. 저장:
- export_config.json:
{
  "checkpoint": "...",
  "onnx_path": "...",
  "opset": 17,
  "max_length": 512,
  "outputs": ["logits", "last_hidden_state"],
  "inputs": ["input_ids", "attention_mask", "bbox", "pixel_values"],
  "source": "CORD-only",
  "user_data_used": false
}

11. processor files도 out_dir에 저장:
processor.save_pretrained(out_dir)
model.config.save_pretrained(out_dir) 가능하면 저장
checkpoint의 labels.json도 복사하거나 config id2label/label2id를 저장

12. CLI:
python scripts/quantization/export_layoutlmv3_to_onnx.py \
  --checkpoint models/layoutlmv3-cord-full/best \
  --cord_bio_dir processed_data/cord_bio \
  --cord_raw_data_dir ../receipt_training_data2 \
  --split validation \
  --sample_index 0 \
  --out_dir models/layoutlmv3-cord-onnx/fp32 \
  --opset 17 \
  --max_length 512 \
  --device cpu \
  --local_files_only \
  --overwrite \
  --debug

13. CPU에서 export한다.
- export는 cuda가 아니어도 된다.
- dtype은 FP32.

14. syntax check:
python -m py_compile scripts/quantization/export_layoutlmv3_to_onnx.py

15. 최종 보고:
- onnx path
- model file size
- logits shape
- hidden shape
- checker pass 여부
- smoke test pass 여부
```

## 주의

`last_hidden_state` output을 빼면 rel-g cache를 만들 수 없다. 이번 export는 반드시 `logits + last_hidden_state`를 출력해야 한다.
