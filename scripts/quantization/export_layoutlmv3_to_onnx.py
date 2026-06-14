import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
from transformers import AutoModelForTokenClassification, AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.quantization.quant_common import (
    fail,
    file_size_mb,
    finite_report,
    load_cord_jsonl,
    load_labels,
    load_raw_dataset,
    numpy_inputs_from_encoding,
    prepare_cord_sample,
    save_json,
    onnx_total_size_mb,
)


class LayoutLMv3TokenAndHiddenWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

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


def parse_args():
    parser = argparse.ArgumentParser(description="Export CORD-only LayoutLMv3 token classifier to ONNX FP32.")
    parser.add_argument("--checkpoint", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--cord_bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--cord_raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out_dir", default="models/layoutlmv3-cord-onnx/fp32")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def prepare_out_dir(out_dir, overwrite):
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            fail(f"{out_dir} already exists. Use --overwrite to replace it.")
        print(f"Removing existing output directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if args.local_files_only and not checkpoint.exists():
        fail(f"checkpoint not found: {checkpoint}")
    if args.device == "cuda" and not torch.cuda.is_available():
        fail("CUDA requested but torch.cuda.is_available() is False")
    device = torch.device(args.device)
    out_dir = prepare_out_dir(args.out_dir, args.overwrite)
    onnx_path = out_dir / "model.onnx"
    labels_payload, _, label2id, _ = load_labels(checkpoint)
    records = load_cord_jsonl(args.cord_bio_dir, args.split, args.sample_index + 1)
    if args.sample_index >= len(records):
        fail(f"sample_index {args.sample_index} out of range for {args.split}; records loaded={len(records)}")
    raw_dataset = load_raw_dataset(args.cord_raw_data_dir)
    sample = prepare_cord_sample(records[args.sample_index], raw_dataset, label2id, args.split)

    processor = AutoProcessor.from_pretrained(str(checkpoint), apply_ocr=False, local_files_only=args.local_files_only)
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint), local_files_only=args.local_files_only)
    model.to(device)
    model.eval()
    wrapper = LayoutLMv3TokenAndHiddenWrapper(model).to(device).eval()
    encoding = processor(
        sample["image"],
        sample["words"],
        boxes=sample["normalized_boxes"],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    bbox = encoding["bbox"].to(device)
    pixel_values = encoding["pixel_values"].to(device)
    print(f"checkpoint: {checkpoint}")
    print(f"out_dir: {out_dir}")
    print(f"opset: {args.opset}")
    print(f"sample: {sample['id']} words={len(sample['words'])}")
    print("input shapes:")
    for name, tensor in {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "bbox": bbox,
        "pixel_values": pixel_values,
    }.items():
        print(f"  {name}: {tuple(tensor.shape)} {tensor.dtype}")

    with torch.no_grad():
        torch_logits, torch_hidden = wrapper(input_ids, attention_mask, bbox, pixel_values)
    torch.onnx.export(
        wrapper,
        (input_ids, attention_mask, bbox, pixel_values),
        str(onnx_path),
        input_names=["input_ids", "attention_mask", "bbox", "pixel_values"],
        output_names=["logits", "last_hidden_state"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "bbox": {0: "batch"},
            "pixel_values": {0: "batch"},
            "logits": {0: "batch"},
            "last_hidden_state": {0: "batch"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
    )
    checker_passed = False
    checker_error = None
    try:
        onnx.checker.check_model(str(onnx_path))
        checker_passed = True
    except Exception as exc:
        checker_error = repr(exc)
        print(f"WARNING: onnx checker failed: {checker_error}")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_outputs = session.run(None, numpy_inputs_from_encoding(encoding))
    output_names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(output_names, ort_outputs))
    smoke = {
        "providers": session.get_providers(),
        "output_names": output_names,
        "finite_report": finite_report(outputs),
        "torch_logits_shape": list(torch_logits.shape),
        "torch_hidden_shape": list(torch_hidden.shape),
        "onnx_logits_shape": list(outputs["logits"].shape),
        "onnx_hidden_shape": list(outputs["last_hidden_state"].shape),
        "logits_max_abs_diff": float(np.max(np.abs(outputs["logits"] - torch_logits.detach().cpu().numpy()))),
        "hidden_max_abs_diff": float(np.max(np.abs(outputs["last_hidden_state"] - torch_hidden.detach().cpu().numpy()))),
    }
    smoke["passed"] = (
        checker_passed
        and smoke["onnx_logits_shape"] == list(torch_logits.shape)
        and smoke["onnx_hidden_shape"] == list(torch_hidden.shape)
        and not smoke["finite_report"]["logits"]["has_nan"]
        and not smoke["finite_report"]["last_hidden_state"]["has_nan"]
    )
    processor.save_pretrained(out_dir)
    model.config.save_pretrained(out_dir)
    save_json(out_dir / "labels.json", labels_payload)
    export_config = {
        "checkpoint": str(checkpoint),
        "onnx_path": str(onnx_path),
        "opset": args.opset,
        "max_length": args.max_length,
        "outputs": ["logits", "last_hidden_state"],
        "inputs": ["input_ids", "attention_mask", "bbox", "pixel_values"],
        "source": "CORD-only",
        "user_data_used": False,
        "model_size_mb": file_size_mb(onnx_path),
        "model_total_size_mb": onnx_total_size_mb(onnx_path),
        "uses_external_data": Path(str(onnx_path) + ".data").exists(),
        "checker_passed": checker_passed,
        "checker_error": checker_error,
    }
    save_json(out_dir / "export_config.json", export_config)
    save_json(out_dir / "smoke_test_report.json", smoke)
    print(f"onnx path: {onnx_path}")
    print(f"model size MB: {export_config['model_total_size_mb']:.2f}")
    print(f"logits shape: {smoke['onnx_logits_shape']}")
    print(f"hidden shape: {smoke['onnx_hidden_shape']}")
    print(f"checker_passed: {checker_passed}")
    print(f"smoke passed: {smoke['passed']}")
    if not smoke["passed"]:
        fail("ONNX FP32 export smoke test failed.")
    print("LayoutLMv3 ONNX FP32 export passed.")


if __name__ == "__main__":
    main()
