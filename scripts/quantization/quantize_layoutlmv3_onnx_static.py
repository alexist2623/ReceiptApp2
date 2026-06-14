import argparse
import json
import shutil
import sys
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from transformers import AutoProcessor

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
    onnx_total_size_mb,
    prepare_cord_sample,
    save_json,
)


class LayoutLMv3CordCalibrationReader(CalibrationDataReader):
    def __init__(self, records, raw_dataset, processor, label2id, split, max_length=512, input_names=None):
        self.records = records
        self.raw_dataset = raw_dataset
        self.processor = processor
        self.label2id = label2id
        self.split = split
        self.max_length = max_length
        self.input_names = set(input_names or [])
        self._index = 0
        self.num_returned = 0

    def get_next(self):
        if self._index >= len(self.records):
            return None
        record = self.records[self._index]
        self._index += 1
        sample = prepare_cord_sample(record, self.raw_dataset, self.label2id, self.split)
        encoding = self.processor(
            sample["image"],
            sample["words"],
            boxes=sample["normalized_boxes"],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = numpy_inputs_from_encoding(encoding)
        if self.input_names:
            inputs = {key: value for key, value in inputs.items() if key in self.input_names}
        self.num_returned += 1
        return inputs

    def rewind(self):
        self._index = 0


def parse_args():
    parser = argparse.ArgumentParser(description="Optional ONNX Runtime static INT8 PTQ for LayoutLMv3 ONNX.")
    parser.add_argument("--input_onnx", default="models/layoutlmv3-cord-onnx/fp32/model.onnx")
    parser.add_argument("--checkpoint_for_processor", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--cord_bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--cord_raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--calibration_split", default="train")
    parser.add_argument("--calibration_samples", type=int, default=100)
    parser.add_argument("--sample_split", default="validation")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out_dir", default="models/layoutlmv3-cord-onnx/int8_static")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--calibration_method", default="minmax", choices=("minmax", "entropy", "percentile"))
    parser.add_argument("--quant_format", default="qdq", choices=("qdq", "qoperator"))
    parser.add_argument("--activation_type", default="qint8", choices=("qint8", "quint8"))
    parser.add_argument("--weight_type", default="qint8", choices=("qint8", "quint8"))
    parser.add_argument("--per_channel", action="store_true")
    parser.add_argument("--op_types", default=None, help="Comma-separated op types, e.g. MatMul,Gemm.")
    parser.add_argument("--nodes_to_exclude", default=None, help="JSON or text file with node names to exclude.")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def prepare_out_dir(path, overwrite):
    out_dir = Path(path)
    if out_dir.exists():
        if not overwrite:
            fail(f"{out_dir} already exists. Use --overwrite to replace it.")
        print(f"Removing existing output directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def quant_type(name):
    return QuantType.QInt8 if name == "qint8" else QuantType.QUInt8


def quant_format(name):
    return QuantFormat.QDQ if name == "qdq" else QuantFormat.QOperator


def calibration_method(name):
    mapping = {
        "minmax": CalibrationMethod.MinMax,
        "entropy": CalibrationMethod.Entropy,
        "percentile": CalibrationMethod.Percentile,
    }
    return mapping[name]


def parse_op_types(value):
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


def load_nodes_to_exclude(path):
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        fail(f"nodes_to_exclude file not found: {path}")
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("nodes_to_exclude", payload.get("nodes", []))
        return [str(item) for item in payload]
    return [line.strip() for line in text.splitlines() if line.strip()]


def run_smoke(output_onnx, checkpoint, cord_bio_dir, cord_raw_data_dir, split, sample_index, max_length, local_files_only):
    labels_payload, _, label2id, _ = load_labels(checkpoint)
    records = load_cord_jsonl(cord_bio_dir, split, sample_index + 1)
    raw_dataset = load_raw_dataset(cord_raw_data_dir)
    sample = prepare_cord_sample(records[sample_index], raw_dataset, label2id, split)
    processor = AutoProcessor.from_pretrained(str(checkpoint), apply_ocr=False, local_files_only=local_files_only)
    encoding = processor(
        sample["image"],
        sample["words"],
        boxes=sample["normalized_boxes"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    session = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
    input_names = {item.name for item in session.get_inputs()}
    output_names = [output.name for output in session.get_outputs()]
    ort_outputs = session.run(None, {k: v for k, v in numpy_inputs_from_encoding(encoding).items() if k in input_names})
    outputs = dict(zip(output_names, ort_outputs))
    report = finite_report(outputs)
    passed = (
        "logits" in outputs
        and "last_hidden_state" in outputs
        and not report["logits"]["has_nan"]
        and not report["logits"]["has_inf"]
        and not report["last_hidden_state"]["has_nan"]
        and not report["last_hidden_state"]["has_inf"]
    )
    return labels_payload, {
        "sample_id": sample["id"],
        "providers": session.get_providers(),
        "input_names": sorted(input_names),
        "output_names": output_names,
        "finite_report": report,
        "logits_shape": list(outputs["logits"].shape),
        "last_hidden_state_shape": list(outputs["last_hidden_state"].shape),
        "passed": passed,
    }


def main():
    args = parse_args()
    input_onnx = Path(args.input_onnx)
    checkpoint = Path(args.checkpoint_for_processor)
    if not input_onnx.exists():
        fail(f"input ONNX not found: {input_onnx}. Run ONNX FP32 export first.")
    if args.local_files_only and not checkpoint.exists():
        fail(f"checkpoint_for_processor not found: {checkpoint}")
    out_dir = prepare_out_dir(args.out_dir, args.overwrite)
    output_onnx = out_dir / "model.onnx"
    raw_dataset = load_raw_dataset(args.cord_raw_data_dir)
    labels_payload, _, label2id, _ = load_labels(checkpoint)
    calibration_records = load_cord_jsonl(args.cord_bio_dir, args.calibration_split, args.calibration_samples)
    processor = AutoProcessor.from_pretrained(str(checkpoint), apply_ocr=False, local_files_only=args.local_files_only)
    input_names = [item.name for item in ort.InferenceSession(str(input_onnx), providers=["CPUExecutionProvider"]).get_inputs()]
    reader = LayoutLMv3CordCalibrationReader(
        calibration_records,
        raw_dataset,
        processor,
        label2id,
        args.calibration_split,
        max_length=args.max_length,
        input_names=input_names,
    )
    op_types = parse_op_types(args.op_types)
    nodes_to_exclude = load_nodes_to_exclude(args.nodes_to_exclude)
    print(f"input_onnx: {input_onnx}")
    print(f"output_onnx: {output_onnx}")
    print(f"calibration split: {args.calibration_split}")
    print(f"calibration samples: {len(calibration_records)}")
    print(f"quant_format: {args.quant_format}")
    print(f"activation_type: {args.activation_type}")
    print(f"weight_type: {args.weight_type}")
    print(f"per_channel: {args.per_channel}")
    print(f"op_types: {op_types}")
    print(f"nodes_to_exclude count: {len(nodes_to_exclude or [])}")
    quant_warning = None
    try:
        quantize_static(
            model_input=str(input_onnx),
            model_output=str(output_onnx),
            calibration_data_reader=reader,
            quant_format=quant_format(args.quant_format),
            op_types_to_quantize=op_types,
            per_channel=args.per_channel,
            activation_type=quant_type(args.activation_type),
            weight_type=quant_type(args.weight_type),
            nodes_to_exclude=nodes_to_exclude,
            calibrate_method=calibration_method(args.calibration_method),
            use_external_data_format=True,
        )
    except TypeError as exc:
        quant_warning = f"quantize_static fallback after TypeError: {exc}"
        print(f"WARNING: {quant_warning}")
        reader.rewind()
        quantize_static(
            model_input=str(input_onnx),
            model_output=str(output_onnx),
            calibration_data_reader=reader,
            quant_format=quant_format(args.quant_format),
            op_types_to_quantize=op_types,
            per_channel=args.per_channel,
            activation_type=quant_type(args.activation_type),
            weight_type=quant_type(args.weight_type),
            nodes_to_exclude=nodes_to_exclude,
            calibrate_method=calibration_method(args.calibration_method),
        )
    if not output_onnx.exists():
        fail("quantize_static did not create output model.")

    labels_payload, smoke = run_smoke(
        output_onnx,
        checkpoint,
        args.cord_bio_dir,
        args.cord_raw_data_dir,
        args.sample_split,
        args.sample_index,
        args.max_length,
        args.local_files_only,
    )
    input_size = onnx_total_size_mb(input_onnx)
    output_size = onnx_total_size_mb(output_onnx)
    reduction = ((input_size - output_size) / input_size * 100.0) if input_size and output_size else None
    quant_config = {
        "input_onnx": str(input_onnx),
        "output_onnx": str(output_onnx),
        "method": "static",
        "calibration_split": args.calibration_split,
        "calibration_samples_requested": args.calibration_samples,
        "calibration_samples_used": reader.num_returned,
        "calibration_method": args.calibration_method,
        "quant_format": args.quant_format,
        "activation_type": "QInt8" if args.activation_type == "qint8" else "QUInt8",
        "weight_type": "QInt8" if args.weight_type == "qint8" else "QUInt8",
        "per_channel": args.per_channel,
        "op_types_to_quantize": op_types,
        "nodes_to_exclude": nodes_to_exclude,
        "user_data_used": False,
        "source": "CORD-only",
        "input_size_mb": input_size,
        "output_size_mb": output_size,
        "output_main_file_size_mb": file_size_mb(output_onnx),
        "uses_external_data": Path(str(output_onnx) + ".data").exists(),
        "size_reduction_percent": reduction,
        "quant_warning": quant_warning,
    }
    processor.save_pretrained(out_dir)
    save_json(out_dir / "labels.json", labels_payload)
    save_json(out_dir / "quant_config.json", quant_config)
    save_json(out_dir / "calibration_summary.json", quant_config)
    save_json(out_dir / "smoke_test_report.json", smoke)
    print(f"input size MB: {input_size:.2f}")
    print(f"output size MB: {output_size:.2f}")
    print(f"size reduction percent: {reduction:.2f}")
    print(f"logits shape: {smoke['logits_shape']}")
    print(f"hidden shape: {smoke['last_hidden_state_shape']}")
    print(f"smoke passed: {smoke['passed']}")
    if not smoke["passed"]:
        fail("Static INT8 ONNX smoke test failed.")
    print("LayoutLMv3 ONNX static INT8 PTQ passed.")


if __name__ == "__main__":
    main()
