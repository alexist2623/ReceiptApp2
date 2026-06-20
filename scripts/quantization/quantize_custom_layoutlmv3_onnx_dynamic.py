import argparse
import shutil
import sys
from pathlib import Path

import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.quantization.custom_quant_common import encode_custom_sample, load_custom_records, prepare_custom_sample
from scripts.quantization.quant_common import (
    fail,
    file_size_mb,
    finite_report,
    load_labels,
    numpy_inputs_from_encoding,
    onnx_total_size_mb,
    save_json,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Apply dynamic INT8 quantization to custom LayoutLMv3 ONNX.")
    parser.add_argument("--input_onnx", default="models/layoutlmv3-item-policy-onnx/fp32/model.onnx")
    parser.add_argument("--checkpoint_for_processor", default="models/layoutlmv3-item-policy-mixed-100/best")
    parser.add_argument("--input_dir", required=True, nargs="+")
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out_dir", default="models/layoutlmv3-item-policy-onnx/int8_dynamic")
    parser.add_argument("--weight_type", default="qint8", choices=("qint8", "quint8"))
    parser.add_argument("--per_channel", action="store_true")
    parser.add_argument("--disable_per_channel", action="store_true")
    parser.add_argument("--op_types", default=None)
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


def parse_op_types(value):
    if not value:
        return None
    return [part.strip() for part in value.split(",") if part.strip()]


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
    op_types = parse_op_types(args.op_types)
    use_per_channel = bool(args.per_channel and not args.disable_per_channel)
    quant_warning = None
    print(f"input_onnx: {input_onnx}")
    print(f"output_onnx: {output_onnx}")
    print(f"weight_type: {args.weight_type}")
    print(f"per_channel: {use_per_channel}")
    print(f"op_types: {op_types}")
    try:
        quantize_dynamic(
            model_input=str(input_onnx),
            model_output=str(output_onnx),
            weight_type=quant_type(args.weight_type),
            per_channel=use_per_channel,
            op_types_to_quantize=op_types,
        )
    except TypeError as exc:
        quant_warning = f"dynamic quantization fallback after TypeError: {exc}"
        print(f"WARNING: {quant_warning}")
        quantize_dynamic(model_input=str(input_onnx), model_output=str(output_onnx), weight_type=quant_type(args.weight_type))
    if not output_onnx.exists():
        fail("quantize_dynamic did not create output model.")

    labels_payload, _, label2id, _ = load_labels(checkpoint)
    records, excluded = load_custom_records(args.input_dir, args.exclude_dir_name)
    sample = prepare_custom_sample(records[args.sample_index], label2id)
    processor = AutoProcessor.from_pretrained(str(checkpoint), apply_ocr=False, local_files_only=args.local_files_only)
    encoding = encode_custom_sample(processor, sample, include_labels=False)
    session = ort.InferenceSession(str(output_onnx), providers=["CPUExecutionProvider"])
    output_names = [output.name for output in session.get_outputs()]
    ort_outputs = session.run(None, numpy_inputs_from_encoding(encoding))
    outputs = dict(zip(output_names, ort_outputs))
    finite = finite_report(outputs)
    smoke = {
        "sample_id": sample["id"],
        "providers": session.get_providers(),
        "output_names": output_names,
        "finite_report": finite,
        "logits_shape": list(outputs["logits"].shape),
        "last_hidden_state_shape": list(outputs["last_hidden_state"].shape),
        "passed": "logits" in outputs
        and "last_hidden_state" in outputs
        and not finite["logits"]["has_nan"]
        and not finite["last_hidden_state"]["has_nan"],
    }
    input_size = onnx_total_size_mb(input_onnx)
    output_size = onnx_total_size_mb(output_onnx)
    reduction = ((input_size - output_size) / input_size * 100.0) if input_size and output_size else None
    processor.save_pretrained(out_dir)
    save_json(out_dir / "labels.json", labels_payload)
    save_json(
        out_dir / "quant_config.json",
        {
            "input_onnx": str(input_onnx),
            "output_onnx": str(output_onnx),
            "method": "dynamic",
            "weight_type": "QInt8" if args.weight_type == "qint8" else "QUInt8",
            "per_channel": use_per_channel,
            "op_types_to_quantize": op_types,
            "source": "custom-labeled-receipts",
            "user_data_used": True,
            "excluded_dir_name": args.exclude_dir_name,
            "excluded_count": len(excluded),
            "input_size_mb": input_size,
            "output_size_mb": output_size,
            "output_main_file_size_mb": file_size_mb(output_onnx),
            "uses_external_data": Path(str(output_onnx) + ".data").exists(),
            "size_reduction_percent": reduction,
            "quant_warning": quant_warning,
        },
    )
    save_json(out_dir / "smoke_test_report.json", smoke)
    save_json(out_dir / "model_size_report.json", {"input_size_mb": input_size, "output_size_mb": output_size, "size_reduction_percent": reduction})
    print(f"input size MB: {input_size:.2f}")
    print(f"output size MB: {output_size:.2f}")
    print(f"size reduction percent: {reduction:.2f}")
    print(f"logits shape: {smoke['logits_shape']}")
    print(f"hidden shape: {smoke['last_hidden_state_shape']}")
    print(f"smoke passed: {smoke['passed']}")
    if not smoke["passed"]:
        fail("Custom dynamic INT8 ONNX smoke test failed.")
    print("Custom LayoutLMv3 ONNX dynamic INT8 quantization passed.")


if __name__ == "__main__":
    main()
