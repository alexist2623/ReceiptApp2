import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import onnx
import onnxruntime as ort
from onnxruntime.quantization import QuantType, quantize_dynamic
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.quantization.quant_common import (
    fail,
    finite_report,
    load_cord_jsonl,
    load_labels,
    load_raw_dataset,
    numpy_inputs_from_encoding,
    onnx_total_size_mb,
    prepare_cord_sample,
    save_json,
)


VARIANTS = [
    {
        "name": "matmul_gemm_only",
        "op_types": ["MatMul", "Gemm"],
        "per_channel": True,
        "weight_type": "qint8",
        "exclude_patterns": [],
    },
    {
        "name": "matmul_only",
        "op_types": ["MatMul"],
        "per_channel": True,
        "weight_type": "qint8",
        "exclude_patterns": [],
    },
    {
        "name": "gemm_only",
        "op_types": ["Gemm"],
        "per_channel": True,
        "weight_type": "qint8",
        "exclude_patterns": [],
    },
    {
        "name": "exclude_layernorm_embedding",
        "op_types": ["MatMul", "Gemm"],
        "per_channel": True,
        "weight_type": "qint8",
        "exclude_patterns": ["layernorm", "layer_norm", "embed", "embedding", "gather"],
    },
    {
        "name": "exclude_classifier",
        "op_types": ["MatMul", "Gemm"],
        "per_channel": True,
        "weight_type": "qint8",
        "exclude_patterns": ["classifier", "logits", "score"],
    },
    {
        "name": "safest_dynamic",
        "op_types": ["MatMul", "Gemm"],
        "per_channel": True,
        "weight_type": "qint8",
        "exclude_patterns": ["layernorm", "layer_norm", "embed", "embedding", "gather", "classifier", "logits", "score"],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run selective ONNX dynamic quantization variants.")
    parser.add_argument("--input_onnx", default="models/layoutlmv3-cord-onnx/fp32/model.onnx")
    parser.add_argument("--checkpoint_for_processor", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--cord_bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--cord_raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample_index", type=int, default=0)
    parser.add_argument("--out_root", default="models/layoutlmv3-cord-onnx/selective")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--run_eval", action="store_true")
    parser.add_argument("--eval_max_samples", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def prepare_out_root(path, overwrite):
    out_root = Path(path)
    if out_root.exists():
        if not overwrite:
            fail(f"{out_root} already exists. Use --overwrite to replace it.")
        print(f"Removing existing output directory: {out_root}")
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    return out_root


def quant_type(name):
    return QuantType.QInt8 if name == "qint8" else QuantType.QUInt8


def node_search_blob(node):
    return " ".join([node.name, node.op_type, *node.input, *node.output]).lower()


def inspect_graph(input_onnx):
    model = onnx.load(str(input_onnx), load_external_data=False)
    op_counts = Counter(node.op_type for node in model.graph.node)
    nodes = []
    for node in model.graph.node:
        nodes.append(
            {
                "name": node.name,
                "op_type": node.op_type,
                "inputs": list(node.input),
                "outputs": list(node.output),
            }
        )
    return {
        "num_nodes": len(nodes),
        "op_counts": dict(sorted(op_counts.items())),
        "nodes": nodes,
    }


def matching_nodes(graph_info, patterns):
    if not patterns:
        return []
    matches = []
    for node in graph_info["nodes"]:
        blob = " ".join([node["name"], node["op_type"], *node["inputs"], *node["outputs"]]).lower()
        if any(pattern.lower() in blob for pattern in patterns):
            if node["name"]:
                matches.append(node["name"])
    return sorted(set(matches))


def run_quantize(input_onnx, output_onnx, variant, nodes_to_exclude):
    kwargs = {
        "model_input": str(input_onnx),
        "model_output": str(output_onnx),
        "weight_type": quant_type(variant["weight_type"]),
        "per_channel": bool(variant["per_channel"]),
        "op_types_to_quantize": variant["op_types"],
    }
    if nodes_to_exclude:
        kwargs["nodes_to_exclude"] = nodes_to_exclude
    try:
        quantize_dynamic(**kwargs)
        return None
    except TypeError as exc:
        warning = f"quantize_dynamic fallback without nodes_to_exclude after TypeError: {exc}"
        print(f"WARNING: {warning}")
        kwargs.pop("nodes_to_exclude", None)
        quantize_dynamic(**kwargs)
        return warning


def smoke_test(output_onnx, checkpoint, cord_bio_dir, cord_raw_data_dir, split, sample_index, max_length, local_files_only):
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
    outputs = dict(
        zip(
            output_names,
            session.run(None, {key: value for key, value in numpy_inputs_from_encoding(encoding).items() if key in input_names}),
        )
    )
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


def run_eval_for_variant(args, output_onnx, variant_dir):
    out_dir = variant_dir / "eval"
    cmd = [
        sys.executable,
        "scripts/quantization/eval_layoutlmv3_cord_onnx.py",
        "--onnx_model",
        str(output_onnx),
        "--checkpoint_for_processor",
        str(args.checkpoint_for_processor),
        "--cord_bio_dir",
        str(args.cord_bio_dir),
        "--cord_raw_data_dir",
        str(args.cord_raw_data_dir),
        "--split",
        str(args.split),
        "--max_samples",
        str(args.eval_max_samples),
        "--out_dir",
        str(out_dir),
    ]
    if args.local_files_only:
        cmd.append("--local_files_only")
    result = subprocess.run(cmd, cwd=ROOT_DIR, text=True, capture_output=True)
    return {
        "command": cmd,
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "metrics_path": str(out_dir / f"metrics_{args.split}.json"),
    }


def main():
    args = parse_args()
    input_onnx = Path(args.input_onnx)
    checkpoint = Path(args.checkpoint_for_processor)
    if not input_onnx.exists():
        fail(f"input ONNX not found: {input_onnx}. Run ONNX FP32 export first.")
    if args.local_files_only and not checkpoint.exists():
        fail(f"checkpoint_for_processor not found: {checkpoint}")
    out_root = prepare_out_root(args.out_root, args.overwrite)
    graph_info = inspect_graph(input_onnx)
    input_size = onnx_total_size_mb(input_onnx)
    save_json(out_root / "onnx_graph_inspection.json", graph_info)
    labels_payload = None
    rows = []
    for variant in VARIANTS:
        name = variant["name"]
        variant_dir = out_root / name
        variant_dir.mkdir(parents=True, exist_ok=True)
        output_onnx = variant_dir / "model.onnx"
        nodes_to_exclude = matching_nodes(graph_info, variant["exclude_patterns"])
        print(f"Running variant: {name}")
        print(f"  op_types: {variant['op_types']}")
        print(f"  exclude nodes: {len(nodes_to_exclude)}")
        warning = run_quantize(input_onnx, output_onnx, variant, nodes_to_exclude)
        if not output_onnx.exists():
            fail(f"Variant {name} did not create output ONNX.")
        labels_payload, smoke = smoke_test(
            output_onnx,
            checkpoint,
            args.cord_bio_dir,
            args.cord_raw_data_dir,
            args.split,
            args.sample_index,
            args.max_length,
            args.local_files_only,
        )
        output_size = onnx_total_size_mb(output_onnx)
        reduction = ((input_size - output_size) / input_size * 100.0) if input_size and output_size else None
        eval_result = run_eval_for_variant(args, output_onnx, variant_dir) if args.run_eval else None
        quant_config = {
            "variant": name,
            "input_onnx": str(input_onnx),
            "output_onnx": str(output_onnx),
            "method": "dynamic_selective",
            "op_types_to_quantize": variant["op_types"],
            "weight_type": "QInt8" if variant["weight_type"] == "qint8" else "QUInt8",
            "per_channel": variant["per_channel"],
            "exclude_patterns": variant["exclude_patterns"],
            "nodes_to_exclude": nodes_to_exclude,
            "nodes_to_exclude_count": len(nodes_to_exclude),
            "input_size_mb": input_size,
            "output_size_mb": output_size,
            "size_reduction_percent": reduction,
            "user_data_used": False,
            "source": "CORD-only",
            "quant_warning": warning,
            "eval_result": eval_result,
        }
        save_json(variant_dir / "quant_config.json", quant_config)
        save_json(variant_dir / "smoke_test_report.json", smoke)
        if labels_payload is not None:
            save_json(variant_dir / "labels.json", labels_payload)
        rows.append(
            {
                "variant": name,
                "model_path": str(output_onnx),
                "op_types_to_quantize": variant["op_types"],
                "nodes_to_exclude_count": len(nodes_to_exclude),
                "input_size_mb": input_size,
                "output_size_mb": output_size,
                "size_reduction_percent": reduction,
                "smoke_passed": smoke["passed"],
                "eval_metrics_path": eval_result["metrics_path"] if eval_result else None,
            }
        )
        print(f"  size MB: {output_size:.2f}")
        print(f"  smoke passed: {smoke['passed']}")
        if not smoke["passed"]:
            fail(f"Variant {name} smoke test failed.")
    summary = {
        "input_onnx": str(input_onnx),
        "out_root": str(out_root),
        "input_size_mb": input_size,
        "variants": rows,
        "graph_op_counts": graph_info["op_counts"],
        "recommended_eval_order": [
            "safest_dynamic",
            "matmul_gemm_only",
            "exclude_classifier",
            "exclude_layernorm_embedding",
            "matmul_only",
            "gemm_only",
        ],
        "selection_note": "Final choice should be based on end-to-end item_price_pair_f1, not only model size.",
    }
    save_json(out_root / "selective_summary.json", summary)
    if labels_payload is not None:
        save_json(out_root / "labels.json", labels_payload)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:5000])
    print(f"summary path: {out_root / 'selective_summary.json'}")
    print("Selective quantization experiments passed.")


if __name__ == "__main__":
    main()
