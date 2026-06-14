import argparse
import csv
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.quantization.quant_common import save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize CORD-only LayoutLMv3 quantization outputs.")
    parser.add_argument("--out_dir", default="outputs/quantization/reports")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_optional(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def field_f1(metrics, field):
    return (metrics or {}).get("field_metrics", {}).get(field, {}).get("f1")


def row_for(name, model_path, metrics, notes, relg=None):
    return {
        "model_variant": name,
        "model_path": model_path,
        "model_size_mb": (metrics or {}).get("model_size_mb"),
        "split": (metrics or {}).get("split"),
        "num_samples": (metrics or {}).get("num_samples"),
        "token_seqeval_f1": (metrics or {}).get("seqeval_f1"),
        "ITEM_NAME_f1": field_f1(metrics, "ITEM_NAME"),
        "ITEM_PRICE_f1": field_f1(metrics, "ITEM_PRICE"),
        "TOTAL_PRICE_f1": field_f1(metrics, "TOTAL_PRICE"),
        "avg_latency_ms": (metrics or {}).get("latency", {}).get("avg_ms") if metrics else None,
        "p95_latency_ms": (metrics or {}).get("latency", {}).get("p95_ms") if metrics else None,
        "relg_item_price_pair_f1": (relg or {}).get("item_price_pair_f1"),
        "relg_hard_negative_fp": (relg or {}).get("hard_negative_false_positive_count"),
        "notes": notes,
    }


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_md(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# CORD-only Quantization Summary\n\n")
        if not rows:
            handle.write("No rows found.\n")
            return
        handle.write("| " + " | ".join(headers) + " |\n")
        handle.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            handle.write("| " + " | ".join(str(row.get(h, "")) for h in headers) + " |\n")


def main():
    args = parse_args()
    split = args.split
    pytorch = load_optional(f"outputs/quantization/cord_baseline_pytorch_fp32/metrics_{split}.json")
    onnx_fp32 = load_optional(f"outputs/quantization/cord_onnx_fp32/metrics_{split}.json")
    onnx_int8 = load_optional(f"outputs/quantization/cord_onnx_int8_dynamic/metrics_{split}.json")
    relg_int8 = load_optional(f"outputs/quantization/cord_relg_hidden_impact/relg_metrics_onnx_int8_{split}.json")
    rows = [
        row_for("pytorch_fp32", "models/layoutlmv3-cord-full/best", pytorch, "Baseline CORD-only PyTorch FP32"),
        row_for("onnx_fp32", "models/layoutlmv3-cord-onnx/fp32/model.onnx", onnx_fp32, "ONNX FP32 export"),
        row_for("onnx_int8_dynamic", "models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx", onnx_int8, "Dynamic INT8", relg=relg_int8),
    ]
    out_dir = Path(args.out_dir)
    payload = {"split": split, "rows": rows}
    save_json(out_dir / "quantization_summary.json", payload)
    write_csv(out_dir / "quantization_summary.csv", rows)
    write_md(out_dir / "quantization_summary.md", rows)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"summary path: {out_dir / 'quantization_summary.json'}")


if __name__ == "__main__":
    main()
