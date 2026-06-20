import argparse
import json
from pathlib import Path


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value):
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def metric(payload, key):
    if not payload:
        return None
    return payload.get(key)


def parse_args():
    parser = argparse.ArgumentParser(description="Summarize custom LayoutLMv3 ONNX/INT8 quantization comparison.")
    parser.add_argument("--pytorch_e2e_metrics", default="outputs/custom_item_policy_predicted_overlay/metrics_summary.json")
    parser.add_argument("--onnx_fp32_metrics", default="outputs/quantization/custom_item_policy_onnx_fp32/metrics_all.json")
    parser.add_argument("--onnx_int8_metrics", default="outputs/quantization/custom_item_policy_onnx_int8_dynamic/metrics_all.json")
    parser.add_argument("--onnx_fp32_e2e_metrics", default="outputs/quantization/custom_item_policy_onnx_fp32_e2e/metrics_summary.json")
    parser.add_argument("--onnx_int8_e2e_metrics", default="outputs/quantization/custom_item_policy_onnx_int8_dynamic_e2e/metrics_summary.json")
    parser.add_argument("--fp32_export_config", default="models/layoutlmv3-item-policy-onnx/fp32/export_config.json")
    parser.add_argument("--int8_quant_config", default="models/layoutlmv3-item-policy-onnx/int8_dynamic/quant_config.json")
    parser.add_argument("--out_dir", default="outputs/quantization/custom_item_policy_comparison")
    return parser.parse_args()


def row(name, payload, fields):
    return {"name": name, **{field: metric(payload, field) for field in fields}}


def main():
    args = parse_args()
    pytorch_e2e = load_json(args.pytorch_e2e_metrics)
    onnx_fp32 = load_json(args.onnx_fp32_metrics)
    onnx_int8 = load_json(args.onnx_int8_metrics)
    onnx_fp32_e2e = load_json(args.onnx_fp32_e2e_metrics)
    onnx_int8_e2e = load_json(args.onnx_int8_e2e_metrics)
    fp32_export = load_json(args.fp32_export_config)
    int8_quant = load_json(args.int8_quant_config)
    token_fields = ["num_samples", "seqeval_f1", "token_accuracy", "word_label_accuracy"]
    e2e_fields = [
        "num_samples",
        "word_label_accuracy",
        "menu_price_pair_precision",
        "menu_price_pair_recall",
        "menu_price_pair_f1",
        "summary_amount_pair_f1",
        "hard_negative_false_positive_count",
        "total_subtotal_false_positive_count",
        "dependent_collision_count",
    ]
    report = {
        "model_size": {
            "fp32_size_mb": (fp32_export or {}).get("model_total_size_mb"),
            "int8_size_mb": (int8_quant or {}).get("output_size_mb"),
            "size_reduction_percent": (int8_quant or {}).get("size_reduction_percent"),
        },
        "layoutlmv3_token_word_metrics": [
            row("ONNX_FP32", onnx_fp32, token_fields),
            row("ONNX_INT8_DYNAMIC", onnx_int8, token_fields),
        ],
        "e2e_relg_metrics": [
            row("PyTorch_FP32_baseline", pytorch_e2e, e2e_fields),
            row("ONNX_FP32", onnx_fp32_e2e, e2e_fields),
            row("ONNX_INT8_DYNAMIC", onnx_int8_e2e, e2e_fields),
        ],
    }
    if onnx_fp32 and onnx_int8:
        report["layoutlmv3_int8_drop_vs_onnx_fp32"] = {
            "seqeval_f1": onnx_fp32.get("seqeval_f1") - onnx_int8.get("seqeval_f1"),
            "word_label_accuracy": onnx_fp32.get("word_label_accuracy") - onnx_int8.get("word_label_accuracy"),
        }
    if onnx_fp32_e2e and onnx_int8_e2e:
        report["e2e_int8_drop_vs_onnx_fp32"] = {
            "menu_price_pair_f1": onnx_fp32_e2e.get("menu_price_pair_f1") - onnx_int8_e2e.get("menu_price_pair_f1"),
            "word_label_accuracy": onnx_fp32_e2e.get("word_label_accuracy") - onnx_int8_e2e.get("word_label_accuracy"),
        }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "quantization_comparison.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# Custom Item Policy Quantization Comparison",
        "",
        "## Model Size",
        f"- FP32 ONNX MB: {fmt(report['model_size']['fp32_size_mb'])}",
        f"- INT8 dynamic ONNX MB: {fmt(report['model_size']['int8_size_mb'])}",
        f"- size reduction percent: {fmt(report['model_size']['size_reduction_percent'])}",
        "",
        "## LayoutLMv3 Token/Word Metrics",
        "| model | samples | seqeval_f1 | token_accuracy | word_label_accuracy |",
        "|---|---:|---:|---:|---:|",
    ]
    for item in report["layoutlmv3_token_word_metrics"]:
        lines.append(
            f"| {item['name']} | {fmt(item.get('num_samples'))} | {fmt(item.get('seqeval_f1'))} | {fmt(item.get('token_accuracy'))} | {fmt(item.get('word_label_accuracy'))} |"
        )
    lines.extend(["", "## E2E Rel-G Metrics", "| model | samples | word_acc | menu_price_P | menu_price_R | menu_price_F1 | summary_F1 | hard_neg_fp | collision |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for item in report["e2e_relg_metrics"]:
        lines.append(
            f"| {item['name']} | {fmt(item.get('num_samples'))} | {fmt(item.get('word_label_accuracy'))} | {fmt(item.get('menu_price_pair_precision'))} | {fmt(item.get('menu_price_pair_recall'))} | {fmt(item.get('menu_price_pair_f1'))} | {fmt(item.get('summary_amount_pair_f1'))} | {fmt(item.get('hard_negative_false_positive_count'))} | {fmt(item.get('dependent_collision_count'))} |"
        )
    if "layoutlmv3_int8_drop_vs_onnx_fp32" in report:
        drop = report["layoutlmv3_int8_drop_vs_onnx_fp32"]
        lines.extend(["", "## INT8 Drop vs ONNX FP32", f"- LayoutLMv3 seqeval_f1 drop: {fmt(drop.get('seqeval_f1'))}", f"- LayoutLMv3 word_label_accuracy drop: {fmt(drop.get('word_label_accuracy'))}"])
    if "e2e_int8_drop_vs_onnx_fp32" in report:
        drop = report["e2e_int8_drop_vs_onnx_fp32"]
        lines.extend([f"- E2E menu_price_pair_f1 drop: {fmt(drop.get('menu_price_pair_f1'))}", f"- E2E word_label_accuracy drop: {fmt(drop.get('word_label_accuracy'))}"])
    (out_dir / "quantization_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print((out_dir / "quantization_comparison.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
