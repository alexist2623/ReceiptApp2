import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Write an oracle-span rel-g evaluation report.")
    parser.add_argument("--metrics", default="outputs/span_relg_eval/metrics_summary.json")
    parser.add_argument("--sweep", default="outputs/span_relg_eval_sweep/threshold_sweep.json")
    parser.add_argument("--run_config", default="outputs/span_relg_eval/run_config.json")
    parser.add_argument("--overlay_dir", default="outputs/span_relg_overlay")
    parser.add_argument("--out", default="outputs/span_relg_eval/oracle_span_report.md")
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def metric(metrics, key):
    value = metrics.get(key)
    return "n/a" if value is None else f"{value:.6f}" if isinstance(value, float) else str(value)


def verdict(metrics):
    f1 = float(metrics.get("menu_price_pair_f1", 0.0))
    hard = int(metrics.get("hard_negative_false_positive_count", 0))
    collisions = int(metrics.get("dependent_collision_count", 0))
    if f1 >= 0.90:
        base = "Oracle span rel-g parser is very strong for MENU_NM -> MENU_PRICE."
    elif f1 >= 0.80:
        base = "Oracle span rel-g parser is usable, but error analysis is needed."
    else:
        base = "Oracle span rel-g parser needs model, feature, or target generation review."
    notes = []
    if hard:
        notes.append("Hard negative total/subtotal false positives remain and should be monitored.")
    if collisions:
        notes.append("Dependent collisions are present; threshold or decoding collision avoidance may need tuning.")
    return base + (" " + " ".join(notes) if notes else "")


def markdown_table(rows):
    lines = ["| threshold | edge F1 | menu-price precision | menu-price recall | menu-price F1 | hard FP | collisions |", "|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("threshold")),
                    metric(row, "edge_f1"),
                    metric(row, "menu_price_pair_precision"),
                    metric(row, "menu_price_pair_recall"),
                    metric(row, "menu_price_pair_f1"),
                    str(row.get("hard_negative_false_positive_count")),
                    str(row.get("dependent_collision_count")),
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    metrics = load_json(args.metrics)
    sweep = load_json(args.sweep) if Path(args.sweep).exists() else {"thresholds": [], "best_threshold": None}
    run_config = load_json(args.run_config) if Path(args.run_config).exists() else {}
    overlay_dir = Path(args.overlay_dir)
    overlays = sorted(str(path) for path in overlay_dir.glob("*.png"))[:6]
    best = sweep.get("best_threshold") or {}

    lines = [
        "# Oracle Span Rel-G Evaluation Report",
        "",
        f"- dataset_dir: `{run_config.get('dataset_dir', metrics.get('dataset_dir'))}`",
        f"- checkpoint: `{run_config.get('checkpoint', metrics.get('checkpoint'))}`",
        f"- resolved cache path: `{run_config.get('resolved_split_cache_path', 'n/a')}`",
        f"- resolved config path: `{run_config.get('resolved_config_path', 'n/a')}`",
        f"- test sample count: {metrics.get('num_samples')}",
        "",
        f"## Threshold {metrics.get('threshold', 'selected')} Metrics",
        "",
        f"- edge precision/recall/F1: {metric(metrics, 'edge_precision')} / {metric(metrics, 'edge_recall')} / {metric(metrics, 'edge_f1')}",
        f"- MENU_NM -> MENU_PRICE precision/recall/F1: {metric(metrics, 'menu_price_pair_precision')} / {metric(metrics, 'menu_price_pair_recall')} / {metric(metrics, 'menu_price_pair_f1')}",
        f"- hard negative false positives: {metrics.get('hard_negative_false_positive_count')}",
        f"- dependent collisions: {metrics.get('dependent_collision_count')}",
        f"- no_price_item_count: {metrics.get('no_price_item_count')}",
        f"- multiple_price_item_count: {metrics.get('multiple_price_item_count')}",
        "",
        "## Threshold Sweep",
        "",
        markdown_table(sweep.get("thresholds", [])),
        "",
        "## Best Threshold",
        "",
        f"- best threshold: {best.get('threshold')}",
        f"- best MENU_NM -> MENU_PRICE F1: {metric(best, 'menu_price_pair_f1')}",
        f"- best precision/recall: {metric(best, 'menu_price_pair_precision')} / {metric(best, 'menu_price_pair_recall')}",
        "",
        "## Error Notes",
        "",
        f"- total/subtotal attached as menu price: {'yes' if metrics.get('hard_negative_false_positive_count', 0) else 'no'}",
        f"- hard negative false positive count: {metrics.get('hard_negative_false_positive_count')}",
        f"- dependent collision count: {metrics.get('dependent_collision_count')}",
        "",
        "## Overlay Examples",
        "",
    ]
    lines.extend(f"- `{path}`" for path in overlays)
    lines.extend(["", "## Verdict", "", verdict(metrics), ""])

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"oracle report path: {out}")


if __name__ == "__main__":
    main()
