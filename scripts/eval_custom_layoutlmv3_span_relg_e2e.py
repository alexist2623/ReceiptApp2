import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.bio_repair import repair_bio_boundaries
from ml.angle_geometry import build_angle_features_for_words, angle_feature_dim_for_mode
from ml.metrics.field_metrics import compute_field_metrics
from ml.receipt_schema import canonicalize_field, canonicalize_label
from ml.span_relg.decode import decode_edges_to_items
from ml.span_relg.feature_cache import build_cache_sample
from ml.span_relg.io_utils import load_json, resolve_field_vocab
from ml.span_relg.metrics import aggregate_metrics
from ml.span_relg.span_utils import bio_predictions_to_spans
from ml.span_relg.visualization import draw_grouped_relation_mapping_overlay, draw_span_relg_overlay
from scripts.build_user_span_relg_dataset import (
    assign_group_keys_from_relations,
    collect_label_pairs,
    make_sample_info,
)
from scripts.eval_predicted_span_relg_e2e import (
    THRESHOLDS,
    attach_boxes_to_predictions,
    classify_edges,
    edge_payload,
    filter_candidate_spans,
    gold_probs,
    load_layoutlmv3,
    load_rel_model,
    resolve_kind_vocab,
    run_layout_prediction,
    run_rel_model,
    simple_item,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate custom receipt LayoutLMv3 predicted spans through span rel-g.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--layoutlm_checkpoint", default="models/layoutlmv3-mixed-public-user/best")
    parser.add_argument("--relg_checkpoint", default="models/span-relg-context/best")
    parser.add_argument("--span_relg_dataset_dir", default="processed_data/user_span_relg_non_temp")
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--split", default="all", choices=("train", "validation", "test", "all"))
    parser.add_argument("--label_schema", default="schemas/receipt_labels_v2.json")
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", default="outputs/custom_predicted_span_relg_e2e")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--sweep_thresholds", action="store_true")
    parser.add_argument("--save_overlay_limit", type=int, default=30)
    parser.add_argument("--repair_bio_boundaries", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def append_jsonl(handle, payload):
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def load_manifest_ids(path, split):
    if not path:
        return None
    payload = load_json(path)
    splits = payload.get("splits", payload) if isinstance(payload, dict) else {}
    if split == "all":
        ids = set()
        if isinstance(splits, dict):
            for values in splits.values():
                for item in values if isinstance(values, list) else []:
                    ids.add(str(item.get("id") if isinstance(item, dict) else item))
        return ids or None
    values = splits.get(split) if isinstance(splits, dict) else None
    if not isinstance(values, list):
        return set()
    return {str(item.get("id") if isinstance(item, dict) else item) for item in values}


def select_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def assign_predicted_span_groups(predicted_spans, payload, word_original_indices):
    relation_spans = []
    for span in predicted_spans:
        clone = dict(span)
        clone["word_indices"] = [
            word_original_indices[idx]
            for idx in span.get("word_indices", [])
            if idx < len(word_original_indices)
        ]
        relation_spans.append(clone)
    assigned = assign_group_keys_from_relations(relation_spans, payload)
    group_by_span_id = {span.get("span_id"): span.get("group_key") for span in relation_spans}
    for span in predicted_spans:
        span["group_key"] = group_by_span_id.get(span.get("span_id"))
    return assigned


def canonicalize_prediction_labels(predictions):
    for prediction in predictions:
        raw_label = prediction.get("label")
        prediction["raw_label"] = raw_label
        prediction["label"] = canonicalize_label(raw_label)
    return predictions


def gold_label_for_predictions(sample_info):
    return [prediction.get("label", "O") for prediction in sample_info.get("predictions", [])]


def summarize(samples, probs_by_sample, threshold, word_counts, args, device, field_metrics):
    metrics = aggregate_metrics(samples, probs_by_sample, threshold)
    labels = []
    for sample in samples:
        labels.extend(int(value.item()) for value in sample["pair_labels"])
    no_price_count = 0
    multiple_price_count = 0
    for sample, probs in zip(samples, probs_by_sample):
        decoded = decode_edges_to_items(sample, probs, threshold=threshold)
        for item in decoded.get("items", []):
            if item.get("price") is None:
                no_price_count += 1
            price_edges = [edge for edge in item.get("rel_g_edges", []) if canonicalize_field(edge.get("dep_field")) == "ITEM_PRICE"]
            if len(price_edges) > 1:
                multiple_price_count += 1
    return {
        "num_samples": len(samples),
        "threshold": threshold,
        "num_candidate_pairs": len(labels),
        "num_positive_pairs": sum(labels),
        "num_negative_pairs": len(labels) - sum(labels),
        "word_label_accuracy": word_counts["correct"] / word_counts["total"] if word_counts["total"] else 0.0,
        "field_metrics": field_metrics,
        "edge_precision": metrics["edge"]["precision"],
        "edge_recall": metrics["edge"]["recall"],
        "edge_f1": metrics["edge"]["f1"],
        "menu_price_pair_precision": metrics["menu_price_pair"]["precision"],
        "menu_price_pair_recall": metrics["menu_price_pair"]["recall"],
        "menu_price_pair_f1": metrics["menu_price_pair"]["f1"],
        "summary_amount_pair_precision": metrics["summary_amount_pair"]["precision"],
        "summary_amount_pair_recall": metrics["summary_amount_pair"]["recall"],
        "summary_amount_pair_f1": metrics["summary_amount_pair"]["f1"],
        "hard_negative_false_positive_count": metrics["hard_negative_false_positive_count"],
        "total_subtotal_false_positive_count": metrics["total_subtotal_false_positive_count"],
        "tax_false_positive_count": metrics.get("tax_false_positive_count", 0),
        "dependent_collision_count": metrics["dependent_collision_count"],
        "no_price_item_count": no_price_count,
        "multiple_price_item_count": multiple_price_count,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "layoutlm_checkpoint": args.layoutlm_checkpoint,
        "relg_checkpoint": args.relg_checkpoint,
        "nested_metrics": metrics,
    }


def threshold_sweep(samples, probs_by_sample):
    rows = []
    best = None
    for threshold in THRESHOLDS:
        metrics = aggregate_metrics(samples, probs_by_sample, threshold)
        row = {
            "threshold": threshold,
            "edge_precision": metrics["edge"]["precision"],
            "edge_recall": metrics["edge"]["recall"],
            "edge_f1": metrics["edge"]["f1"],
            "menu_price_pair_precision": metrics["menu_price_pair"]["precision"],
            "menu_price_pair_recall": metrics["menu_price_pair"]["recall"],
            "menu_price_pair_f1": metrics["menu_price_pair"]["f1"],
            "hard_negative_false_positive_count": metrics["hard_negative_false_positive_count"],
            "dependent_collision_count": metrics["dependent_collision_count"],
        }
        rows.append(row)
        if (
            best is None
            or row["menu_price_pair_f1"] > best["menu_price_pair_f1"]
            or (
                row["menu_price_pair_f1"] == best["menu_price_pair_f1"]
                and row["menu_price_pair_precision"] > best["menu_price_pair_precision"]
            )
            or (
                row["menu_price_pair_f1"] == best["menu_price_pair_f1"]
                and row["menu_price_pair_precision"] == best["menu_price_pair_precision"]
                and row["hard_negative_false_positive_count"] < best["hard_negative_false_positive_count"]
            )
        ):
            best = row
    return {"thresholds": rows, "best_threshold": best}


def write_gallery(out_dir, cards):
    rows = []
    for card in cards:
        image_rel = html.escape(str(Path("overlay") / Path(card["image"]).name))
        debug_rel = html.escape(str(Path("overlay") / Path(card["debug_json"]).name))
        rows.append(
            "<article>"
            f"<h2>{html.escape(card['data_id'])}</h2>"
            f"<p>pairs={card['num_pairs']} positives={card['num_positive_pairs']} hard_neg={card['hard_negative_count']}</p>"
            f"<a href='{debug_rel}'>debug JSON</a><br><img src='{image_rel}' alt='{html.escape(card['data_id'])} overlay'>"
            "</article>"
        )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Custom Predicted Span Rel-G</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}article{margin-bottom:32px;border-bottom:1px solid #ddd;padding-bottom:24px}"
        "img{max-width:100%;height:auto;border:1px solid #ccc}</style></head><body>"
        "<h1>Custom Predicted Span Rel-G Gallery</h1>"
        + "\n".join(rows)
        + "</body></html>"
    )
    (Path(out_dir) / "index.html").write_text(page, encoding="utf-8")


def write_report(out_dir, summary, sweep):
    lines = [
        "# Custom Predicted Span Rel-G E2E Report",
        "",
        f"- samples: {summary['num_samples']}",
        f"- threshold: {summary['threshold']}",
        f"- word_label_accuracy: {summary['word_label_accuracy']:.6f}",
        f"- menu_price_pair_precision: {summary['menu_price_pair_precision']:.6f}",
        f"- menu_price_pair_recall: {summary['menu_price_pair_recall']:.6f}",
        f"- menu_price_pair_f1: {summary['menu_price_pair_f1']:.6f}",
        f"- summary_amount_pair_f1: {summary['summary_amount_pair_f1']:.6f}",
        f"- hard_negative_false_positive_count: {summary['hard_negative_false_positive_count']}",
        f"- dependent_collision_count: {summary['dependent_collision_count']}",
    ]
    if sweep:
        best = sweep.get("best_threshold") or {}
        lines.extend(["", "## Best Threshold", f"- threshold: {best.get('threshold')}", f"- menu_price_pair_f1: {best.get('menu_price_pair_f1')}"])
    (Path(out_dir) / "custom_e2e_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    for path, label in [
        (args.input_dir, "input_dir"),
        (args.layoutlm_checkpoint, "layoutlm_checkpoint"),
        (args.relg_checkpoint, "relg_checkpoint"),
        (args.span_relg_dataset_dir, "span_relg_dataset_dir"),
        (args.label_schema, "label_schema"),
    ]:
        if not Path(path).exists():
            fail(f"{label} not found: {path}")
    device = select_device(args.device)
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"selected device: {device}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    records, excluded = collect_label_pairs(args.input_dir, args.exclude_dir_name)
    manifest_ids = load_manifest_ids(args.split_manifest, args.split)
    if manifest_ids is not None:
        records = [record for record in records if record["id"] in manifest_ids]
    records = records[: args.max_samples] if args.max_samples is not None else records
    if not records:
        fail("No custom labeled records found.")
    field_vocab = resolve_field_vocab(args.span_relg_dataset_dir, args.relg_checkpoint)
    field2id = {str(key): int(value) for key, value in field_vocab["vocab"].items()}
    kind2id, kind_source = resolve_kind_vocab(args.span_relg_dataset_dir, args.relg_checkpoint)
    processor, layout_model, _label2id, id2label, label_source = load_layoutlmv3(args.layoutlm_checkpoint, args.local_files_only, device)
    rel_model, rel_config, rel_config_path = load_rel_model(args.relg_checkpoint, device)
    include_context_tokens = rel_config.get("include_context_tokens", "all")
    span_pooling = rel_config.get("span_pooling", "first")

    out_dir = Path(args.out_dir)
    overlay_dir = out_dir / "overlay"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    probs_by_sample = []
    word_counts = Counter()
    failures = []
    gallery = []
    true_sequences = []
    pred_sequences = []
    raw_true_sequences = []
    raw_pred_sequences = []
    repair_reports = []

    spans_path = out_dir / "predicted_spans.jsonl"
    edges_path = out_dir / "edge_predictions.jsonl"
    items_path = out_dir / "item_predictions.jsonl"
    predictions_path = out_dir / "predictions.jsonl"
    match_path = out_dir / "span_matching_debug.jsonl"
    errors_path = out_dir / "errors.jsonl"
    with spans_path.open("w", encoding="utf-8") as spans_handle, edges_path.open("w", encoding="utf-8") as edges_handle, items_path.open("w", encoding="utf-8") as items_handle, predictions_path.open("w", encoding="utf-8") as pred_handle, match_path.open("w", encoding="utf-8") as match_handle, errors_path.open("w", encoding="utf-8") as errors_handle:
        for index, record in enumerate(tqdm(records, desc="custom predicted e2e", unit="sample")):
            data_id = record["id"]
            try:
                gold_sample = make_sample_info(record, repair_labels=args.repair_bio_boundaries)
                if getattr(layout_model, "uses_angle_features", False):
                    angle_config = getattr(layout_model, "angle_feature_config", {})
                    angle_mode = angle_config.get("angle_encoding_mode") or "sincos_scalar"
                    angle_dim = int(angle_config.get("angle_feature_dim") or angle_feature_dim_for_mode(angle_mode))
                    existing = gold_sample.get("angle_features") or []
                    existing_dim = len(existing[0]) if existing else 0
                    if existing_dim != angle_dim:
                        angle_result = build_angle_features_for_words(
                            gold_sample.get("word_payloads") or [],
                            boxes=gold_sample.get("boxes") or [],
                            image_width=gold_sample.get("width"),
                            image_height=gold_sample.get("height"),
                            mode=angle_mode,
                        )
                        gold_sample["angle_features"] = angle_result["angle_features"]
                        gold_sample["angle_debug"] = angle_result["word_angles"]
                payload = load_json(record["label_json"])
                layout = run_layout_prediction(
                    gold_sample["image"],
                    gold_sample["words"],
                    gold_sample["normalized_boxes"],
                    processor,
                    layout_model,
                    device,
                    id2label,
                    args.max_length,
                    angle_features=(
                        gold_sample.get("angle_features")
                        if getattr(layout_model, "uses_angle_features", False)
                        else None
                    ),
                )
                predictions = attach_boxes_to_predictions(layout["predictions"], gold_sample["boxes"], gold_sample["normalized_boxes"])
                canonicalize_prediction_labels(predictions)
                gold_labels = gold_label_for_predictions(gold_sample)
                raw_true_sequences.append(list(gold_labels))
                raw_pred_sequences.append([prediction["label"] for prediction in predictions])
                if args.repair_bio_boundaries:
                    repaired_labels, repair_report = repair_bio_boundaries(
                        [prediction["label"] for prediction in predictions],
                        words=gold_sample["words"],
                        boxes=gold_sample["boxes"],
                    )
                    for prediction, label in zip(predictions, repaired_labels):
                        prediction["label"] = label
                    repair_report["data_id"] = data_id
                    repair_reports.append(repair_report)
                true_sequences.append(gold_labels)
                pred_sequences.append([prediction["label"] for prediction in predictions])
                for pred, gold in zip(predictions, gold_labels):
                    word_counts["total"] += 1
                    if pred["label"] == gold:
                        word_counts["correct"] += 1
                spans = bio_predictions_to_spans(predictions, gold_sample["width"], gold_sample["height"])
                filtered_spans, dropped_spans = filter_candidate_spans(spans, field2id)
                assigned = assign_predicted_span_groups(filtered_spans, payload, gold_sample["word_original_indices"])
                sample_info = {
                    "image": gold_sample["image"],
                    "width": gold_sample["width"],
                    "height": gold_sample["height"],
                    "words": gold_sample["words"],
                    "boxes": gold_sample["boxes"],
                    "normalized_boxes": gold_sample["normalized_boxes"],
                    "angle_features": gold_sample.get("angle_features"),
                    "predictions": predictions,
                    "spans": filtered_spans,
                }
                cache = build_cache_sample(
                    data_id,
                    "custom",
                    index,
                    sample_info,
                    layout["word_hidden"],
                    field2id,
                    kind2id,
                    include_context_tokens=include_context_tokens,
                    span_pooling=span_pooling,
                )
                cache["source_image"] = record["image"]
                cache["source_label_json"] = record["label_json"]
                probs = run_rel_model(rel_model, cache, device)
                samples.append(cache)
                probs_by_sample.append(probs)
                decoded = decode_edges_to_items(cache, probs, threshold=args.threshold)
                gold_decoded = decode_edges_to_items(cache, gold_probs(cache), threshold=0.5)
                append_jsonl(spans_handle, {"data_id": data_id, "predicted_spans": filtered_spans, "dropped_spans": dropped_spans, "assigned_relation_groups": assigned})
                append_jsonl(edges_handle, {"data_id": data_id, "candidate_edges": edge_payload(cache, probs, args.threshold)})
                append_jsonl(items_handle, {"data_id": data_id, "items": [simple_item(item) for item in decoded.get("items", [])], "gold_items": [simple_item(item) for item in gold_decoded.get("items", [])]})
                append_jsonl(pred_handle, {"data_id": data_id, "words": gold_sample["words"], "gold_labels": gold_labels, "pred_labels": [item["label"] for item in predictions], "decoded_items": [simple_item(item) for item in decoded.get("items", [])]})
                append_jsonl(match_handle, {"data_id": data_id, "word_preview": [{"word": item["text"], "pred": item["label"], "gold": gold_labels[item["word_idx"]], "confidence": item["confidence"]} for item in predictions[:80]], "predicted_spans": filtered_spans})
                if len(gallery) < args.save_overlay_limit:
                    pred_edges, missed_edges = classify_edges(cache, probs, args.threshold)
                    overlay_edges = pred_edges + missed_edges
                    overlay_path = overlay_dir / f"{data_id}_pred_span_relg.png"
                    graph_overlay_path = overlay_dir / f"{data_id}_pred_span_relg_graph.png"
                    debug_path = overlay_dir / f"{data_id}_pred_span_relg_debug.json"
                    draw_grouped_relation_mapping_overlay(
                        gold_sample["image"],
                        cache,
                        overlay_edges,
                        overlay_path,
                        title=f"{data_id} custom predicted rel-g grouped mapping",
                    )
                    draw_span_relg_overlay(
                        gold_sample["image"],
                        cache,
                        overlay_edges,
                        graph_overlay_path,
                        title=f"{data_id} custom predicted rel-g graph",
                    )
                    save_json(debug_path, {"data_id": data_id, "pred_edges": pred_edges, "missed_edges": missed_edges, "predicted_spans": filtered_spans, "decoded_items": decoded.get("items", []), "token_debug": layout["token_debug"]})
                    gallery.append(
                        {
                            "data_id": data_id,
                            "image": str(overlay_path),
                            "debug_json": str(debug_path),
                            "num_pairs": len(cache.get("pair_meta", [])),
                            "num_positive_pairs": int(cache["pair_labels"].sum().item()),
                            "hard_negative_count": sum(1 for edge in pred_edges if edge.get("hard_negative")),
                        }
                    )
            except Exception as exc:
                failure = {"data_id": data_id, "error": repr(exc)}
                failures.append(failure)
                append_jsonl(errors_handle, failure)
                print(f"WARNING: skipped {data_id}: {exc}")

    if not samples:
        fail("No custom samples were processed successfully.")
    field_metrics_raw = compute_field_metrics(raw_true_sequences, raw_pred_sequences)
    field_metrics = compute_field_metrics(true_sequences, pred_sequences)
    summary = summarize(samples, probs_by_sample, args.threshold, word_counts, args, device, field_metrics)
    sweep = threshold_sweep(samples, probs_by_sample) if args.sweep_thresholds else None
    save_json(out_dir / "metrics_summary.json", summary)
    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "field_metrics_raw.json", field_metrics_raw)
    if args.repair_bio_boundaries:
        save_json(out_dir / "field_metrics_repaired.json", field_metrics)
        save_json(out_dir / "bio_repair_report.json", repair_reports)
    save_json(
        out_dir / "item_pair_metrics.json",
        {
            "menu_price_pair_precision": summary["menu_price_pair_precision"],
            "menu_price_pair_recall": summary["menu_price_pair_recall"],
            "menu_price_pair_f1": summary["menu_price_pair_f1"],
            "summary_amount_pair_precision": summary["summary_amount_pair_precision"],
            "summary_amount_pair_recall": summary["summary_amount_pair_recall"],
            "summary_amount_pair_f1": summary["summary_amount_pair_f1"],
        },
    )
    save_json(
        out_dir / "hard_negative_metrics.json",
        {
            key: value
            for key, value in summary["nested_metrics"].items()
            if "hard_negative" in key or key.endswith("_false_positive_count") or key == "dependent_collision_count"
        },
    )
    if sweep:
        save_json(out_dir / "threshold_sweep.json", sweep)
    save_json(out_dir / "failure_cases.json", failures)
    save_json(out_dir / "hard_negative_errors.json", summary["nested_metrics"].get("hard_negative_false_positive_examples", []))
    save_json(out_dir / "run_config.json", {**vars(args), "excluded_temp_count": len(excluded), "field_vocab_source": field_vocab["source"], "field_vocab_key": field_vocab["key"], "kind_vocab_source": kind_source, "relg_config_path": str(rel_config_path), "layout_label_source": label_source})
    write_gallery(out_dir, gallery)
    write_report(out_dir, summary, sweep)
    print(f"metrics_summary path: {out_dir / 'metrics_summary.json'}")
    print(f"overlay gallery path: {out_dir / 'index.html'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:5000])
    print("Custom predicted-span rel-g e2e evaluation passed.")


if __name__ == "__main__":
    main()
