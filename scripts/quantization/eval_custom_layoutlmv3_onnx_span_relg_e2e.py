import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.metrics.field_metrics import compute_field_metrics
from ml.span_relg.decode import decode_edges_to_items
from ml.span_relg.feature_cache import build_cache_sample
from ml.span_relg.io_utils import load_json, resolve_field_vocab
from ml.span_relg.span_utils import bio_predictions_to_spans
from ml.span_relg.visualization import draw_span_relg_overlay
from scripts.build_user_span_relg_dataset import assign_group_keys_from_relations, make_sample_info
from scripts.eval_custom_layoutlmv3_span_relg_e2e import (
    append_jsonl,
    canonicalize_prediction_labels,
    gold_label_for_predictions,
    load_manifest_ids,
    save_json,
    select_device,
    summarize,
    threshold_sweep,
    write_gallery,
    write_report,
)
from scripts.eval_predicted_span_relg_e2e import (
    attach_boxes_to_predictions,
    classify_edges,
    edge_payload,
    filter_candidate_spans,
    gold_probs,
    load_rel_model,
    resolve_kind_vocab,
    run_rel_model,
    simple_item,
)
from scripts.quantization.custom_quant_common import load_custom_records, make_onnx_session, run_onnx_layout_prediction
from scripts.quantization.quant_common import load_labels
from transformers import AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate custom receipt ONNX LayoutLMv3 predicted spans through span rel-g.")
    parser.add_argument("--onnx_model", required=True)
    parser.add_argument("--checkpoint_for_processor", "--layoutlm_checkpoint", dest="checkpoint_for_processor", default="models/layoutlmv3-item-policy-mixed-100/best")
    parser.add_argument("--relg_checkpoint", default="models/span-relg-item-policy-mixed/best")
    parser.add_argument("--span_relg_dataset_dir", default="processed_data/span_relg_item_policy_mixed")
    parser.add_argument("--input_dir", required=True, nargs="+")
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--split", default="all", choices=("train", "validation", "test", "all"))
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--out_dir", default="outputs/quantization/custom_onnx_e2e")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--onnx_provider", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--sweep_thresholds", action="store_true")
    parser.add_argument("--save_overlay_limit", type=int, default=30)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


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


def main():
    args = parse_args()
    args.layoutlm_checkpoint = args.checkpoint_for_processor
    for path, label in [
        (args.onnx_model, "onnx_model"),
        (args.checkpoint_for_processor, "checkpoint_for_processor"),
        (args.relg_checkpoint, "relg_checkpoint"),
        (args.span_relg_dataset_dir, "span_relg_dataset_dir"),
    ]:
        if not Path(path).exists():
            fail(f"{label} not found: {path}")
    device = select_device(args.device)
    print(f"onnx_model: {args.onnx_model}")
    print(f"checkpoint_for_processor: {args.checkpoint_for_processor}")
    print(f"relg_checkpoint: {args.relg_checkpoint}")
    print(f"selected rel-g device: {device}")

    records, excluded = load_custom_records(args.input_dir, args.exclude_dir_name, max_samples=None)
    manifest_ids = load_manifest_ids(args.split_manifest, args.split)
    if manifest_ids is not None:
        records = [record for record in records if record["id"] in manifest_ids]
    records = records[: args.max_samples] if args.max_samples is not None else records
    if not records:
        fail("No custom labeled records found.")
    labels_payload, _label_list, _label2id, id2label = load_labels(args.checkpoint_for_processor)
    processor = AutoProcessor.from_pretrained(args.checkpoint_for_processor, apply_ocr=False, local_files_only=args.local_files_only)
    onnx_session = make_onnx_session(args.onnx_model, args.onnx_provider)
    field_vocab = resolve_field_vocab(args.span_relg_dataset_dir, args.relg_checkpoint)
    field2id = {str(key): int(value) for key, value in field_vocab["vocab"].items()}
    kind2id, kind_source = resolve_kind_vocab(args.span_relg_dataset_dir, args.relg_checkpoint)
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

    spans_path = out_dir / "predicted_spans.jsonl"
    edges_path = out_dir / "edge_predictions.jsonl"
    items_path = out_dir / "item_predictions.jsonl"
    predictions_path = out_dir / "predictions.jsonl"
    errors_path = out_dir / "errors.jsonl"
    with spans_path.open("w", encoding="utf-8") as spans_handle, edges_path.open("w", encoding="utf-8") as edges_handle, items_path.open("w", encoding="utf-8") as items_handle, predictions_path.open("w", encoding="utf-8") as pred_handle, errors_path.open("w", encoding="utf-8") as errors_handle:
        for index, record in enumerate(tqdm(records, desc="custom onnx e2e", unit="sample")):
            data_id = record["id"]
            try:
                gold_sample = make_sample_info(record)
                payload = load_json(record["label_json"])
                layout = run_onnx_layout_prediction(
                    gold_sample["image"],
                    gold_sample["words"],
                    gold_sample["normalized_boxes"],
                    processor,
                    onnx_session,
                    id2label,
                    args.max_length,
                )
                predictions = attach_boxes_to_predictions(layout["predictions"], gold_sample["boxes"], gold_sample["normalized_boxes"])
                canonicalize_prediction_labels(predictions)
                gold_labels = gold_label_for_predictions(gold_sample)
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
                if len(gallery) < args.save_overlay_limit:
                    pred_edges, missed_edges = classify_edges(cache, probs, args.threshold)
                    overlay_edges = pred_edges + missed_edges
                    overlay_path = overlay_dir / f"{data_id}_onnx_pred_span_relg.png"
                    debug_path = overlay_dir / f"{data_id}_onnx_pred_span_relg_debug.json"
                    draw_span_relg_overlay(gold_sample["image"], cache, overlay_edges, overlay_path, title=f"{data_id} ONNX predicted rel-g")
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
        fail("No custom ONNX samples were processed successfully.")
    field_metrics = compute_field_metrics(true_sequences, pred_sequences)
    summary = summarize(samples, probs_by_sample, args.threshold, word_counts, args, device, field_metrics)
    summary.update(
        {
            "onnx_model": args.onnx_model,
            "checkpoint_for_processor": args.checkpoint_for_processor,
            "onnx_providers": onnx_session.get_providers(),
            "excluded_temp_count": len(excluded),
            "failed_samples": len(failures),
        }
    )
    sweep = threshold_sweep(samples, probs_by_sample) if args.sweep_thresholds else None
    save_json(out_dir / "metrics_summary.json", summary)
    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "field_metrics.json", field_metrics)
    if sweep:
        save_json(out_dir / "threshold_sweep.json", sweep)
    save_json(out_dir / "failure_cases.json", failures)
    save_json(out_dir / "hard_negative_errors.json", summary["nested_metrics"].get("hard_negative_false_positive_examples", []))
    save_json(
        out_dir / "run_config.json",
        {
            **vars(args),
            "excluded_temp_count": len(excluded),
            "field_vocab_source": field_vocab["source"],
            "field_vocab_key": field_vocab["key"],
            "kind_vocab_source": kind_source,
            "relg_config_path": str(rel_config_path),
            "layout_labels": labels_payload,
        },
    )
    write_gallery(out_dir, gallery)
    write_report(out_dir, summary, sweep)
    print(f"metrics_summary path: {out_dir / 'metrics_summary.json'}")
    print(f"overlay gallery path: {out_dir / 'index.html'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:5000])
    print("Custom ONNX predicted-span rel-g e2e evaluation passed.")


if __name__ == "__main__":
    main()
