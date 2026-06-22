import argparse
import html
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.angle_geometry import (
    ANGLE_FEATURE_DIM,
    align_angle_features_to_tokens,
    angle_feature_dim_for_mode,
    build_angle_features_for_words,
)
from ml.layoutlmv3_angle_model import load_angle_aware_token_classifier
from ml.span_relg.cord_spans import extract_cord_words_and_lines
from ml.span_relg.decode import decode_edges_to_items
from ml.span_relg.feature_cache import build_cache_sample
from ml.span_relg.io_utils import load_json, resolve_field_vocab, resolve_model_config
from ml.span_relg.metrics import aggregate_metrics
from ml.span_relg.model import SpanRelGModel
from ml.span_relg.schema import ALL_FIELDS, is_candidate_dep_field, is_dependent_field, is_head_field
from ml.span_relg.span_utils import bio_predictions_to_spans
from ml.span_relg.visualization import draw_span_relg_overlay
from ml.receipt_schema import canonicalize_field, field_for_vocab


THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
MODEL_CONFIG_KEYS = {"hidden_dim", "num_fields", "num_kinds", "d_model", "num_layers", "num_heads", "dropout"}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predicted-span LayoutLMv3 -> span rel-g on CORD.")
    parser.add_argument("--raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--layoutlm_checkpoint", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--relg_checkpoint", default="models/span-relg-context/best")
    parser.add_argument("--span_relg_dataset_dir", default="processed_data/span_relg")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--out_dir", default="outputs/predicted_span_relg_e2e")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--use_angle_features", default="auto", choices=("auto", "true", "false"))
    parser.add_argument("--sweep_thresholds", action="store_true")
    parser.add_argument("--save_overlay_limit", type=int, default=30)
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


def select_device(device_arg):
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_layout_labels(checkpoint, model):
    labels_path = Path(checkpoint) / "labels.json"
    if labels_path.exists():
        payload = load_json(labels_path)
        id2label = {int(idx): str(label) for idx, label in payload["id2label"].items()}
        label2id = {str(label): int(idx) for label, idx in payload["label2id"].items()}
        return label2id, id2label, str(labels_path)
    id2label = {int(idx): str(label) for idx, label in model.config.id2label.items()}
    label2id = {str(label): int(idx) for idx, label in model.config.label2id.items()}
    return label2id, id2label, "model.config"


def checkpoint_looks_angle_aware(checkpoint):
    checkpoint = Path(checkpoint)
    if (checkpoint / "angle_model_config.json").exists():
        return True
    config_path = checkpoint / "config.json"
    if config_path.exists():
        try:
            payload = load_json(config_path)
            return bool(payload.get("use_angle_features") or payload.get("angle_feature_dim"))
        except Exception:
            return False
    return False


def load_angle_config(checkpoint):
    checkpoint = Path(checkpoint)
    for name in ("angle_model_config.json", "config.json"):
        path = checkpoint / name
        if not path.exists():
            continue
        try:
            payload = load_json(path)
        except Exception:
            continue
        if payload.get("use_angle_features") or payload.get("angle_feature_dim") or payload.get("angle_encoding_mode"):
            mode = payload.get("angle_encoding_mode", "sincos_scalar")
            return {
                "source": str(path),
                "angle_encoding_mode": mode,
                "angle_feature_dim": int(payload.get("angle_feature_dim", angle_feature_dim_for_mode(mode))),
            }
    return {"source": None, "angle_encoding_mode": "none", "angle_feature_dim": 0}


def load_layoutlmv3(checkpoint, local_files_only, device, use_angle_features="auto"):
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        apply_ocr=False,
        local_files_only=local_files_only,
    )
    angle_config = load_angle_config(checkpoint)
    use_angle = checkpoint_looks_angle_aware(checkpoint) if use_angle_features == "auto" else use_angle_features == "true"
    if use_angle:
        model = load_angle_aware_token_classifier(
            checkpoint,
            local_files_only=local_files_only,
            ignore_mismatched_sizes=True,
            angle_feature_dim=angle_config.get("angle_feature_dim") or None,
            angle_encoding_mode=angle_config.get("angle_encoding_mode") or None,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            checkpoint,
            local_files_only=local_files_only,
        )
    label2id, id2label, label_source = load_layout_labels(checkpoint, model)
    model.config.label2id = label2id
    model.config.id2label = id2label
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    model.uses_angle_features = use_angle
    model.angle_feature_config = angle_config if use_angle else {"angle_feature_dim": 0, "angle_encoding_mode": "none"}
    return processor, model, label2id, id2label, label_source


def load_rel_model(checkpoint, device):
    checkpoint = Path(checkpoint)
    if not (checkpoint / "model.pt").exists():
        fail(f"span rel-g checkpoint missing. Run train_span_relg.py first. Missing: {checkpoint / 'model.pt'}")
    config_path = resolve_model_config(checkpoint)
    config = load_json(config_path)
    model_kwargs = {key: value for key, value in config.items() if key in MODEL_CONFIG_KEYS}
    model = SpanRelGModel(**model_kwargs)
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, config, config_path


def resolve_kind_vocab(dataset_dir, checkpoint):
    for path in (Path(dataset_dir) / "schema.json", Path(checkpoint) / "schema.json"):
        if path.exists():
            payload = load_json(path)
            if "kind2id" in payload:
                return {str(key): int(value) for key, value in payload["kind2id"].items()}, str(path)
    return {"SPAN": 0, "TOKEN": 1}, "default"


def make_group_key(line):
    group_id = line.get("group_id")
    if group_id is None:
        return None
    return str(group_id)


def make_gold_word_labels(extracted):
    labels = ["O"] * len(extracted["words"])
    for line in extracted["lines"]:
        field = line.get("field", "O")
        if field == "O":
            continue
        for offset, word_idx in enumerate(line["word_indices"]):
            labels[word_idx] = f"{'B' if offset == 0 else 'I'}-{field}"
    return labels


def run_layout_prediction(image, words, boxes, processor, model, device, id2label, max_length, angle_features=None):
    encoding = processor(
        image,
        words,
        boxes=boxes,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    model_inputs = {
        key: value.to(device)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"}
    }
    if getattr(model, "uses_angle_features", False) or angle_features is not None:
        angle_feature_dim = int(getattr(model, "angle_feature_config", {}).get("angle_feature_dim") or ANGLE_FEATURE_DIM)
        token_angle = align_angle_features_to_tokens(
            encoding,
            angle_features or [],
            batch_index=0,
            feature_dim=angle_feature_dim,
        )
        model_inputs["angle_features"] = token_angle.unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
    logits = outputs.logits[0].detach().cpu()
    probs = torch.softmax(logits, dim=-1)
    hidden = outputs.hidden_states[-1][0].detach().cpu() if outputs.hidden_states else outputs.last_hidden_state[0].detach().cpu()
    pred_ids = logits.argmax(dim=-1)
    attention_mask = encoding["attention_mask"][0].detach().cpu()
    token_strings = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())

    first_token_for_word = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or attention_mask[token_idx].item() == 0:
            continue
        first_token_for_word.setdefault(int(word_idx), token_idx)
    missing = [idx for idx in range(len(words)) if idx not in first_token_for_word]
    if missing:
        raise ValueError(f"Missing token alignment for {len(missing)} words; first missing={missing[:20]}")

    predictions = []
    word_hidden = []
    token_debug = []
    for word_idx, word in enumerate(words):
        token_idx = first_token_for_word[word_idx]
        pred_id = int(pred_ids[token_idx].item())
        confidence = float(probs[token_idx, pred_id].item())
        predictions.append({"word_idx": word_idx, "text": word, "label": id2label[pred_id], "confidence": confidence})
        word_hidden.append(hidden[token_idx])
    for token_idx, word_idx in enumerate(word_ids[:120]):
        pred_id = int(pred_ids[token_idx].item())
        token_debug.append(
            {
                "token_idx": token_idx,
                "token": token_strings[token_idx],
                "word_idx": word_idx,
                "word": None if word_idx is None or word_idx >= len(words) else words[word_idx],
                "pred_label": id2label[pred_id],
                "confidence": float(probs[token_idx, pred_id].item()),
            }
        )
    return {
        "predictions": predictions,
        "word_hidden": torch.stack(word_hidden, dim=0),
        "word_token_indices": [first_token_for_word[idx] for idx in range(len(words))],
        "encoding_shapes": {key: list(value.shape) for key, value in encoding.items() if hasattr(value, "shape")},
        "angle_features_shape": list(model_inputs["angle_features"].shape) if "angle_features" in model_inputs else None,
        "token_debug": token_debug,
    }


def attach_boxes_to_predictions(predictions, boxes, normalized_boxes):
    output = []
    for idx, prediction in enumerate(predictions):
        item = dict(prediction)
        item["box"] = boxes[idx]
        item["normalized_box"] = normalized_boxes[idx]
        output.append(item)
    return output


def match_predicted_spans(spans, gold_lines):
    for span in spans:
        word_indices = set(span.get("word_indices", []))
        best_line = None
        best_overlap = 0
        for line in gold_lines:
            overlap = len(word_indices & set(line.get("word_indices", [])))
            if overlap > best_overlap:
                best_line = line
                best_overlap = overlap
        span["matched_gold_line_id"] = best_line.get("line_id") if best_line and best_overlap else None
        span["matched_gold_field"] = best_line.get("field") if best_line and best_overlap else None
        span["matched_gold_category"] = best_line.get("category") if best_line and best_overlap else None
        span["matched_gold_overlap"] = best_overlap
        span["group_id"] = best_line.get("group_id") if best_line and best_overlap else None
        span["sub_group_id"] = best_line.get("sub_group_id") if best_line and best_overlap else None
        span["row_id"] = best_line.get("row_id") if best_line and best_overlap else None
        span["group_key"] = make_group_key(best_line) if best_line and best_overlap else None
    return spans


def filter_candidate_spans(spans, field2id):
    kept = []
    dropped = []
    for span in spans:
        field = canonicalize_field(span.get("field"))
        if field_for_vocab(field, field2id) is None:
            dropped.append({"field": field, "text": span.get("text"), "reason": "field_not_in_vocab"})
            continue
        if not (is_head_field(field) or is_candidate_dep_field(field)):
            dropped.append({"field": field, "text": span.get("text"), "reason": "not_relg_candidate"})
            continue
        span = dict(span)
        span["field"] = field
        kept.append(span)
    for span_id, span in enumerate(kept):
        span["span_id"] = span_id
    return kept, dropped


def span_metric_counts(data_id, spans, gold_lines, field):
    field = canonicalize_field(field)
    pred = [span for span in spans if canonicalize_field(span.get("field")) == field]
    gold = [line for line in gold_lines if canonicalize_field(line.get("field")) == field]
    correct_pred = [span for span in pred if canonicalize_field(span.get("matched_gold_field")) == field]
    matched_gold = {
        (data_id, span.get("matched_gold_line_id"))
        for span in correct_pred
        if span.get("matched_gold_line_id") is not None
    }
    return {
        "pred": len(pred),
        "gold": len(gold),
        "correct_pred": len(correct_pred),
        "matched_gold": matched_gold,
    }


def run_rel_model(model, cache, device):
    node_hidden = cache["node_hidden"].unsqueeze(0).float().to(device)
    node_field_ids = cache["node_field_ids"].unsqueeze(0).long().to(device)
    node_kind_ids = cache["node_kind_ids"].unsqueeze(0).long().to(device)
    node_boxes = cache["node_boxes"].unsqueeze(0).float().to(device)
    node_mask = torch.ones(1, cache["node_hidden"].shape[0], dtype=torch.bool, device=device)
    pairs = cache["candidate_pairs"].long()
    if pairs.numel():
        candidate_pairs = torch.cat([torch.zeros(pairs.shape[0], 1, dtype=torch.long), pairs], dim=1).to(device)
    else:
        candidate_pairs = torch.empty((0, 3), dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(
            node_hidden=node_hidden,
            node_field_ids=node_field_ids,
            node_kind_ids=node_kind_ids,
            node_boxes=node_boxes,
            node_mask=node_mask,
            candidate_pairs=candidate_pairs,
        )
    return output["probs"].detach().cpu().tolist()


def edge_payload(sample, probs, threshold):
    rows = []
    for idx, meta in enumerate(sample.get("pair_meta", [])):
        prob = float(probs[idx])
        gold = int(sample["pair_labels"][idx].item())
        pred = int(prob >= threshold)
        rows.append({**meta, "gold": gold, "prob": prob, "pred": pred, "correct": bool(gold == pred)})
    return rows


def gold_probs(sample):
    return [float(value.item()) for value in sample["pair_labels"]]


def simple_item(item):
    def text(value):
        return value.get("text") if isinstance(value, dict) else value

    return {
        "item_name": text(item.get("item_name") or item.get("menu_name")),
        "menu_name": text(item.get("menu_name") or item.get("item_name")),
        "price": text(item.get("price")),
        "count": text(item.get("count")),
        "unit_price": text(item.get("unit_price")),
        "rel_g_edges": item.get("rel_g_edges", []),
        "warnings": item.get("warnings", []),
    }


def decoded_item_errors(samples, probs_by_sample, threshold):
    no_price = []
    multiple_price = []
    collision_errors = []
    for sample, probs in zip(samples, probs_by_sample):
        decoded = decode_edges_to_items(sample, probs, threshold=threshold)
        for item in decoded.get("items", []):
            name_value = item.get("item_name") or item.get("menu_name")
            menu_text = name_value.get("text") if isinstance(name_value, dict) else name_value
            if item.get("price") is None:
                no_price.append({"data_id": sample.get("data_id"), "index": sample.get("index"), "menu_name": menu_text, "item": simple_item(item)})
            price_edges = [edge for edge in item.get("rel_g_edges", []) if edge.get("dep_field") in {"ITEM_PRICE", "MENU_PRICE"}]
            if len(price_edges) > 1:
                multiple_price.append({"data_id": sample.get("data_id"), "index": sample.get("index"), "menu_name": menu_text, "price_edges": price_edges})
        selected_by_dep = {}
        for idx, meta in enumerate(sample.get("pair_meta", [])):
            if float(probs[idx]) < threshold:
                continue
            selected_by_dep.setdefault(meta.get("dep_span_id"), []).append({"data_id": sample.get("data_id"), **meta, "prob": float(probs[idx])})
        for dep_span_id, edges in selected_by_dep.items():
            if len(edges) > 1:
                collision_errors.append({"data_id": sample.get("data_id"), "dep_span_id": dep_span_id, "edges": edges})
    return no_price, multiple_price, collision_errors


def classify_edges(sample, probs, threshold):
    pred_edges = []
    missed_edges = []
    for idx, meta in enumerate(sample.get("pair_meta", [])):
        prob = float(probs[idx])
        gold = int(sample["pair_labels"][idx].item()) == 1
        pred = prob >= threshold
        base = dict(meta)
        base.update({"pair_index": idx, "prob": prob, "gold": int(gold), "pred": int(pred), "correct": bool(gold and pred)})
        if pred:
            status = "correct" if gold else "wrong"
            edge = dict(base, status=status)
            if not gold and (str(meta.get("dep_field", "")).startswith("TOTAL_") or str(meta.get("dep_field", "")).startswith("SUBTOTAL_")):
                edge["hard_negative"] = True
            pred_edges.append(edge)
        elif gold:
            missed_edges.append(dict(base, status="missed"))
    return pred_edges, missed_edges


def summarize_metrics(args, samples, probs_by_sample, threshold, word_counts, span_counts, field2id, label_source, device):
    metrics = aggregate_metrics(samples, probs_by_sample, threshold)
    no_price, multiple_price, collision_errors = decoded_item_errors(samples, probs_by_sample, threshold)
    labels = []
    for sample in samples:
        labels.extend(int(value.item()) for value in sample["pair_labels"])
    def span_precision(field):
        stat = span_counts[field]
        return stat["correct_pred"] / stat["pred"] if stat["pred"] else 0.0
    def span_recall(field):
        stat = span_counts[field]
        return len(stat["matched_gold"]) / stat["gold"] if stat["gold"] else 0.0
    summary = {
        "split": args.split,
        "num_samples": len(samples),
        "threshold": threshold,
        "word_label_accuracy": word_counts["correct"] / word_counts["total"] if word_counts["total"] else 0.0,
        "item_name_span_precision": span_precision("ITEM_NAME"),
        "item_name_span_recall": span_recall("ITEM_NAME"),
        "item_price_span_precision": span_precision("ITEM_PRICE"),
        "item_price_span_recall": span_recall("ITEM_PRICE"),
        "menu_nm_span_precision": span_precision("ITEM_NAME"),
        "menu_nm_span_recall": span_recall("ITEM_NAME"),
        "menu_price_span_precision": span_precision("ITEM_PRICE"),
        "menu_price_span_recall": span_recall("ITEM_PRICE"),
        "num_candidate_pairs": len(labels),
        "num_positive_pairs": sum(labels),
        "num_negative_pairs": len(labels) - sum(labels),
        "edge_precision": metrics["edge"]["precision"],
        "edge_recall": metrics["edge"]["recall"],
        "edge_f1": metrics["edge"]["f1"],
        "item_price_pair_precision": metrics.get("item_price_pair", metrics["menu_price_pair"])["precision"],
        "item_price_pair_recall": metrics.get("item_price_pair", metrics["menu_price_pair"])["recall"],
        "item_price_pair_f1": metrics.get("item_price_pair", metrics["menu_price_pair"])["f1"],
        "menu_price_pair_precision": metrics["menu_price_pair"]["precision"],
        "menu_price_pair_recall": metrics["menu_price_pair"]["recall"],
        "menu_price_pair_f1": metrics["menu_price_pair"]["f1"],
        "e2e_item_price_pair_f1": metrics.get("item_price_pair", metrics["menu_price_pair"])["f1"],
        "hard_negative_false_positive_count": metrics["hard_negative_false_positive_count"],
        "total_subtotal_false_positive_count": metrics["hard_negative_false_positive_count"],
        "dependent_collision_count": metrics["dependent_collision_count"],
        "no_price_item_count": len(no_price),
        "multiple_price_item_count": len(multiple_price),
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "layoutlm_checkpoint": args.layoutlm_checkpoint,
        "relg_checkpoint": args.relg_checkpoint,
        "span_relg_dataset_dir": args.span_relg_dataset_dir,
        "layout_label_source": label_source,
        "num_fields": len(field2id),
        "nested_metrics": metrics,
    }
    return summary, metrics, no_price, multiple_price, collision_errors


def threshold_sweep(samples, probs_by_sample):
    rows = []
    best = None
    for threshold in THRESHOLDS:
        metrics = aggregate_metrics(samples, probs_by_sample, threshold)
        edge = metrics["edge"]
        menu_price = metrics["menu_price_pair"]
        row = {
            "threshold": threshold,
            "edge_precision": edge["precision"],
            "edge_recall": edge["recall"],
            "edge_f1": edge["f1"],
            "menu_price_pair_precision": menu_price["precision"],
            "menu_price_pair_recall": menu_price["recall"],
            "menu_price_pair_f1": menu_price["f1"],
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


def write_html_gallery(out_dir, cards):
    rows = []
    for card in cards:
        image_rel = html.escape(str(Path("overlays") / Path(card["image"]).name))
        debug_rel = html.escape(str(Path("overlays") / Path(card["debug_json"]).name))
        rows.append(
            "<article>"
            f"<h2>{html.escape(card['data_id'])}</h2>"
            f"<p>index={card['index']} pred_items={card['predicted_item_count']} gold_items={card['gold_item_count']} "
            f"menu_price_correct={card['menu_price_correct']} hard_neg={card['hard_negative_count']}</p>"
            f"<a href='{debug_rel}'>debug JSON</a><br>"
            f"<img src='{image_rel}' alt='{html.escape(card['data_id'])} overlay'>"
            "</article>"
        )
    page = (
        "<!doctype html><html><head><meta charset='utf-8'><title>Predicted Span Rel-G E2E</title>"
        "<style>body{font-family:Arial,sans-serif;margin:24px}article{margin-bottom:32px;border-bottom:1px solid #ddd;padding-bottom:24px}"
        "img{max-width:100%;height:auto;border:1px solid #ccc}</style></head><body>"
        "<h1>Predicted Span Rel-G E2E Gallery</h1>"
        + "\n".join(rows)
        + "</body></html>"
    )
    (Path(out_dir) / "index.html").write_text(page, encoding="utf-8")


def write_oracle_vs_predicted_report(out_dir, oracle_metrics_path, predicted_summary):
    out_dir = Path(out_dir)
    oracle = load_json(oracle_metrics_path) if Path(oracle_metrics_path).exists() else None
    oracle_f1 = oracle.get("menu_price_pair_f1") if oracle else None
    pred_f1 = predicted_summary.get("menu_price_pair_f1")
    drop = oracle_f1 - pred_f1 if oracle_f1 is not None and pred_f1 is not None else None
    lines = [
        "# Oracle Span vs Predicted Span Rel-G Report",
        "",
        "## Oracle Span Metric",
    ]
    if oracle:
        lines.extend(
            [
                f"- menu_price_pair_precision: {oracle.get('menu_price_pair_precision'):.6f}",
                f"- menu_price_pair_recall: {oracle.get('menu_price_pair_recall'):.6f}",
                f"- menu_price_pair_f1: {oracle_f1:.6f}",
            ]
        )
    else:
        lines.append("- Oracle metrics not found.")
    lines.extend(
        [
            "",
            "## Predicted Span E2E Metric",
            f"- menu_price_pair_precision: {predicted_summary.get('menu_price_pair_precision'):.6f}",
            f"- menu_price_pair_recall: {predicted_summary.get('menu_price_pair_recall'):.6f}",
            f"- menu_price_pair_f1: {pred_f1:.6f}",
            f"- word_label_accuracy: {predicted_summary.get('word_label_accuracy'):.6f}",
            f"- MENU_NM span precision/recall: {predicted_summary.get('menu_nm_span_precision'):.6f} / {predicted_summary.get('menu_nm_span_recall'):.6f}",
            f"- MENU_PRICE span precision/recall: {predicted_summary.get('menu_price_span_precision'):.6f} / {predicted_summary.get('menu_price_span_recall'):.6f}",
            "",
            "## Gap",
            f"- oracle_minus_predicted_f1: {drop:.6f}" if drop is not None else "- oracle_minus_predicted_f1: unavailable",
            "",
            "## Likely Drop Causes",
            "- LayoutLMv3 field span misses",
            "- MENU_PRICE vs TOTAL_TOTAL_PRICE confusion",
            "- span boundary mismatch",
            "- rel-g connection error after predicted span extraction",
            "",
            "## Recommended Next Actions",
            "- If span precision/recall is low, improve LayoutLMv3 fine-tuning or add user-domain data.",
            "- If rel-g errors remain high with good spans, improve rel-g features/model.",
            "- If total/subtotal false positives are high, strengthen hard negative handling.",
            "- If collisions are high, tune threshold and decoding collision avoidance.",
        ]
    )
    (out_dir / "oracle_vs_predicted_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    raw_data_dir = Path(args.raw_data_dir)
    layout_checkpoint = Path(args.layoutlm_checkpoint)
    relg_checkpoint = Path(args.relg_checkpoint)
    span_relg_dataset_dir = Path(args.span_relg_dataset_dir)
    if not raw_data_dir.exists():
        fail(f"CORD-v2 dataset not found at {raw_data_dir}.")
    if not layout_checkpoint.exists():
        fail(f"LayoutLMv3 checkpoint missing: {layout_checkpoint}")
    if not relg_checkpoint.exists():
        fail(f"Span rel-g checkpoint missing: {relg_checkpoint}")
    if not span_relg_dataset_dir.exists():
        fail(f"Span rel-g dataset dir missing: {span_relg_dataset_dir}")

    device = select_device(args.device)
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"selected device: {device}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    raw_dataset = load_from_disk(str(raw_data_dir))
    if args.split not in raw_dataset:
        fail(f"Split {args.split!r} not found. Available: {list(raw_dataset.keys())}")
    total = len(raw_dataset[args.split])
    limit = min(total, args.max_samples) if args.max_samples is not None else total

    field_vocab = resolve_field_vocab(span_relg_dataset_dir, relg_checkpoint)
    field2id = {str(key): int(value) for key, value in field_vocab["vocab"].items()}
    kind2id, kind_source = resolve_kind_vocab(span_relg_dataset_dir, relg_checkpoint)
    processor, layout_model, label2id, id2label, label_source = load_layoutlmv3(
        layout_checkpoint,
        args.local_files_only,
        device,
        use_angle_features=args.use_angle_features,
    )
    rel_model, rel_config, rel_config_path = load_rel_model(relg_checkpoint, device)
    include_context_tokens = rel_config.get("include_context_tokens", "all")
    span_pooling = rel_config.get("span_pooling", "first")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)

    print(f"raw_data_dir: {raw_data_dir}")
    print(f"layoutlm_checkpoint: {layout_checkpoint}")
    print(f"relg_checkpoint: {relg_checkpoint}")
    print(f"relg_config_path: {rel_config_path}")
    print(f"field vocab source: {field_vocab['source']}::{field_vocab['key']}")
    print(f"kind vocab source: {kind_source}")
    print(f"layout label source: {label_source}")
    print(f"use_angle_features: {args.use_angle_features}")
    print(f"layout_model_uses_angle_features: {getattr(layout_model, 'uses_angle_features', False)}")
    print(f"angle_feature_dim: {ANGLE_FEATURE_DIM}")
    print(f"processing split={args.split} samples={limit}/{total}")

    samples = []
    probs_by_sample = []
    word_counts = Counter()
    span_counts = {
        "ITEM_NAME": {"pred": 0, "gold": 0, "correct_pred": 0, "matched_gold": set()},
        "ITEM_PRICE": {"pred": 0, "gold": 0, "correct_pred": 0, "matched_gold": set()},
    }
    failure_cases = []
    gallery_cards = []

    predicted_spans_path = out_dir / "predicted_spans.jsonl"
    edges_path = out_dir / "edge_predictions.jsonl"
    items_path = out_dir / "item_predictions.jsonl"
    matching_path = out_dir / "span_matching_debug.jsonl"
    with predicted_spans_path.open("w", encoding="utf-8") as spans_handle, edges_path.open("w", encoding="utf-8") as edges_handle, items_path.open("w", encoding="utf-8") as items_handle, matching_path.open("w", encoding="utf-8") as matching_handle:
        for index in tqdm(range(limit), desc="predicted-span e2e", unit="sample"):
            data_id = f"{args.split}_{index:06d}"
            try:
                raw_sample = raw_dataset[args.split][index]
                extracted = extract_cord_words_and_lines(raw_sample)
                if getattr(layout_model, "uses_angle_features", False):
                    angle_config = getattr(layout_model, "angle_feature_config", {})
                    angle_mode = angle_config.get("angle_encoding_mode") or "sincos_scalar"
                    angle_dim = int(angle_config.get("angle_feature_dim") or angle_feature_dim_for_mode(angle_mode))
                    existing = extracted.get("angle_features") or []
                    existing_dim = len(existing[0]) if existing else 0
                    if existing_dim != angle_dim:
                        angle_result = build_angle_features_for_words(
                            extracted.get("word_payloads") or [],
                            boxes=extracted.get("boxes") or [],
                            image_width=extracted.get("width"),
                            image_height=extracted.get("height"),
                            mode=angle_mode,
                        )
                        extracted["angle_features"] = angle_result["angle_features"]
                        extracted["angle_debug"] = angle_result["word_angles"]
                gold_labels = make_gold_word_labels(extracted)
                layout = run_layout_prediction(
                    extracted["image"],
                    extracted["words"],
                    extracted["normalized_boxes"],
                    processor,
                    layout_model,
                    device,
                    id2label,
                    args.max_length,
                    angle_features=(
                        extracted.get("angle_features")
                        if getattr(layout_model, "uses_angle_features", False) or args.use_angle_features == "true"
                        else None
                    ),
                )
                predictions = attach_boxes_to_predictions(layout["predictions"], extracted["boxes"], extracted["normalized_boxes"])
                for pred, gold in zip(predictions, gold_labels):
                    word_counts["total"] += 1
                    if pred["label"] == gold:
                        word_counts["correct"] += 1
                spans = bio_predictions_to_spans(predictions, extracted["width"], extracted["height"])
                match_predicted_spans(spans, extracted["lines"])
                filtered_spans, dropped_spans = filter_candidate_spans(spans, field2id)
                for field in ("ITEM_NAME", "ITEM_PRICE"):
                    counts = span_metric_counts(data_id, filtered_spans, extracted["lines"], field)
                    span_counts[field]["pred"] += counts["pred"]
                    span_counts[field]["gold"] += counts["gold"]
                    span_counts[field]["correct_pred"] += counts["correct_pred"]
                    span_counts[field]["matched_gold"].update(counts["matched_gold"])

                sample_info = {
                    "image": extracted["image"],
                    "width": extracted["width"],
                    "height": extracted["height"],
                    "words": extracted["words"],
                    "boxes": extracted["boxes"],
                    "normalized_boxes": extracted["normalized_boxes"],
                    "angle_features": extracted.get("angle_features"),
                    "predictions": predictions,
                    "spans": filtered_spans,
                }
                cache = build_cache_sample(
                    data_id,
                    args.split,
                    index,
                    sample_info,
                    layout["word_hidden"],
                    field2id,
                    kind2id,
                    include_context_tokens=include_context_tokens,
                    span_pooling=span_pooling,
                )
                cache["word_token_indices"] = layout["word_token_indices"]
                cache["encoding_shapes"] = layout["encoding_shapes"]
                probs = run_rel_model(rel_model, cache, device)
                samples.append(cache)
                probs_by_sample.append(probs)

                decoded = decode_edges_to_items(cache, probs, threshold=args.threshold)
                gold_decoded = decode_edges_to_items(cache, gold_probs(cache), threshold=0.5)
                append_jsonl(spans_handle, {"data_id": data_id, "split": args.split, "index": index, "predicted_spans": filtered_spans, "dropped_spans": dropped_spans})
                append_jsonl(edges_handle, {"data_id": data_id, "split": args.split, "index": index, "candidate_edges": edge_payload(cache, probs, args.threshold)})
                append_jsonl(
                    items_handle,
                    {
                        "data_id": data_id,
                        "split": args.split,
                        "index": index,
                        "items": [simple_item(item) for item in decoded.get("items", [])],
                        "gold_items": [simple_item(item) for item in gold_decoded.get("items", [])],
                    },
                )
                append_jsonl(
                    matching_handle,
                    {
                        "data_id": data_id,
                        "word_label_preview": [
                            {"word": item["text"], "pred": item["label"], "gold": gold_labels[item["word_idx"]], "confidence": item["confidence"]}
                            for item in predictions[:80]
                        ],
                        "predicted_spans": filtered_spans,
                        "gold_lines": extracted["lines"],
                    },
                )
                if len(gallery_cards) < args.save_overlay_limit:
                    pred_edges, missed_edges = classify_edges(cache, probs, args.threshold)
                    overlay_edges = pred_edges + missed_edges
                    overlay_path = overlay_dir / f"{data_id}_pred_span_relg.png"
                    debug_path = overlay_dir / f"{data_id}_pred_span_relg_debug.json"
                    draw_span_relg_overlay(extracted["image"], cache, overlay_edges, overlay_path, title=f"{data_id} predicted span rel-g")
                    hard_count = sum(1 for edge in pred_edges if edge.get("hard_negative"))
                    menu_price_correct = sum(1 for edge in pred_edges if edge.get("dep_field") in {"ITEM_PRICE", "MENU_PRICE"} and edge.get("correct"))
                    save_json(
                        debug_path,
                        {
                            "data_id": data_id,
                            "split": args.split,
                            "index": index,
                            "threshold": args.threshold,
                            "predicted_spans": filtered_spans,
                            "dropped_spans": dropped_spans,
                            "pred_edges": pred_edges,
                            "missed_edges": missed_edges,
                            "decoded_items": decoded.get("items", []),
                            "token_debug": layout["token_debug"],
                        },
                    )
                    gallery_cards.append(
                        {
                            "data_id": data_id,
                            "index": index,
                            "image": str(overlay_path),
                            "debug_json": str(debug_path),
                            "predicted_item_count": len(decoded.get("items", [])),
                            "gold_item_count": len(gold_decoded.get("items", [])),
                            "menu_price_correct": menu_price_correct,
                            "hard_negative_count": hard_count,
                        }
                    )
                if args.debug and index < 2:
                    print(f"\nDEBUG {data_id}")
                    print(f"  words={len(extracted['words'])} pred_spans={len(filtered_spans)} pairs={len(cache['pair_meta'])}")
                    for span in filtered_spans[:20]:
                        print(f"  span {span['field']} {span['text']!r} match={span.get('matched_gold_field')} group={span.get('group_key')}")
                    for meta, prob in list(zip(cache["pair_meta"], probs))[:20]:
                        print(f"  edge {meta['head_text']!r} -> {meta['dep_text']!r} {meta['dep_field']} y={meta['label']} p={prob:.3f}")
            except Exception as exc:
                failure = {"data_id": data_id, "split": args.split, "index": index, "error": repr(exc)}
                failure_cases.append(failure)
                print(f"WARNING: skipped {data_id}: {exc}")

    if not samples:
        fail("No samples were processed successfully.")

    summary, metrics, no_price, multiple_price, collision_errors = summarize_metrics(
        args, samples, probs_by_sample, args.threshold, word_counts, span_counts, field2id, label_source, device
    )
    save_json(out_dir / "metrics_summary.json", summary)
    save_json(out_dir / "hard_negative_errors.json", metrics.get("hard_negative_false_positive_examples", []))
    save_json(
        out_dir / "failure_cases.json",
        {
            "skipped_samples": failure_cases,
            "no_price_examples": no_price[:50],
            "multiple_price_examples": multiple_price[:50],
            "collision_examples": collision_errors[:50],
        },
    )
    save_json(
        out_dir / "run_config.json",
        {
            **vars(args),
            "resolved_relg_config_path": str(rel_config_path),
            "field_vocab_source": field_vocab["source"],
            "field_vocab_key": field_vocab["key"],
            "kind_vocab_source": kind_source,
            "layout_label_source": label_source,
            "use_angle_features": args.use_angle_features,
            "layout_model_uses_angle_features": bool(getattr(layout_model, "uses_angle_features", False)),
            "angle_feature_dim": ANGLE_FEATURE_DIM,
        },
    )
    if args.sweep_thresholds:
        save_json(out_dir / "threshold_sweep.json", threshold_sweep(samples, probs_by_sample))
    write_html_gallery(out_dir, gallery_cards)
    write_oracle_vs_predicted_report(out_dir, "outputs/span_relg_eval/metrics_summary.json", summary)

    print(f"metrics_summary path: {out_dir / 'metrics_summary.json'}")
    print(f"predicted_spans path: {predicted_spans_path}")
    print(f"edge_predictions path: {edges_path}")
    print(f"item_predictions path: {items_path}")
    print(f"span_matching_debug path: {matching_path}")
    print(f"overlay gallery path: {out_dir / 'index.html'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:5000])
    print("Predicted-span rel-g e2e evaluation passed.")


if __name__ == "__main__":
    main()
