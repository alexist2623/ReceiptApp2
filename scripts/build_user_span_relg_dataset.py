import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.bio_repair import repair_bio_boundaries
from ml.receipt_schema import canonicalize_field
from ml.span_relg.feature_cache import build_cache_sample, compute_word_hidden, load_layoutlmv3
from ml.span_relg.schema import ALL_FIELDS, DEP_FIELDS, HEAD_FIELDS, is_dependent_field, is_head_field
from ml.span_relg.span_utils import bio_predictions_to_spans
from scripts.smoke_finetune_user_labels_v2 import clamp_box, load_json, normalize_box, parse_box


def parse_args():
    parser = argparse.ArgumentParser(description="Build span rel-g cache from non-Temp user labeled receipt JSON files.")
    parser.add_argument(
        "--input_dir",
        required=True,
        nargs="+",
        help="One or more directories containing *_receipt_ocr folders.",
    )
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--layout_checkpoint", default="models/layoutlmv3-user-labels-non-temp/best")
    parser.add_argument("--out_dir", default="processed_data/user_span_relg_non_temp")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument("--validation_count", type=int, default=None)
    parser.add_argument("--split_manifest", default=None, help="Optional JSON/JSONL file assigning capture ids to train/validation/test.")
    parser.add_argument("--repair_bio_boundaries", action="store_true", help="Repair safe same-field B/I fragmentation before span extraction.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--include_context_tokens", default="all", choices=("all", "o_only", "none"))
    parser.add_argument("--span_pooling", default="first", choices=("first", "mean"))
    parser.add_argument("--overwrite", action="store_true")
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


def collect_label_pairs(input_dir, exclude_dir_name):
    input_dirs = input_dir if isinstance(input_dir, list) else [input_dir]
    all_labels = []
    for root in input_dirs:
        root_path = Path(root)
        if not root_path.exists():
            fail(f"input_dir not found: {root_path}")
        all_labels.extend(sorted(root_path.rglob("*_labeled_v2_1.json")))
    excluded = [path for path in all_labels if exclude_dir_name and exclude_dir_name in path.parts]
    labels = [path for path in all_labels if path not in excluded]
    pairs = []
    for label_path in labels:
        capture_id = label_path.name.replace("_labeled_v2_1.json", "")
        image_path = label_path.with_name(f"{capture_id}.jpg")
        if not image_path.exists():
            fail(f"image not found for {label_path}: {image_path}")
        pairs.append({"id": capture_id, "image": str(image_path), "label_json": str(label_path)})
    if len(pairs) < 2:
        fail("Need at least two non-Temp labeled samples for train/validation split.")
    return pairs, excluded


def split_pairs(pairs, validation_ratio, validation_count):
    pairs = list(sorted(pairs, key=lambda item: item["id"]))
    if validation_count is None:
        validation_count = max(1, int(round(len(pairs) * validation_ratio)))
    validation_count = max(1, min(int(validation_count), len(pairs) - 1))
    return {"train": pairs[:-validation_count], "validation": pairs[-validation_count:]}


def _manifest_entries(path):
    path = Path(path)
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    payload = load_json(path)
    if isinstance(payload, list):
        return payload
    entries = []
    splits = payload.get("splits", payload) if isinstance(payload, dict) else {}
    if isinstance(splits, dict):
        for split, values in splits.items():
            if isinstance(values, dict) and "records" in values:
                values = values["records"]
            if isinstance(values, dict):
                values = list(values.values())
            if not isinstance(values, list):
                continue
            for item in values:
                if isinstance(item, str):
                    entries.append({"id": item, "split": split})
                elif isinstance(item, dict):
                    entry = dict(item)
                    entry.setdefault("split", split)
                    entries.append(entry)
    return entries


def split_pairs_from_manifest(pairs, split_manifest):
    if not split_manifest:
        return None
    id_to_pair = {pair["id"]: pair for pair in pairs}
    path_to_pair = {str(Path(pair["label_json"]).resolve()): pair for pair in pairs}
    out = {}
    missing = []
    for entry in _manifest_entries(split_manifest):
        split = str(entry.get("split", "")).strip()
        if not split:
            continue
        key = entry.get("id") or entry.get("capture_id") or entry.get("data_id")
        label_path = entry.get("label_json") or entry.get("path")
        pair = id_to_pair.get(str(key)) if key is not None else None
        if pair is None and label_path:
            pair = path_to_pair.get(str(Path(label_path).resolve()))
        if pair is None:
            missing.append(entry)
            continue
        out.setdefault(split, []).append(pair)
    if missing:
        print(f"WARNING: split_manifest entries not matched: {len(missing)}")
    return out


def load_image(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def word_text(word):
    return str(word.get("text", "")).strip()


def label_for_word(payload, word, idx):
    if isinstance(word, dict) and word.get("label") is not None:
        return word.get("label")
    labels = payload.get("labels")
    if isinstance(labels, list) and idx < len(labels):
        return labels[idx]
    return "O"


def normalize_indices(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        try:
            return [int(value)]
        except ValueError:
            return []
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


def relation_items(payload):
    relations = []
    for key in ("item_relations", "summary_relations", "payment_relations"):
        values = payload.get(key)
        if isinstance(values, list):
            relations.extend(values)
    if relations:
        return relations

    all_relations = payload.get("relations")
    if isinstance(all_relations, list) and all_relations:
        return all_relations

    edges = payload.get("rel_g_edges")
    return edges if isinstance(edges, list) else []


def find_span_by_indices(spans, indices):
    wanted = set(int(idx) for idx in indices)
    if not wanted:
        return None
    best = None
    best_score = 0
    for span in spans:
        overlap = len(wanted & set(int(idx) for idx in span.get("word_indices", [])))
        if overlap > best_score:
            best = span
            best_score = overlap
    return best if best_score else None


def filter_relg_spans(spans):
    field_vocab = set(ALL_FIELDS)
    kept = []
    skipped = []
    for span in spans:
        field = canonicalize_field(span.get("field"))
        if field in field_vocab and field != "O":
            span = dict(span)
            span["field"] = field
            kept.append(span)
            continue
        skipped.append(
            {
                "span_id": span.get("span_id"),
                "field": field,
                "raw_field": span.get("raw_field", span.get("field")),
                "text": span.get("text", ""),
                "reason": "field not used by rel-g field vocab",
            }
        )
    return kept, skipped


def assign_group_keys_from_relations(spans, payload):
    for span in spans:
        span["group_key"] = None
    assigned = 0
    for rel_idx, rel in enumerate(relation_items(payload)):
        head_field = canonicalize_field(rel.get("head_field"))
        dep_field = canonicalize_field(rel.get("tail_field") or rel.get("dep_field") or rel.get("dependent_field"))
        if not is_head_field(head_field) or not is_dependent_field(dep_field):
            continue
        head_indices = normalize_indices(rel.get("head_word_indices") or rel.get("head_word_idx"))
        dep_indices = normalize_indices(
            rel.get("tail_word_indices")
            or rel.get("dep_word_indices")
            or rel.get("dependent_word_indices")
            or rel.get("tail_word_idx")
            or rel.get("dep_word_idx")
            or rel.get("dependent_word_idx")
        )
        head_span = find_span_by_indices(spans, head_indices)
        dep_span = find_span_by_indices(spans, dep_indices)
        if head_span is None or dep_span is None:
            continue
        group_key = rel.get("group_id") or head_span.get("group_key") or f"item_rel_{rel_idx:04d}"
        head_span["group_key"] = group_key
        dep_span["group_key"] = group_key
        assigned += 1
    return assigned


def make_sample_info(record, repair_labels=False):
    image = load_image(record["image"])
    width, height = image.size
    payload = load_json(record["label_json"])
    raw_words = payload.get("words")
    if not isinstance(raw_words, list):
        raise ValueError("label_json missing words list")
    words = []
    boxes = []
    normalized_boxes = []
    predictions = []
    word_original_indices = []
    skipped = []
    for idx, word in enumerate(raw_words):
        if not isinstance(word, dict):
            skipped.append({"word_idx": idx, "reason": "word is not object"})
            continue
        text = word_text(word)
        box = clamp_box(parse_box(word.get("box")), width, height)
        if not text or box is None:
            skipped.append({"word_idx": idx, "reason": "empty text or invalid box"})
            continue
        norm = normalize_box(box, width, height)
        new_idx = len(words)
        words.append(text)
        boxes.append(box)
        normalized_boxes.append(norm)
        word_original_indices.append(idx)
        predictions.append(
            {
                "text": text,
                "box": box,
                "normalized_box": norm,
                "label": label_for_word(payload, word, idx),
                "confidence": 1.0,
                "original_word_idx": idx,
                "word_idx": new_idx,
            }
        )
    repair_report = None
    if repair_labels:
        labels = [prediction["label"] for prediction in predictions]
        repaired, repair_report = repair_bio_boundaries(labels, words=words, boxes=boxes)
        for prediction, label in zip(predictions, repaired):
            prediction["label"] = label
    spans, skipped_spans = filter_relg_spans(bio_predictions_to_spans(predictions, width, height))
    original_to_new = {prediction["original_word_idx"]: idx for idx, prediction in enumerate(predictions)}
    for span in spans:
        span["original_word_indices"] = [
            predictions[idx]["original_word_idx"] for idx in span.get("word_indices", []) if idx < len(predictions)
        ]
    # Relations are stored against original word indices, so temporarily match with original indices.
    relation_spans = []
    for span in spans:
        clone = dict(span)
        clone["word_indices"] = span["original_word_indices"]
        relation_spans.append(clone)
    assigned = assign_group_keys_from_relations(relation_spans, payload)
    group_by_span_id = {span["span_id"]: span.get("group_key") for span in relation_spans}
    for span in spans:
        span["group_key"] = group_by_span_id.get(span["span_id"])
    return {
        "image": image,
        "width": width,
        "height": height,
        "words": words,
        "boxes": boxes,
        "normalized_boxes": normalized_boxes,
        "word_original_indices": word_original_indices,
        "predictions": predictions,
        "spans": spans,
        "skipped": skipped,
        "skipped_spans": skipped_spans,
        "bio_repair_report": repair_report,
        "assigned_relation_groups": assigned,
    }


def prepare_out_dir(path, overwrite):
    out_dir = Path(path)
    if out_dir.exists():
        if not overwrite:
            fail(f"{out_dir} already exists. Use --overwrite to rebuild it.")
        print(f"Removing existing output directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def main():
    args = parse_args()
    out_dir = prepare_out_dir(args.out_dir, args.overwrite)
    if args.local_files_only and not Path(args.layout_checkpoint).exists():
        fail(f"layout checkpoint not found: {args.layout_checkpoint}")
    pairs, excluded = collect_label_pairs(args.input_dir, args.exclude_dir_name)
    split_records = split_pairs_from_manifest(pairs, args.split_manifest) or split_pairs(pairs, args.validation_ratio, args.validation_count)
    print(f"input_dir: {args.input_dir}")
    print(f"layout_checkpoint: {args.layout_checkpoint}")
    print(f"out_dir: {out_dir}")
    print(f"excluded Temp labels: {len(excluded)}")
    print({split: len(records) for split, records in split_records.items()})
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    processor, layout_model, device = load_layoutlmv3(args.layout_checkpoint, args.local_files_only, args.device)
    print(f"selected device: {device}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")
    field2id = {field: idx for idx, field in enumerate(ALL_FIELDS)}
    kind2id = {"SPAN": 0, "TOKEN": 1}
    manifest = {"splits": {}, "records": []}
    summary = {
        "input_dir": args.input_dir,
        "exclude_dir_name": args.exclude_dir_name,
        "excluded_temp_count": len(excluded),
        "layout_checkpoint": args.layout_checkpoint,
        "out_dir": str(out_dir),
        "max_length": args.max_length,
        "include_context_tokens": args.include_context_tokens,
        "span_pooling": args.span_pooling,
        "split_manifest": args.split_manifest,
        "repair_bio_boundaries": args.repair_bio_boundaries,
        "splits": {},
        "skipped_samples": [],
        "skipped_spans": [],
        "field_counts": {},
        "pair_field_counts": {},
    }
    hidden_dim = None
    field_counts = Counter()
    pair_field_counts = Counter()
    for split, records in split_records.items():
        split_dir = out_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        written = []
        counters = Counter()
        for index, record in enumerate(tqdm(records, desc=f"user relg cache {split}", unit="sample")):
            data_id = record["id"]
            try:
                sample_info = make_sample_info(record, repair_labels=args.repair_bio_boundaries)
                word_features = compute_word_hidden(
                    sample_info["image"],
                    sample_info["words"],
                    sample_info["normalized_boxes"],
                    processor,
                    layout_model,
                    device,
                    max_length=args.max_length,
                )
                cache = build_cache_sample(
                    data_id,
                    split,
                    index,
                    sample_info,
                    word_features["word_hidden"],
                    field2id,
                    kind2id,
                    include_context_tokens=args.include_context_tokens,
                    span_pooling=args.span_pooling,
                )
                cache["source_image"] = record["image"]
                cache["source_label_json"] = record["label_json"]
                cache["word_token_indices"] = word_features["word_token_indices"]
                cache["encoding_shapes"] = word_features["encoding_shapes"]
                if cache["candidate_pairs"].numel() == 0:
                    raise ValueError("No candidate rel-g pairs found.")
                sample_path = split_dir / f"{data_id}.pt"
                torch.save(cache, sample_path)
                rec = {"id": data_id, "split": split, "index": index, "path": str(sample_path)}
                written.append(rec)
                manifest["records"].append(rec)
                counters["input_samples"] += 1
                counters["written_samples"] += 1
                counters["nodes"] += len(cache["nodes"])
                counters["candidate_pairs"] += int(cache["pair_labels"].numel())
                counters["positive_pairs"] += int(cache["pair_labels"].sum().item())
                counters["negative_pairs"] += int(cache["pair_labels"].numel() - cache["pair_labels"].sum().item())
                counters["assigned_relation_groups"] += sample_info["assigned_relation_groups"]
                counters["skipped_non_relg_spans"] += len(sample_info.get("skipped_spans", []))
                for skipped_span in sample_info.get("skipped_spans", [])[:20]:
                    summary["skipped_spans"].append({"id": data_id, "split": split, **skipped_span})
                if sample_info.get("bio_repair_report"):
                    counters["bio_boundary_repairs"] += sample_info["bio_repair_report"].get("num_repairs", 0)
                for node in cache["nodes"]:
                    if node.get("node_kind") == "SPAN":
                        field_counts[node.get("field")] += 1
                pair_field_counts.update(cache.get("pair_fields", []))
                if hidden_dim is None:
                    hidden_dim = int(cache["node_hidden"].shape[-1])
                if args.debug:
                    print(
                        f"{data_id}: spans={sum(1 for n in cache['nodes'] if n.get('node_kind') == 'SPAN')} "
                        f"pairs={cache['pair_labels'].numel()} positives={int(cache['pair_labels'].sum().item())}"
                    )
            except Exception as exc:
                counters["input_samples"] += 1
                counters["skipped_samples"] += 1
                issue = {"id": data_id, "split": split, "error": repr(exc)}
                summary["skipped_samples"].append(issue)
                print(f"WARNING: skipped {data_id}: {exc}")
        manifest["splits"][split] = written
        summary["splits"][split] = dict(counters)
    schema = {
        "field_list": ALL_FIELDS,
        "field2id": field2id,
        "kind2id": kind2id,
        "hidden_dim": hidden_dim,
        "candidate_head_fields": HEAD_FIELDS,
        "candidate_dep_fields": DEP_FIELDS,
        "notes": [
            "Built from user hand-labeled receipt JSON files.",
            "Temp directory inputs are excluded.",
            "LayoutLMv3 is used only as a frozen feature extractor for rel-g cache building.",
            "Pair labels are binary relation labels derived from item_relations/summary_relations/payment_relations/relations/rel_g_edges.",
            "Optional BIO boundary repair can be enabled with --repair_bio_boundaries.",
        ],
    }
    summary["field_counts"] = dict(field_counts)
    summary["pair_field_counts"] = dict(pair_field_counts)
    summary["num_cached_samples"] = len(manifest["records"])
    summary["hidden_dim"] = hidden_dim
    save_json(out_dir / "schema.json", schema)
    save_json(out_dir / "manifest.json", manifest)
    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "cache_build_report.json", summary)
    save_json(out_dir / "skipped_spans.json", summary["skipped_spans"])
    save_json(out_dir / "field_counts.json", {"field_counts": dict(field_counts), "pair_field_counts": dict(pair_field_counts)})
    save_json(
        out_dir / "relation_assignment_report.json",
        {
            "splits": {
                split: {
                    "assigned_relation_groups": values.get("assigned_relation_groups", 0),
                    "positive_pairs": values.get("positive_pairs", 0),
                    "candidate_pairs": values.get("candidate_pairs", 0),
                }
                for split, values in summary["splits"].items()
            }
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:5000])
    print(f"schema path: {out_dir / 'schema.json'}")
    print(f"manifest path: {out_dir / 'manifest.json'}")
    print(f"summary path: {out_dir / 'summary.json'}")
    print("User span rel-g cache build passed.")


if __name__ == "__main__":
    main()
