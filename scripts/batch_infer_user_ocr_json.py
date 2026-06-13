import argparse
import csv
import html
import json
import re
import sys
from collections import Counter
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import canonicalize_label, label_to_field

from infer_user_ocr_json import (
    compare_model_labels,
    fail,
    load_image,
    load_labels,
    load_ocr_json,
    prepare_ocr_words,
    run_inference,
    save_json,
    save_overlay,
    select_device,
    raw_label_to_field,
)


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
JSON_EXTENSIONS = (".json",)
OCR_SUFFIX_PATTERNS = (
    "_ocr",
    "-ocr",
    ".ocr",
    "_ocr_result",
    "-ocr-result",
    "_ocr_results",
    "-ocr-results",
    "_vision",
    "-vision",
    "_mlkit",
    "-mlkit",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Batch LayoutLMv3 inference for user receipt image/OCR JSON pairs."
    )
    parser.add_argument("--input_dir", default="user_receipts", help="Directory containing images and OCR JSON files.")
    parser.add_argument("--image_dir", default=None, help="Optional image directory. Defaults to input_dir.")
    parser.add_argument("--ocr_dir", default=None, help="Optional OCR JSON directory. Defaults to input_dir.")
    parser.add_argument("--pairs_csv", default=None, help="Optional CSV with image and ocr_json columns.")
    parser.add_argument("--checkpoint", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--labels", default="processed_data/cord_bio/labels.json")
    parser.add_argument("--out_dir", default="outputs/user_ocr_batch_inference")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Discover pairs and write empty summary without inference.")
    parser.add_argument(
        "--show_text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show OCR text in overlay labels.",
    )
    parser.add_argument("--max_text_len", type=int, default=25)
    parser.add_argument("--hide_o", action="store_true")
    parser.add_argument("--draw_conf_threshold", type=float, default=0.0)
    parser.add_argument("--assume_boxes_normalized", action="store_true")
    parser.add_argument("--box_format", default="auto", choices=("auto", "xyxy", "xywh", "quad"))
    parser.add_argument("--image_width_key", default="image_width")
    parser.add_argument("--image_height_key", default="image_height")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def require_path(path, message):
    if not Path(path).exists():
        fail(message)


def safe_stem(path):
    stem = Path(path).stem
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return stem or "receipt"


def normalized_pair_key(path):
    stem = Path(path).stem
    lowered = stem.lower()
    changed = True
    while changed:
        changed = False
        for suffix in OCR_SUFFIX_PATTERNS:
            if lowered.endswith(suffix):
                stem = stem[: -len(suffix)]
                lowered = lowered[: -len(suffix)]
                changed = True
    return re.sub(r"[\s._-]+", "", lowered)


def iter_files(directory, extensions, recursive=True):
    directory = Path(directory)
    if not directory.exists():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    )


def load_pairs_csv(path):
    pairs = []
    base = Path(path).parent
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "ocr_json"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            fail(f"{path} is missing required CSV columns: {sorted(missing)}")
        for row_index, row in enumerate(reader, start=1):
            image_path = Path(row["image"])
            ocr_path = Path(row["ocr_json"])
            if not image_path.is_absolute():
                image_path = base / image_path
            if not ocr_path.is_absolute():
                ocr_path = base / ocr_path
            pairs.append(
                {
                    "id": row.get("id") or safe_stem(image_path),
                    "image": image_path,
                    "ocr_json": ocr_path,
                    "source": f"pairs_csv:{row_index}",
                    "warnings": [],
                }
            )
    return pairs


def discover_pairs(args):
    if args.pairs_csv:
        require_path(args.pairs_csv, f"pairs_csv not found: {args.pairs_csv}")
        return load_pairs_csv(args.pairs_csv)

    input_dir = Path(args.input_dir)
    image_dir = Path(args.image_dir) if args.image_dir else input_dir
    ocr_dir = Path(args.ocr_dir) if args.ocr_dir else input_dir

    images = iter_files(image_dir, IMAGE_EXTENSIONS, recursive=args.recursive)
    jsons = iter_files(ocr_dir, JSON_EXTENSIONS, recursive=args.recursive)
    json_by_key = {}
    for json_path in jsons:
        json_by_key.setdefault(normalized_pair_key(json_path), []).append(json_path)

    pairs = []
    for image_path in images:
        key = normalized_pair_key(image_path)
        candidates = json_by_key.get(key, [])
        warnings = []
        if not candidates:
            direct_candidates = [
                image_path.with_suffix(".json"),
                image_path.with_name(f"{image_path.stem}_ocr.json"),
                image_path.with_name(f"{image_path.stem}.ocr.json"),
            ]
            candidates = [path for path in direct_candidates if path.exists()]
        if not candidates:
            continue
        candidates = sorted(candidates)
        if len(candidates) > 1:
            warnings.append(f"Multiple OCR JSON candidates found; using {candidates[0]}")
        pairs.append(
            {
                "id": safe_stem(image_path),
                "image": image_path,
                "ocr_json": candidates[0],
                "source": "stem_match",
                "warnings": warnings,
            }
        )
    return pairs


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "id",
        "status",
        "image_path",
        "ocr_json_path",
        "prediction_json",
        "overlay_png",
        "ocr_debug_json",
        "valid_word_count",
        "skipped_empty_text_count",
        "skipped_invalid_box_count",
        "top_labels",
        "o_ratio",
        "avg_confidence",
        "min_confidence",
        "max_confidence",
        "warning_count",
        "error",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def make_predictions_payload(image_path, ocr_json_path, checkpoint, image, predictions, label_counts):
    width, height = image.size
    return {
        "image_path": str(image_path),
        "ocr_json_path": str(ocr_json_path),
        "checkpoint": str(checkpoint),
        "image_width": width,
        "image_height": height,
        "num_words": len(predictions),
        "predictions": predictions,
        "label_counts": dict(label_counts),
    }


def summarize_confidence(predictions):
    if not predictions:
        return None, None, None
    values = [float(item["confidence"]) for item in predictions]
    return sum(values) / len(values), min(values), max(values)


def process_pair(pair, args, processor, model, device, id2label, out_dir):
    image_path = Path(pair["image"])
    ocr_json_path = Path(pair["ocr_json"])
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if not ocr_json_path.exists():
        raise FileNotFoundError(f"OCR JSON not found: {ocr_json_path}")

    image = load_image(image_path)
    ocr_obj = load_ocr_json(ocr_json_path)
    words, pixel_boxes, normalized_boxes, metadata, ocr_debug = prepare_ocr_words(ocr_obj, image.size, args)

    pred_labels, confidences, token_debug, word_ids = run_inference(
        image=image,
        words=words,
        normalized_boxes=normalized_boxes,
        processor=processor,
        model=model,
        device=device,
        id2label=id2label,
        max_length=args.max_length,
    )

    predictions = []
    for idx, word in enumerate(words):
        raw_label = pred_labels[idx]
        canonical_label = canonicalize_label(raw_label)
        predictions.append(
            {
                "word_idx": idx,
                "text": word,
                "box": pixel_boxes[idx],
                "normalized_box": normalized_boxes[idx],
                "label": raw_label,
                "canonical_label": canonical_label,
                "field": raw_label_to_field(raw_label),
                "canonical_field": label_to_field(canonical_label),
                "confidence": confidences[idx],
                "source": metadata[idx]["source"],
                "line_index": metadata[idx]["line_index"],
                "block_index": metadata[idx]["block_index"],
            }
        )

    label_counts = Counter(pred_labels)
    safe_id = safe_stem(pair["id"])
    prediction_path = out_dir / "predictions" / f"{safe_id}_prediction.json"
    overlay_path = out_dir / "overlays" / f"{safe_id}_overlay.png"
    debug_path = out_dir / "debug" / f"{safe_id}_ocr_debug.json"

    save_json(
        prediction_path,
        make_predictions_payload(image_path, ocr_json_path, args.checkpoint, image, predictions, label_counts),
    )
    save_json(debug_path, ocr_debug)
    save_overlay(image, predictions, args, overlay_path)

    avg_conf, min_conf, max_conf = summarize_confidence(predictions)
    o_count = label_counts.get("O", 0)
    top_labels = ", ".join(f"{label}:{count}" for label, count in label_counts.most_common(8))
    warnings = list(pair.get("warnings") or []) + list(ocr_debug.get("warnings") or [])
    if o_count == len(predictions) and predictions:
        warnings.append("All predictions are O; possible label collapse or out-of-domain OCR input.")

    return {
        "id": pair["id"],
        "status": "ok",
        "image_path": str(image_path),
        "ocr_json_path": str(ocr_json_path),
        "prediction_json": str(prediction_path),
        "overlay_png": str(overlay_path),
        "ocr_debug_json": str(debug_path),
        "valid_word_count": len(predictions),
        "skipped_empty_text_count": ocr_debug.get("skipped_empty_text_count", 0),
        "skipped_invalid_box_count": ocr_debug.get("skipped_invalid_box_count", 0),
        "label_counts": dict(label_counts),
        "top_labels": top_labels,
        "o_ratio": o_count / len(predictions) if predictions else 0.0,
        "avg_confidence": avg_conf,
        "min_confidence": min_conf,
        "max_confidence": max_conf,
        "warnings": warnings,
        "warning_count": len(warnings),
        "first_predictions": predictions[:30],
        "error": "",
    }


def html_rel(path, base):
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except ValueError:
        return Path(path).as_posix()


def write_gallery(path, rows, out_dir):
    cards = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        overlay_rel = html_rel(row["overlay_png"], out_dir)
        prediction_rel = html_rel(row["prediction_json"], out_dir)
        debug_rel = html_rel(row["ocr_debug_json"], out_dir)
        warnings = "<br>".join(html.escape(str(w)) for w in row.get("warnings", [])) or "none"
        preview_items = row.get("first_predictions", [])[:12]
        preview = "<br>".join(
            html.escape(
                f"{item['word_idx']:02d} {item['text']} -> {item['label']} ({item['confidence']:.2f})"
            )
            for item in preview_items
        )
        cards.append(
            f"""
            <section class="card">
              <h2>{html.escape(str(row['id']))}</h2>
              <img src="{html.escape(overlay_rel)}" alt="{html.escape(str(row['id']))} overlay">
              <p><b>words</b>: {row.get('valid_word_count', 0)} | <b>O ratio</b>: {row.get('o_ratio', 0):.3f} | <b>avg conf</b>: {row.get('avg_confidence') or 0:.3f}</p>
              <p><b>top labels</b>: {html.escape(row.get('top_labels', ''))}</p>
              <p><b>warnings</b>: {warnings}</p>
              <p><a href="{html.escape(prediction_rel)}">prediction JSON</a> | <a href="{html.escape(debug_rel)}">OCR debug JSON</a></p>
              <pre>{preview}</pre>
            </section>
            """
        )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>User OCR LayoutLMv3 Batch Gallery</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f7f7f4; color: #202124; }}
    h1 {{ margin-bottom: 4px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 18px; }}
    .card {{ background: white; border: 1px solid #d7d7d2; border-radius: 8px; padding: 14px; }}
    img {{ width: 100%; max-height: 720px; object-fit: contain; background: #111; }}
    pre {{ white-space: pre-wrap; font-size: 12px; background: #f1f1ee; padding: 8px; }}
  </style>
</head>
<body>
  <h1>User OCR LayoutLMv3 Batch Gallery</h1>
  <p>Generated overlays for quick visual inspection of OCR boxes and predicted labels.</p>
  <div class="grid">
    {''.join(cards)}
  </div>
</body>
</html>
"""
    Path(path).write_text(content, encoding="utf-8")


def write_empty_outputs(args, pairs, out_dir, reason):
    rows = []
    summary = {
        "checkpoint": args.checkpoint,
        "labels": args.labels,
        "input_dir": args.input_dir,
        "image_dir": args.image_dir,
        "ocr_dir": args.ocr_dir,
        "pairs_csv": args.pairs_csv,
        "num_pairs_found": len(pairs),
        "num_processed": 0,
        "num_failed": 0,
        "reason": reason,
        "rows": rows,
    }
    save_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "summary.csv", rows)
    write_gallery(out_dir / "gallery.html", rows, out_dir)
    return summary


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"WSL/conda Python path: {sys.executable}")
    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"cuda device name: {torch.cuda.get_device_name(0)}")
    device = select_device(args.device)
    print(f"selected device: {device}")
    print(f"checkpoint path: {args.checkpoint}")
    print(f"labels.json path: {args.labels}")
    print(f"input_dir: {args.input_dir}")
    print(f"image_dir: {args.image_dir or args.input_dir}")
    print(f"ocr_dir: {args.ocr_dir or args.input_dir}")

    require_path(args.checkpoint, f"Checkpoint not found: {args.checkpoint}. Run step 5 first.")
    require_path(args.labels, f"labels.json not found: {args.labels}. Run step 3 first.")

    pairs = discover_pairs(args)
    if args.limit is not None:
        pairs = pairs[: args.limit]
    print(f"discovered pair count: {len(pairs)}")
    for pair in pairs[:10]:
        print(f"  pair id={pair['id']} image={pair['image']} ocr_json={pair['ocr_json']}")

    run_config = {
        "input_dir": args.input_dir,
        "image_dir": args.image_dir,
        "ocr_dir": args.ocr_dir,
        "pairs_csv": args.pairs_csv,
        "checkpoint": args.checkpoint,
        "labels": args.labels,
        "out_dir": args.out_dir,
        "max_length": args.max_length,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "recursive": args.recursive,
        "limit": args.limit,
        "dry_run": args.dry_run,
        "assume_boxes_normalized": args.assume_boxes_normalized,
        "box_format": args.box_format,
        "hide_o": args.hide_o,
        "draw_conf_threshold": args.draw_conf_threshold,
    }
    save_json(out_dir / "run_config.json", run_config)

    if not pairs:
        print("No user receipt image/OCR JSON pairs found. Skipping actual inference.")
        write_empty_outputs(args, pairs, out_dir, "no_pairs_found")
        print(f"summary JSON path: {out_dir / 'summary.json'}")
        print(f"summary CSV path: {out_dir / 'summary.csv'}")
        print(f"HTML gallery path: {out_dir / 'gallery.html'}")
        return

    if args.dry_run:
        print("Dry run requested. Skipping model load and inference.")
        write_empty_outputs(args, pairs, out_dir, "dry_run")
        print(f"summary JSON path: {out_dir / 'summary.json'}")
        print(f"summary CSV path: {out_dir / 'summary.csv'}")
        print(f"HTML gallery path: {out_dir / 'gallery.html'}")
        return

    label_list, _label2id, id2label = load_labels(args.labels)
    print(f"labels.json num_labels: {len(label_list)}")
    print("processor apply_ocr=False")
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        apply_ocr=False,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint,
        local_files_only=args.local_files_only,
    )
    compare_model_labels(model, label_list, id2label)
    model.to(device)
    model.eval()

    rows = []
    failures = 0
    for pair in pairs:
        print(f"processing: {pair['id']}")
        try:
            row = process_pair(pair, args, processor, model, device, id2label, out_dir)
        except Exception as exc:
            failures += 1
            print(f"ERROR processing {pair['id']}: {exc}", file=sys.stderr)
            row = {
                "id": pair["id"],
                "status": "error",
                "image_path": str(pair["image"]),
                "ocr_json_path": str(pair["ocr_json"]),
                "prediction_json": "",
                "overlay_png": "",
                "ocr_debug_json": "",
                "valid_word_count": 0,
                "skipped_empty_text_count": 0,
                "skipped_invalid_box_count": 0,
                "top_labels": "",
                "o_ratio": "",
                "avg_confidence": "",
                "min_confidence": "",
                "max_confidence": "",
                "warnings": pair.get("warnings", []),
                "warning_count": len(pair.get("warnings", [])),
                "first_predictions": [],
                "error": str(exc),
            }
        rows.append(row)

    total_words = sum(int(row.get("valid_word_count") or 0) for row in rows if row.get("status") == "ok")
    aggregate_labels = Counter()
    for row in rows:
        aggregate_labels.update(row.get("label_counts", {}))
    summary = {
        "checkpoint": args.checkpoint,
        "labels": args.labels,
        "input_dir": args.input_dir,
        "image_dir": args.image_dir,
        "ocr_dir": args.ocr_dir,
        "pairs_csv": args.pairs_csv,
        "num_pairs_found": len(pairs),
        "num_processed": sum(1 for row in rows if row.get("status") == "ok"),
        "num_failed": failures,
        "num_words": total_words,
        "aggregate_label_counts": dict(aggregate_labels),
        "aggregate_o_ratio": aggregate_labels.get("O", 0) / total_words if total_words else 0.0,
        "rows": rows,
    }
    save_json(out_dir / "summary.json", summary)
    write_csv(out_dir / "summary.csv", rows)
    write_gallery(out_dir / "gallery.html", rows, out_dir)

    print(f"processed count: {summary['num_processed']}")
    print(f"failed count: {summary['num_failed']}")
    print(f"total word count: {summary['num_words']}")
    print(f"aggregate label distribution: {aggregate_labels.most_common(20)}")
    print(f"summary JSON path: {out_dir / 'summary.json'}")
    print(f"summary CSV path: {out_dir / 'summary.csv'}")
    print(f"HTML gallery path: {out_dir / 'gallery.html'}")
    print("User OCR batch inference step passed.")


if __name__ == "__main__":
    main()
