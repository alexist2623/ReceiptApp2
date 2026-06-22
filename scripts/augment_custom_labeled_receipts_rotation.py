import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.angle_geometry import (
    angle_deg_from_quad,
    box_to_quad,
    clamp_box,
    normalize_box_1000,
    parse_quad,
    quad_to_axis_aligned_box,
    rotate_image_and_quads,
)
from ml.receipt_schema import canonicalize_label, get_bio_label_list


IGNORE_LABEL = "IGNORE"


def parse_args():
    parser = argparse.ArgumentParser(description="Rotate hand-labeled receipt JSON into angle-aware BIO JSONL.")
    parser.add_argument("--input_dir", required=True, help="Directory containing *_receipt_ocr folders.")
    parser.add_argument("--out_dir", default="processed_data/custom_rotated_receipt_v2_bio")
    parser.add_argument("--overlay_dir", default="outputs/custom_rotated_receipt_overlay")
    parser.add_argument("--rotation_degrees", default="-10,-5,0,5,10")
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--validation_count", type=int, default=3)
    parser.add_argument("--test_count", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overlay_limit", type=int, default=10)
    parser.add_argument("--jpeg_quality", type=int, default=95)
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


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_degrees(value):
    degrees = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if item:
            degrees.append(float(item))
    if not degrees:
        fail("--rotation_degrees produced no values")
    return degrees


def collect_pairs(input_dir, exclude_dir_name):
    root = Path(input_dir)
    if not root.exists():
        fail(f"input_dir not found: {root}")
    labels = sorted(root.rglob("*_labeled_v2_1.json"))
    excluded = [path for path in labels if exclude_dir_name and exclude_dir_name in path.parts]
    labels = [path for path in labels if path not in excluded]
    pairs = []
    missing_images = []
    for label_path in labels:
        capture_id = label_path.name.replace("_labeled_v2_1.json", "")
        image_path = label_path.with_name(f"{capture_id}.jpg")
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        pairs.append({"capture_id": capture_id, "image": image_path, "label_json": label_path})
    return pairs, excluded, missing_images


def split_pairs(pairs, validation_count, test_count, seed):
    pairs = list(sorted(pairs, key=lambda item: item["capture_id"]))
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if len(pairs) <= 2:
        return {"train": pairs, "validation": [], "test": []}
    test_count = max(0, min(int(test_count), len(pairs) - 2))
    validation_count = max(0, min(int(validation_count), len(pairs) - test_count - 1))
    test = pairs[:test_count]
    validation = pairs[test_count : test_count + validation_count]
    train = pairs[test_count + validation_count :]
    return {"train": train, "validation": validation, "test": test}


def word_text(word):
    for key in ("text", "value", "word"):
        if isinstance(word, dict) and word.get(key) is not None:
            return str(word.get(key)).strip()
    return ""


def word_label(payload, word, idx):
    label = word.get("label") if isinstance(word, dict) else None
    if label is None and isinstance(payload.get("labels"), list) and idx < len(payload["labels"]):
        label = payload["labels"][idx]
    return canonicalize_label(label or "O")


def word_quad_or_box(word):
    for key in ("quad", "cornerPoints", "corner_points", "vertices", "points", "polygon"):
        quad = parse_quad(word.get(key)) if isinstance(word, dict) else None
        if quad:
            return quad
    box = quad_to_axis_aligned_box(word.get("box")) if isinstance(word, dict) else None
    return box_to_quad(box)


def load_source_sample(pair, counters):
    image = ImageOps.exif_transpose(Image.open(pair["image"])).convert("RGB")
    width, height = image.size
    payload = load_json(pair["label_json"])
    json_width = payload.get("image_width") or payload.get("width")
    json_height = payload.get("image_height") or payload.get("height")
    if json_width and json_height and (int(json_width) != width or int(json_height) != height):
        counters["image_size_mismatch"] += 1
    raw_words = payload.get("words")
    if not isinstance(raw_words, list):
        counters["missing_words"] += 1
        return None
    valid_labels = set(get_bio_label_list()) | {IGNORE_LABEL}
    words = []
    boxes = []
    quads = []
    labels = []
    original_indices = []
    word_payloads = []
    for idx, word in enumerate(raw_words):
        if not isinstance(word, dict):
            counters["invalid_word_objects"] += 1
            continue
        text = word_text(word)
        label = word_label(payload, word, idx)
        if label not in valid_labels:
            counters[f"unknown_label:{label}"] += 1
            continue
        quad = word_quad_or_box(word)
        box = clamp_box(quad_to_axis_aligned_box(quad), width, height)
        if not text:
            counters["empty_text"] += 1
            continue
        if box is None or quad is None:
            counters["invalid_boxes"] += 1
            continue
        words.append(text)
        boxes.append(box)
        quads.append(quad)
        labels.append(label)
        original_indices.append(idx)
        word_payloads.append(word)
    if not words:
        counters["empty_samples"] += 1
        return None
    return {
        "capture_id": pair["capture_id"],
        "image": image,
        "image_path": pair["image"],
        "label_json": pair["label_json"],
        "width": width,
        "height": height,
        "words": words,
        "boxes": boxes,
        "quads": quads,
        "labels": labels,
        "original_indices": original_indices,
        "word_payloads": word_payloads,
    }


def flat_quad(quad):
    values = []
    for x, y in quad:
        values.extend([int(round(float(x))), int(round(float(y)))])
    return values


def color_for_label(label):
    if label == "O":
        return (120, 120, 120)
    field = label[2:] if label.startswith(("B-", "I-")) else label
    if field.startswith("ITEM_NAME"):
        return (40, 160, 80)
    if field.startswith("ITEM_PRICE"):
        return (30, 105, 210)
    if field.startswith("TAX"):
        return (230, 125, 25)
    if field.startswith("TOTAL"):
        return (220, 45, 70)
    if field.startswith("SUBTOTAL"):
        return (175, 55, 200)
    return (70, 140, 180)


def draw_rectangle(draw, box, color, width=2):
    try:
        draw.rectangle(box, outline=color, width=width)
    except TypeError:
        for offset in range(width):
            draw.rectangle([box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset], outline=color)


def draw_overlay(image, record, out_path, *, limit_text=18):
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    font = ImageFont.load_default()
    for idx, (word, label, box, payload) in enumerate(
        zip(record["words"], record["labels"], record["boxes"], record["word_payloads"])
    ):
        color = color_for_label(label)
        quad = parse_quad(payload.get("quad"))
        if quad:
            points = [(int(round(x)), int(round(y))) for x, y in quad]
            try:
                draw.line(points + [points[0]], fill=color, width=3)
            except TypeError:
                draw.line(points + [points[0]], fill=color)
        draw_rectangle(draw, box, color, width=1)
        if label != "O":
            text = f"{idx}:{label} {str(word)[:limit_text]}"
            x0, y0, _x1, _y1 = box
            tx = max(0, int(x0))
            ty = max(0, int(y0) - 12)
            draw.rectangle([tx, ty, tx + min(360, len(text) * 6 + 4), ty + 11], fill=(255, 255, 255))
            draw.text((tx + 2, ty), text, fill=color, font=font)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(out_path)


def convert_variant(source, split, variant_idx, degree, out_dir, overlay_dir, overlay_slots, args, counters):
    if abs(float(degree)) < 1e-9:
        rotated_image = source["image"].copy()
        rotated_quads = source["quads"]
        matrix = {"angle_deg": 0.0, "input_width": source["width"], "input_height": source["height"]}
    else:
        rotated_image, rotated_quads, matrix = rotate_image_and_quads(source["image"], source["quads"], degree)
    width, height = rotated_image.size
    variant_id = f"{source['capture_id']}_rot_{variant_idx:02d}_{int(round(float(degree) * 10)):04d}"
    image_rel = Path("images") / split / f"{variant_id}.jpg"
    image_out = out_dir / image_rel
    image_out.parent.mkdir(parents=True, exist_ok=True)
    rotated_image.save(image_out, quality=args.jpeg_quality)

    words = []
    boxes = []
    normalized_boxes = []
    labels = []
    word_payloads = []
    original_indices = []
    for word, label, original_idx, quad, source_payload, source_box in zip(
        source["words"],
        source["labels"],
        source["original_indices"],
        rotated_quads,
        source["word_payloads"],
        source["boxes"],
    ):
        box = clamp_box(quad_to_axis_aligned_box(quad), width, height)
        if box is None:
            counters["rotated_invalid_boxes"] += 1
            continue
        normalized = normalize_box_1000(box, width, height)
        angle = angle_deg_from_quad(quad)
        payload = dict(source_payload)
        payload.update(
            {
                "text": word,
                "box": box,
                "normalized_box": normalized,
                "quad": flat_quad(quad),
                "angle_deg": angle,
                "rotation_deg": float(degree),
                "source_image_id": source["capture_id"],
                "source_word_index": original_idx,
                "source_box": source_box,
                "label": label,
            }
        )
        words.append(word)
        boxes.append(box)
        normalized_boxes.append(normalized)
        labels.append(label)
        word_payloads.append(payload)
        original_indices.append(original_idx)
    if not words:
        counters["empty_rotated_records"] += 1
        return None
    record = {
        "source": "custom_rotated",
        "id": variant_id,
        "capture_id": source["capture_id"],
        "source_image_id": source["capture_id"],
        "source_image": str(source["image_path"]),
        "source_label_json": str(source["label_json"]),
        "image": str(image_out),
        "split": split,
        "rotation_deg": float(degree),
        "rotation_matrix": matrix,
        "image_width": width,
        "image_height": height,
        "words": words,
        "boxes": boxes,
        "normalized_boxes": normalized_boxes,
        "word_payloads": word_payloads,
        "labels": labels,
        "original_word_indices": original_indices,
    }
    if len(overlay_slots) < args.overlay_limit:
        overlay_path = overlay_dir / f"{variant_id}_overlay.jpg"
        draw_overlay(rotated_image, record, overlay_path)
        record["overlay_path"] = str(overlay_path)
        overlay_slots.append(str(overlay_path))
    return record


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    overlay_dir = Path(args.overlay_dir)
    if args.overwrite:
        for path in (out_dir, overlay_dir):
            if path.exists():
                import shutil

                shutil.rmtree(path)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)
    degrees = parse_degrees(args.rotation_degrees)
    pairs, excluded, missing_images = collect_pairs(args.input_dir, args.exclude_dir_name)
    if not pairs:
        fail("No custom labeled pairs found.")
    splits = split_pairs(pairs, args.validation_count, args.test_count, args.seed)
    counters = Counter()
    converted = {}
    overlays = []
    for split, split_pairs_list in splits.items():
        rows = []
        for pair in split_pairs_list:
            source = load_source_sample(pair, counters)
            if source is None:
                continue
            for variant_idx, degree in enumerate(degrees):
                row = convert_variant(source, split, variant_idx, degree, out_dir, overlay_dir, overlays, args, counters)
                if row is not None:
                    rows.append(row)
        converted[split] = rows
        write_jsonl(out_dir / f"{split}.jsonl", rows)
    label_counts = Counter()
    rotation_counts = Counter()
    for rows in converted.values():
        for row in rows:
            label_counts.update(row["labels"])
            rotation_counts.update([str(row["rotation_deg"])])
    report = {
        "input_dir": str(Path(args.input_dir)),
        "out_dir": str(out_dir),
        "overlay_dir": str(overlay_dir),
        "rotation_degrees": degrees,
        "exclude_dir_name": args.exclude_dir_name,
        "excluded_count": len(excluded),
        "missing_images": missing_images,
        "split_pair_counts": {split: len(values) for split, values in splits.items()},
        "split_record_counts": {split: len(rows) for split, rows in converted.items()},
        "label_counts": dict(label_counts),
        "rotation_counts": dict(rotation_counts),
        "overlay_paths": overlays,
        "counters": dict(counters),
        "notes": [
            "This dataset is for LayoutLMv3 BIO training, not rel-g training.",
            "Each word_payload contains rotated axis-aligned box, rotated quad, angle_deg, and rotation_deg.",
            "Temp directory inputs are excluded by default.",
        ],
    }
    save_json(out_dir / "schema_report.json", report)
    save_json(overlay_dir / "overlay_summary.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2)[:8000])
    print("Custom labeled receipt rotation augmentation passed.")


if __name__ == "__main__":
    main()
