import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.angle_geometry import (
    angle_deg_from_quad,
    box_to_quad,
    clamp_box,
    normalize_box_1000,
    quad_to_axis_aligned_box,
    rotate_image_and_quads,
)
from ml.receipt_schema import get_bio_label_list
from scripts.convert_wildreceipt_to_receipt_v2_bio import (
    IGNORE_LABEL,
    annotation_label,
    annotation_text,
    canonical_field_from_wild,
    get_annotations,
    get_image_path,
    load_class_map,
    make_bio_labels,
    parse_box,
    read_split_file,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create angle-aware ReceiptApp2 BIO JSONL from WildReceipt by rotating images and word boxes."
    )
    parser.add_argument("--wildreceipt_root", required=True)
    parser.add_argument("--out_dir", default="processed_data/wildreceipt_rotated_receipt_v2_bio")
    parser.add_argument("--rotation_degrees", default="-10,-5,0,5,10")
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore_ambiguous_others", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--jpeg_quality", type=int, default=95)
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


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_degrees(value):
    degrees = []
    for item in str(value).replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        degrees.append(float(item))
    if not degrees:
        fail("--rotation_degrees produced no values")
    return degrees


def source_annotations(root, record, args, counters):
    image_path = get_image_path(root, record)
    if image_path is None or not image_path.exists():
        counters["missing_images"] += 1
        return None
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    words = []
    boxes = []
    fields = []
    raw_labels = []
    annotations = []
    for ann in get_annotations(record):
        if not isinstance(ann, dict):
            counters["invalid_annotations"] += 1
            continue
        text = annotation_text(ann).strip()
        if not text:
            counters["empty_text"] += 1
            continue
        box = clamp_box(parse_box(ann), width, height)
        if box is None:
            counters["invalid_boxes"] += 1
            continue
        raw_label = annotation_label(ann, getattr(args, "class_id_to_label", {}))
        field, reason = canonical_field_from_wild(raw_label, args.ignore_ambiguous_others, text=text)
        counters[f"map_reason:{reason}"] += 1
        words.append(text)
        boxes.append(box)
        fields.append(field)
        raw_labels.append(raw_label)
        annotations.append(ann)
    if not words:
        counters["empty_records"] += 1
        return None
    receipt_id = record.get("receipt_id") or record.get("id") or Path(image_path).stem
    return {
        "image": image,
        "image_path": image_path,
        "width": width,
        "height": height,
        "receipt_id": str(receipt_id),
        "words": words,
        "boxes": boxes,
        "fields": fields,
        "raw_labels": raw_labels,
        "annotations": annotations,
    }


def flat_quad(quad):
    values = []
    for x, y in quad:
        values.extend([int(round(float(x))), int(round(float(y)))])
    return values


def convert_variant(source, split_name, source_index, variant_index, rotation_deg, out_dir, args, counters):
    source_quads = [box_to_quad(box) for box in source["boxes"]]
    if any(quad is None for quad in source_quads):
        counters["missing_source_quads"] += 1
        return None
    if abs(float(rotation_deg)) < 1e-9:
        rotated_image = source["image"].copy()
        rotated_quads = source_quads
        matrix = {
            "input_width": source["width"],
            "input_height": source["height"],
            "output_width": source["width"],
            "output_height": source["height"],
            "angle_deg": 0.0,
        }
    else:
        rotated_image, rotated_quads, matrix = rotate_image_and_quads(source["image"], source_quads, rotation_deg)
    width, height = rotated_image.size
    boxes = []
    normalized_boxes = []
    word_payloads = []
    keep_words = []
    keep_fields = []
    keep_raw_labels = []
    keep_labels_source_boxes = []
    for word_idx, (word, field, raw_label, quad, source_box) in enumerate(
        zip(source["words"], source["fields"], source["raw_labels"], rotated_quads, source["boxes"])
    ):
        if quad is None:
            counters["rotation_missing_quad"] += 1
            continue
        box = clamp_box(quad_to_axis_aligned_box(quad), width, height)
        if box is None:
            counters["rotated_invalid_boxes"] += 1
            continue
        angle = angle_deg_from_quad(quad)
        normalized = normalize_box_1000(box, width, height)
        quad_values = flat_quad(quad)
        keep_words.append(word)
        keep_fields.append(field)
        keep_raw_labels.append(raw_label)
        keep_labels_source_boxes.append(box)
        boxes.append(box)
        normalized_boxes.append(normalized)
        word_payloads.append(
            {
                "text": word,
                "box": box,
                "normalized_box": normalized,
                "quad": quad_values,
                "angle_deg": angle,
                "rotation_deg": float(rotation_deg),
                "source_image_id": source["receipt_id"],
                "source_word_index": word_idx,
                "source_box": source_box,
                "raw_label": raw_label,
                "field": field,
            }
        )
    if not keep_words:
        counters["empty_rotated_records"] += 1
        return None
    labels = make_bio_labels(keep_fields, keep_labels_source_boxes)
    variant_id = f"wildreceipt_{split_name}_{source_index:06d}_rot_{variant_index:02d}_{int(round(float(rotation_deg) * 10)):04d}"
    image_rel = Path("images") / split_name / f"{variant_id}.jpg"
    image_out = out_dir / image_rel
    image_out.parent.mkdir(parents=True, exist_ok=True)
    rotated_image.save(image_out, quality=args.jpeg_quality)
    return {
        "source": "wildreceipt_rotated",
        "id": variant_id,
        "receipt_id": source["receipt_id"],
        "source_image_id": source["receipt_id"],
        "source_image": str(source["image_path"]),
        "image": str(image_out),
        "split": split_name,
        "source_index": source_index,
        "rotation_deg": float(rotation_deg),
        "rotation_matrix": matrix,
        "image_width": width,
        "image_height": height,
        "words": keep_words,
        "boxes": boxes,
        "normalized_boxes": normalized_boxes,
        "word_payloads": word_payloads,
        "labels": labels,
        "raw_labels": keep_raw_labels,
    }


def main():
    args = parse_args()
    root = Path(args.wildreceipt_root)
    if not root.exists():
        fail(f"wildreceipt_root not found: {root}")
    train_path = root / "train.txt"
    test_path = root / "test.txt"
    if not train_path.exists() or not test_path.exists():
        fail("wildreceipt_root must contain train.txt and test.txt")
    args.class_id_to_label = load_class_map(root / "class_list.txt")
    rotation_degrees = parse_degrees(args.rotation_degrees)
    train_raw, bad_train = read_split_file(train_path)
    test_raw, bad_test = read_split_file(test_path)
    if args.max_samples:
        train_raw = train_raw[: args.max_samples]
        test_raw = test_raw[: args.max_samples]
    rng = random.Random(args.seed)
    train_raw = list(train_raw)
    rng.shuffle(train_raw)
    val_count = max(1, int(round(len(train_raw) * args.validation_ratio))) if len(train_raw) > 1 else 0
    validation_raw = train_raw[:val_count]
    train_raw = train_raw[val_count:]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    counters = Counter({"bad_train_lines": bad_train, "bad_test_lines": bad_test})
    valid_labels = set(get_bio_label_list()) | {IGNORE_LABEL}
    converted = {}
    for split_name, rows in (("train", train_raw), ("validation", validation_raw), ("test", test_raw)):
        records = []
        for source_index, record in enumerate(rows):
            source = source_annotations(root, record, args, counters)
            if source is None:
                continue
            for variant_index, degree in enumerate(rotation_degrees):
                converted_record = convert_variant(source, split_name, source_index, variant_index, degree, out_dir, args, counters)
                if converted_record is None:
                    continue
                unknown = [label for label in converted_record["labels"] if label not in valid_labels]
                if unknown:
                    fail(f"Invalid emitted labels in {converted_record['id']}: {unknown[:10]}")
                records.append(converted_record)
        converted[split_name] = records
        write_jsonl(out_dir / f"{split_name}.jsonl", records)

    label_counts = Counter()
    raw_counts = Counter()
    angle_counts = Counter()
    for records in converted.values():
        for record in records:
            label_counts.update(record["labels"])
            raw_counts.update(record.get("raw_labels", []))
            angle_counts.update([str(record.get("rotation_deg"))])
    save_json(
        out_dir / "schema_report.json",
        {
            "wildreceipt_root": str(root),
            "out_dir": str(out_dir),
            "rotation_degrees": rotation_degrees,
            "validation_ratio": args.validation_ratio,
            "seed": args.seed,
            "split_counts": {split: len(records) for split, records in converted.items()},
            "bio_label_counts": dict(label_counts),
            "raw_label_counts": dict(raw_counts),
            "rotation_counts": dict(angle_counts),
            "counters": dict(counters),
            "notes": [
                "Each record stores axis-aligned boxes plus word_payloads[].quad and angle_deg.",
                "Original WildReceipt boxes have no token-level angle; rotation_deg is synthetic whole-page augmentation.",
                "Unsupported or ambiguous WildReceipt labels are emitted as IGNORE by default.",
            ],
        },
    )
    print(
        json.dumps(
            {
                "split_counts": {split: len(records) for split, records in converted.items()},
                "rotation_degrees": rotation_degrees,
                "counters": dict(counters),
                "out_dir": str(out_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
