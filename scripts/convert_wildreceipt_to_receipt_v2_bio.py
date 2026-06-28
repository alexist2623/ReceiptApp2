import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import canonicalize_field, get_bio_label_list
from scripts.standardize_item_name_policy import (
    is_item_category_header_like,
    is_item_placeholder_like,
    is_loyalty_or_membership_like,
    is_service_name_like,
)


IGNORE_LABEL = "IGNORE"


VALUE_LABEL_MAP = {
    "STORE_NAME_VALUE": "STORE_NAME",
    "STORE_ADDR_VALUE": "STORE_ADDRESS",
    "STORE_ADDRESS_VALUE": "STORE_ADDRESS",
    "TEL_VALUE": "STORE_PHONE",
    "PHONE_VALUE": "STORE_PHONE",
    "DATE_VALUE": "DATE",
    "TIME_VALUE": "TIME",
    "PROD_ITEM_VALUE": "ITEM_NAME",
    "PROD_QUANTITY_VALUE": "ITEM_QTY",
    "PROD_QTY_VALUE": "ITEM_QTY",
    "PROD_PRICE_VALUE": "ITEM_PRICE",
    "PROD_UNIT_PRICE_VALUE": "ITEM_UNIT_PRICE",
    "SUBTOTAL_VALUE": "SUBTOTAL_PRICE",
    "TAX_VALUE": "TAX_PRICE",
    "TIPS_VALUE": "TIP_PRICE",
    "TIP_VALUE": "TIP_PRICE",
    "TOTAL_VALUE": "TOTAL_PRICE",
}

KEY_LABEL_MAP = {
    "SUBTOTAL_KEY": "SUBTOTAL_NAME",
    "TAX_KEY": "TAX_NAME",
    "TIPS_KEY": "TIP_NAME",
    "TIP_KEY": "TIP_NAME",
    "TOTAL_KEY": "TOTAL_NAME",
    "STORE_NAME_KEY": "STORE_NAME",
    "STORE_ADDR_KEY": "STORE_ADDRESS",
    "STORE_ADDRESS_KEY": "STORE_ADDRESS",
    "TEL_KEY": "STORE_PHONE",
    "PHONE_KEY": "STORE_PHONE",
    "DATE_KEY": "DATE",
    "TIME_KEY": "TIME",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Convert raw WildReceipt annotations to ReceiptApp2 BIO JSONL.")
    parser.add_argument("--wildreceipt_root", required=True)
    parser.add_argument("--out_dir", default="processed_data/wildreceipt_bio")
    parser.add_argument("--validation_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore_ambiguous_others", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max_samples", type=int, default=None)
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


def parse_json_line(line):
    line = line.strip()
    if not line:
        return None
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        parts = line.split("\t")
        if len(parts) >= 2:
            try:
                payload = json.loads(parts[-1])
                if isinstance(payload, dict) and "file_name" not in payload:
                    payload["file_name"] = parts[0]
                return payload
            except json.JSONDecodeError:
                return None
    return None


def read_split_file(path):
    records = []
    bad_lines = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            obj = parse_json_line(line)
            if obj is None:
                if line.strip():
                    bad_lines += 1
                continue
            records.append(obj)
    return records, bad_lines


def load_class_map(path):
    mapping = {}
    path = Path(path)
    if not path.exists():
        return mapping
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            mapping[int(parts[0])] = parts[1].strip()
        except ValueError:
            continue
    return mapping


def get_image_path(root, record):
    candidates = []
    for key in ("image", "image_path", "img_path", "file_name", "filename", "file"):
        if record.get(key):
            candidates.append(Path(str(record[key])))
    if record.get("receipt_id"):
        candidates.append(Path("image_files") / f"{record['receipt_id']}.jpg")
    for candidate in candidates:
        paths = [candidate] if candidate.is_absolute() else [root / candidate, root / "image_files" / candidate]
        for path in paths:
            if path.exists():
                return path
    return root / str(candidates[0]) if candidates else None


def get_annotations(record):
    for key in ("annotations", "words", "tokens", "items", "entities"):
        value = record.get(key)
        if isinstance(value, list):
            return value
    return []


def parse_box(annotation):
    raw = None
    for key in ("box", "bbox", "bounding_box", "boundingBox", "quad", "points", "polygon", "vertices"):
        if annotation.get(key) is not None:
            raw = annotation[key]
            break
    if raw is None:
        keys = {"x0", "y0", "x1", "y1", "left", "top", "right", "bottom", "x", "y", "width", "height"}
        if keys.intersection(annotation):
            raw = annotation
    if raw is None:
        return None
    if isinstance(raw, dict):
        if {"left", "top", "right", "bottom"}.issubset(raw):
            vals = [raw["left"], raw["top"], raw["right"], raw["bottom"]]
        elif {"x0", "y0", "x1", "y1"}.issubset(raw):
            vals = [raw["x0"], raw["y0"], raw["x1"], raw["y1"]]
        elif {"x", "y", "width", "height"}.issubset(raw):
            vals = [raw["x"], raw["y"], raw["x"] + raw["width"], raw["y"] + raw["height"]]
        elif {"x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"}.issubset(raw):
            xs = [raw[f"x{i}"] for i in range(1, 5)]
            ys = [raw[f"y{i}"] for i in range(1, 5)]
            vals = [min(xs), min(ys), max(xs), max(ys)]
        elif "vertices" in raw:
            return parse_box({"vertices": raw["vertices"]})
        else:
            return None
    elif isinstance(raw, list):
        if len(raw) == 4 and all(isinstance(item, dict) for item in raw):
            xs = [item.get("x", 0) for item in raw]
            ys = [item.get("y", 0) for item in raw]
            vals = [min(xs), min(ys), max(xs), max(ys)]
        elif len(raw) == 8:
            xs = raw[0::2]
            ys = raw[1::2]
            vals = [min(xs), min(ys), max(xs), max(ys)]
        elif len(raw) == 4:
            vals = raw
        else:
            return None
    else:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in vals]
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def clamp_box(box, width, height):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(x0), width - 1))
    x1 = max(0, min(int(x1), width - 1))
    y0 = max(0, min(int(y0), height - 1))
    y1 = max(0, min(int(y1), height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def normalize_box(box, width, height):
    x0, y0, x1, y1 = box
    out = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]
    return [max(0, min(1000, value)) for value in out]


def annotation_text(annotation):
    for key in ("text", "transcription", "word", "value"):
        if annotation.get(key) is not None:
            return str(annotation[key])
    return ""


def annotation_label(annotation, class_id_to_label=None):
    class_id_to_label = class_id_to_label or {}
    for key in ("label", "class", "category", "tag"):
        if annotation.get(key) is not None:
            value = annotation[key]
            try:
                int_value = int(value)
            except (TypeError, ValueError):
                int_value = None
            if int_value is not None and int_value in class_id_to_label:
                return class_id_to_label[int_value]
            return str(value)
    return "Others"


def repair_item_name_field_from_text(field, text):
    field = canonicalize_field(field)
    if field != "ITEM_NAME":
        return field, None
    if is_service_name_like(text):
        return "SERVICE_NAME", "item_name_policy_service_name"
    if is_loyalty_or_membership_like(text):
        return "PAYMENT_INFO", "item_name_policy_loyalty_or_membership"
    if is_item_placeholder_like(text):
        return "ITEM_ETC", "item_name_policy_placeholder"
    if is_item_category_header_like(text):
        return "ITEM_CATEGORY", "item_name_policy_category_header"
    return field, None


def canonical_field_from_wild(label, ignore_ambiguous_others=True, text=None):
    raw = str(label or "").strip()
    normalized = raw.upper().replace("-", "_").replace(" ", "_")
    if normalized in VALUE_LABEL_MAP:
        field, repair_reason = repair_item_name_field_from_text(VALUE_LABEL_MAP[normalized], text or "")
        return field, repair_reason or "mapped_value"
    if normalized in KEY_LABEL_MAP:
        return canonicalize_field(KEY_LABEL_MAP[normalized]), "mapped_key"
    if normalized in {"O", "OTHER", "OTHERS", "IGNORE", "IGNORED"}:
        return (IGNORE_LABEL if ignore_ambiguous_others else "O"), "ambiguous_other"
    return IGNORE_LABEL, "unsupported"


def same_line(prev_box, box):
    prev_center = (prev_box[1] + prev_box[3]) / 2.0
    center = (box[1] + box[3]) / 2.0
    height = max(prev_box[3] - prev_box[1], box[3] - box[1], 1)
    return abs(prev_center - center) <= height * 0.6


def close_gap(prev_box, box):
    height = max(prev_box[3] - prev_box[1], box[3] - box[1], 1)
    gap = box[0] - prev_box[2]
    return gap <= height * 3.0


def make_bio_labels(fields, boxes):
    labels = []
    prev_field = None
    prev_box = None
    for field, box in zip(fields, boxes):
        if field == IGNORE_LABEL:
            labels.append(IGNORE_LABEL)
        elif field == "O":
            labels.append("O")
        else:
            prefix = "B"
            if prev_field == field and prev_box is not None and same_line(prev_box, box) and close_gap(prev_box, box):
                prefix = "I"
            labels.append(f"{prefix}-{field}")
        prev_field = field if field not in {IGNORE_LABEL, "O"} else None
        prev_box = box if field not in {IGNORE_LABEL, "O"} else None
    return labels


def convert_record(root, record, split_name, index, args, counters):
    image_path = get_image_path(root, record)
    if image_path is None or not image_path.exists():
        counters["missing_images"] += 1
        return None
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    words = []
    boxes = []
    normalized_boxes = []
    fields = []
    raw_labels = []
    for ann in get_annotations(record):
        if not isinstance(ann, dict):
            counters["invalid_annotations"] += 1
            continue
        text = annotation_text(ann)
        if not text.strip():
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
        normalized_boxes.append(normalize_box(box, width, height))
        fields.append(field)
        raw_labels.append(raw_label)
    if not words:
        counters["empty_records"] += 1
        return None
    labels = make_bio_labels(fields, boxes)
    receipt_id = record.get("receipt_id") or record.get("id") or Path(image_path).stem
    return {
        "source": "wildreceipt",
        "id": f"wildreceipt_{split_name}_{index:06d}",
        "receipt_id": str(receipt_id),
        "image": str(image_path),
        "words": words,
        "boxes": boxes,
        "normalized_boxes": normalized_boxes,
        "labels": labels,
        "raw_labels": raw_labels,
        "image_width": width,
        "image_height": height,
    }


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    root = Path(args.wildreceipt_root)
    if not root.exists():
        fail(f"wildreceipt_root not found: {root}")
    train_path = root / "train.txt"
    test_path = root / "test.txt"
    class_list_path = root / "class_list.txt"
    if not train_path.exists() or not test_path.exists():
        fail("wildreceipt_root must contain train.txt and test.txt")
    args.class_id_to_label = load_class_map(class_list_path)

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
    valid_labels = set(get_bio_label_list()) | {IGNORE_LABEL}
    counters = Counter({"bad_train_lines": bad_train, "bad_test_lines": bad_test})
    converted = {}
    for split_name, rows in (("train", train_raw), ("validation", validation_raw), ("test", test_raw)):
        records = []
        for index, record in enumerate(rows):
            converted_record = convert_record(root, record, split_name, index, args, counters)
            if converted_record is None:
                continue
            unknown = [label for label in converted_record["labels"] if label not in valid_labels]
            if unknown:
                fail(f"Invalid emitted labels in {converted_record['id']}: {unknown[:10]}")
            records.append(converted_record)
        converted[split_name] = records
        write_jsonl(out_dir / f"{split_name}.jsonl", records)

    label_counts = Counter()
    field_counts = Counter()
    raw_counts = Counter()
    for records in converted.values():
        for record in records:
            label_counts.update(record["labels"])
            raw_counts.update(record.get("raw_labels", []))
            for label in record["labels"]:
                if label in {"O", IGNORE_LABEL}:
                    field_counts[label] += 1
                else:
                    field_counts[label[2:]] += 1
    save_json(out_dir / "label_counts.json", {
        "bio_label_counts": dict(label_counts),
        "canonical_field_counts": dict(field_counts),
        "raw_label_counts": dict(raw_counts),
    })
    class_list = []
    if class_list_path.exists():
        class_list = [line.strip() for line in class_list_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    save_json(out_dir / "schema_report.json", {
        "wildreceipt_root": str(root),
        "out_dir": str(out_dir),
        "validation_ratio": args.validation_ratio,
        "seed": args.seed,
        "ignore_ambiguous_others": args.ignore_ambiguous_others,
        "split_counts": {split: len(records) for split, records in converted.items()},
        "class_list": class_list,
        "class_id_to_label": {str(key): value for key, value in args.class_id_to_label.items()},
        "mapping": {"value": VALUE_LABEL_MAP, "key": KEY_LABEL_MAP},
        "counters": dict(counters),
        "notes": [
            "Unsupported or ambiguous WildReceipt labels are emitted as IGNORE by default.",
            "Training code must mask IGNORE word labels to -100.",
        ],
    })
    print(json.dumps({"split_counts": {split: len(records) for split, records in converted.items()}, "counters": dict(counters)}, indent=2))
    print(f"wrote: {out_dir}")


if __name__ == "__main__":
    main()
