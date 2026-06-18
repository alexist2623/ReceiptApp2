import argparse
import json
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import SCHEMA_VERSION, canonicalize_field, field_to_bio


WILDRECEIPT_TO_FIELD = {
    "Ignore": "O",
    "Store_name_value": "STORE_NAME",
    "Store_name_key": "STORE_NAME",
    "Store_addr_value": "STORE_ADDRESS",
    "Store_addr_key": "STORE_ADDRESS",
    "Tel_value": "STORE_PHONE",
    "Tel_key": "STORE_PHONE",
    "Date_value": "DATE",
    "Date_key": "DATE",
    "Time_value": "TIME",
    "Time_key": "TIME",
    "Prod_item_value": "ITEM_NAME",
    "Prod_item_key": "O",
    "Prod_quantity_value": "ITEM_QTY",
    "Prod_quantity_key": "O",
    "Prod_price_value": "ITEM_PRICE",
    "Prod_price_key": "O",
    "Subtotal_value": "SUBTOTAL_PRICE",
    "Subtotal_key": "SUBTOTAL_NAME",
    "Tax_value": "TAX_PRICE",
    "Tax_key": "TAX_NAME",
    "Tips_value": "TIP_PRICE",
    "Tips_key": "TIP_NAME",
    "Total_value": "TOTAL_PRICE",
    "Total_key": "TOTAL_NAME",
    "Others": "O",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert WildReceipt JSONL annotations into the repo's *_receipt_ocr folder layout."
    )
    parser.add_argument(
        "--wildreceipt_dir",
        required=True,
        help="Directory containing train.txt, test.txt, class_list.txt, and image_files/.",
    )
    parser.add_argument(
        "--out_dir",
        required=True,
        help="Output directory for <capture_id>_receipt_ocr folders.",
    )
    parser.add_argument("--splits", nargs="+", default=["train", "test"], help="Split txt files to convert.")
    parser.add_argument(
        "--label_suffix",
        default="_init_labeled.json",
        help="Primary requested label JSON suffix.",
    )
    parser.add_argument(
        "--write_labeled_v2_1_copy",
        action="store_true",
        help="Also write <capture_id>_labeled_v2_1.json for existing training scripts.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Debug limit per split.")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_class_map(path):
    if not path.exists():
        fail(f"class_list.txt not found: {path}")
    mapping = {}
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            continue
        try:
            mapping[int(parts[0])] = parts[1].strip()
        except ValueError:
            continue
    if not mapping:
        fail(f"Could not parse class map from {path}")
    return mapping


def quad_to_box(quad):
    if not isinstance(quad, list) or len(quad) not in {4, 8}:
        return None
    try:
        values = [float(value) for value in quad]
    except (TypeError, ValueError):
        return None
    if len(values) == 4:
        x0, y0, x1, y1 = values
        xs = [x0, x1]
        ys = [y0, y1]
    else:
        xs = values[0::2]
        ys = values[1::2]
    x0, x1 = int(round(min(xs))), int(round(max(xs)))
    y0, y1 = int(round(min(ys))), int(round(max(ys)))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def clamp_box(box, width, height):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(x0), int(width) - 1))
    x1 = max(0, min(int(x1), int(width) - 1))
    y0 = max(0, min(int(y0), int(height) - 1))
    y1 = max(0, min(int(y1), int(height) - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def normalize_box(box, width, height):
    x0, y0, x1, y1 = box
    values = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]
    return [max(0, min(value, 1000)) for value in values]


def same_line(prev, current):
    if prev is None:
        return False
    prev_box = prev.get("box")
    cur_box = current.get("box")
    if not prev_box or not cur_box:
        return False
    prev_cy = (prev_box[1] + prev_box[3]) / 2.0
    cur_cy = (cur_box[1] + cur_box[3]) / 2.0
    prev_h = max(1, prev_box[3] - prev_box[1])
    cur_h = max(1, cur_box[3] - cur_box[1])
    return abs(prev_cy - cur_cy) <= max(8.0, min(prev_h, cur_h) * 0.65)


def assign_bio_labels(words):
    previous_by_field = {}
    labels = []
    for item in words:
        field = canonicalize_field(item.get("field"))
        if field == "O":
            label = "O"
        else:
            b_label, i_label = field_to_bio(field)
            prev = previous_by_field.get(field)
            label = i_label if same_line(prev, item) else b_label
            previous_by_field[field] = item
        labels.append(label)
    return labels


def safe_capture_id(split, image_rel_path, used_ids):
    stem = Path(image_rel_path).stem
    capture_id = stem
    if capture_id not in used_ids:
        used_ids.add(capture_id)
        return capture_id
    prefixed = f"wildreceipt_{split}_{stem}"
    if prefixed not in used_ids:
        used_ids.add(prefixed)
        return prefixed
    index = 2
    while f"{prefixed}_{index}" in used_ids:
        index += 1
    capture_id = f"{prefixed}_{index}"
    used_ids.add(capture_id)
    return capture_id


def convert_record(record, class_map, wildreceipt_dir, split, out_dir, capture_id):
    width = int(record["width"])
    height = int(record["height"])
    image_rel_path = record["file_name"]
    source_image = wildreceipt_dir / image_rel_path
    if not source_image.exists():
        return None, {"reason": "image missing", "image": str(source_image)}

    converted_words = []
    skipped = []
    for word_idx, annotation in enumerate(record.get("annotations", [])):
        if not isinstance(annotation, dict):
            skipped.append({"word_idx": word_idx, "reason": "annotation is not an object"})
            continue
        text = str(annotation.get("text", ""))
        raw_label_id = annotation.get("label")
        raw_label_name = class_map.get(int(raw_label_id), f"UNKNOWN_{raw_label_id}") if raw_label_id is not None else "UNKNOWN"
        field = canonicalize_field(WILDRECEIPT_TO_FIELD.get(raw_label_name, "O"))
        box = clamp_box(quad_to_box(annotation.get("box")), width, height)
        if not text.strip():
            skipped.append({"word_idx": word_idx, "reason": "empty text", "wildreceipt_label": raw_label_name})
            continue
        if box is None:
            skipped.append({"word_idx": word_idx, "text": text, "reason": "invalid box", "box": annotation.get("box")})
            continue
        converted_words.append(
            {
                "word_idx": len(converted_words),
                "text": text,
                "box": box,
                "normalized_box": normalize_box(box, width, height),
                "field": field,
                "wildreceipt_label_id": raw_label_id,
                "wildreceipt_label_name": raw_label_name,
                "source_box_quad": annotation.get("box"),
            }
        )

    labels = assign_bio_labels(converted_words)
    for item, label in zip(converted_words, labels):
        item["label"] = label

    folder = out_dir / f"{capture_id}_receipt_ocr"
    image_out = folder / f"{capture_id}.jpg"
    label_payload = {
        "schema_version": SCHEMA_VERSION,
        "capture_id": capture_id,
        "source_dataset": "wildreceipt",
        "source_split": split,
        "source_file_name": image_rel_path,
        "image_width": width,
        "image_height": height,
        "words": converted_words,
        "labels": labels,
        "relations": [],
        "item_relations": [],
        "summary_relations": [],
        "payment_relations": [],
        "rel_g_edges": [],
        "conversion": {
            "label_mapping": WILDRECEIPT_TO_FIELD,
            "primary_label_suffix": None,
            "skipped_annotations": skipped,
        },
    }
    ocr_payload = {
        "schemaVersion": "wildreceipt_to_user_ocr_v1",
        "captureId": capture_id,
        "image_width": width,
        "image_height": height,
        "words": [
            {
                "text": item["text"],
                "box": item["box"],
                "source": "wildreceipt",
                "wildreceipt_label_id": item["wildreceipt_label_id"],
                "wildreceipt_label_name": item["wildreceipt_label_name"],
            }
            for item in converted_words
        ],
    }
    return {
        "folder": folder,
        "image_source": source_image,
        "image_out": image_out,
        "label_payload": label_payload,
        "ocr_payload": ocr_payload,
        "word_count": len(converted_words),
        "skipped_count": len(skipped),
        "label_counts": dict(Counter(labels)),
        "field_counts": dict(Counter(item["field"] for item in converted_words)),
    }, None


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    wildreceipt_dir = Path(args.wildreceipt_dir)
    out_dir = Path(args.out_dir)
    if not wildreceipt_dir.exists():
        fail(f"wildreceipt_dir not found: {wildreceipt_dir}")
    if not (wildreceipt_dir / "image_files").exists():
        fail(f"image_files directory not found under: {wildreceipt_dir}")

    class_map = load_class_map(wildreceipt_dir / "class_list.txt")
    out_dir.mkdir(parents=True, exist_ok=True)

    used_ids = set()
    manifest = {
        "source_dir": str(wildreceipt_dir.resolve()),
        "out_dir": str(out_dir.resolve()),
        "label_suffix": args.label_suffix,
        "write_labeled_v2_1_copy": bool(args.write_labeled_v2_1_copy),
        "splits": {},
        "records": [],
        "label_counts": {},
        "field_counts": {},
        "errors": [],
    }
    total_labels = Counter()
    total_fields = Counter()

    for split in args.splits:
        split_path = wildreceipt_dir / f"{split}.txt"
        if not split_path.exists():
            fail(f"split file not found: {split_path}")
        split_records = []
        split_errors = []
        split_label_counts = Counter()
        split_field_counts = Counter()
        with split_path.open("r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if args.limit is not None and index >= args.limit:
                    break
                raw = line.strip()
                if not raw:
                    continue
                try:
                    record = json.loads(raw)
                except json.JSONDecodeError as exc:
                    split_errors.append({"line": index + 1, "reason": f"json decode error: {exc}"})
                    continue
                capture_id = safe_capture_id(split, record.get("file_name", f"{split}_{index:06d}"), used_ids)
                converted, error = convert_record(record, class_map, wildreceipt_dir, split, out_dir, capture_id)
                if error:
                    error.update({"split": split, "line": index + 1, "capture_id": capture_id})
                    split_errors.append(error)
                    continue
                if not args.dry_run:
                    folder = converted["folder"]
                    if folder.exists() and not args.overwrite:
                        fail(f"output folder already exists: {folder}. Use --overwrite to replace JSON/image files.")
                    folder.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(converted["image_source"], converted["image_out"])
                    primary_label = folder / f"{capture_id}{args.label_suffix}"
                    converted["label_payload"]["conversion"]["primary_label_suffix"] = args.label_suffix
                    write_json(primary_label, converted["label_payload"])
                    if args.write_labeled_v2_1_copy:
                        write_json(folder / f"{capture_id}_labeled_v2_1.json", converted["label_payload"])
                    write_json(folder / f"{capture_id}_ocr.json", converted["ocr_payload"])
                record_summary = {
                    "capture_id": capture_id,
                    "split": split,
                    "folder": str((out_dir / f"{capture_id}_receipt_ocr").resolve()),
                    "image": f"{capture_id}.jpg",
                    "label_json": f"{capture_id}{args.label_suffix}",
                    "labeled_v2_1_json": f"{capture_id}_labeled_v2_1.json" if args.write_labeled_v2_1_copy else None,
                    "ocr_json": f"{capture_id}_ocr.json",
                    "source_file_name": record.get("file_name"),
                    "word_count": converted["word_count"],
                    "skipped_annotation_count": converted["skipped_count"],
                }
                split_records.append(record_summary)
                manifest["records"].append(record_summary)
                split_label_counts.update(converted["label_counts"])
                split_field_counts.update(converted["field_counts"])
                total_labels.update(converted["label_counts"])
                total_fields.update(converted["field_counts"])

        split_manifest = {
            "num_records": len(split_records),
            "num_errors": len(split_errors),
            "label_counts": dict(split_label_counts),
            "field_counts": dict(split_field_counts),
            "records": split_records,
            "errors": split_errors,
        }
        manifest["splits"][split] = {
            "num_records": split_manifest["num_records"],
            "num_errors": split_manifest["num_errors"],
            "label_counts": split_manifest["label_counts"],
            "field_counts": split_manifest["field_counts"],
            "manifest_file": f"{split}_manifest.json",
        }
        manifest["errors"].extend({"split": split, **error} for error in split_errors)
        if not args.dry_run:
            write_json(out_dir / f"{split}_manifest.json", split_manifest)

    manifest["label_counts"] = dict(total_labels)
    manifest["field_counts"] = dict(total_fields)
    manifest["num_records"] = len(manifest["records"])
    manifest["num_errors"] = len(manifest["errors"])
    if not args.dry_run:
        write_json(out_dir / "manifest.json", manifest)

    print(f"source_dir: {wildreceipt_dir.resolve()}")
    print(f"out_dir: {out_dir.resolve()}")
    print(f"converted records: {manifest['num_records']}")
    print(f"errors: {manifest['num_errors']}")
    for split, info in manifest["splits"].items():
        print(f"{split}: records={info['num_records']} errors={info['num_errors']}")
    print("top labels:", total_labels.most_common(20))
    print("top fields:", total_fields.most_common(20))
    print("done.")


if __name__ == "__main__":
    main()
