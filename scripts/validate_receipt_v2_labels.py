import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.smoke_finetune_user_labels_v2 import load_label_schema


VALID_IGNORE = {"IGNORE", None}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate ReceiptApp2 BIO JSONL records.")
    parser.add_argument("--input", required=True, help="JSONL file to validate.")
    parser.add_argument("--label_schema", default="schemas/receipt_labels_v2.json")
    parser.add_argument("--allow_missing_images", action="store_true")
    parser.add_argument("--max_errors", type=int, default=50)
    return parser.parse_args()


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def valid_box(box):
    return (
        isinstance(box, list)
        and len(box) == 4
        and all(isinstance(value, (int, float)) for value in box)
        and float(box[2]) > float(box[0])
        and float(box[3]) > float(box[1])
    )


def main():
    args = parse_args()
    _payload, _label_list, label2id, _id2label = load_label_schema(args.label_schema)
    valid_labels = set(label2id) | {"IGNORE"}
    errors = []
    counters = Counter()
    for line_no, record in load_jsonl(args.input):
        counters["records"] += 1
        words = record.get("words")
        boxes = record.get("boxes")
        norm = record.get("normalized_boxes")
        labels = record.get("labels")
        lengths = [len(value) for value in (words, boxes, norm, labels) if isinstance(value, list)]
        if len(lengths) != 4 or len(set(lengths)) != 1:
            errors.append({"line": line_no, "id": record.get("id"), "error": "words/boxes/normalized_boxes/labels length mismatch"})
            continue
        image = record.get("image") or record.get("image_path")
        if image and not args.allow_missing_images and not Path(image).exists():
            errors.append({"line": line_no, "id": record.get("id"), "error": f"image missing: {image}"})
        for idx, label in enumerate(labels):
            if label not in valid_labels:
                errors.append({"line": line_no, "id": record.get("id"), "word_idx": idx, "error": f"invalid label: {label}"})
            counters[f"label:{label}"] += 1
        for idx, box in enumerate(boxes):
            if not valid_box(box):
                errors.append({"line": line_no, "id": record.get("id"), "word_idx": idx, "error": f"invalid box: {box}"})
        for idx, box in enumerate(norm):
            if not valid_box(box) or any(int(value) < 0 or int(value) > 1000 for value in box):
                errors.append({"line": line_no, "id": record.get("id"), "word_idx": idx, "error": f"invalid normalized box: {box}"})
        if len(errors) >= args.max_errors:
            break
    report = {
        "input": args.input,
        "records": counters["records"],
        "valid": not errors,
        "error_count": len(errors),
        "errors": errors[: args.max_errors],
        "label_counts": {key[len("label:") :]: value for key, value in counters.items() if key.startswith("label:")},
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
