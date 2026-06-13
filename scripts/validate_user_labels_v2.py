import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate schema-v2 user labeled OCR JSONL.")
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--schema", default="schemas/receipt_labels_v2.json")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def valid_box(box, width, height):
    if not isinstance(box, list) or len(box) != 4:
        return False
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return False
    if width and (x0 < 0 or x1 > width):
        return False
    if height and (y0 < 0 or y1 > height):
        return False
    return True


def main():
    args = parse_args()
    schema = load_json(args.schema)
    allowed = set(schema["label_list"])
    label_counts = Counter()
    field_counts = Counter()
    warnings = []
    records = 0
    store_samples = item_samples = item_price_samples = 0
    with Path(args.jsonl).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            records += 1
            record = json.loads(line)
            words = record.get("words", [])
            labels = record.get("labels", [])
            width = record.get("image_width")
            height = record.get("image_height")
            if len(words) != len(labels):
                warnings.append(f"line {line_number}: labels length {len(labels)} != words length {len(words)}")
                continue
            previous_field = None
            fields_in_sample = set()
            for idx, (word, label) in enumerate(zip(words, labels)):
                if label not in allowed:
                    warnings.append(f"line {line_number} word {idx}: unknown label {label}")
                    continue
                label_counts[label] += 1
                field = label[2:] if label.startswith(("B-", "I-")) else "O"
                field_counts[field] += 1
                fields_in_sample.add(field)
                if label.startswith("I-") and previous_field != field:
                    warnings.append(f"line {line_number} word {idx}: {label} starts without previous B/I-{field}")
                previous_field = field if label.startswith(("B-", "I-")) else None
                if not valid_box(word.get("box"), width, height):
                    warnings.append(f"line {line_number} word {idx}: invalid box {word.get('box')}")
            if "STORE_NAME" in fields_in_sample:
                store_samples += 1
            if "ITEM_NAME" in fields_in_sample:
                item_samples += 1
            if "ITEM_PRICE" in fields_in_sample:
                item_price_samples += 1

    summary = {
        "jsonl": args.jsonl,
        "schema": args.schema,
        "records": records,
        "label_counts": dict(label_counts),
        "field_counts": dict(field_counts),
        "store_name_sample_count": store_samples,
        "item_name_sample_count": item_samples,
        "item_price_sample_count": item_price_samples,
        "warning_count": len(warnings),
        "warnings": warnings[:200],
    }
    out = Path(args.out) if args.out else Path(args.jsonl).with_suffix(".validation_summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"validation summary path: {out}")
    print(f"records: {records}")
    print(f"warning_count: {len(warnings)}")


if __name__ == "__main__":
    main()
