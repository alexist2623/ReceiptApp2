import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ml.receipt_schema import canonicalize_label, label_to_field
except Exception:  # pragma: no cover - standalone fallback
    canonicalize_label = None

    def label_to_field(label):
        value = str(label or "O").strip()
        if not value or value == "O":
            return "O"
        if value.startswith(("B-", "I-")):
            value = value[2:]
        return value.upper().replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


RELATION_KEYS = (
    "relations",
    "item_relations",
    "summary_relations",
    "payment_relations",
    "rel_g_edges",
)

REL_G_DEP_FIELDS = {
    "ITEM_PRICE",
    "ITEM_QTY",
    "ITEM_UNIT_PRICE",
    "ITEM_DISCOUNT",
    "ITEM_CODE",
    "ITEM_SKU",
    "ITEM_CATEGORY",
    "ITEM_OPTION",
    "ITEM_TAX_FLAG",
    "ITEM_ETC",
}

SUMMARY_PAYMENT_FIELDS = {
    "SUBTOTAL_NAME",
    "SUBTOTAL_PRICE",
    "TAX_NAME",
    "TAX_RATE",
    "TAX_PRICE",
    "DISCOUNT_NAME",
    "DISCOUNT_PRICE",
    "SERVICE_NAME",
    "SERVICE_PRICE",
    "TOTAL_NAME",
    "TOTAL_PRICE",
    "CASH_NAME",
    "CASH_PRICE",
    "CHANGE_NAME",
    "CHANGE_PRICE",
    "CARD_NAME",
    "CARD_PRICE",
    "TIP_NAME",
    "TIP_PRICE",
    "PAYMENT_METHOD",
    "PAYMENT_CARD",
    "PAYMENT_AUTH_CODE",
    "PAYMENT_INFO",
    "APPROVAL_CODE",
    "TRANSACTION_ID",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Validate augmented hand-labeled receipt image/OCR/relation folders.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--label_schema", default="schemas/receipt_labels_v2.json")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--out", default="outputs/augmentation_validation_summary.json")
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def strip_label_suffix(stem):
    for suffix in ("_labeled_v2_1", "_labeled_v2", "_labeled", "_labels_v2", "_labels"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def read_json_size(payload):
    width = payload.get("image_width") or payload.get("width")
    height = payload.get("image_height") or payload.get("height")
    image = payload.get("image")
    if isinstance(image, dict):
        width = image.get("width", width)
        height = image.get("height", height)
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        return None, None


def parse_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def valid_box(box, width, height):
    parsed = parse_box(box)
    if parsed is None:
        return False
    x0, y0, x1, y1 = parsed
    return x0 >= 0 and y0 >= 0 and x1 <= width and y1 <= height


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
        result = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                continue
        return result
    return []


def word_text(words, indices):
    values = []
    for idx in normalize_indices(indices):
        if 0 <= idx < len(words):
            values.append(str(words[idx].get("text", "")))
    return " ".join(value for value in values if value)


def relation_tail_indices(relation):
    for key in (
        "tail_word_indices",
        "dep_word_indices",
        "dependent_word_indices",
        "tail_word_idx",
        "dep_word_idx",
        "dependent_word_idx",
    ):
        if key in relation:
            return normalize_indices(relation.get(key)), key
    return [], None


def relation_tail_text(relation):
    for key in ("tail_text", "dep_text", "dependent_text"):
        if key in relation:
            return str(relation.get(key) or ""), key
    return "", None


def relation_tail_field(relation):
    for key in ("tail_field", "dep_field", "dependent_field"):
        if relation.get(key) is not None:
            return str(relation.get(key)), key
    return "", None


def load_allowed_labels(path):
    schema_path = Path(path)
    if not schema_path.exists():
        return None
    schema = load_json(schema_path)
    labels = schema.get("label_list")
    return set(labels) if isinstance(labels, list) else None


def validate_bio(labels):
    errors = []
    previous_field = None
    for idx, raw_label in enumerate(labels):
        label = canonicalize_label(raw_label) if canonicalize_label else str(raw_label or "O")
        if label == "O":
            previous_field = None
            continue
        if not label.startswith(("B-", "I-")):
            errors.append(f"word {idx}: label has invalid BIO prefix: {raw_label}")
            previous_field = None
            continue
        field = label_to_field(label)
        if label.startswith("I-") and previous_field != field:
            errors.append(f"word {idx}: {label} starts without previous B/I-{field}")
        previous_field = field
    return errors


def validate_one(label_path, allowed_labels):
    folder = label_path.parent
    capture_id = strip_label_suffix(label_path.stem)
    image_path = folder / f"{capture_id}.jpg"
    ocr_path = folder / f"{capture_id}_ocr.json"
    errors = []
    warnings = []
    if not image_path.exists():
        errors.append(f"image not found: {image_path}")
    if not ocr_path.exists():
        errors.append(f"ocr json not found: {ocr_path}")
    if errors:
        return {"capture_id": capture_id, "label_json": str(label_path), "errors": errors, "warnings": warnings}

    label_payload = load_json(label_path)
    ocr_payload = load_json(ocr_path)
    with Image.open(image_path) as image:
        image_width, image_height = image.size

    for name, payload in (("label_json", label_payload), ("ocr_json", ocr_payload)):
        width, height = read_json_size(payload)
        if width != image_width or height != image_height:
            errors.append(f"{name} size {width}x{height} != image {image_width}x{image_height}")

    words = label_payload.get("words")
    if not isinstance(words, list) or not words:
        errors.append("label_json words missing/empty")
        words = []
    labels = label_payload.get("labels")
    if isinstance(labels, list):
        if len(labels) != len(words):
            errors.append(f"labels length {len(labels)} != words length {len(words)}")
    else:
        labels = [word.get("label", "O") if isinstance(word, dict) else "O" for word in words]
        warnings.append("root labels missing; using words[].label for BIO checks")

    if len(labels) == len(words):
        errors.extend(validate_bio(labels))
    label_counts = Counter()
    for idx, word in enumerate(words):
        if not isinstance(word, dict):
            errors.append(f"word {idx}: not an object")
            continue
        label = labels[idx] if idx < len(labels) else word.get("label", "O")
        label_counts[label] += 1
        if allowed_labels is not None and label not in allowed_labels:
            errors.append(f"word {idx}: label not in schema: {label}")
        if not valid_box(word.get("box"), image_width, image_height):
            errors.append(f"word {idx}: invalid box {word.get('box')}")

    ocr_words = ocr_payload.get("words")
    if isinstance(ocr_words, list) and len(ocr_words) != len(words):
        errors.append(f"ocr words length {len(ocr_words)} != label words length {len(words)}")

    relation_counts = {}
    relation_errors = []
    rel_g_summary_payment_edges = 0
    for key in RELATION_KEYS:
        values = label_payload.get(key)
        if values is None:
            relation_counts[key] = 0
            continue
        if not isinstance(values, list):
            errors.append(f"{key}: must be a list")
            continue
        relation_counts[key] = len(values)
        for ordinal, relation in enumerate(values):
            if not isinstance(relation, dict):
                relation_errors.append(f"{key}[{ordinal}]: not an object")
                continue
            relation_id = relation.get("relation_id") or relation.get("edge_id") or f"{key}[{ordinal}]"
            head_indices = normalize_indices(relation.get("head_word_indices", relation.get("head_word_idx")))
            tail_indices, tail_index_key = relation_tail_indices(relation)
            if not head_indices:
                relation_errors.append(f"{relation_id}: missing head_word_indices")
            if not tail_indices:
                relation_errors.append(f"{relation_id}: missing tail/dep word indices")
            for idx in head_indices + tail_indices:
                if idx < 0 or idx >= len(words):
                    relation_errors.append(f"{relation_id}: word index out of range: {idx}")
            head_text = str(relation.get("head_text") or "")
            if head_indices and head_text and head_text != word_text(words, head_indices):
                relation_errors.append(f"{relation_id}: head_text mismatch")
            tail_text, _ = relation_tail_text(relation)
            if tail_indices and tail_text and tail_text != word_text(words, tail_indices):
                relation_errors.append(f"{relation_id}: tail/dep text mismatch")
            if key == "rel_g_edges":
                head_field = str(relation.get("head_field") or "")
                dep_field, _ = relation_tail_field(relation)
                if head_field != "ITEM_NAME":
                    relation_errors.append(f"{relation_id}: rel_g_edges head_field must be ITEM_NAME, got {head_field}")
                if dep_field not in REL_G_DEP_FIELDS:
                    relation_errors.append(f"{relation_id}: rel_g_edges dep/tail field not allowed: {dep_field}")
                if head_field in SUMMARY_PAYMENT_FIELDS or dep_field in SUMMARY_PAYMENT_FIELDS:
                    rel_g_summary_payment_edges += 1
                    relation_errors.append(f"{relation_id}: summary/payment field appears in rel_g_edges")
                if tail_index_key in {"tail_word_idx", "dep_word_idx", "dependent_word_idx"}:
                    warnings.append(f"{relation_id}: singleton tail index key used ({tail_index_key})")
    errors.extend(relation_errors)

    return {
        "capture_id": capture_id,
        "image": str(image_path),
        "ocr_json": str(ocr_path),
        "label_json": str(label_path),
        "image_width": image_width,
        "image_height": image_height,
        "word_count": len(words),
        "label_counts": dict(label_counts),
        "relation_counts": relation_counts,
        "rel_g_summary_payment_edges": rel_g_summary_payment_edges,
        "errors": errors,
        "warnings": warnings,
    }


def validate_jsonl(input_dir, record_count):
    path = Path(input_dir) / "all_labeled_v2_1_augmented.jsonl"
    errors = []
    count = 0
    if not path.exists():
        errors.append(f"jsonl not found: {path}")
        return {"path": str(path), "count": count, "errors": errors}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                json.loads(line)
                count += 1
            except json.JSONDecodeError as exc:
                errors.append(f"line {line_number}: {exc}")
    if count != record_count:
        errors.append(f"jsonl count {count} != discovered label count {record_count}")
    return {"path": str(path), "count": count, "errors": errors}


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: input_dir not found: {input_dir}", file=sys.stderr)
        raise SystemExit(1)
    allowed_labels = load_allowed_labels(args.label_schema)
    label_paths = sorted(input_dir.rglob("*_labeled_v2_1.json"))
    results = [validate_one(path, allowed_labels) for path in label_paths]
    jsonl_summary = validate_jsonl(input_dir, len(label_paths))
    error_count = sum(len(item.get("errors", [])) for item in results) + len(jsonl_summary["errors"])
    warning_count = sum(len(item.get("warnings", [])) for item in results)
    summary = {
        "input_dir": str(input_dir),
        "label_schema": args.label_schema,
        "record_count": len(label_paths),
        "error_count": error_count,
        "warning_count": warning_count,
        "jsonl": jsonl_summary,
        "results": results,
    }
    save_json(args.out, summary)
    print(f"input_dir: {input_dir}")
    print(f"record_count: {len(label_paths)}")
    print(f"error_count: {error_count}")
    print(f"warning_count: {warning_count}")
    print(f"summary path: {args.out}")
    if error_count:
        print("first errors:")
        for item in results:
            for error in item.get("errors", [])[:5]:
                print(f"  - {item.get('capture_id')}: {error}")
        for error in jsonl_summary["errors"][:10]:
            print(f"  - jsonl: {error}")
        if args.strict:
            raise SystemExit(1)
    print("Augmented receipt validation passed.")


if __name__ == "__main__":
    main()
