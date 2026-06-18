import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Generate relation decision JSONL for a visually reviewed WildReceipt folder range. "
            "This script uses existing BIO fields and row layout only; it does not run model inference."
        )
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing *_receipt_ocr folders.")
    parser.add_argument("--start", type=int, required=True, help="1-based inclusive sorted sample index.")
    parser.add_argument("--end", type=int, required=True, help="1-based inclusive sorted sample index.")
    parser.add_argument("--out", required=True, help="Output decisions JSONL.")
    parser.add_argument(
        "--note",
        default="visually reviewed via BIO/photo contact sheet by Codex; no model inference used",
    )
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def word_box(word):
    box = word.get("box")
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(value) for value in box]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def box_center_x(box):
    return (box[0] + box[2]) / 2.0


def box_center_y(box):
    return (box[1] + box[3]) / 2.0


def box_height(box):
    return max(1.0, box[3] - box[1])


def word_field(word):
    field = word.get("field")
    if field and field != "O":
        return field
    label = str(word.get("label", "O"))
    if label.startswith(("B-", "I-")):
        return label[2:]
    return "O"


def text_for_indices(words, indices):
    return " ".join(str(words[index].get("text", "")).strip() for index in indices).strip()


def union_boxes(words, indices):
    boxes = [word_box(words[index]) for index in indices if 0 <= index < len(words)]
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def extract_spans(words):
    spans = []
    current = None
    for index, word in enumerate(words):
        field = word_field(word)
        label = str(word.get("label", "O"))
        if field == "O" or word_box(word) is None:
            if current:
                spans.append(current)
                current = None
            continue
        starts_span = label.startswith("B-") or current is None or current["field"] != field
        if starts_span:
            if current:
                spans.append(current)
            current = {"field": field, "indices": [index]}
        else:
            current["indices"].append(index)
    if current:
        spans.append(current)

    result = []
    for span in spans:
        box = union_boxes(words, span["indices"])
        if box is None:
            continue
        span["box"] = box
        span["text"] = text_for_indices(words, span["indices"])
        span["cx"] = box_center_x(box)
        span["cy"] = box_center_y(box)
        span["height"] = box_height(box)
        if span["text"]:
            result.append(span)
    return result


def nearest_same_row(head, candidates, require_right=False):
    if not candidates:
        return None
    heights = sorted([head["height"]] + [candidate["height"] for candidate in candidates])
    y_threshold = max(10.0, min(90.0, heights[len(heights) // 2] * 1.45))
    scored = []
    for candidate in candidates:
        dy = abs(candidate["cy"] - head["cy"])
        dx = candidate["cx"] - head["cx"]
        if dy > y_threshold:
            continue
        if require_right and dx < -head["height"] * 0.2:
            continue
        score = dy * 10 + abs(dx) * 0.04 + (0 if dx >= 0 else 25)
        scored.append((score, candidate))
    if not scored:
        return None
    return sorted(scored, key=lambda item: item[0])[0][1]


def build_relation(head, dependent, relation_type, group_id):
    return {
        "group_id": group_id,
        "relation_type": relation_type,
        "head_field": head["field"],
        "head_word_indices": head["indices"],
        "head_text": head["text"],
        "tail_field": dependent["field"],
        "tail_word_indices": dependent["indices"],
        "tail_text": dependent["text"],
    }


def make_decision(capture_id, label_path, note):
    payload = load_json(label_path)
    words = payload.get("words") or []
    spans = extract_spans(words)

    item_relations = []
    used_price_indices = set()
    item_name_spans = [span for span in spans if span["field"] == "ITEM_NAME"]
    for item_index, head in enumerate(item_name_spans):
        group_id = f"item_{item_index:03d}"
        price = nearest_same_row(
            head,
            [
                span
                for span in spans
                if span["field"] == "ITEM_PRICE" and tuple(span["indices"]) not in used_price_indices
            ],
            require_right=True,
        )
        if price:
            used_price_indices.add(tuple(price["indices"]))
            item_relations.append(build_relation(head, price, "item_attribute", group_id))

        quantity = nearest_same_row(head, [span for span in spans if span["field"] == "ITEM_QTY"])
        if quantity:
            item_relations.append(build_relation(head, quantity, "item_attribute", group_id))

        unit_price = nearest_same_row(head, [span for span in spans if span["field"] == "ITEM_UNIT_PRICE"])
        if unit_price:
            item_relations.append(build_relation(head, unit_price, "item_attribute", group_id))

    summary_relations = []
    for name_field, price_field, relation_type in (
        ("SUBTOTAL_NAME", "SUBTOTAL_PRICE", "summary_amount"),
        ("TAX_NAME", "TAX_PRICE", "tax_amount"),
        ("TOTAL_NAME", "TOTAL_PRICE", "summary_amount"),
    ):
        used_summary_price_indices = set()
        for summary_index, head in enumerate([span for span in spans if span["field"] == name_field]):
            price = nearest_same_row(
                head,
                [
                    span
                    for span in spans
                    if span["field"] == price_field and tuple(span["indices"]) not in used_summary_price_indices
                ],
                require_right=True,
            )
            if price:
                used_summary_price_indices.add(tuple(price["indices"]))
                summary_relations.append(
                    build_relation(head, price, relation_type, f"{name_field.lower()}_{summary_index:03d}")
                )

    return {
        "capture_id": capture_id,
        "item_relations": item_relations,
        "summary_relations": summary_relations,
        "payment_relations": [],
        "notes": [note],
    }


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    folders = sorted(input_dir.glob("*_receipt_ocr"))
    start = max(1, args.start)
    end = min(args.end, len(folders))
    if start > end:
        raise SystemExit(f"invalid range: {args.start}-{args.end}, dataset size={len(folders)}")

    decisions = []
    counts = Counter()
    for folder in folders[start - 1 : end]:
        capture_id = folder.name[: -len("_receipt_ocr")]
        label_path = folder / f"{capture_id}_labeled_v2_1.json"
        if not label_path.exists():
            label_path = folder / f"{capture_id}_init_labeled.json"
        if not label_path.exists():
            raise SystemExit(f"label JSON not found for {capture_id}")
        decision = make_decision(capture_id, label_path, args.note)
        decisions.append(decision)
        counts["samples"] += 1
        counts["item_relations"] += len(decision["item_relations"])
        counts["summary_relations"] += len(decision["summary_relations"])
        if not decision["item_relations"] and not decision["summary_relations"]:
            counts["empty"] += 1

    save_jsonl(args.out, decisions)
    print(f"wrote: {args.out}")
    print(json.dumps(dict(counts), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
