import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import get_bio_label_list

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Create a schema-v2 user OCR labeling template JSONL.")
    parser.add_argument("--input_dir", default="user_receipts")
    parser.add_argument("--out_jsonl", default="labeling/user_receipts_label_template.jsonl")
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_box(box):
    if isinstance(box, list) and len(box) == 4:
        return [int(round(float(value))) for value in box]
    return None


def extract_words(payload):
    words = payload.get("words") if isinstance(payload, dict) else None
    if not isinstance(words, list):
        return []
    out = []
    for idx, item in enumerate(words):
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        box = safe_box(item.get("box") or item.get("bbox"))
        if not text or box is None:
            continue
        out.append({"word_idx": len(out), "text": text, "box": box, "source_word_idx": idx})
    return out


def find_image_for_json(json_path):
    stem = json_path.stem
    for suffix in ("_ocr", "-ocr"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    for ext in IMAGE_EXTENSIONS:
        candidate = json_path.with_name(stem + ext)
        if candidate.exists():
            return candidate
    return None


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"input_dir not found: {input_dir}")
    allowed_labels = get_bio_label_list()
    records = []
    for json_path in sorted(input_dir.rglob("*.json")):
        payload = load_json(json_path)
        words = extract_words(payload)
        if not words:
            continue
        image_path = find_image_for_json(json_path)
        width = payload.get("image_width") or payload.get("width") or payload.get("image", {}).get("width")
        height = payload.get("image_height") or payload.get("height") or payload.get("image", {}).get("height")
        capture_id = payload.get("captureId") or payload.get("capture_id") or json_path.stem.replace("_ocr", "")
        records.append(
            {
                "capture_id": capture_id,
                "image_path": str(image_path) if image_path else None,
                "ocr_json_path": str(json_path),
                "image_width": width,
                "image_height": height,
                "words": words,
                "labels": ["O"] * len(words),
                "allowed_labels": allowed_labels,
                "labeling_notes": [
                    "Do not merge OCR tokens.",
                    "For split price, label '$' as B-ITEM_PRICE and '16.99' as I-ITEM_PRICE.",
                    "For store name, label all store name tokens as B/I-STORE_NAME.",
                    "For item name, label product/category/item text as B/I-ITEM_NAME.",
                    "For total price, use B/I-TOTAL_PRICE.",
                    "For tax, use B/I-TAX_PRICE.",
                ],
            }
        )
    out = Path(args.out_jsonl)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"labeling template path: {out}")
    print(f"records: {len(records)}")


if __name__ == "__main__":
    main()
