import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ml.receipt_schema import canonicalize_label, label_to_field as schema_label_to_field, normalize_span_text
except Exception:  # pragma: no cover - fallback keeps this inspection tool standalone.
    canonicalize_label = None
    schema_label_to_field = None

    def normalize_span_text(field, text):
        return " ".join(str(text or "").split())


FIELD_COLORS = {
    "STORE_NAME": (255, 0, 255),
    "STORE_ADDRESS": (128, 0, 128),
    "STORE_PHONE": (138, 43, 226),
    "STORE_ID": (0, 82, 204),
    "ITEM_CATEGORY": (255, 140, 0),
    "ITEM_NAME": (0, 150, 72),
    "ITEM_CODE": (0, 188, 212),
    "ITEM_SKU": (0, 188, 212),
    "ITEM_PRICE": (220, 38, 38),
    "ITEM_UNIT_PRICE": (255, 87, 34),
    "ITEM_QTY": (214, 170, 0),
    "ITEM_TAX_FLAG": (121, 85, 72),
    "SUBTOTAL_PRICE": (250, 128, 114),
    "TAX_PRICE": (255, 69, 0),
    "TOTAL_PRICE": (255, 20, 147),
    "CARD_PRICE": (255, 105, 180),
    "CHANGE_PRICE": (148, 0, 211),
    "PAYMENT_METHOD": (14, 165, 233),
    "PAYMENT_INFO": (107, 114, 128),
    "PAYMENT_CARD": (65, 105, 225),
    "PAYMENT_AUTH_CODE": (0, 0, 128),
    "RECEIPT_ID": (0, 128, 128),
    "REGISTER_ID": (0, 128, 128),
    "DATE": (0, 100, 0),
    "TIME": (0, 100, 0),
    "O": (190, 190, 190),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay manually labeled receipt BIO JSON on the source image.")
    parser.add_argument("--image", required=True, help="Original receipt image path.")
    parser.add_argument("--label_json", required=True, help="Manual labeled receipt JSON path.")
    parser.add_argument("--out", default=None, help="Overlay PNG path.")
    parser.add_argument("--summary_out", default=None, help="Label summary JSON path.")
    parser.add_argument("--show_o", action="store_true", help="Draw O labels as light gray boxes.")
    parser.add_argument("--show_text", action="store_true", help="Show word text next to labels.")
    parser.add_argument("--max_text_len", type=int, default=20)
    parser.add_argument("--font_size", type=int, default=22)
    parser.add_argument("--line_width", type=int, default=4)
    parser.add_argument("--draw_legend", action="store_true", default=True)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def capture_id_from_payload(payload, label_json):
    for key in ("capture_id", "captureId", "id"):
        if payload.get(key):
            return str(payload[key])
    stem = Path(label_json).stem
    for suffix in ("_labeled_v2", "_labeled", "_labels_v2", "_labels"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def default_paths(args, payload):
    capture_id = capture_id_from_payload(payload, args.label_json)
    out_dir = Path("outputs/labeled_overlay")
    out = Path(args.out) if args.out else out_dir / f"{capture_id}_labeled_overlay.png"
    summary = Path(args.summary_out) if args.summary_out else out_dir / f"{capture_id}_label_summary.json"
    return capture_id, out, summary


def load_font(size):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def label_to_field(label):
    if label is None:
        return "O"
    value = str(label).strip()
    if not value or value == "O":
        return "O"
    if schema_label_to_field is not None:
        return schema_label_to_field(canonicalize_label(value) if canonicalize_label else value)
    if value.startswith(("B-", "I-")):
        value = value[2:]
    return value.upper().replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


def label_prefix(label):
    value = str(label or "O").strip()
    if value.startswith("B-"):
        return "B"
    if value.startswith("I-"):
        return "I"
    return "O"


def deterministic_color(field):
    digest = hashlib.md5(str(field).encode("utf-8")).digest()
    return (80 + digest[0] % 150, 70 + digest[1] % 150, 70 + digest[2] % 150)


def field_color(field):
    return FIELD_COLORS.get(field, deterministic_color(field))


def parse_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return None, "box must be [left, top, right, bottom]"
    try:
        parsed = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return None, "box contains non-numeric value"
    x0, y0, x1, y1 = parsed
    if x1 <= x0 or y1 <= y0:
        return parsed, "box has non-positive width or height"
    return parsed, None


def clamp_for_draw(box, width, height):
    if box is None or len(box) != 4:
        return None
    x0, y0, x1, y1 = box
    x0 = max(0, min(int(x0), width - 1))
    x1 = max(0, min(int(x1), width - 1))
    y0 = max(0, min(int(y0), height - 1))
    y1 = max(0, min(int(y1), height - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 == x0:
        x1 = min(width - 1, x1 + 6)
    if y1 == y0:
        y1 = min(height - 1, y1 + 6)
    return [x0, y0, x1, y1]


def draw_rect_width(draw, box, color, width=4):
    for offset in range(max(1, int(width))):
        draw.rectangle(
            [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
            outline=color + (255,),
        )


def draw_dashed_rect(draw, box, color, width=4, dash=18):
    for offset in range(max(1, int(width))):
        x0, y0, x1, y1 = box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset
        for x in range(x0, x1, dash * 2):
            draw.line([(x, y0), (min(x + dash, x1), y0)], fill=color + (255,), width=1)
            draw.line([(x, y1), (min(x + dash, x1), y1)], fill=color + (255,), width=1)
        for y in range(y0, y1, dash * 2):
            draw.line([(x0, y), (x0, min(y + dash, y1))], fill=color + (255,), width=1)
            draw.line([(x1, y), (x1, min(y + dash, y1))], fill=color + (255,), width=1)


def text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def draw_label(draw, xy, text, color, font, image_width, image_height):
    x, y = xy
    tw, th = text_size(draw, text, font)
    pad = 5
    x = max(0, min(int(x), max(0, image_width - tw - pad * 2 - 1)))
    y = max(0, min(int(y), max(0, image_height - th - pad * 2 - 1)))
    draw.rectangle([x, y, x + tw + pad * 2, y + th + pad * 2], fill=(255, 255, 255, 210), outline=color + (255,))
    draw.text((x + pad, y + pad), text, fill=color + (255,), font=font)


def truncate_text(text, max_len):
    text = str(text or "")
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 1)] + "…"


def load_labeled_words(payload):
    words = payload.get("words")
    if not isinstance(words, list):
        fail("label_json must contain a words list.")
    top_labels = payload.get("labels")
    if top_labels is not None:
        if not isinstance(top_labels, list):
            fail("top-level labels must be a list when present.")
        if len(top_labels) != len(words):
            fail(f"words length ({len(words)}) and labels length ({len(top_labels)}) differ.")
    loaded = []
    for idx, word in enumerate(words):
        if not isinstance(word, dict):
            fail(f"words[{idx}] must be an object.")
        label = word.get("label")
        if label is None and top_labels is not None:
            label = top_labels[idx]
        if label is None:
            fail(f"words[{idx}] has no label and no top-level labels fallback is available.")
        loaded.append(
            {
                "word_idx": int(word.get("word_idx", idx)),
                "text": str(word.get("text", "")),
                "box_raw": word.get("box"),
                "label": str(label),
            }
        )
    return loaded


def build_span_preview(words, warnings):
    spans = []
    current = None
    for item in words:
        label = item["label"]
        field = label_to_field(label)
        prefix = label_prefix(label)
        if field == "O":
            if current is not None:
                spans.append(current)
                current = None
            continue
        if prefix == "B" or current is None or current["field"] != field:
            if current is not None:
                spans.append(current)
            if prefix == "I":
                warnings.append(f"word_idx={item['word_idx']}: {label} started without matching previous span; treated as B-{field}.")
            current = {"field": field, "texts": [item["text"]], "word_indices": [item["word_idx"]]}
        else:
            current["texts"].append(item["text"])
            current["word_indices"].append(item["word_idx"])
    if current is not None:
        spans.append(current)
    for span in spans:
        raw_text = " ".join(span["texts"])
        span["text"] = raw_text
        span["normalized_text"] = normalize_span_text(span["field"], raw_text)
    return spans


def draw_legend(draw, fields, font, image_width):
    if not fields:
        return
    rows = [(field, field_color(field)) for field in fields]
    row_h = max(28, font.size + 10 if hasattr(font, "size") else 28)
    width = 420
    height = row_h * len(rows) + 18
    x = max(8, image_width - width - 18)
    y = 18
    draw.rectangle([x, y, x + width, y + height], fill=(255, 255, 255, 220), outline=(30, 30, 30, 255))
    yy = y + 10
    for field, color in rows:
        draw.rectangle([x + 12, yy + 4, x + 34, yy + 24], fill=color + (255,), outline=(0, 0, 0, 255))
        draw.text((x + 44, yy), field, fill=(20, 20, 20, 255), font=font)
        yy += row_h


def main():
    args = parse_args()
    image_path = Path(args.image)
    label_json_path = Path(args.label_json)
    if not image_path.exists():
        fail(f"Image file not found: {image_path}")
    if not label_json_path.exists():
        fail(f"label_json not found: {label_json_path}")

    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    payload = load_json(label_json_path)
    capture_id, out_path, summary_path = default_paths(args, payload)
    words = load_labeled_words(payload)
    warnings = []
    invalid_boxes = []
    label_counts = Counter()
    field_counts = Counter()
    appeared_fields = []

    json_width = payload.get("image_width") or payload.get("width")
    json_height = payload.get("image_height") or payload.get("height")
    if json_width is not None and json_height is not None:
        if int(json_width) != width or int(json_height) != height:
            warnings.append(f"image size mismatch: JSON={json_width}x{json_height}, actual={width}x{height}")

    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = load_font(args.font_size)

    invalid_marker_y = 18
    for item in words:
        label = item["label"]
        field = label_to_field(label)
        label_counts[label] += 1
        field_counts[field] += 1
        if field not in appeared_fields:
            appeared_fields.append(field)
        parsed_box, box_error = parse_box(item["box_raw"])
        if box_error:
            invalid_boxes.append(
                {
                    "word_idx": item["word_idx"],
                    "text": item["text"],
                    "label": label,
                    "box": item["box_raw"],
                    "reason": box_error,
                }
            )
        if field == "O" and not args.show_o and not box_error:
            continue
        color = (220, 20, 60) if box_error else field_color(field)
        box = clamp_for_draw(parsed_box, width, height)
        if box is None:
            box = [10, invalid_marker_y, min(width - 1, 240), min(height - 1, invalid_marker_y + 28)]
            invalid_marker_y += 34
        if box_error:
            draw_dashed_rect(draw, box, color, width=args.line_width)
        else:
            draw_rect_width(draw, box, color, width=args.line_width)
        text = f"{item['word_idx']} {label}"
        if args.show_text:
            text += f": {truncate_text(item['text'], args.max_text_len)}"
        draw_label(draw, (box[0], max(0, box[1] - args.font_size - 14)), text, color, font, width, height)

    legend_fields = [field for field in appeared_fields if field != "O" or args.show_o]
    if args.draw_legend:
        draw_legend(draw, legend_fields, font, width)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(out_path)
    span_preview = build_span_preview(words, warnings)
    summary = {
        "image_path": str(image_path),
        "label_json": str(label_json_path),
        "capture_id": capture_id,
        "image_width": width,
        "image_height": height,
        "json_image_width": json_width,
        "json_image_height": json_height,
        "num_words": len(words),
        "num_labeled_non_o": sum(count for label, count in label_counts.items() if label != "O"),
        "label_counts": dict(label_counts),
        "field_counts": dict(field_counts),
        "invalid_boxes": invalid_boxes,
        "warnings": warnings,
        "span_preview": span_preview,
    }
    save_json(summary_path, summary)

    print(f"image: {image_path}")
    print(f"label_json: {label_json_path}")
    print(f"overlay PNG path: {out_path}")
    print(f"summary JSON path: {summary_path}")
    print(f"label_counts: {dict(label_counts)}")
    print(f"field_counts: {dict(field_counts)}")
    print(f"invalid_box_count: {len(invalid_boxes)}")
    if warnings:
        print("warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    print("span preview:")
    for span in span_preview:
        print(f"  {span['field']}: {span['normalized_text']}")
    if args.debug:
        print(f"appeared_fields: {appeared_fields}")
        if invalid_boxes:
            print("invalid boxes:")
            for invalid in invalid_boxes[:50]:
                print(f"  {invalid}")
    print("Labeled receipt JSON overlay passed.")


if __name__ == "__main__":
    main()
