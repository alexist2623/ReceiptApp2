import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


RELATION_COLORS = {
    "item_attribute": (34, 150, 83),
    "summary_amount": (217, 70, 239),
    "tax_amount": (245, 124, 0),
    "payment_attribute": (25, 118, 210),
    "payment_amount": (25, 118, 210),
    "rel_g_positive": (220, 38, 38),
}

FIELD_COLORS = {
    "ITEM_NAME": (0, 150, 72),
    "ITEM_PRICE": (220, 38, 38),
    "ITEM_QTY": (214, 170, 0),
    "ITEM_UNIT_PRICE": (255, 87, 34),
    "SUBTOTAL_NAME": (217, 70, 239),
    "SUBTOTAL_PRICE": (250, 128, 114),
    "TAX_NAME": (245, 124, 0),
    "TAX_PRICE": (255, 69, 0),
    "TOTAL_NAME": (255, 20, 147),
    "TOTAL_PRICE": (255, 20, 147),
    "PAYMENT_METHOD": (14, 165, 233),
    "PAYMENT_CARD": (65, 105, 225),
    "CARD_PRICE": (255, 105, 180),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Overlay hand-labeled receipt relations on a receipt image.")
    parser.add_argument("--image", required=True, help="Receipt image path.")
    parser.add_argument("--label_json", required=True, help="Labeled receipt JSON path.")
    parser.add_argument("--out", default=None, help="Output relation overlay PNG path.")
    parser.add_argument("--summary_out", default=None, help="Output relation summary JSON path.")
    parser.add_argument(
        "--relation_source",
        choices=["all", "relations", "item_relations", "summary_relations", "payment_relations", "rel_g_edges"],
        default="all",
    )
    parser.add_argument("--coordinate_mode", choices=["strict", "auto-scale"], default="strict")
    parser.add_argument("--font_size", type=int, default=20)
    parser.add_argument("--line_width", type=int, default=4)
    parser.add_argument("--max_text_len", type=int, default=34)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(message if message.startswith("ERROR:") else f"ERROR: {message}", file=sys.stderr)
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
        if isinstance(payload, dict) and payload.get(key):
            return str(payload[key])
    stem = Path(label_json).stem
    for suffix in ("_labeled_v2_1", "_labeled_v2", "_labeled", "_labels_v2", "_labels"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def default_paths(image_path, label_json_path, out_path=None, summary_path=None):
    payload = load_json(label_json_path)
    capture_id = capture_id_from_payload(payload, label_json_path)
    out_dir = Path("outputs/relation_overlay")
    out = Path(out_path) if out_path else out_dir / f"{capture_id}_relations_overlay.png"
    summary = Path(summary_path) if summary_path else out_dir / f"{capture_id}_relations_summary.json"
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


def read_label_size(payload):
    width = payload.get("image_width") or payload.get("width")
    height = payload.get("image_height") or payload.get("height")
    image = payload.get("image")
    if (width is None or height is None) and isinstance(image, dict):
        width = image.get("width") if width is None else width
        height = image.get("height") if height is None else height
    if width is None or height is None:
        fail("label JSON missing image_width/image_height.")
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        fail(f"label JSON image size must be integer-like, got {width!r}x{height!r}.")


def coordinate_transform(payload, actual_width, actual_height, coordinate_mode):
    label_width, label_height = read_label_size(payload)
    if label_width == actual_width and label_height == actual_height:
        return {
            "label_width": label_width,
            "label_height": label_height,
            "scale_x": 1.0,
            "scale_y": 1.0,
            "box_transform_applied": False,
        }
    if coordinate_mode == "strict":
        fail(
            f"ERROR: coordinate mismatch: actual image={actual_width}x{actual_height}, "
            f"label_json={label_width}x{label_height}. "
            "Use --coordinate_mode auto-scale for visualization only, or fix image/json coordinate space."
        )
    scale_x = actual_width / label_width
    scale_y = actual_height / label_height
    print("COORDINATE MISMATCH DETECTED")
    print(f"actual image: {actual_width}x{actual_height}")
    print(f"label json: {label_width}x{label_height}")
    print("mode: auto-scale")
    print(f"scale_x={scale_x:.6f} scale_y={scale_y:.6f}")
    return {
        "label_width": label_width,
        "label_height": label_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "box_transform_applied": True,
    }


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


def box_from_corner_points(points):
    if not isinstance(points, list) or not points:
        return None
    xs = []
    ys = []
    for point in points:
        if isinstance(point, list) and len(point) >= 2:
            xs.append(point[0])
            ys.append(point[1])
    if not xs or not ys:
        return None
    return parse_box([min(xs), min(ys), max(xs), max(ys)])


def scale_box(box, scale_x, scale_y):
    if box is None:
        return None
    return [
        int(round(box[0] * scale_x)),
        int(round(box[1] * scale_y)),
        int(round(box[2] * scale_x)),
        int(round(box[3] * scale_y)),
    ]


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


def union_boxes(boxes):
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def center(box):
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


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


def build_word_lookup(words, scale_x, scale_y, image_width, image_height):
    lookup = {}
    for idx, word in enumerate(words):
        if not isinstance(word, dict):
            continue
        raw_box = parse_box(word.get("box")) or box_from_corner_points(word.get("cornerPoints"))
        draw_box = clamp_box(scale_box(raw_box, scale_x, scale_y), image_width, image_height)
        entry = {
            "text": str(word.get("text", "")),
            "box": draw_box,
            "raw": word,
        }
        lookup[idx] = entry
        for key in ("word_idx", "globalWordIndex"):
            if word.get(key) is not None:
                try:
                    lookup[int(word[key])] = entry
                except (TypeError, ValueError):
                    pass
    return lookup


def get_relation_sources(payload, relation_source):
    if relation_source == "relations":
        return [("relations", payload.get("relations") or [])]
    if relation_source in {"item_relations", "summary_relations", "payment_relations", "rel_g_edges"}:
        return [(relation_source, payload.get(relation_source) or [])]

    relations = payload.get("relations")
    if isinstance(relations, list) and relations:
        return [("relations", relations)]

    combined = []
    for key in ("item_relations", "summary_relations", "payment_relations"):
        values = payload.get(key)
        if isinstance(values, list):
            combined.append((key, values))
    if not combined and isinstance(payload.get("rel_g_edges"), list):
        combined.append(("rel_g_edges", payload.get("rel_g_edges") or []))
    return combined


def relation_field(relation, *keys):
    for key in keys:
        if relation.get(key) is not None:
            return relation.get(key)
    return None


def normalize_relation(relation, source_name, ordinal):
    relation_type = relation_field(relation, "relation_type", "edge_type") or (
        "rel_g_positive" if source_name == "rel_g_edges" else "unknown"
    )
    head_indices = normalize_indices(relation_field(relation, "head_word_indices", "head_word_idx"))
    tail_indices = normalize_indices(
        relation_field(
            relation,
            "tail_word_indices",
            "dep_word_indices",
            "dependent_word_indices",
            "tail_word_idx",
            "dep_word_idx",
            "dependent_word_idx",
        )
    )
    head_field = relation_field(relation, "head_field") or "HEAD"
    tail_field = relation_field(relation, "tail_field", "dep_field", "dependent_field") or "TAIL"
    head_text = relation_field(relation, "head_text") or ""
    tail_text = relation_field(relation, "tail_text", "dep_text", "dependent_text") or ""
    return {
        "source": source_name,
        "relation_id": relation_field(relation, "relation_id", "edge_id") or f"{source_name}_{ordinal:04d}",
        "relation_type": str(relation_type),
        "head_field": str(head_field),
        "head_word_indices": head_indices,
        "head_text": str(head_text),
        "tail_field": str(tail_field),
        "tail_word_indices": tail_indices,
        "tail_text": str(tail_text),
        "raw": relation,
    }


def relation_color(relation_type):
    if relation_type in RELATION_COLORS:
        return RELATION_COLORS[relation_type]
    if relation_type.startswith("tax"):
        return RELATION_COLORS["tax_amount"]
    if relation_type.startswith("payment"):
        return RELATION_COLORS["payment_attribute"]
    if relation_type.startswith("summary"):
        return RELATION_COLORS["summary_amount"]
    return (96, 96, 96)


def field_color(field):
    return FIELD_COLORS.get(str(field), (100, 100, 100))


def truncate_text(value, max_len):
    value = str(value or "")
    if len(value) <= max_len:
        return value
    return value[: max(0, max_len - 3)] + "..."


def text_size(draw, text, font):
    try:
        bbox = draw.textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.textsize(text, font=font)


def draw_text_label(draw, xy, text, color, font, image_width, image_height):
    x, y = xy
    tw, th = text_size(draw, text, font)
    pad = 4
    x = max(0, min(int(x), max(0, image_width - tw - pad * 2 - 1)))
    y = max(0, min(int(y), max(0, image_height - th - pad * 2 - 1)))
    draw.rectangle([x, y, x + tw + pad * 2, y + th + pad * 2], fill=(255, 255, 255, 220), outline=color + (255,))
    draw.text((x + pad, y + pad), text, fill=color + (255,), font=font)


def draw_rect(draw, box, color, width=2):
    for offset in range(max(1, int(width))):
        draw.rectangle(
            [box[0] - offset, box[1] - offset, box[2] + offset, box[3] + offset],
            outline=color + (255,),
        )


def draw_arrow(draw, start, end, color, width=4):
    sx, sy = start
    ex, ey = end
    draw.line([start, end], fill=color + (230,), width=max(1, int(width)))
    angle = math.atan2(ey - sy, ex - sx)
    arrow_len = max(10, width * 4)
    arrow_angle = math.pi / 7
    p1 = (
        ex - arrow_len * math.cos(angle - arrow_angle),
        ey - arrow_len * math.sin(angle - arrow_angle),
    )
    p2 = (
        ex - arrow_len * math.cos(angle + arrow_angle),
        ey - arrow_len * math.sin(angle + arrow_angle),
    )
    draw.polygon([end, p1, p2], fill=color + (230,))


def validate_relation_boxes(relation, word_lookup):
    if not relation["head_word_indices"]:
        return None, None, "missing head_word_indices"
    if not relation["tail_word_indices"]:
        return None, None, "missing tail_word_indices or dep_word_indices"

    head_entries = []
    for idx in relation["head_word_indices"]:
        entry = word_lookup.get(idx)
        if entry is None:
            return None, None, f"head word index out of range: {idx}"
        if entry["box"] is None:
            return None, None, f"head word box missing/invalid: {idx}"
        head_entries.append(entry)

    tail_entries = []
    for idx in relation["tail_word_indices"]:
        entry = word_lookup.get(idx)
        if entry is None:
            return None, None, f"tail word index out of range: {idx}"
        if entry["box"] is None:
            return None, None, f"tail word box missing/invalid: {idx}"
        tail_entries.append(entry)

    head_box = union_boxes([entry["box"] for entry in head_entries])
    tail_box = union_boxes([entry["box"] for entry in tail_entries])
    if head_box is None:
        return None, None, "zero area head box"
    if tail_box is None:
        return None, None, "zero area tail box"
    return head_box, tail_box, None


def overlay_relations(
    image_path,
    label_json_path,
    out_path=None,
    summary_path=None,
    relation_source="all",
    coordinate_mode="strict",
    font_size=20,
    line_width=4,
    max_text_len=34,
    debug=False,
):
    image_path = Path(image_path)
    label_json_path = Path(label_json_path)
    if not image_path.exists():
        fail(f"Image file not found: {image_path}")
    if not label_json_path.exists():
        fail(f"label_json not found: {label_json_path}")

    payload = load_json(label_json_path)
    capture_id, out_path, summary_path = default_paths(image_path, label_json_path, out_path, summary_path)
    image = Image.open(image_path).convert("RGB")
    image_width, image_height = image.size
    coord = coordinate_transform(payload, image_width, image_height, coordinate_mode)
    words = payload.get("words")
    if not isinstance(words, list):
        fail("label_json must contain a words list.")

    word_lookup = build_word_lookup(words, coord["scale_x"], coord["scale_y"], image_width, image_height)
    source_batches = get_relation_sources(payload, relation_source)
    normalized = []
    for source_name, relations in source_batches:
        if not isinstance(relations, list):
            continue
        for ordinal, relation in enumerate(relations):
            if isinstance(relation, dict):
                normalized.append(normalize_relation(relation, source_name, ordinal))

    overlay = image.convert("RGBA")
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = load_font(font_size)

    relation_list = []
    skipped = []
    type_counts = Counter()
    for relation in normalized:
        head_box, tail_box, reason = validate_relation_boxes(relation, word_lookup)
        if reason:
            skipped.append(
                {
                    "relation_id": relation["relation_id"],
                    "source": relation["source"],
                    "relation_type": relation["relation_type"],
                    "reason": reason,
                    "head_word_indices": relation["head_word_indices"],
                    "tail_word_indices": relation["tail_word_indices"],
                }
            )
            continue
        color = relation_color(relation["relation_type"])
        type_counts[relation["relation_type"]] += 1
        draw_rect(draw, head_box, field_color(relation["head_field"]), width=max(1, line_width - 2))
        draw_rect(draw, tail_box, field_color(relation["tail_field"]), width=max(1, line_width - 2))
        start = center(head_box)
        end = center(tail_box)
        draw_arrow(draw, start, end, color, width=line_width)
        label = f"{relation['head_field']} -> {relation['tail_field']}"
        midpoint = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
        draw_text_label(draw, midpoint, label, color, font, image_width, image_height)
        if debug:
            draw_text_label(
                draw,
                (head_box[0], max(0, head_box[1] - font_size - 12)),
                truncate_text(relation["head_text"], max_text_len),
                field_color(relation["head_field"]),
                font,
                image_width,
                image_height,
            )
            draw_text_label(
                draw,
                (tail_box[0], min(image_height - 1, tail_box[3] + 4)),
                truncate_text(relation["tail_text"], max_text_len),
                field_color(relation["tail_field"]),
                font,
                image_width,
                image_height,
            )
        relation_list.append(
            {
                "relation_id": relation["relation_id"],
                "source": relation["source"],
                "relation_type": relation["relation_type"],
                "head_field": relation["head_field"],
                "head_word_indices": relation["head_word_indices"],
                "head_text": relation["head_text"],
                "head_box": head_box,
                "tail_field": relation["tail_field"],
                "tail_word_indices": relation["tail_word_indices"],
                "tail_text": relation["tail_text"],
                "tail_box": tail_box,
            }
        )

    out_path = Path(out_path)
    summary_path = Path(summary_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.convert("RGB").save(out_path)
    summary = {
        "image_path": str(image_path),
        "label_json": str(label_json_path),
        "capture_id": capture_id,
        "image_actual_width": image_width,
        "image_actual_height": image_height,
        "label_json_width": coord["label_width"],
        "label_json_height": coord["label_height"],
        "coordinate_mode": coordinate_mode,
        "scale_x": coord["scale_x"],
        "scale_y": coord["scale_y"],
        "box_transform_applied": coord["box_transform_applied"],
        "relation_source": relation_source,
        "relation_count": len(relation_list),
        "skipped_relation_count": len(skipped),
        "skipped_relations": skipped,
        "relation_type_counts": dict(type_counts),
        "relations": relation_list,
        "overlay_path": str(out_path),
    }
    save_json(summary_path, summary)

    print(f"image: {image_path}")
    print(f"label_json: {label_json_path}")
    print(f"relation_source: {relation_source}")
    print(f"coordinate_mode: {coordinate_mode}")
    print(f"relation_count: {len(relation_list)}")
    print(f"skipped_relation_count: {len(skipped)}")
    if skipped:
        print("skipped reasons:")
        for item in skipped[:20]:
            print(f"  - {item['relation_id']}: {item['reason']}")
    print(f"overlay PNG path: {out_path}")
    print(f"summary JSON path: {summary_path}")
    print("Labeled relation overlay passed.")
    return summary


def main():
    args = parse_args()
    overlay_relations(
        image_path=args.image,
        label_json_path=args.label_json,
        out_path=args.out,
        summary_path=args.summary_out,
        relation_source=args.relation_source,
        coordinate_mode=args.coordinate_mode,
        font_size=args.font_size,
        line_width=args.line_width,
        max_text_len=args.max_text_len,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
