import json
from io import BytesIO
from pathlib import Path

from PIL import Image

from .geometry import clamp_box, normalize_box_1000, union_boxes
from .schema import category_to_field, is_candidate_dep_field, is_head_field


def parse_ground_truth(gt) -> dict:
    if isinstance(gt, dict):
        return gt
    if isinstance(gt, str):
        return json.loads(gt)
    raise TypeError(f"Unsupported ground_truth type: {type(gt)}")


def ensure_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, bytes):
        return Image.open(BytesIO(image)).convert("RGB")
    if isinstance(image, dict):
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
        if image.get("bytes"):
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
    if hasattr(image, "convert"):
        converted = image.convert("RGB")
        if isinstance(converted, Image.Image):
            return converted
    raise TypeError(f"Unsupported image type: {type(image)}")


def quad_to_box(quad_or_box):
    if quad_or_box is None:
        return None
    if isinstance(quad_or_box, dict):
        if all(key in quad_or_box for key in ("left", "top", "right", "bottom")):
            return [
                int(round(float(quad_or_box["left"]))),
                int(round(float(quad_or_box["top"]))),
                int(round(float(quad_or_box["right"]))),
                int(round(float(quad_or_box["bottom"]))),
            ]
        if all(key in quad_or_box for key in ("x0", "y0", "x1", "y1")):
            return [
                int(round(float(quad_or_box["x0"]))),
                int(round(float(quad_or_box["y0"]))),
                int(round(float(quad_or_box["x1"]))),
                int(round(float(quad_or_box["y1"]))),
            ]
        if all(key in quad_or_box for key in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")):
            xs = [int(round(float(quad_or_box[f"x{i}"]))) for i in range(1, 5)]
            ys = [int(round(float(quad_or_box[f"y{i}"]))) for i in range(1, 5)]
            return [min(xs), min(ys), max(xs), max(ys)]
    if isinstance(quad_or_box, (list, tuple)):
        values = [int(round(float(v))) for v in quad_or_box]
        if len(values) == 4:
            return values
        if len(values) == 8:
            xs = values[0::2]
            ys = values[1::2]
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _word_text(word):
    for key in ("text", "value", "word"):
        if isinstance(word, dict) and word.get(key) is not None:
            return str(word.get(key))
    return ""


def _word_box(word):
    if not isinstance(word, dict):
        return None
    for key in ("quad", "box", "bbox"):
        if word.get(key) is not None:
            return quad_to_box(word.get(key))
    return quad_to_box(word)


def extract_cord_words_and_lines(sample) -> dict:
    image = ensure_pil_rgb(sample["image"])
    width, height = image.size
    gt = parse_ground_truth(sample["ground_truth"])
    valid_lines = gt.get("valid_line")
    if not isinstance(valid_lines, list):
        raise ValueError("ground_truth.valid_line not found")

    words = []
    boxes = []
    normalized_boxes = []
    lines = []
    for line_id, line in enumerate(valid_lines):
        raw_words = line.get("words", []) if isinstance(line, dict) else []
        word_indices = []
        for word in raw_words:
            text = _word_text(word).strip()
            box = clamp_box(_word_box(word), width, height)
            if not text or box is None:
                continue
            word_indices.append(len(words))
            words.append(text)
            boxes.append(box)
            normalized_boxes.append(normalize_box_1000(box, width, height))
        if not word_indices:
            continue
        line_box = union_boxes([boxes[idx] for idx in word_indices])
        category = line.get("category", "O")
        field = category_to_field(category)
        lines.append(
            {
                "line_id": line_id,
                "category": category,
                "field": field,
                "text": line.get("text", ""),
                "word_indices": word_indices,
                "box": line_box,
                "normalized_box": normalize_box_1000(line_box, width, height),
                "group_id": line.get("group_id"),
                "sub_group_id": line.get("sub_group_id"),
                "row_id": line.get("row_id"),
            }
        )

    return {
        "image": image,
        "width": width,
        "height": height,
        "words": words,
        "boxes": boxes,
        "normalized_boxes": normalized_boxes,
        "lines": lines,
    }


def make_group_key(span, strategy="group"):
    group_id = span.get("group_id")
    if group_id is None:
        return None
    if strategy == "group":
        return str(group_id)
    if strategy == "group_sub":
        return f"{group_id}:{span.get('sub_group_id')}"
    if strategy == "group_row":
        return f"{group_id}:{span.get('row_id')}"
    raise ValueError(f"Unsupported group key strategy: {strategy}")


def make_gold_spans_from_cord(sample, group_key_strategy="group") -> dict:
    extracted = extract_cord_words_and_lines(sample)
    spans = []
    for line in extracted["lines"]:
        field = line["field"]
        if not (is_head_field(field) or is_candidate_dep_field(field)):
            continue
        span = {
            "span_id": len(spans),
            "field": field,
            "text": line["text"] or " ".join(extracted["words"][idx] for idx in line["word_indices"]),
            "word_indices": line["word_indices"],
            "first_word_idx": line["word_indices"][0],
            "box": line["box"],
            "normalized_box": line["normalized_box"],
            "confidence": 1.0,
            "category": line["category"],
            "group_id": line["group_id"],
            "sub_group_id": line["sub_group_id"],
            "row_id": line["row_id"],
            "line_id": line["line_id"],
        }
        span["group_key"] = make_group_key(span, group_key_strategy)
        spans.append(span)
    extracted["spans"] = spans
    return extracted

