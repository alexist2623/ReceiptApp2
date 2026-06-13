import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from transformers import AutoModelForTokenClassification, AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import canonicalize_label, label_to_field


BOX_KEYS = ("box", "bbox", "bounding_box", "boundingBox", "rect", "quad", "vertices")
TEXT_KEYS = ("text", "value", "word", "description")


def raw_label_to_field(label):
    value = str(label or "O").strip()
    if not value or value == "O":
        return "O"
    if value.startswith(("B-", "I-")):
        value = value[2:]
    return value.upper().replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run fine-tuned LayoutLMv3 inference on a user receipt image and OCR JSON."
    )
    parser.add_argument("--image", required=True, help="User receipt image path.")
    parser.add_argument("--ocr_json", required=True, help="OCR JSON path.")
    parser.add_argument("--checkpoint", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--labels", default="processed_data/cord_bio/labels.json")
    parser.add_argument("--out_dir", default="outputs/user_ocr_inference")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--show_text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show OCR text in overlay labels.",
    )
    parser.add_argument("--max_text_len", type=int, default=25)
    parser.add_argument("--hide_o", action="store_true")
    parser.add_argument("--draw_conf_threshold", type=float, default=0.0)
    parser.add_argument("--assume_boxes_normalized", action="store_true")
    parser.add_argument("--box_format", default="auto", choices=("auto", "xyxy", "xywh", "quad"))
    parser.add_argument("--image_width_key", default="image_width")
    parser.add_argument("--image_height_key", default="image_height")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_path(path, message):
    if not Path(path).exists():
        fail(message)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_image(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def load_ocr_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_json_image_size(ocr_obj, args):
    if not isinstance(ocr_obj, dict):
        return None, None
    width_keys = (args.image_width_key, "image_width", "width", "img_width")
    height_keys = (args.image_height_key, "image_height", "height", "img_height")
    json_width = next((ocr_obj.get(key) for key in width_keys if ocr_obj.get(key) is not None), None)
    json_height = next((ocr_obj.get(key) for key in height_keys if ocr_obj.get(key) is not None), None)
    try:
        json_width = int(round(float(json_width))) if json_width is not None else None
        json_height = int(round(float(json_height))) if json_height is not None else None
    except (TypeError, ValueError):
        json_width, json_height = None, None
    return json_width, json_height


def text_from_item(item):
    if not isinstance(item, dict):
        return ""
    for key in TEXT_KEYS:
        value = item.get(key)
        if value is not None:
            return str(value)
    return ""


def box_from_item(item):
    if not isinstance(item, dict):
        return item
    for key in BOX_KEYS:
        if item.get(key) is not None:
            return item.get(key)
    coordinate_keys = {
        "left",
        "top",
        "right",
        "bottom",
        "x0",
        "y0",
        "x1",
        "y1",
        "x",
        "y",
        "width",
        "height",
        "x2",
        "y2",
        "x3",
        "y3",
        "x4",
        "y4",
    }
    if coordinate_keys.intersection(item.keys()):
        return item
    return None


def make_ocr_item(item, source, line_index=None, block_index=None, word_index=None):
    return {
        "text": text_from_item(item),
        "box_raw": box_from_item(item),
        "line_index": line_index,
        "block_index": block_index,
        "word_index": word_index,
        "source": source,
        "raw": item,
    }


def add_items_from_list(items, source, output, line_index=None, block_index=None):
    if not isinstance(items, list):
        return
    for word_index, item in enumerate(items):
        output.append(
            make_ocr_item(
                item,
                source=source,
                line_index=line_index,
                block_index=block_index,
                word_index=word_index,
            )
        )


def extract_from_lines(lines, source_prefix, output, block_index=None):
    if not isinstance(lines, list):
        return
    for line_index, line in enumerate(lines):
        if not isinstance(line, dict):
            continue
        for key in ("words", "tokens", "elements"):
            if isinstance(line.get(key), list):
                add_items_from_list(
                    line[key],
                    source=f"{source_prefix}.{key}",
                    output=output,
                    line_index=line_index,
                    block_index=block_index,
                )
        if box_from_item(line) is not None and text_from_item(line).strip():
            output.append(
                make_ocr_item(
                    line,
                    source=source_prefix,
                    line_index=line_index,
                    block_index=block_index,
                    word_index=None,
                )
            )


def extract_ocr_items(ocr_obj):
    items = []
    if isinstance(ocr_obj, list):
        add_items_from_list(ocr_obj, "list", items)
        return items
    if not isinstance(ocr_obj, dict):
        return items

    top_level_words = ocr_obj.get("words")
    if isinstance(top_level_words, list) and top_level_words:
        add_items_from_list(top_level_words, "words", items)
        return items

    top_level_tokens = ocr_obj.get("tokens")
    if isinstance(top_level_tokens, list) and top_level_tokens:
        add_items_from_list(top_level_tokens, "tokens", items)
        return items

    extract_from_lines(ocr_obj.get("lines"), "lines", items)

    for block_key in ("textBlocks", "blocks"):
        blocks = ocr_obj.get(block_key)
        if not isinstance(blocks, list):
            continue
        for block_index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            extract_from_lines(block.get("lines"), f"{block_key}.lines", items, block_index=block_index)
            for key in ("words", "tokens", "elements"):
                if isinstance(block.get(key), list):
                    add_items_from_list(
                        block[key],
                        source=f"{block_key}.{key}",
                        output=items,
                        block_index=block_index,
                    )
    return items


def to_number(value):
    if value is None:
        raise ValueError("missing value")
    return int(round(float(value)))


def box_from_points(points):
    xs = []
    ys = []
    for point in points:
        if isinstance(point, dict):
            if point.get("x") is None or point.get("y") is None:
                continue
            xs.append(to_number(point.get("x")))
            ys.append(to_number(point.get("y")))
        elif isinstance(point, (list, tuple)) and len(point) >= 2:
            xs.append(to_number(point[0]))
            ys.append(to_number(point[1]))
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def parse_box(box_raw, item=None, box_format="auto"):
    target = box_raw if box_raw is not None else item
    try:
        if isinstance(target, dict):
            if isinstance(target.get("vertices"), list):
                return box_from_points(target["vertices"])
            if all(key in target for key in ("left", "top", "right", "bottom")):
                return [
                    to_number(target["left"]),
                    to_number(target["top"]),
                    to_number(target["right"]),
                    to_number(target["bottom"]),
                ]
            if all(key in target for key in ("x0", "y0", "x1", "y1")):
                return [
                    to_number(target["x0"]),
                    to_number(target["y0"]),
                    to_number(target["x1"]),
                    to_number(target["y1"]),
                ]
            if all(key in target for key in ("x", "y", "width", "height")):
                x = to_number(target["x"])
                y = to_number(target["y"])
                return [x, y, x + to_number(target["width"]), y + to_number(target["height"])]
            if all(key in target for key in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")):
                xs = [to_number(target[f"x{i}"]) for i in range(1, 5)]
                ys = [to_number(target[f"y{i}"]) for i in range(1, 5)]
                return [min(xs), min(ys), max(xs), max(ys)]
            return None

        if isinstance(target, list):
            if target and all(isinstance(point, dict) for point in target):
                return box_from_points(target)
            if len(target) == 8:
                values = [to_number(value) for value in target]
                xs = values[0::2]
                ys = values[1::2]
                return [min(xs), min(ys), max(xs), max(ys)]
            if len(target) == 4:
                values = [to_number(value) for value in target]
                if box_format == "quad":
                    return None
                if box_format == "xywh":
                    x, y, width, height = values
                    return [x, y, x + width, y + height]
                x0, y0, x1, y1 = values
                if box_format == "auto" and (x1 < x0 or y1 < y0):
                    return [x0, y0, x0 + x1, y0 + y1]
                return [x0, y0, x1, y1]
        return None
    except (TypeError, ValueError):
        return None


def clamp_box(box, width, height):
    x0, y0, x1, y1 = [int(v) for v in box]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def clamp_normalized_box(box):
    x0, y0, x1, y1 = [int(v) for v in box]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = max(0, min(x0, 1000))
    x1 = max(0, min(x1, 1000))
    y0 = max(0, min(y0, 1000))
    y1 = max(0, min(y1, 1000))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def normalize_box(box, width, height):
    x0, y0, x1, y1 = box
    normalized = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]
    return clamp_normalized_box(normalized)


def normalized_to_pixel_box(box, width, height):
    x0, y0, x1, y1 = box
    pixel = [
        int(round(x0 * width / 1000)),
        int(round(y0 * height / 1000)),
        int(round(x1 * width / 1000)),
        int(round(y1 * height / 1000)),
    ]
    return clamp_box(pixel, width, height)


def detect_schema(ocr_obj, raw_items):
    if isinstance(ocr_obj, list):
        return "list"
    if not isinstance(ocr_obj, dict):
        return "unknown"
    paths = sorted({item["source"] for item in raw_items})
    return ", ".join(paths) if paths else "unknown"


def coordinate_warnings(parsed_raw_boxes, json_width, json_height, width, height, assume_normalized):
    warnings = []
    if json_width is not None and json_height is not None and (json_width != width or json_height != height):
        warnings.append(
            "OCR JSON image size differs from EXIF-transposed image size: "
            f"json={json_width}x{json_height}, actual={width}x{height}. Boxes may be shifted or scaled."
        )
    if not parsed_raw_boxes:
        return warnings

    xs = [box[0] for box in parsed_raw_boxes] + [box[2] for box in parsed_raw_boxes]
    ys = [box[1] for box in parsed_raw_boxes] + [box[3] for box in parsed_raw_boxes]
    out_of_bounds = sum(1 for box in parsed_raw_boxes if box[2] > width or box[3] > height)
    if not assume_normalized and out_of_bounds > max(2, len(parsed_raw_boxes) // 2):
        warnings.append(
            "Many OCR boxes exceed image dimensions. Check whether OCR coordinates are normalized, scaled, or from a rotated image."
        )
    if not assume_normalized and max(xs + ys) <= 1000 and (width > 1500 or height > 1500):
        warnings.append(
            "All OCR coordinates are within 0..1000 while the image is large; boxes may already be normalized. "
            "Try --assume_boxes_normalized."
        )
    return warnings


def prepare_ocr_words(ocr_obj, image_size, args):
    width, height = image_size
    json_width, json_height = get_json_image_size(ocr_obj, args)
    raw_items = extract_ocr_items(ocr_obj)
    schema = detect_schema(ocr_obj, raw_items)
    words = []
    pixel_boxes = []
    normalized_boxes = []
    metadata = []
    parsed_raw_boxes = []
    skipped_empty = 0
    skipped_invalid = 0
    skipped_examples = []

    for raw_idx, item in enumerate(raw_items):
        text = str(item.get("text", ""))
        if not text.strip():
            skipped_empty += 1
            if len(skipped_examples) < 20:
                skipped_examples.append({"reason": "empty_text", "raw_index": raw_idx, "source": item.get("source")})
            continue

        parsed_box = parse_box(item.get("box_raw"), item=item.get("raw"), box_format=args.box_format)
        if parsed_box is None:
            skipped_invalid += 1
            if len(skipped_examples) < 20:
                skipped_examples.append(
                    {
                        "reason": "invalid_box",
                        "raw_index": raw_idx,
                        "text": text,
                        "source": item.get("source"),
                        "box_raw": item.get("box_raw"),
                    }
                )
            continue
        parsed_raw_boxes.append(parsed_box)

        if args.assume_boxes_normalized:
            normalized_box = clamp_normalized_box(parsed_box)
            pixel_box = normalized_to_pixel_box(normalized_box, width, height) if normalized_box else None
        else:
            pixel_box = clamp_box(parsed_box, width, height)
            normalized_box = normalize_box(pixel_box, width, height) if pixel_box else None

        if pixel_box is None or normalized_box is None:
            skipped_invalid += 1
            if len(skipped_examples) < 20:
                skipped_examples.append(
                    {
                        "reason": "invalid_box_after_clamp",
                        "raw_index": raw_idx,
                        "text": text,
                        "source": item.get("source"),
                        "box": parsed_box,
                    }
                )
            continue

        words.append(text.strip())
        pixel_boxes.append(pixel_box)
        normalized_boxes.append(normalized_box)
        metadata.append(
            {
                "source": item.get("source"),
                "line_index": item.get("line_index"),
                "block_index": item.get("block_index"),
                "word_index": item.get("word_index"),
            }
        )

    warnings = coordinate_warnings(
        parsed_raw_boxes,
        json_width=json_width,
        json_height=json_height,
        width=width,
        height=height,
        assume_normalized=args.assume_boxes_normalized,
    )

    if not words:
        fail("No valid OCR words found after filtering. Check OCR JSON text and boxes.")

    debug_payload = {
        "raw_item_count": len(raw_items),
        "valid_word_count": len(words),
        "skipped_empty_text_count": skipped_empty,
        "skipped_invalid_box_count": skipped_invalid,
        "json_image_width": json_width,
        "json_image_height": json_height,
        "actual_image_width": width,
        "actual_image_height": height,
        "detected_schema": schema,
        "warnings": warnings,
        "skipped_item_examples": skipped_examples,
    }
    return words, pixel_boxes, normalized_boxes, metadata, debug_payload


def load_labels(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    label_list = payload["label_list"]
    label2id = {str(label): int(idx) for label, idx in payload["label2id"].items()}
    id2label = {int(idx): str(label) for idx, label in payload["id2label"].items()}
    if "O" not in label_list:
        fail("labels.json is invalid: 'O' is missing from label_list.")
    if label2id.get("O") != 0:
        fail("labels.json is invalid: label2id['O'] must be 0.")
    return label_list, label2id, id2label


def select_device(device_arg):
    cuda_available = torch.cuda.is_available()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not cuda_available:
            fail("--device cuda requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if cuda_available else "cpu")


def compare_model_labels(model, label_list, id2label):
    print(f"loaded model num_labels: {model.config.num_labels}")
    if model.config.num_labels != len(label_list):
        print(
            f"WARNING: model num_labels={model.config.num_labels} differs from "
            f"labels.json num_labels={len(label_list)}"
        )
    config_id2label = getattr(model.config, "id2label", {}) or {}
    mismatches = []
    for idx, label in id2label.items():
        config_label = config_id2label.get(idx, config_id2label.get(str(idx)))
        if config_label is not None and config_label != label:
            mismatches.append((idx, config_label, label))
    if mismatches:
        print(f"WARNING: checkpoint id2label differs from labels.json. First mismatches: {mismatches[:5]}")
    model.config.id2label = id2label


def run_inference(image, words, normalized_boxes, processor, model, device, id2label, max_length):
    encoding = processor(
        image,
        words,
        boxes=normalized_boxes,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    print("encoding shapes:")
    for key, value in encoding.items():
        if hasattr(value, "shape"):
            print(f"  {key}: {list(value.shape)}")

    try:
        word_ids = encoding.word_ids(batch_index=0)
    except Exception as exc:
        fail(f"BatchEncoding.word_ids is required for word-level prediction restore: {exc}")
    if word_ids is None:
        fail("BatchEncoding.word_ids returned None. A fast tokenizer is required.")

    model_inputs = {
        key: value.to(device)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"}
    }
    with torch.no_grad():
        logits = model(**model_inputs).logits.detach().cpu()[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)

    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())
    attention_mask = encoding["attention_mask"][0].detach().cpu()
    labels = [None] * len(words)
    confidences = [None] * len(words)
    token_debug = []

    for token_idx, word_idx in enumerate(word_ids):
        pred_id = int(pred_ids[token_idx].item())
        pred_label = id2label[pred_id]
        confidence = float(probs[token_idx, pred_id].item())
        word_text = None if word_idx is None or word_idx >= len(words) else words[word_idx]
        token_debug.append(
            {
                "token_idx": token_idx,
                "token": tokens[token_idx],
                "word_idx": word_idx,
                "word": word_text,
                "pred_label": pred_label,
                "confidence": confidence,
            }
        )
        if word_idx is None or word_idx >= len(words) or attention_mask[token_idx].item() == 0:
            continue
        if labels[word_idx] is None:
            labels[word_idx] = pred_label
            confidences[word_idx] = confidence

    missing = [idx for idx, label in enumerate(labels) if label is None]
    if missing:
        print(f"ERROR: missing word-level predictions for word indices: {missing[:20]}", file=sys.stderr)
        print(f"word_ids preview: {word_ids[:80]}", file=sys.stderr)
        fail("Could not restore word-level predictions for every word. Increase --max_length or inspect OCR tokenization.")
    return labels, confidences, token_debug, word_ids


def label_to_color(label):
    if label == "O":
        return (135, 135, 135)
    digest = hashlib.md5(label.encode("utf-8")).digest()
    return (40 + digest[0] % 180, 40 + digest[1] % 180, 40 + digest[2] % 180)


def draw_rectangle(draw, box, color, width=2):
    try:
        draw.rectangle(box, outline=color, width=width)
    except TypeError:
        x0, y0, x1, y1 = box
        for offset in range(width):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)


def text_size(draw, text, font):
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.multiline_textsize(text, font=font)


def draw_label(draw, box, text, color, font, image_size):
    width, height = image_size
    text_w, text_h = text_size(draw, text, font)
    x0, y0, _, y1 = box
    x = max(0, min(x0, width - text_w - 6))
    y = y0 - text_h - 6
    if y < 0:
        y = y1 + 2
    y = max(0, min(y, height - text_h - 6))
    draw.rectangle([x, y, x + text_w + 5, y + text_h + 5], fill=(255, 255, 255))
    draw.multiline_text((x + 2, y + 2), text, fill=color, font=font)


def save_overlay(image, predictions, args, output_path):
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for item in predictions:
        label = item["label"]
        confidence = item["confidence"]
        if args.hide_o and label == "O":
            continue
        if confidence < args.draw_conf_threshold:
            continue
        color = label_to_color(label)
        draw_rectangle(draw, item["box"], color, width=2)
        lines = [f"{label} {confidence:.2f}"]
        if args.show_text:
            lines.append(item["text"][: args.max_text_len])
        draw_label(draw, item["box"], "\n".join(lines), color, font, output.size)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def min_max_boxes(boxes):
    values = [coord for box in boxes for coord in box]
    return {"min": min(values), "max": max(values)} if values else {"min": None, "max": None}


def main():
    args = parse_args()
    require_path(args.image, f"Image file not found: {args.image}")
    require_path(args.ocr_json, f"OCR JSON not found: {args.ocr_json}")
    require_path(args.checkpoint, f"Checkpoint not found: {args.checkpoint}. Run step 5 first.")
    require_path(args.labels, f"labels.json not found: {args.labels}. Run step 3 first.")

    print(f"WSL/conda Python path: {sys.executable}")
    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"cuda device name: {torch.cuda.get_device_name(0)}")
    device = select_device(args.device)
    print(f"selected device: {device}")
    print(f"checkpoint path: {args.checkpoint}")
    print(f"labels.json path: {args.labels}")
    print(f"image path: {args.image}")
    print(f"OCR JSON path: {args.ocr_json}")

    image = load_image(args.image)
    width, height = image.size
    print(f"image size: {width}x{height}")

    ocr_obj = load_ocr_json(args.ocr_json)
    if args.debug and isinstance(ocr_obj, dict):
        print(f"OCR JSON top-level keys: {list(ocr_obj.keys())}")

    words, pixel_boxes, normalized_boxes, metadata, ocr_debug = prepare_ocr_words(ocr_obj, image.size, args)
    print(f"OCR JSON schema detected: {ocr_debug['detected_schema']}")
    print(f"raw OCR item count: {ocr_debug['raw_item_count']}")
    print(f"valid word count: {ocr_debug['valid_word_count']}")
    print(
        "skipped count: "
        f"empty_text={ocr_debug['skipped_empty_text_count']} "
        f"invalid_box={ocr_debug['skipped_invalid_box_count']}"
    )
    for warning in ocr_debug["warnings"]:
        print(f"WARNING: {warning}")
    print("first 20 words + boxes:")
    for idx, word in enumerate(words[:20]):
        print(f"  [{idx}] word={word!r} box={pixel_boxes[idx]} norm={normalized_boxes[idx]}")

    label_list, _label2id, id2label = load_labels(args.labels)
    print(f"labels.json num_labels: {len(label_list)}")
    print("processor apply_ocr=False")
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        apply_ocr=False,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint,
        local_files_only=args.local_files_only,
    )
    compare_model_labels(model, label_list, id2label)
    model.to(device)
    model.eval()

    pred_labels, confidences, token_debug, word_ids = run_inference(
        image=image,
        words=words,
        normalized_boxes=normalized_boxes,
        processor=processor,
        model=model,
        device=device,
        id2label=id2label,
        max_length=args.max_length,
    )

    predictions = []
    for idx, word in enumerate(words):
        raw_label = pred_labels[idx]
        canonical_label = canonicalize_label(raw_label)
        predictions.append(
            {
                "word_idx": idx,
                "text": word,
                "box": pixel_boxes[idx],
                "normalized_box": normalized_boxes[idx],
                "label": raw_label,
                "canonical_label": canonical_label,
                "field": raw_label_to_field(raw_label),
                "canonical_field": label_to_field(canonical_label),
                "confidence": confidences[idx],
                "source": metadata[idx]["source"],
                "line_index": metadata[idx]["line_index"],
                "block_index": metadata[idx]["block_index"],
            }
        )
    label_counts = Counter(pred_labels)
    print("first 30 word predictions:")
    for item in predictions[:30]:
        print(
            f"  [{item['word_idx']}] text={item['text']!r} label={item['label']} "
            f"conf={item['confidence']:.4f} box={item['box']}"
        )
    print("label distribution:")
    for label, count in label_counts.most_common(30):
        print(f"  {label}: {count}")

    out_dir = Path(args.out_dir)
    stem = Path(args.image).stem
    prediction_path = out_dir / f"{stem}_prediction.json"
    overlay_path = out_dir / f"{stem}_overlay.png"
    debug_path = out_dir / f"{stem}_ocr_debug.json"
    run_config_path = out_dir / "run_config.json"

    prediction_payload = {
        "image_path": str(args.image),
        "ocr_json_path": str(args.ocr_json),
        "checkpoint": str(args.checkpoint),
        "image_width": width,
        "image_height": height,
        "num_words": len(words),
        "predictions": predictions,
        "label_counts": dict(label_counts),
    }
    save_json(prediction_path, prediction_payload)
    save_json(debug_path, ocr_debug)
    save_json(
        run_config_path,
        {
            "image": args.image,
            "ocr_json": args.ocr_json,
            "checkpoint": args.checkpoint,
            "labels": args.labels,
            "max_length": args.max_length,
            "device": args.device,
            "local_files_only": args.local_files_only,
            "assume_boxes_normalized": args.assume_boxes_normalized,
            "box_format": args.box_format,
        },
    )
    save_overlay(image, predictions, args, overlay_path)

    print(f"prediction JSON path: {prediction_path}")
    print(f"overlay PNG path: {overlay_path}")
    print(f"OCR debug JSON path: {debug_path}")

    if args.debug:
        print(f"detected OCR schema path: {ocr_debug['detected_schema']}")
        print(f"skipped item examples: {ocr_debug['skipped_item_examples']}")
        print("first 80 token alignment:")
        for item in token_debug[:80]:
            print(
                f"  token_idx={item['token_idx']} token={item['token']!r} "
                f"word_idx={item['word_idx']} word={item['word']!r} "
                f"pred={item['pred_label']} conf={item['confidence']:.4f}"
            )
        print(f"word_ids unique count: {len({word_id for word_id in word_ids if word_id is not None})}")
        print(f"normalized box min/max: {min_max_boxes(normalized_boxes)}")
        print(f"pixel box min/max: {min_max_boxes(pixel_boxes)}")

    print("User OCR JSON inference step passed.")


if __name__ == "__main__":
    main()
