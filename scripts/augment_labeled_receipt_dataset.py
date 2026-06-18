import argparse
import copy
import hashlib
import io
import json
import math
import random
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from ml.receipt_schema import canonicalize_label, label_to_field, normalize_span_text
except Exception:  # pragma: no cover - keeps the augmentation tool usable standalone.
    canonicalize_label = None

    def label_to_field(label):
        value = str(label or "O").strip()
        if not value or value == "O":
            return "O"
        if value.startswith(("B-", "I-")):
            value = value[2:]
        return value.upper().replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")

    def normalize_span_text(field, text):
        return " ".join(str(text or "").split())


PRICE_LIKE_FIELDS = {
    "ITEM_PRICE",
    "ITEM_UNIT_PRICE",
    "ITEM_DISCOUNT",
    "SUBTOTAL_PRICE",
    "TAX_PRICE",
    "DISCOUNT_PRICE",
    "SERVICE_PRICE",
    "TOTAL_PRICE",
    "CASH_PRICE",
    "CHANGE_PRICE",
    "CARD_PRICE",
    "TIP_PRICE",
}

NUMERIC_FIELDS = PRICE_LIKE_FIELDS | {
    "ITEM_QTY",
    "TAX_RATE",
    "DATE",
    "TIME",
    "RECEIPT_ID",
    "TRANSACTION_ID",
    "PAYMENT_AUTH_CODE",
    "APPROVAL_CODE",
}

RELATION_KEYS = (
    "relations",
    "item_relations",
    "summary_relations",
    "payment_relations",
    "rel_g_edges",
)

CONFUSIONS = [
    ("O", "0"),
    ("0", "O"),
    ("I", "1"),
    ("1", "I"),
    ("l", "1"),
    ("1", "l"),
    ("S", "5"),
    ("5", "S"),
    ("B", "8"),
    ("8", "B"),
    ("G", "6"),
    ("6", "G"),
    ("Z", "2"),
    ("2", "Z"),
    ("T", "I"),
    ("m", "rn"),
    ("rn", "m"),
]


class VariantSkip(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create augmented copies of hand-labeled receipt OCR/image folders while preserving labels and relations."
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--variants_per_sample", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--apply_to_splits", choices=["train", "all"], default="train")
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--parent_split_policy", choices=["no_leakage"], default="no_leakage")
    parser.add_argument("--geometry_prob", type=float, default=0.8)
    parser.add_argument("--rotation_deg_min", type=float, default=-3.0)
    parser.add_argument("--rotation_deg_max", type=float, default=3.0)
    parser.add_argument("--translate_x_frac", type=float, default=0.025)
    parser.add_argument("--translate_y_frac", type=float, default=0.025)
    parser.add_argument("--scale_min", type=float, default=0.97)
    parser.add_argument("--scale_max", type=float, default=1.03)
    parser.add_argument("--text_noise_prob", type=float, default=0.3)
    parser.add_argument("--char_noise_prob", type=float, default=0.03)
    parser.add_argument("--word_noise_prob", type=float, default=0.05)
    parser.add_argument("--numeric_noise_prob", type=float, default=0.2)
    parser.add_argument(
        "--numeric_mode",
        choices=["independent", "price_like", "none", "receipt_consistent"],
        default="price_like",
    )
    parser.add_argument("--mask_prob", type=float, default=0.15)
    parser.add_argument("--mask_word_prob", type=float, default=0.03)
    parser.add_argument("--mask_token", default="****")
    parser.add_argument("--image_noise_prob", type=float, default=0.4)
    parser.add_argument("--blur_prob", type=float, default=0.2)
    parser.add_argument("--noise_prob", type=float, default=0.2)
    parser.add_argument("--jpeg_prob", type=float, default=0.2)
    parser.add_argument("--make_overlays", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional smoke-test limit after discovery/split filtering.")
    return parser.parse_args()


def fail(message):
    print(message if message.startswith("ERROR:") else f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload, compact=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        if compact:
            json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"))
        else:
            json.dump(payload, handle, ensure_ascii=False, indent=2)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_short_hash(value):
    return hashlib.sha1(str(value).encode("utf-8")).hexdigest()[:8]


def safe_rmtree(path, input_dir):
    target = Path(path).resolve()
    source = Path(input_dir).resolve()
    if target == source:
        fail("--output_dir must not be the same as --input_dir when --overwrite is used.")
    if source in target.parents:
        fail("--output_dir must not be inside --input_dir for this augmentation run.")
    if len(target.parts) <= 2:
        fail(f"Refusing to remove suspicious output_dir: {target}")
    shutil.rmtree(target)


def is_excluded(path, exclude_name):
    if not exclude_name:
        return False
    needle = exclude_name.lower()
    return any(needle in part.lower() for part in Path(path).parts)


def strip_label_suffix(stem):
    for suffix in ("_labeled_v2_1", "_labeled_v2", "_labeled", "_labels_v2", "_labels"):
        if stem.lower().endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def find_samples(input_dir, exclude_name, strict=False):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        fail(f"input_dir not found: {input_dir}")
    samples = []
    skipped = []
    for label_path in sorted(input_dir.rglob("*_labeled_v2_1.json")):
        if is_excluded(label_path, exclude_name):
            skipped.append({"label_json": str(label_path), "reason": f"excluded by {exclude_name}"})
            continue
        folder = label_path.parent
        if not folder.name.endswith("_receipt_ocr"):
            skipped.append({"label_json": str(label_path), "reason": "not inside *_receipt_ocr folder"})
            continue
        capture_id = strip_label_suffix(label_path.stem)
        image_path = folder / f"{capture_id}.jpg"
        ocr_path = folder / f"{capture_id}_ocr.json"
        missing = []
        if not image_path.exists():
            missing.append(str(image_path))
        if not ocr_path.exists():
            missing.append(str(ocr_path))
        if missing:
            item = {"label_json": str(label_path), "capture_id": capture_id, "reason": "missing files", "missing": missing}
            skipped.append(item)
            if strict:
                fail(f"Missing paired files for {label_path}: {missing}")
            continue
        samples.append(
            {
                "capture_id": capture_id,
                "folder": str(folder),
                "image": str(image_path),
                "ocr_json": str(ocr_path),
                "label_json": str(label_path),
            }
        )
    return samples, skipped


def load_split_manifest(path):
    if not path:
        return {}, "not_provided"
    manifest_path = Path(path)
    if not manifest_path.exists():
        fail(f"split_manifest not found: {manifest_path}")
    obj = load_json(manifest_path)
    mapping = {}
    if isinstance(obj, dict):
        if "samples" in obj and isinstance(obj["samples"], list):
            for item in obj["samples"]:
                if not isinstance(item, dict):
                    continue
                capture_id = item.get("capture_id") or item.get("captureId") or item.get("id")
                split = item.get("split")
                if capture_id and split:
                    mapping[str(capture_id)] = str(split)
        else:
            for key, value in obj.items():
                if isinstance(value, list):
                    for item in value:
                        capture_id = item.get("capture_id") if isinstance(item, dict) else item
                        if capture_id:
                            mapping[str(capture_id)] = str(key)
                elif isinstance(value, str):
                    mapping[str(key)] = value
    return mapping, str(manifest_path)


def should_process_split(capture_id, split_mapping, apply_to_splits):
    split = split_mapping.get(capture_id)
    if split is None:
        split = "train"
    if apply_to_splits == "all":
        return True, split
    return split == "train", split


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


def validate_coordinate_sizes(image_size, label_payload, ocr_payload, strict=False):
    warnings = []
    image_width, image_height = image_size
    for name, payload in (("label_json", label_payload), ("ocr_json", ocr_payload)):
        width, height = read_json_size(payload)
        if width is None or height is None:
            message = f"{name} missing image_width/image_height"
            if strict:
                raise VariantSkip(message)
            warnings.append(message)
            continue
        if width != image_width or height != image_height:
            message = f"{name} size {width}x{height} != actual image {image_width}x{image_height}"
            if strict:
                raise VariantSkip(message)
            warnings.append(message)
    return warnings


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


def box_to_corners(box):
    x0, y0, x1, y1 = box
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def corners_to_box(points):
    if not points:
        return None
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]


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
    valid = [box for box in boxes if box]
    if not valid:
        return None
    return [
        min(box[0] for box in valid),
        min(box[1] for box in valid),
        max(box[2] for box in valid),
        max(box[3] for box in valid),
    ]


def transform_point(point, matrix):
    x, y = point
    a, b, c, d, e, f = matrix
    return [a * x + b * y + c, d * x + e * y + f]


def build_affine(width, height, rng, args):
    if rng.random() >= args.geometry_prob:
        angle = 0.0
        scale = 1.0
        tx = 0.0
        ty = 0.0
    else:
        angle = rng.uniform(args.rotation_deg_min, args.rotation_deg_max)
        scale = rng.uniform(args.scale_min, args.scale_max)
        tx = rng.uniform(-args.translate_x_frac, args.translate_x_frac) * width
        ty = rng.uniform(-args.translate_y_frac, args.translate_y_frac) * height

    theta = math.radians(angle)
    cos_v = math.cos(theta) * scale
    sin_v = math.sin(theta) * scale
    cx = width / 2.0
    cy = height / 2.0
    a, b = cos_v, -sin_v
    d, e = sin_v, cos_v
    c = cx + tx - a * cx - b * cy
    f = cy + ty - d * cx - e * cy
    det = a * e - b * d
    if abs(det) < 1e-8:
        raise VariantSkip("affine matrix is singular")
    ia = e / det
    ib = -b / det
    id_ = -d / det
    ie = a / det
    ic = -(ia * c + ib * f)
    iff = -(id_ * c + ie * f)
    return {
        "angle_deg": angle,
        "scale": scale,
        "translate_x": tx,
        "translate_y": ty,
        "forward": [a, b, c, d, e, f],
        "inverse": [ia, ib, ic, id_, ie, iff],
    }


def transform_image(image, affine):
    return image.transform(
        image.size,
        Image.Transform.AFFINE,
        data=tuple(affine["inverse"]),
        resample=Image.Resampling.BICUBIC,
        fillcolor=(255, 255, 255),
    )


def transform_word_box(word, affine, width, height):
    raw_box = parse_box(word.get("box"))
    if raw_box is None:
        raise VariantSkip(f"invalid word box before transform: {word.get('box')}")
    raw_corners = word.get("cornerPoints")
    if not isinstance(raw_corners, list) or len(raw_corners) < 4:
        raw_corners = box_to_corners(raw_box)
    transformed = [transform_point(point, affine["forward"]) for point in raw_corners]
    new_box = clamp_box(corners_to_box(transformed), width, height)
    if new_box is None:
        raise VariantSkip("word box became invalid after geometry transform")
    new_corners = []
    for point in transformed:
        x = max(0, min(int(round(point[0])), width - 1))
        y = max(0, min(int(round(point[1])), height - 1))
        new_corners.append([x, y])
    if new_box[2] - new_box[0] < 2 or new_box[3] - new_box[1] < 2:
        raise VariantSkip("word box became too small after geometry transform")
    return new_box, new_corners


def maybe_degrade_image(image, rng, args):
    operations = []
    out = image
    if rng.random() < args.image_noise_prob:
        if rng.random() < args.blur_prob:
            radius = rng.uniform(0.1, 0.8)
            out = out.filter(ImageFilter.GaussianBlur(radius=radius))
            operations.append({"op": "gaussian_blur", "radius": radius})
        if rng.random() < 0.5:
            brightness = rng.uniform(0.9, 1.1)
            out = ImageEnhance.Brightness(out).enhance(brightness)
            operations.append({"op": "brightness", "factor": brightness})
        if rng.random() < 0.5:
            contrast = rng.uniform(0.9, 1.1)
            out = ImageEnhance.Contrast(out).enhance(contrast)
            operations.append({"op": "contrast", "factor": contrast})
        if rng.random() < args.noise_prob:
            sigma = rng.uniform(1.0, 6.0)
            array = np.asarray(out).astype(np.float32)
            noise_array = np.random.default_rng(rng.randrange(2**32)).normal(0, sigma, array.shape)
            array = np.clip(array + noise_array, 0, 255).astype(np.uint8)
            out = Image.fromarray(array, mode="RGB")
            operations.append({"op": "gaussian_noise", "sigma": sigma})
        if rng.random() < args.jpeg_prob:
            quality = rng.randint(65, 95)
            buffer = io.BytesIO()
            out.save(buffer, format="JPEG", quality=quality)
            buffer.seek(0)
            out = Image.open(buffer).convert("RGB")
            operations.append({"op": "jpeg_recompress", "quality": quality})
    return out, operations


def field_from_word(word, labels=None, idx=None):
    label = word.get("label")
    if label is None and labels is not None and idx is not None and idx < len(labels):
        label = labels[idx]
    return label_to_field(label)


def mutate_numeric_text(text, field, rng, mode):
    if mode == "none":
        return text, False
    if mode == "receipt_consistent":
        raise VariantSkip("numeric_mode=receipt_consistent is not supported for this augmentation tool")
    if not any(ch.isdigit() for ch in text):
        return text, False
    chars = list(text)
    digit_positions = [idx for idx, ch in enumerate(chars) if ch.isdigit()]
    if not digit_positions:
        return text, False
    if mode == "price_like" and field in PRICE_LIKE_FIELDS:
        positions = digit_positions[-2:] if len(digit_positions) >= 2 else digit_positions
        idx = rng.choice(positions)
    else:
        idx = rng.choice(digit_positions)
    old = chars[idx]
    choices = [str(i) for i in range(10) if str(i) != old]
    chars[idx] = rng.choice(choices)
    return "".join(chars), True


def mutate_text_ocr_like(text, field, rng, args):
    if not text or len(text.strip()) == 0:
        return text, False
    if field in PRICE_LIKE_FIELDS:
        local_char_prob = min(args.char_noise_prob, 0.015)
        allow_delete_swap = False
    else:
        local_char_prob = args.char_noise_prob
        allow_delete_swap = True
    value = text
    changed = False
    for src, dst in CONFUSIONS:
        if src in value and rng.random() < local_char_prob:
            value = value.replace(src, dst, 1)
            changed = True
            break
    if allow_delete_swap and len(value) > 3 and rng.random() < local_char_prob * 0.5:
        idx = rng.randrange(len(value))
        value = value[:idx] + value[idx + 1 :]
        changed = True
    if allow_delete_swap and len(value) > 3 and rng.random() < local_char_prob * 0.7:
        idx = rng.randrange(len(value) - 1)
        chars = list(value)
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
        value = "".join(chars)
        changed = True
    if value.isalpha() and rng.random() < local_char_prob:
        value = value.lower() if value[:1].isupper() else value.upper()
        changed = True
    return value, changed


def mutate_words(words, labels, rng, args):
    new_texts = []
    changes = []
    for idx, word in enumerate(words):
        original = str(word.get("text", ""))
        field = field_from_word(word, labels, idx)
        text = original
        changed_ops = []
        if args.mask_prob > 0 and rng.random() < args.mask_prob * args.mask_word_prob:
            text = args.mask_token
            changed_ops.append("mask")
        elif field in NUMERIC_FIELDS and rng.random() < args.numeric_noise_prob:
            text, changed = mutate_numeric_text(text, field, rng, args.numeric_mode)
            if changed:
                changed_ops.append("numeric")
        elif rng.random() < args.text_noise_prob * args.word_noise_prob:
            text, changed = mutate_text_ocr_like(text, field, rng, args)
            if changed:
                changed_ops.append("text_noise")
        new_texts.append(text)
        if changed_ops:
            changes.append({"word_idx": idx, "field": field, "from": original, "to": text, "ops": changed_ops})
    return new_texts, changes


def update_image_metadata(payload, aug_capture_id, image_name, image_width, image_height, image_sha, augmentation_meta):
    if not isinstance(payload, dict):
        return
    original_created = payload.get("createdAtUtc")
    payload["captureId"] = aug_capture_id
    payload["capture_id"] = aug_capture_id
    payload["createdAtUtc"] = utc_now()
    payload["image_width"] = image_width
    payload["image_height"] = image_height
    if isinstance(payload.get("image"), dict):
        payload["image"]["width"] = image_width
        payload["image"]["height"] = image_height
        for key in ("fileName", "filename", "name"):
            if key in payload["image"]:
                payload["image"][key] = image_name
        payload["image"]["sha256"] = image_sha
    else:
        payload["image"] = {"fileName": image_name, "width": image_width, "height": image_height, "sha256": image_sha}
    if "image_path" in payload:
        payload["image_path"] = image_name
    payload["augmentation"] = dict(augmentation_meta)
    payload["augmentation"]["original_createdAtUtc"] = original_created


def refresh_root_words(payload, transformed_words):
    payload["words"] = copy.deepcopy(transformed_words)
    payload["labels"] = [word.get("label", "O") for word in transformed_words]
    payload["label_counts"] = dict(Counter(payload["labels"]))
    payload["field_counts"] = dict(Counter(label_to_field(label) for label in payload["labels"]))


def refresh_ocr_words(payload, transformed_words):
    if not isinstance(payload.get("words"), list):
        payload["words"] = []
    existing = payload["words"]
    if len(existing) == len(transformed_words):
        refreshed = []
        for source, transformed in zip(existing, transformed_words):
            item = copy.deepcopy(source)
            item["text"] = transformed.get("text", "")
            item["box"] = transformed.get("box")
            item["cornerPoints"] = transformed.get("cornerPoints")
            refreshed.append(item)
        payload["words"] = refreshed
    else:
        refreshed = []
        for transformed in transformed_words:
            item = {
                key: transformed.get(key)
                for key in ("wordId", "blockId", "lineId", "wordIndexInLine", "globalWordIndex", "text", "box", "cornerPoints")
                if key in transformed
            }
            refreshed.append(item)
        payload["words"] = refreshed


def word_text(words, indices):
    values = []
    for idx in normalize_indices(indices):
        if 0 <= idx < len(words):
            values.append(str(words[idx].get("text", "")))
    return " ".join(value for value in values if value)


def span_ids_for_indices(spans, indices):
    wanted = set(normalize_indices(indices))
    result = []
    for span in spans:
        span_indices = set(normalize_indices(span.get("word_indices")))
        if span_indices & wanted:
            result.append(span.get("span_id"))
    return [item for item in result if item is not None]


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


def rebuild_spans(words):
    spans = []
    current = None
    for idx, word in enumerate(words):
        label = str(word.get("label", "O") or "O")
        if canonicalize_label is not None:
            label = canonicalize_label(label)
        field = label_to_field(label)
        prefix = "O"
        if label.startswith("B-"):
            prefix = "B"
        elif label.startswith("I-"):
            prefix = "I"
        if field == "O" or prefix == "O":
            if current:
                spans.append(current)
                current = None
            continue
        if prefix == "B" or current is None or current["field"] != field:
            if current:
                spans.append(current)
            current = {"field": field, "word_indices": [], "label_sequence": []}
        current["word_indices"].append(idx)
        current["label_sequence"].append(label)
    if current:
        spans.append(current)

    for span_idx, span in enumerate(spans):
        indices = span["word_indices"]
        text = word_text(words, indices)
        span["span_id"] = f"span_{span_idx:04d}"
        span["text"] = text
        span["normalized_text"] = normalize_span_text(span["field"], text)
        span["box"] = union_boxes([parse_box(words[idx].get("box")) for idx in indices])
    return spans


def refresh_lines_and_blocks(payload):
    words = payload.get("words") or []
    by_word_id = {word.get("wordId"): word for word in words if isinstance(word, dict) and word.get("wordId") is not None}
    by_line_id = defaultdict(list)
    by_block_id = defaultdict(list)
    for word in words:
        if not isinstance(word, dict):
            continue
        if word.get("lineId") is not None:
            by_line_id[word.get("lineId")].append(word)
        if word.get("blockId") is not None:
            by_block_id[word.get("blockId")].append(word)

    line_boxes = {}
    line_texts = {}
    if isinstance(payload.get("lines"), list):
        for line in payload["lines"]:
            if not isinstance(line, dict):
                continue
            line_words = []
            for word_id in line.get("wordIds") or []:
                if word_id in by_word_id:
                    line_words.append(by_word_id[word_id])
            if not line_words and line.get("lineId") in by_line_id:
                line_words = by_line_id[line.get("lineId")]
            box = union_boxes([parse_box(word.get("box")) for word in line_words])
            text = " ".join(str(word.get("text", "")) for word in line_words).strip()
            if box:
                line["box"] = box
                line["cornerPoints"] = box_to_corners(box)
            line["text"] = text
            if line.get("lineId") is not None:
                line_boxes[line.get("lineId")] = box
                line_texts[line.get("lineId")] = text

    if isinstance(payload.get("blocks"), list):
        for block in payload["blocks"]:
            if not isinstance(block, dict):
                continue
            boxes = []
            texts = []
            for line_id in block.get("lineIds") or []:
                if line_id in line_boxes and line_boxes[line_id]:
                    boxes.append(line_boxes[line_id])
                    texts.append(line_texts.get(line_id, ""))
            if not boxes and block.get("blockId") in by_block_id:
                block_words = by_block_id[block.get("blockId")]
                boxes = [parse_box(word.get("box")) for word in block_words]
                texts = [" ".join(str(word.get("text", "")) for word in block_words).strip()]
            box = union_boxes(boxes)
            if box:
                block["box"] = box
                block["cornerPoints"] = box_to_corners(box)
            block["text"] = "\n".join(text for text in texts if text)


def refresh_relation_texts(payload):
    words = payload.get("words") or []
    spans = payload.get("spans") or []
    relation_ids_as_head = defaultdict(list)
    relation_ids_as_tail = defaultdict(list)
    for key in RELATION_KEYS:
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        for ordinal, relation in enumerate(values):
            if not isinstance(relation, dict):
                continue
            relation_id = relation.get("relation_id") or relation.get("edge_id") or f"{key}_{ordinal:04d}"
            if "head_word_indices" in relation:
                relation["head_text"] = word_text(words, relation.get("head_word_indices"))
                relation["head_span_ids"] = span_ids_for_indices(spans, relation.get("head_word_indices"))
                for idx in normalize_indices(relation.get("head_word_indices")):
                    relation_ids_as_head[str(idx)].append(relation_id)
            elif "head_word_idx" in relation:
                relation["head_text"] = word_text(words, [relation.get("head_word_idx")])
                relation["head_span_ids"] = span_ids_for_indices(spans, [relation.get("head_word_idx")])
                relation_ids_as_head[str(relation.get("head_word_idx"))].append(relation_id)

            tail_keys = [
                ("tail_word_indices", "tail_text", "tail_span_ids"),
                ("dep_word_indices", "dep_text", "dep_span_ids"),
                ("dependent_word_indices", "dependent_text", "dependent_span_ids"),
                ("aux_word_indices", "aux_text", "aux_span_ids"),
            ]
            for index_key, text_key, span_key in tail_keys:
                if index_key in relation:
                    relation[text_key] = word_text(words, relation.get(index_key))
                    relation[span_key] = span_ids_for_indices(spans, relation.get(index_key))
                    for idx in normalize_indices(relation.get(index_key)):
                        relation_ids_as_tail[str(idx)].append(relation_id)
            if "dependent_word_idx" in relation and "dependent_word_indices" not in relation:
                relation["dependent_text"] = word_text(words, [relation.get("dependent_word_idx")])
                relation["dependent_span_ids"] = span_ids_for_indices(spans, [relation.get("dependent_word_idx")])
                relation_ids_as_tail[str(relation.get("dependent_word_idx"))].append(relation_id)
            if "tail_word_idx" in relation and "tail_word_indices" not in relation:
                relation["tail_text"] = word_text(words, [relation.get("tail_word_idx")])
                relation["tail_span_ids"] = span_ids_for_indices(spans, [relation.get("tail_word_idx")])
                relation_ids_as_tail[str(relation.get("tail_word_idx"))].append(relation_id)
    if relation_ids_as_head:
        payload["relation_ids_as_head"] = {key: sorted(set(values)) for key, values in relation_ids_as_head.items()}
    if relation_ids_as_tail:
        payload["relation_ids_as_tail"] = {key: sorted(set(values)) for key, values in relation_ids_as_tail.items()}
    payload["relation_counts"] = {
        key: len(payload.get(key) or []) for key in ("relations", "item_relations", "summary_relations", "payment_relations", "rel_g_edges")
    }


def build_transformed_words(label_words, labels, new_texts, affine, width, height):
    transformed = []
    for idx, word in enumerate(label_words):
        if not isinstance(word, dict):
            raise VariantSkip(f"word {idx} is not an object")
        updated = copy.deepcopy(word)
        box, corners = transform_word_box(updated, affine, width, height)
        updated["text"] = new_texts[idx]
        updated["box"] = box
        updated["cornerPoints"] = corners
        updated["word_idx"] = idx
        updated["globalWordIndex"] = updated.get("globalWordIndex", idx)
        label = updated.get("label")
        if label is None and idx < len(labels):
            label = labels[idx]
        updated["label"] = label or "O"
        updated["field"] = label_to_field(updated["label"])
        transformed.append(updated)
    return transformed


def write_overlays(image_path, label_path, folder, aug_capture_id, python_executable, debug=False):
    results = {}
    word_overlay = folder / f"{aug_capture_id}_labeled_v2_1_overlay.png"
    relation_overlay = folder / f"{aug_capture_id}_relations_overlay.png"
    relation_summary = folder / f"{aug_capture_id}_relations_summary.json"
    commands = [
        [
            python_executable,
            str(ROOT_DIR / "scripts" / "overlay_labeled_receipt_json.py"),
            "--image",
            str(image_path),
            "--label_json",
            str(label_path),
            "--out",
            str(word_overlay),
            "--summary_out",
            str(folder / f"{aug_capture_id}_label_summary.json"),
            "--coordinate_mode",
            "strict",
        ],
        [
            python_executable,
            str(ROOT_DIR / "scripts" / "overlay_labeled_relations.py"),
            "--image",
            str(image_path),
            "--label_json",
            str(label_path),
            "--out",
            str(relation_overlay),
            "--summary_out",
            str(relation_summary),
            "--coordinate_mode",
            "strict",
            "--relation_source",
            "all",
        ],
    ]
    for command in commands:
        completed = subprocess.run(command, cwd=str(ROOT_DIR), text=True, capture_output=True)
        if completed.returncode != 0:
            results.setdefault("overlay_errors", []).append(
                {
                    "command": command,
                    "returncode": completed.returncode,
                    "stdout": completed.stdout[-2000:],
                    "stderr": completed.stderr[-2000:],
                }
            )
            if debug:
                print(completed.stdout)
                print(completed.stderr, file=sys.stderr)
        elif debug:
            print(completed.stdout)
    results["word_overlay"] = str(word_overlay) if word_overlay.exists() else None
    results["relation_overlay"] = str(relation_overlay) if relation_overlay.exists() else None
    results["relation_summary"] = str(relation_summary) if relation_summary.exists() else None
    return results


def process_variant(sample, variant_index, output_dir, args, rng):
    capture_id = sample["capture_id"]
    image_path = Path(sample["image"])
    label_path = Path(sample["label_json"])
    ocr_path = Path(sample["ocr_json"])
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    label_payload = load_json(label_path)
    ocr_payload = load_json(ocr_path)
    warnings = validate_coordinate_sizes((width, height), label_payload, ocr_payload, strict=args.strict)
    label_words = label_payload.get("words")
    if not isinstance(label_words, list) or not label_words:
        raise VariantSkip("label_json has no words")
    labels = label_payload.get("labels") if isinstance(label_payload.get("labels"), list) else []
    if labels and len(labels) != len(label_words):
        raise VariantSkip(f"labels length {len(labels)} != words length {len(label_words)}")

    affine = build_affine(width, height, rng, args)
    new_texts, text_changes = mutate_words(label_words, labels, rng, args)
    transformed_words = build_transformed_words(label_words, labels, new_texts, affine, width, height)
    transformed_image = transform_image(image, affine)
    transformed_image, image_ops = maybe_degrade_image(transformed_image, rng, args)

    aug_capture_id = f"{capture_id}_aug_{variant_index:03d}_{stable_short_hash(str(affine) + str(text_changes))}"
    out_folder = Path(output_dir) / f"{aug_capture_id}_receipt_ocr"
    out_folder.mkdir(parents=True, exist_ok=True)
    out_image = out_folder / f"{aug_capture_id}.jpg"
    out_label = out_folder / f"{aug_capture_id}_labeled_v2_1.json"
    out_ocr = out_folder / f"{aug_capture_id}_ocr.json"

    transformed_image.save(out_image, quality=95)
    image_sha = sha256_file(out_image)
    augmentation_meta = {
        "parent_capture_id": capture_id,
        "parent_image": str(image_path),
        "parent_label_json": str(label_path),
        "parent_ocr_json": str(ocr_path),
        "parent_image_sha256": sha256_file(image_path),
        "variant_index": variant_index,
        "seed": args.seed,
        "arithmetic_consistency": "not_guaranteed",
        "geometry": {key: affine[key] for key in ("angle_deg", "scale", "translate_x", "translate_y")},
        "image_operations": image_ops,
        "text_changes": text_changes,
        "created_by": "scripts/augment_labeled_receipt_dataset.py",
    }

    label_aug = copy.deepcopy(label_payload)
    ocr_aug = copy.deepcopy(ocr_payload)
    refresh_root_words(label_aug, transformed_words)
    label_aug["spans"] = rebuild_spans(label_aug["words"])
    refresh_lines_and_blocks(label_aug)
    refresh_relation_texts(label_aug)
    update_image_metadata(label_aug, aug_capture_id, out_image.name, width, height, image_sha, augmentation_meta)

    refresh_ocr_words(ocr_aug, transformed_words)
    refresh_lines_and_blocks(ocr_aug)
    update_image_metadata(ocr_aug, aug_capture_id, out_image.name, width, height, image_sha, augmentation_meta)

    save_json(out_label, label_aug)
    save_json(out_ocr, ocr_aug)

    overlay_results = {}
    if args.make_overlays:
        overlay_results = write_overlays(out_image, out_label, out_folder, aug_capture_id, sys.executable, debug=args.debug)

    relation_array_counts = {key: len(label_aug.get(key) or []) for key in RELATION_KEYS}
    relation_count = relation_array_counts["relations"] or sum(
        relation_array_counts[key] for key in ("item_relations", "summary_relations", "payment_relations")
    )
    record = {
        "capture_id": aug_capture_id,
        "parent_capture_id": capture_id,
        "split": sample.get("split", "train"),
        "folder": str(out_folder),
        "image": str(out_image),
        "ocr_json": str(out_ocr),
        "label_json": str(out_label),
        "image_sha256": image_sha,
        "word_count": len(label_aug["words"]),
        "span_count": len(label_aug.get("spans") or []),
        "relation_count": relation_count,
        "relation_array_counts": relation_array_counts,
        "label_counts": dict(Counter(label_aug.get("labels") or [])),
        "text_change_count": len(text_changes),
        "warnings": warnings,
        "overlay_results": overlay_results,
        "augmentation": augmentation_meta,
    }
    return record, label_aug


def main():
    args = parse_args()
    if args.numeric_mode == "receipt_consistent":
        fail("numeric_mode=receipt_consistent is intentionally unsupported; use price_like, independent, or none.")
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            fail(f"output_dir already exists: {output_dir}. Use --overwrite to replace it.")
        safe_rmtree(output_dir, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_mapping, split_source = load_split_manifest(args.split_manifest)
    samples, discovery_skipped = find_samples(input_dir, args.exclude_dir_name, strict=args.strict)
    rng = random.Random(args.seed)
    records = []
    skipped = list(discovery_skipped)
    jsonl_path = output_dir / "all_labeled_v2_1_augmented.jsonl"

    with jsonl_path.open("w", encoding="utf-8") as jsonl_handle:
        processed_parent_count = 0
        for sample in samples:
            should_process, split = should_process_split(sample["capture_id"], split_mapping, args.apply_to_splits)
            sample["split"] = split
            if not should_process:
                skipped.append({"capture_id": sample["capture_id"], "reason": f"split {split} not selected"})
                continue
            if args.max_samples is not None and processed_parent_count >= args.max_samples:
                skipped.append({"capture_id": sample["capture_id"], "reason": f"skipped by max_samples={args.max_samples}"})
                continue
            processed_parent_count += 1
            variant_successes = 0
            attempts = 0
            max_attempts = max(args.variants_per_sample * 5, args.variants_per_sample + 3)
            while variant_successes < args.variants_per_sample and attempts < max_attempts:
                attempts += 1
                try:
                    record, label_aug = process_variant(sample, variant_successes, output_dir, args, rng)
                    records.append(record)
                    jsonl_handle.write(json.dumps(label_aug, ensure_ascii=False, separators=(",", ":")) + "\n")
                    variant_successes += 1
                    if args.debug:
                        print(f"created {record['capture_id']} words={record['word_count']} relations={record['relation_count']}")
                except VariantSkip as exc:
                    skipped.append(
                        {
                            "capture_id": sample["capture_id"],
                            "variant_attempt": attempts,
                            "reason": str(exc),
                        }
                    )
                    if args.strict:
                        raise
            if variant_successes < args.variants_per_sample:
                message = f"{sample['capture_id']}: created {variant_successes}/{args.variants_per_sample} variants"
                if args.strict:
                    fail(message)
                print(f"WARNING: {message}")

    manifest = {
        "schema_version": "receipt_augmentation_v1",
        "created_at_utc": utc_now(),
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "exclude_dir_name": args.exclude_dir_name,
        "split_manifest": split_source,
        "apply_to_splits": args.apply_to_splits,
        "parent_split_policy": args.parent_split_policy,
        "seed": args.seed,
        "variants_per_sample": args.variants_per_sample,
        "records": records,
        "skipped": skipped,
    }
    summary = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "num_input_samples_discovered": len(samples),
        "num_augmented_records": len(records),
        "num_skipped": len(skipped),
        "jsonl_path": str(jsonl_path),
        "label_counts": dict(
            sum((Counter(record.get("label_counts") or {}) for record in records), Counter())
        ),
        "relation_count_total": sum(record.get("relation_count", 0) for record in records),
        "text_change_count_total": sum(record.get("text_change_count", 0) for record in records),
        "records_by_parent": dict(Counter(record["parent_capture_id"] for record in records)),
        "skipped_reasons": dict(Counter(item.get("reason", "unknown") for item in skipped)),
    }
    save_json(output_dir / "augmentation_manifest.json", manifest)
    save_json(output_dir / "augmentation_summary.json", summary)

    print(f"input_dir: {input_dir}")
    print(f"output_dir: {output_dir}")
    print(f"discovered samples: {len(samples)}")
    print(f"augmented records: {len(records)}")
    print(f"skipped: {len(skipped)}")
    print(f"jsonl: {jsonl_path}")
    print(f"manifest: {output_dir / 'augmentation_manifest.json'}")
    print(f"summary: {output_dir / 'augmentation_summary.json'}")
    if skipped:
        print("first skipped items:")
        for item in skipped[:20]:
            print(f"  - {item}")
    print("Labeled receipt augmentation passed.")


if __name__ == "__main__":
    main()
