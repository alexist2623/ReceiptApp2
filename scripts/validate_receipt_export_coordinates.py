import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
JSON_SUFFIXES = {".json"}
LABEL_MARKERS = ("labeled", "labels", "label")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate receipt export image/OCR/labeled JSON coordinate spaces.")
    parser.add_argument("--input_dir", required=True, help="Unzipped export or labeled bundle directory.")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any coordinate mismatch is found.")
    parser.add_argument("--out_json", default="outputs/coordinate_validation_summary.json")
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


def image_size(path):
    with Image.open(path) as image:
        return list(image.size)


def normalize_capture_id(path):
    stem = path.stem
    lower = stem.lower()
    suffixes = [
        "_labeled_v2_1",
        "_labeled_v2",
        "_labels_v2_1",
        "_labels_v2",
        "_labeled",
        "_labels",
        "_label",
        "_ocr",
        "_server_result",
        "_receipt_ocr",
    ]
    for suffix in suffixes:
        if lower.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def is_label_json(path, payload=None):
    lower = path.stem.lower()
    if any(marker in lower for marker in LABEL_MARKERS):
        return True
    if isinstance(payload, dict) and isinstance(payload.get("labels"), list):
        return True
    return False


def is_ocr_json(path, payload=None):
    lower = path.stem.lower()
    if lower.endswith("_ocr") or "ocr_json" in {part.lower() for part in path.parts}:
        return True
    if isinstance(payload, dict) and payload.get("schemaVersion") == "receipt_ocr_v1":
        return True
    if isinstance(payload, dict) and isinstance(payload.get("ocr"), dict):
        return True
    return False


def read_size_fields(payload):
    if not isinstance(payload, dict):
        return None, []
    sizes = []
    top_w = payload.get("image_width") or payload.get("width")
    top_h = payload.get("image_height") or payload.get("height")
    if top_w is not None and top_h is not None:
        sizes.append(("top_level", coerce_size(top_w, top_h)))
    image = payload.get("image")
    if isinstance(image, dict) and image.get("width") is not None and image.get("height") is not None:
        sizes.append(("image", coerce_size(image.get("width"), image.get("height"))))
    primary = sizes[0][1] if sizes else None
    return primary, sizes


def coerce_size(width, height):
    try:
        return [int(width), int(height)]
    except (TypeError, ValueError):
        return [width, height]


def words_from_payload(payload):
    words = payload.get("words") if isinstance(payload, dict) else None
    return words if isinstance(words, list) else []


def parse_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        return [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None


def check_boxes(words, width, height):
    invalid = 0
    outside = 0
    total = 0
    examples = []
    for idx, word in enumerate(words):
        if not isinstance(word, dict):
            continue
        total += 1
        box = parse_box(word.get("box"))
        if box is None:
            invalid += 1
            if len(examples) < 10:
                examples.append({"word_idx": idx, "text": word.get("text"), "box": word.get("box"), "reason": "invalid_box"})
            continue
        x0, y0, x1, y1 = box
        if x1 <= x0 or y1 <= y0:
            invalid += 1
            if len(examples) < 10:
                examples.append({"word_idx": idx, "text": word.get("text"), "box": box, "reason": "non_positive_size"})
        if x0 < 0 or y0 < 0 or x1 >= width or y1 >= height:
            outside += 1
            if len(examples) < 10:
                examples.append({"word_idx": idx, "text": word.get("text"), "box": box, "reason": "outside_image"})
    denom = max(1, total)
    return {
        "word_count": total,
        "invalid_box_count": invalid,
        "outside_box_count": outside,
        "outside_box_ratio": outside / denom,
        "examples": examples,
    }


def compare_words(ocr_words, label_words):
    errors = []
    warnings = []
    if not ocr_words or not label_words:
        return errors, warnings
    if len(ocr_words) != len(label_words):
        errors.append(f"ocr_json/labeled_json words length mismatch: {len(ocr_words)} vs {len(label_words)}")
    mismatch_examples = []
    for idx, (ocr_word, label_word) in enumerate(zip(ocr_words, label_words)):
        ocr_text = str(ocr_word.get("text", "")) if isinstance(ocr_word, dict) else ""
        label_text = str(label_word.get("text", "")) if isinstance(label_word, dict) else ""
        if ocr_text != label_text and len(mismatch_examples) < 10:
            mismatch_examples.append({"word_idx": idx, "ocr": ocr_text, "label": label_text})
    if mismatch_examples:
        errors.append(f"ocr_json/labeled_json word text mismatch examples: {mismatch_examples}")
    return errors, warnings


def discover(input_dir):
    captures = defaultdict(lambda: {"images": [], "ocr_jsons": [], "label_jsons": [], "other_jsons": []})
    for path in Path(input_dir).rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            if "overlay" in path.stem.lower():
                continue
            captures[normalize_capture_id(path)]["images"].append(path)
        elif suffix in JSON_SUFFIXES:
            payload = None
            try:
                payload = load_json(path)
            except Exception:
                captures[normalize_capture_id(path)]["other_jsons"].append(path)
                continue
            capture_id = normalize_capture_id(path)
            if is_label_json(path, payload):
                captures[capture_id]["label_jsons"].append(path)
            elif is_ocr_json(path, payload):
                captures[capture_id]["ocr_jsons"].append(path)
            else:
                captures[capture_id]["other_jsons"].append(path)
    return captures


def choose_one(paths, preferred_suffix=None):
    if not paths:
        return None
    paths = sorted(paths)
    if preferred_suffix:
        for path in paths:
            if path.name.lower().endswith(preferred_suffix):
                return path
    return paths[0]


def validate_capture(capture_id, files):
    errors = []
    warnings = []
    image_path = choose_one(files["images"])
    ocr_path = choose_one(files["ocr_jsons"])
    label_path = choose_one(files["label_jsons"])
    result = {
        "capture_id": capture_id,
        "image_path": str(image_path) if image_path else None,
        "ocr_json_path": str(ocr_path) if ocr_path else None,
        "label_json_path": str(label_path) if label_path else None,
        "image_size": None,
        "ocr_json_size": None,
        "label_json_size": None,
        "ocr_box_stats": None,
        "label_box_stats": None,
        "status": "passed",
        "errors": errors,
        "warnings": warnings,
    }

    if image_path is None:
        errors.append("image file missing")
        result["status"] = "failed"
        return result
    actual_size = image_size(image_path)
    result["image_size"] = actual_size
    width, height = actual_size

    ocr_payload = load_json(ocr_path) if ocr_path else None
    label_payload = load_json(label_path) if label_path else None

    if ocr_payload is not None:
        ocr_size, ocr_sizes = read_size_fields(ocr_payload)
        result["ocr_json_size"] = ocr_size
        if not ocr_sizes:
            errors.append("ocr_json missing image_width/image_height and image.width/image.height")
        for source, size in ocr_sizes:
            if size != actual_size:
                errors.append(f"image vs ocr_json {source} size mismatch: image={actual_size}, json={size}")
        if len({tuple(size) for _, size in ocr_sizes}) > 1:
            errors.append(f"ocr_json internal image size mismatch: {ocr_sizes}")
        ocr_words = words_from_payload(ocr_payload)
        result["ocr_box_stats"] = check_boxes(ocr_words, width, height)
        if result["ocr_box_stats"]["invalid_box_count"] or result["ocr_box_stats"]["outside_box_count"]:
            errors.append(
                "ocr_json box coordinate issue: "
                f"invalid={result['ocr_box_stats']['invalid_box_count']}, "
                f"outside={result['ocr_box_stats']['outside_box_count']}, "
                f"outside_ratio={result['ocr_box_stats']['outside_box_ratio']:.6f}"
            )
    else:
        warnings.append("ocr_json missing")
        ocr_words = []

    if label_payload is not None:
        label_size, label_sizes = read_size_fields(label_payload)
        result["label_json_size"] = label_size
        if not label_sizes:
            errors.append("label_json missing image_width/image_height and image.width/image.height")
        for source, size in label_sizes:
            if size != actual_size:
                errors.append(f"image vs label_json {source} size mismatch: image={actual_size}, json={size}")
        if len({tuple(size) for _, size in label_sizes}) > 1:
            errors.append(f"label_json internal image size mismatch: {label_sizes}")
        label_words = words_from_payload(label_payload)
        result["label_box_stats"] = check_boxes(label_words, width, height)
        if result["label_box_stats"]["invalid_box_count"] or result["label_box_stats"]["outside_box_count"]:
            errors.append(
                "labeled_json box coordinate issue: "
                f"invalid={result['label_box_stats']['invalid_box_count']}, "
                f"outside={result['label_box_stats']['outside_box_count']}, "
                f"outside_ratio={result['label_box_stats']['outside_box_ratio']:.6f}"
            )
        word_errors, word_warnings = compare_words(ocr_words, label_words)
        errors.extend(word_errors)
        warnings.extend(word_warnings)
    else:
        warnings.append("labeled_json missing")

    if errors:
        result["status"] = "failed"
    return result


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        fail(f"input_dir not found: {input_dir}")
    captures = discover(input_dir)
    results = []
    for capture_id in sorted(captures):
        if not captures[capture_id]["images"]:
            continue
        results.append(validate_capture(capture_id, captures[capture_id]))
    failed = [item for item in results if item["status"] != "passed"]
    summary = {
        "input_dir": str(input_dir),
        "strict": args.strict,
        "total_captures": len(results),
        "passed": len(results) - len(failed),
        "failed": len(failed),
        "captures": results,
    }
    save_json(args.out_json, summary)
    print(f"input_dir: {input_dir}")
    print(f"total_captures: {summary['total_captures']}")
    print(f"passed: {summary['passed']}")
    print(f"failed: {summary['failed']}")
    print(f"summary JSON path: {args.out_json}")
    if failed:
        print("failed captures:")
        for item in failed[:20]:
            print(f"  - {item['capture_id']}: {item['errors']}")
    if args.strict and not results:
        print("No receipt captures with images were found.", file=sys.stderr)
        raise SystemExit(1)
    if args.strict and failed:
        raise SystemExit(1)
    print("Receipt export coordinate validation passed.")


if __name__ == "__main__":
    main()
