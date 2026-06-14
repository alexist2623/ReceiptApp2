import argparse
import copy
import json
import sys
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Rescale a labeled receipt JSON to match the actual image size.")
    parser.add_argument("--image", required=True, help="Actual receipt image path.")
    parser.add_argument("--label_json", required=True, help="Labeled receipt JSON path to rescale.")
    parser.add_argument("--out", required=True, help="Output path for the scaled label JSON.")
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
        return image.size


def read_label_size(payload):
    width = payload.get("image_width") or payload.get("width")
    height = payload.get("image_height") or payload.get("height")
    image = payload.get("image")
    if (width is None or height is None) and isinstance(image, dict):
        width = image.get("width") if width is None else width
        height = image.get("height") if height is None else height
    if width is None or height is None:
        fail("label_json missing image_width/image_height; cannot compute scale.")
    try:
        return int(width), int(height)
    except (TypeError, ValueError):
        fail(f"label_json image size must be integer-like, got {width!r}x{height!r}.")


def scale_box(box, scale_x, scale_y):
    if not isinstance(box, list) or len(box) != 4:
        return box
    try:
        x0, y0, x1, y1 = [float(v) for v in box]
    except (TypeError, ValueError):
        return box
    return [
        int(round(x0 * scale_x)),
        int(round(y0 * scale_y)),
        int(round(x1 * scale_x)),
        int(round(y1 * scale_y)),
    ]


def scale_point(point, scale_x, scale_y):
    if not isinstance(point, list) or len(point) != 2:
        return point
    try:
        return [int(round(float(point[0]) * scale_x)), int(round(float(point[1]) * scale_y))]
    except (TypeError, ValueError):
        return point


def scale_coordinates(value, scale_x, scale_y):
    if isinstance(value, list):
        return [scale_coordinates(item, scale_x, scale_y) for item in value]
    if not isinstance(value, dict):
        return value
    scaled = {}
    for key, item in value.items():
        if key == "box":
            scaled[key] = scale_box(item, scale_x, scale_y)
        elif key == "cornerPoints" and isinstance(item, list):
            scaled[key] = [scale_point(point, scale_x, scale_y) for point in item]
        else:
            scaled[key] = scale_coordinates(item, scale_x, scale_y)
    return scaled


def main():
    args = parse_args()
    image_path = Path(args.image)
    label_path = Path(args.label_json)
    if not image_path.exists():
        fail(f"Image file not found: {image_path}")
    if not label_path.exists():
        fail(f"label_json not found: {label_path}")

    target_width, target_height = image_size(image_path)
    payload = load_json(label_path)
    source_width, source_height = read_label_size(payload)
    scale_x = target_width / source_width
    scale_y = target_height / source_height

    scaled = scale_coordinates(copy.deepcopy(payload), scale_x, scale_y)
    scaled["image_width"] = target_width
    scaled["image_height"] = target_height
    if isinstance(scaled.get("image"), dict):
        scaled["image"]["width"] = target_width
        scaled["image"]["height"] = target_height
    scaled["coordinate_transform"] = {
        "source_width": source_width,
        "source_height": source_height,
        "target_width": target_width,
        "target_height": target_height,
        "scale_x": scale_x,
        "scale_y": scale_y,
        "reason": "rescaled to match actual training image",
    }
    save_json(args.out, scaled)

    print(f"image: {image_path}")
    print(f"label_json: {label_path}")
    print(f"out: {args.out}")
    print(f"source_size: {source_width}x{source_height}")
    print(f"target_size: {target_width}x{target_height}")
    print(f"scale_x={scale_x:.6f} scale_y={scale_y:.6f}")
    print("Labeled receipt JSON rescale passed.")


if __name__ == "__main__":
    main()
