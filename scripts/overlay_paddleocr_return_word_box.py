from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageOps


ROOT_DIR = Path(__file__).resolve().parents[1]
VENDOR_PADDLEOCR_DIR = ROOT_DIR / "tools" / "receipt_ocr_compare" / "vendor" / "PaddleOCR"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PaddleOCR with return_word_box=True and draw word-box overlay.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", default="outputs/paddleocr_return_word_box")
    parser.add_argument("--paddle_model_dir", default="tools/receipt_ocr_compare/models")
    parser.add_argument("--preview_width", type=int, default=1400)
    parser.add_argument("--max_text_len", type=int, default=18)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def setup_installed_paddle_args(model_dir: Path):
    import paddleocr
    from paddleocr.tools.infer import utility

    args = utility.init_args().parse_args([])
    args.det_model_dir = str(model_dir / "paddleocr" / "det")
    args.rec_model_dir = str(model_dir / "paddleocr" / "rec")
    args.rec_char_dict_path = str(Path(paddleocr.__file__).resolve().parent / "ppocr/utils/ppocr_keys_v1.txt")
    args.use_gpu = False
    args.ir_optim = False
    args.enable_mkldnn = False
    args.show_log = False
    args.benchmark = False
    args.use_onnx = False
    args.det = True
    args.rec = True
    args.cls = False
    args.use_angle_cls = False
    args.rec_batch_num = 8
    return args


def setup_vendor_rec_args(model_dir: Path):
    if str(VENDOR_PADDLEOCR_DIR) not in sys.path:
        sys.path.insert(0, str(VENDOR_PADDLEOCR_DIR))
    from tools.infer import utility

    args = utility.init_args().parse_args([])
    args.rec_model_dir = str(model_dir / "paddleocr" / "rec")
    args.rec_char_dict_path = str(VENDOR_PADDLEOCR_DIR / "ppocr" / "utils" / "ppocr_keys_v1.txt")
    args.use_gpu = False
    args.ir_optim = False
    args.enable_mkldnn = False
    args.show_log = False
    args.benchmark = False
    args.use_onnx = False
    args.rec = True
    args.cls = False
    args.use_angle_cls = False
    args.rec_batch_num = 8
    args.return_word_box = True
    return args


def interpolate(point_a: list[float] | tuple[float, float], point_b: list[float] | tuple[float, float], t: float):
    return (float(point_a[0]) + (float(point_b[0]) - float(point_a[0])) * t, float(point_a[1]) + (float(point_b[1]) - float(point_a[1])) * t)


def fallback_word_quads_from_line_text(text: str, quad: list[list[int]]) -> list[dict[str, Any]]:
    """Approximate word boxes by splitting a recognized line along its text quad.

    PaddleOCR's current local recognizer can accept ``return_word_box=True`` through
    the vendored code path, but the installed local model still returns 2-tuples.
    This fallback keeps the same intended coordinate policy: word boxes lie on the
    detected line quadrilateral, split by whitespace token spans.
    """

    text = str(text or "")
    if not text.strip():
        return []
    text_len = max(len(text), 1)
    p0, p1, p2, p3 = [(float(x), float(y)) for x, y in quad]
    words: list[dict[str, Any]] = []
    for match in re.finditer(r"\S+", text):
        start = max(0, match.start()) / text_len
        end = min(text_len, match.end()) / text_len
        q0 = interpolate(p0, p1, start)
        q1 = interpolate(p0, p1, end)
        q2 = interpolate(p3, p2, end)
        q3 = interpolate(p3, p2, start)
        points = [[int(round(x)), int(round(y))] for x, y in (q0, q1, q2, q3)]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        words.append({"text": match.group(0), "quad": points, "box": [min(xs), min(ys), max(xs), max(ys)]})
    return words


def extract_native_word_items(word_info: Any) -> list[dict[str, Any]]:
    if not isinstance(word_info, dict):
        return []
    candidates = [
        ("word_list", "word_box_list"),
        ("text_word", "text_word_region"),
        ("word", "word_box"),
    ]
    for text_key, box_key in candidates:
        texts = word_info.get(text_key)
        boxes = word_info.get(box_key)
        if not isinstance(texts, list) or not isinstance(boxes, list):
            continue
        out = []
        for text, quad in zip(texts, boxes):
            if not isinstance(quad, list) or len(quad) < 4:
                continue
            points = [[int(round(float(x))), int(round(float(y)))] for x, y in quad[:4]]
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            out.append({"text": str(text), "quad": points, "box": [min(xs), min(ys), max(xs), max(ys)]})
        if out:
            return out
    return []


def run_paddleocr_return_word_box(image_path: Path, model_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    from paddleocr.tools.infer.predict_det import TextDetector
    from paddleocr.tools.infer.predict_system import get_rotate_crop_image, sorted_boxes

    installed_args = setup_installed_paddle_args(model_dir)
    image = cv2.imread(str(image_path))
    if image is None:
        fail(f"OpenCV could not read image: {image_path}")
    raw_boxes, _elapsed = TextDetector(installed_args)(image)
    raw_boxes = sorted_boxes(raw_boxes)

    vendor_args = setup_vendor_rec_args(model_dir)
    from tools.infer.predict_rec import TextRecognizer

    recognizer = TextRecognizer(vendor_args)
    crops = [get_rotate_crop_image(image, poly.astype("float32")) for poly in raw_boxes]
    rec_results, _rec_elapsed = recognizer(crops)

    native_word_box_available = False
    line_items: list[dict[str, Any]] = []
    word_items: list[dict[str, Any]] = []
    for line_idx, (poly, rec_result) in enumerate(zip(raw_boxes, rec_results)):
        quad = [[int(round(float(x))), int(round(float(y)))] for x, y in poly.tolist()]
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        text = str(rec_result[0]) if len(rec_result) > 0 else ""
        confidence = float(rec_result[1]) if len(rec_result) > 1 and rec_result[1] is not None else None
        line_items.append(
            {
                "line_idx": line_idx,
                "text": text,
                "confidence": confidence,
                "quad": quad,
                "box": [min(xs), min(ys), max(xs), max(ys)],
                "raw_result_repr": repr(rec_result)[:1000],
            }
        )
        native_words = extract_native_word_items(rec_result[2] if len(rec_result) > 2 else None)
        if native_words:
            native_word_box_available = True
            source = "native_return_word_box"
            words_for_line = native_words
        else:
            source = "fallback_proportional_return_word_box"
            words_for_line = fallback_word_quads_from_line_text(text, quad)
        for line_word_idx, word in enumerate(words_for_line):
            item = dict(word)
            item.update(
                {
                    "line_idx": line_idx,
                    "line_word_idx": line_word_idx,
                    "word_idx": len(word_items),
                    "confidence": confidence,
                    "source": source,
                }
            )
            word_items.append(item)
    return line_items, word_items, native_word_box_available


def load_font(size: int):
    for path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_overlay(image_path: Path, line_items: list[dict[str, Any]], word_items: list[dict[str, Any]], native_available: bool, out_path: Path, max_text_len: int) -> None:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(18)
    small_font = load_font(14)

    for line in line_items:
        quad = [tuple(point) for point in line["quad"]]
        draw.line(quad + [quad[0]], fill=(230, 126, 34, 210), width=3)

    for word in word_items:
        quad = [tuple(point) for point in word["quad"]]
        draw.line(quad + [quad[0]], fill=(0, 140, 255, 235), width=2)
        x0, y0, _x1, _y1 = word["box"]
        label = f"W{word['word_idx']:03d} {word['text'][:max_text_len]}"
        y = max(0, y0 - 18)
        bbox = draw.textbbox((x0, y), label, font=small_font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255, 215))
        draw.text((x0, y), label, fill=(0, 80, 180, 255), font=small_font)

    legend = f"PaddleOCR return_word_box=True | words={len(word_items)} | native={native_available}"
    bbox = draw.textbbox((16, 16), legend, font=font)
    draw.rectangle([bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4], fill=(255, 255, 255, 230))
    draw.text((16, 16), legend, fill=(0, 80, 180, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def make_preview(path: Path, max_width: int) -> Path:
    image = Image.open(path).convert("RGB")
    if max_width > 0 and image.width > max_width:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((max_width, int(image.height * max_width / image.width)), resampling)
    preview = path.with_name(f"{path.stem}_mobile_preview{path.suffix}")
    image.save(preview)
    return preview


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    model_dir = Path(args.paddle_model_dir)
    if not image_path.exists():
        fail(f"Image not found: {image_path}")
    if not model_dir.exists():
        fail(f"Paddle model dir not found: {model_dir}")
    if not VENDOR_PADDLEOCR_DIR.exists():
        fail(f"Vendor PaddleOCR dir not found: {VENDOR_PADDLEOCR_DIR}")

    stem = image_path.stem
    out_dir = Path(args.out_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    line_items, word_items, native_available = run_paddleocr_return_word_box(image_path, model_dir)
    overlay_path = out_dir / f"{stem}_paddleocr_return_word_box_overlay.png"
    json_path = out_dir / f"{stem}_paddleocr_return_word_box_words.json"
    draw_overlay(image_path, line_items, word_items, native_available, overlay_path, args.max_text_len)
    preview_path = make_preview(overlay_path, args.preview_width)
    json_path.write_text(
        json.dumps(
            {
                "image": str(image_path),
                "return_word_box_requested": True,
                "return_word_box_native_available": native_available,
                "note": (
                    "If native word boxes are unavailable, boxes are fallback_proportional_return_word_box: "
                    "whitespace tokens split along each detected line quadrilateral."
                ),
                "line_count": len(line_items),
                "word_count": len(word_items),
                "lines": line_items,
                "words": word_items,
                "overlay": str(overlay_path),
                "preview": str(preview_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"line_count: {len(line_items)}")
    print(f"word_count: {len(word_items)}")
    print(f"native_word_box_available: {native_available}")
    print(f"overlay: {overlay_path}")
    print(f"preview: {preview_path}")
    print(f"json: {json_path}")
    print("PaddleOCR return_word_box overlay passed.")


if __name__ == "__main__":
    main()
