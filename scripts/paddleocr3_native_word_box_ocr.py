from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run PaddleOCR 3.x TextRecognition with return_word_box=True on "
            "pre-detected line crops and map native word boxes back to the receipt image."
        )
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--line_ocr_json",
        required=True,
        help="Line-level OCR JSON containing words[].box/quad from a detector pass.",
    )
    parser.add_argument("--out_dir", default="outputs/paddleocr3_native_word_box")
    parser.add_argument("--model_name", default="PP-OCRv5_mobile_rec")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--preview_width", type=int, default=1400)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_font(size: int):
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ):
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def interpolate(a, b, t: float) -> tuple[float, float]:
    return (float(a[0]) + (float(b[0]) - float(a[0])) * t, float(a[1]) + (float(b[1]) - float(a[1])) * t)


def quad_to_box(points: list[list[int]]) -> list[int]:
    xs = [int(round(float(p[0]))) for p in points]
    ys = [int(round(float(p[1]))) for p in points]
    return [min(xs), min(ys), max(xs), max(ys)]


def line_quad(item: dict[str, Any]) -> list[list[int]] | None:
    quad = item.get("quad")
    if isinstance(quad, list) and len(quad) >= 4:
        return [[int(round(float(p[0]))), int(round(float(p[1])))] for p in quad[:4]]
    box = item.get("box")
    if isinstance(box, list) and len(box) == 4:
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
        return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]
    return None


def crop_box(item: dict[str, Any], width: int, height: int) -> list[int] | None:
    box = item.get("box")
    if not isinstance(box, list) or len(box) != 4:
        quad = line_quad(item)
        if quad is None:
            return None
        box = quad_to_box(quad)
    x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def parse_rec_text(rec_text: Any) -> tuple[str, list[list[str]], list[list[int]], list[str]]:
    if isinstance(rec_text, tuple) and len(rec_text) >= 2 and isinstance(rec_text[1], list) and len(rec_text[1]) >= 4:
        text = str(rec_text[0] or "")
        _scaled_width, word_chars, word_cols, states = rec_text[1][:4]
        return text, word_chars or [], word_cols or [], states or []
    return str(rec_text or ""), [], [], []


def word_quad_from_cols(cols: list[int], quad: list[list[int]], sequence_width: float) -> list[list[int]]:
    if not cols:
        return []
    start = max(0.0, min(cols) / max(sequence_width, 1.0))
    end = min(1.0, (max(cols) + 1) / max(sequence_width, 1.0))
    p0, p1, p2, p3 = quad
    q0 = interpolate(p0, p1, start)
    q1 = interpolate(p0, p1, end)
    q2 = interpolate(p3, p2, end)
    q3 = interpolate(p3, p2, start)
    return [[int(round(x)), int(round(y))] for x, y in (q0, q1, q2, q3)]


def make_preview(path: Path, max_width: int) -> Path:
    image = Image.open(path).convert("RGB")
    if max_width > 0 and image.width > max_width:
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((max_width, int(image.height * max_width / image.width)), resampling)
    preview = path.with_name(f"{path.stem}_mobile_preview{path.suffix}")
    image.save(preview)
    return preview


def draw_overlay(image_path: Path, line_items: list[dict[str, Any]], word_items: list[dict[str, Any]], out_path: Path) -> None:
    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    draw = ImageDraw.Draw(image, "RGBA")
    font = load_font(15)
    title_font = load_font(20)
    for item in line_items:
        quad = item.get("quad")
        if quad:
            pts = [tuple(p) for p in quad]
            draw.line(pts + [pts[0]], fill=(230, 126, 34, 170), width=2)
    for word in word_items:
        quad = word.get("quad")
        if not quad:
            continue
        pts = [tuple(p) for p in quad]
        draw.line(pts + [pts[0]], fill=(0, 140, 255, 235), width=2)
        x0, y0, _x1, _y1 = word["box"]
        label = f"W{word['word_idx']:03d} {word['text'][:18]}"
        y = max(0, y0 - 18)
        bbox = draw.textbbox((x0, y), label, font=font)
        draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255, 220))
        draw.text((x0, y), label, fill=(0, 80, 180, 255), font=font)
    title = f"PaddleOCR 3 native return_word_box=True | words={len(word_items)}"
    bbox = draw.textbbox((16, 16), title, font=title_font)
    draw.rectangle([bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4], fill=(255, 255, 255, 230))
    draw.text((16, 16), title, fill=(0, 80, 180, 255), font=title_font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    line_json_path = Path(args.line_ocr_json)
    if not image_path.exists():
        fail(f"image not found: {image_path}")
    if not line_json_path.exists():
        fail(f"line_ocr_json not found: {line_json_path}")

    from paddleocr import TextRecognition

    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    payload = json.loads(line_json_path.read_text(encoding="utf-8"))
    line_source_items = payload.get("words") or payload.get("lines") or []
    if not line_source_items:
        fail("line_ocr_json has no words/lines items.")

    out_dir = Path(args.out_dir) / image_path.stem
    crop_dir = out_dir / "line_crops"
    crop_dir.mkdir(parents=True, exist_ok=True)

    line_items: list[dict[str, Any]] = []
    crop_paths: list[str] = []
    for line_idx, source in enumerate(line_source_items):
        quad = line_quad(source)
        box = crop_box(source, width, height)
        if quad is None or box is None:
            continue
        x0, y0, x1, y1 = box
        crop_path = crop_dir / f"line_{line_idx:03d}.png"
        image.crop((x0, y0, x1, y1)).save(crop_path)
        line_items.append(
            {
                "line_idx": line_idx,
                "source_text": source.get("text"),
                "source_confidence": source.get("confidence"),
                "box": box,
                "quad": quad,
                "crop_path": str(crop_path),
            }
        )
        crop_paths.append(str(crop_path))

    if not crop_paths:
        fail("No valid line crops.")

    recognizer = TextRecognition(model_name=args.model_name, device=args.device)
    results = recognizer.predict(input=crop_paths, batch_size=args.batch_size, return_word_box=True)

    word_items: list[dict[str, Any]] = []
    for line, result in zip(line_items, results):
        rec_text, word_chars, word_cols, states = parse_rec_text(result.get("rec_text"))
        line["native_rec_text"] = rec_text
        line["native_rec_score"] = float(result.get("rec_score") or 0.0)
        line["native_word_count"] = len(word_chars)
        sequence_width = 1.0
        if isinstance(result.get("rec_text"), tuple) and len(result["rec_text"]) > 1:
            sequence_width = float(result["rec_text"][1][0] or 1.0)
        for line_word_idx, (chars, cols) in enumerate(zip(word_chars, word_cols)):
            text = "".join(str(c) for c in chars).strip()
            if not text:
                continue
            quad = word_quad_from_cols([int(c) for c in cols], line["quad"], sequence_width)
            if not quad:
                continue
            item = {
                "word_idx": len(word_items),
                "line_idx": line["line_idx"],
                "line_word_idx": line_word_idx,
                "text": text,
                "box": quad_to_box(quad),
                "quad": quad,
                "confidence": line["native_rec_score"],
                "source": "paddleocr3_native_return_word_box",
                "state": states[line_word_idx] if line_word_idx < len(states) else None,
            }
            word_items.append(item)

    ocr_json_path = out_dir / f"{image_path.stem}_paddleocr3_native_word_box_ocr.json"
    overlay_path = out_dir / f"{image_path.stem}_paddleocr3_native_word_box_overlay.png"
    draw_overlay(image_path, line_items, word_items, overlay_path)
    preview_path = make_preview(overlay_path, args.preview_width)
    save_json(
        ocr_json_path,
        {
            "image": str(image_path),
            "image_width": width,
            "image_height": height,
            "ocr_engine": "paddleocr3_text_recognition",
            "recognition_model_name": args.model_name,
            "line_ocr_json": str(line_json_path),
            "coordinate_space": "image_pixels",
            "return_word_box_requested": True,
            "return_word_box_native_available": True,
            "raw_box_count": len(line_items),
            "line_count": len(line_items),
            "word_count": len(word_items),
            "lines": line_items,
            "words": word_items,
            "overlay": str(overlay_path),
            "preview": str(preview_path),
        },
    )

    print(f"line_count: {len(line_items)}")
    print(f"word_count: {len(word_items)}")
    print("return_word_box_requested: True")
    print("return_word_box_native_available: True")
    print(f"ocr_json: {ocr_json_path}")
    print(f"overlay: {overlay_path}")
    print(f"preview: {preview_path}")
    print("PaddleOCR 3 native word-box OCR passed.")


if __name__ == "__main__":
    main()
