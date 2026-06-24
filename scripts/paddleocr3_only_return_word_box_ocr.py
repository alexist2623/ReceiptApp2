from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps
from paddleocr import PaddleOCR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR 3 directly with return_word_box=True and export only PaddleOCR-produced word boxes."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--out_dir", default="outputs/paddleocr3_only_return_word_box")
    parser.add_argument("--text_detection_model_name", default="PP-OCRv5_mobile_det")
    parser.add_argument("--text_recognition_model_name", default="PP-OCRv5_mobile_rec")
    parser.add_argument("--lang", default="en")
    parser.add_argument("--max_text_len", type=int, default=24)
    parser.add_argument("--mobile_preview_width", type=int, default=1400)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def as_int_box(box: object) -> list[int] | None:
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def clamp_box(box: list[int], width: int, height: int) -> list[int] | None:
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def draw_box(draw: ImageDraw.ImageDraw, box: list[int], color: tuple[int, int, int], width: int = 2) -> None:
    try:
        draw.rectangle(box, outline=color, width=width)
    except TypeError:
        for i in range(width):
            draw.rectangle([box[0] - i, box[1] - i, box[2] + i, box[3] + i], outline=color)


def draw_overlay(image: Image.Image, words: list[dict], out_path: Path, max_text_len: int) -> None:
    canvas = image.copy().convert("RGB")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for item in words:
        box = item["box"]
        draw_box(draw, box, (40, 190, 95), width=2)
        label = str(item["text"])[:max_text_len]
        text_pos = (box[0], max(0, box[1] - 12))
        draw.rectangle(
            [text_pos[0], text_pos[1], text_pos[0] + max(20, 6 * len(label)), text_pos[1] + 11],
            fill=(255, 255, 255),
        )
        draw.text(text_pos, label, fill=(20, 90, 45), font=font)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def make_mobile_preview(path: Path, max_width: int) -> Path:
    image = Image.open(path).convert("RGB")
    if max_width > 0 and image.width > max_width:
        new_height = int(image.height * max_width / image.width)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((max_width, new_height), resampling)
    preview = path.with_name(f"{path.stem}_mobile_preview{path.suffix}")
    image.save(preview)
    return preview


def flatten_paddle_word_boxes(res: dict, width: int, height: int) -> tuple[list[dict], list[dict]]:
    text_words = res.get("text_word") or []
    text_word_boxes = res.get("text_word_boxes") or []
    rec_texts = res.get("rec_texts") or []
    rec_scores = res.get("rec_scores") or []
    skipped: list[dict] = []
    words: list[dict] = []

    for line_idx, (line_words, line_boxes) in enumerate(zip(text_words, text_word_boxes)):
        if not isinstance(line_words, list) or not isinstance(line_boxes, list):
            skipped.append({"line_idx": line_idx, "reason": "line words/boxes are not lists"})
            continue
        for line_word_idx, (text, raw_box) in enumerate(zip(line_words, line_boxes)):
            text = str(text or "").strip()
            box = as_int_box(raw_box)
            if not text:
                skipped.append({"line_idx": line_idx, "line_word_idx": line_word_idx, "reason": "empty text"})
                continue
            if box is None:
                skipped.append(
                    {
                        "line_idx": line_idx,
                        "line_word_idx": line_word_idx,
                        "text": text,
                        "reason": "invalid PaddleOCR text_word_box",
                        "raw_box": raw_box,
                    }
                )
                continue
            box = clamp_box(box, width, height)
            if box is None:
                skipped.append(
                    {
                        "line_idx": line_idx,
                        "line_word_idx": line_word_idx,
                        "text": text,
                        "reason": "zero-area box after clamp",
                        "raw_box": raw_box,
                    }
                )
                continue
            words.append(
                {
                    "word_idx": len(words),
                    "line_idx": line_idx,
                    "line_word_idx": line_word_idx,
                    "text": text,
                    "box": box,
                    "confidence": rec_scores[line_idx] if line_idx < len(rec_scores) else None,
                    "line_text": rec_texts[line_idx] if line_idx < len(rec_texts) else None,
                    "source": "paddleocr3_return_word_box",
                }
            )
    return words, skipped


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    if not image_path.exists():
        fail(f"image not found: {image_path}")

    image = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
    width, height = image.size
    stem = image_path.stem
    out_dir = Path(args.out_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"image: {image_path}")
    print(f"image_size: {width}x{height}")
    print("paddleocr_mode: direct PaddleOCR.predict(image, return_word_box=True)")
    print(f"text_detection_model_name: {args.text_detection_model_name}")
    print(f"text_recognition_model_name: {args.text_recognition_model_name}")

    ocr = PaddleOCR(
        lang=args.lang,
        text_detection_model_name=args.text_detection_model_name,
        text_recognition_model_name=args.text_recognition_model_name,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
    )
    results = list(ocr.predict(str(image_path), return_word_box=True))
    if not results:
        fail("PaddleOCR returned no results.")
    result = results[0]
    raw_json = result.json
    res = raw_json.get("res") if isinstance(raw_json, dict) else None
    if not isinstance(res, dict):
        fail("PaddleOCR result JSON does not contain res dict.")
    if res.get("return_word_box") is not True:
        fail("PaddleOCR result did not confirm return_word_box=True.")

    words, skipped = flatten_paddle_word_boxes(res, width, height)
    if not words:
        fail("PaddleOCR returned no valid text_word_boxes.")

    raw_json_path = out_dir / f"{stem}_paddleocr3_raw_result.json"
    ocr_json_path = out_dir / f"{stem}_paddleocr3_only_return_word_box_ocr.json"
    overlay_path = out_dir / f"{stem}_paddleocr3_only_return_word_box_overlay.png"
    debug_path = out_dir / f"{stem}_paddleocr3_only_return_word_box_debug.json"

    save_json(raw_json_path, raw_json)
    save_json(
        ocr_json_path,
        {
            "image_width": width,
            "image_height": height,
            "ocr_engine": "paddleocr3",
            "coordinate_space": "image_pixels",
            "return_word_box_requested": True,
            "return_word_box_native_available": True,
            "word_box_source": "PaddleOCR.result.res.text_word_boxes",
            "line_count": len(res.get("rec_texts") or []),
            "word_count": len(words),
            "words": words,
        },
    )
    save_json(
        debug_path,
        {
            "raw_json_path": str(raw_json_path),
            "ocr_json_path": str(ocr_json_path),
            "overlay_path": str(overlay_path),
            "paddle_result_keys": list(res.keys()),
            "line_count": len(res.get("rec_texts") or []),
            "word_count": len(words),
            "skipped_count": len(skipped),
            "skipped": skipped[:200],
            "first_30_words": words[:30],
        },
    )
    draw_overlay(image, words, overlay_path, args.max_text_len)
    preview_path = make_mobile_preview(overlay_path, args.mobile_preview_width)

    print(f"raw_json: {raw_json_path}")
    print(f"ocr_json: {ocr_json_path}")
    print(f"debug_json: {debug_path}")
    print(f"overlay: {overlay_path}")
    print(f"preview: {preview_path}")
    print(f"line_count: {len(res.get('rec_texts') or [])}")
    print(f"word_count: {len(words)}")
    print(f"skipped_count: {len(skipped)}")
    print(f"first_20_words: {[item['text'] for item in words[:20]]}")
    print("PaddleOCR-only return_word_box export passed.")


if __name__ == "__main__":
    main()
