from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    existing_json_path = Path(args.existing_ocr_json)
    out_dir = Path(args.out_dir)

    if not image_path.exists():
        raise SystemExit(f"Image not found: {image_path}")
    if not existing_json_path.exists():
        raise SystemExit(f"Existing OCR JSON not found: {existing_json_path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = image_path.stem
    paddle_overlay = out_dir / f"{stem}_paddleocr_bbox_overlay.png"
    existing_overlay = out_dir / f"{stem}_existing_ocr_bbox_overlay.png"
    summary_path = out_dir / f"{stem}_paddle_vs_existing_summary.json"

    image = Image.open(image_path).convert("RGB")
    existing_words = load_existing_words(existing_json_path)
    paddle_boxes = run_paddle_ocr(
        image_path=image_path,
        model_dir=Path(args.paddle_model_dir),
        recognize=not args.det_only,
    )

    render_existing_overlay(
        image=image,
        words=existing_words,
        output_path=existing_overlay,
        max_text_len=args.max_text_len,
    )
    render_paddle_overlay(
        image=image,
        boxes=paddle_boxes,
        output_path=paddle_overlay,
    )

    summary = {
        "image": str(image_path),
        "existing_ocr_json": str(existing_json_path),
        "image_width": image.width,
        "image_height": image.height,
        "existing_word_count": len(existing_words),
        "paddleocr_box_count": len(paddle_boxes),
        "paddle_overlay": str(paddle_overlay),
        "existing_overlay": str(existing_overlay),
        "note": "PaddleOCR overlay uses native PaddlePaddle predictor. Text is included unless --det_only is used.",
        "existing_words_preview": existing_words[:30],
        "paddle_boxes_preview": paddle_boxes[:30],
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"existing_word_count: {len(existing_words)}")
    print(f"paddleocr_box_count: {len(paddle_boxes)}")
    print(f"paddle_overlay: {paddle_overlay}")
    print(f"existing_overlay: {existing_overlay}")
    print(f"summary: {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Draw separate PaddleOCR and existing OCR bbox overlays for one receipt image.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--existing_ocr_json", required=True)
    parser.add_argument("--out_dir", default="outputs/paddle_vs_existing_ocr")
    parser.add_argument("--paddle_model_dir", default="tools/receipt_ocr_compare/models")
    parser.add_argument("--max_text_len", type=int, default=28)
    parser.add_argument("--det_only", action="store_true")
    return parser.parse_args()


def load_existing_words(path: Path) -> list[dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    raw_words = obj.get("words") or []
    words: list[dict[str, Any]] = []
    for idx, word in enumerate(raw_words):
        text = str(word.get("text", ""))
        box = word.get("box")
        if not text.strip() or not valid_box(box):
            continue
        words.append(
            {
                "index": int(word.get("globalWordIndex", idx)),
                "word_id": word.get("wordId"),
                "line_id": word.get("lineId"),
                "text": text,
                "box": [int(round(v)) for v in box],
            }
        )
    return words


def run_paddle_ocr(*, image_path: Path, model_dir: Path, recognize: bool) -> list[dict[str, Any]]:
    try:
        import cv2
        import paddleocr
        from paddleocr.tools.infer import utility
        from paddleocr.tools.infer.predict_det import TextDetector
        from paddleocr.tools.infer.predict_rec import TextRecognizer
        from paddleocr.tools.infer.predict_system import get_rotate_crop_image
        from paddleocr.tools.infer.predict_system import sorted_boxes
    except Exception as exc:
        raise SystemExit(f"PaddleOCR detector dependencies are not importable: {exc}") from exc

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
    args.rec = recognize
    args.cls = False
    args.use_angle_cls = False
    args.rec_batch_num = 8

    image = cv2.imread(str(image_path))
    if image is None:
        raise SystemExit(f"OpenCV could not read image: {image_path}")

    detector = TextDetector(args)
    raw_boxes, _elapsed = detector(image)
    raw_boxes = sorted_boxes(raw_boxes)
    rec_results: list[tuple[str, float]] = []
    if recognize and len(raw_boxes) > 0:
        recognizer = TextRecognizer(args)
        crops = [get_rotate_crop_image(image, poly.astype("float32")) for poly in raw_boxes]
        rec_results, _rec_elapsed = recognizer(crops)
    boxes: list[dict[str, Any]] = []
    for idx, poly in enumerate(raw_boxes):
        points = [[int(round(float(x))), int(round(float(y)))] for x, y in poly.tolist()]
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        text, confidence = rec_results[idx] if idx < len(rec_results) else (None, None)
        boxes.append(
            {
                "index": idx,
                "text": text,
                "confidence": float(confidence) if confidence is not None else None,
                "box": [min(xs), min(ys), max(xs), max(ys)],
                "quad": points,
            }
        )
    return boxes


def render_existing_overlay(*, image: Image.Image, words: list[dict[str, Any]], output_path: Path, max_text_len: int) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = font_default()
    color = (0, 158, 115)
    draw_legend(draw, "Existing app OCR / ML Kit words", color, font)
    for word in words:
        x0, y0, x1, y1 = word["box"]
        draw.rectangle([x0, y0, x1, y1], outline=(*color, 235), width=2)
        label = f"E{word['index']:03d} {word['text'][:max_text_len]}"
        draw_label(draw, x0, max(0, y0 - 15), label, color, font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def render_paddle_overlay(*, image: Image.Image, boxes: list[dict[str, Any]], output_path: Path) -> None:
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = font_default()
    color = (213, 94, 0)
    draw_legend(draw, "PaddleOCR detection boxes", color, font)
    for box in boxes:
        x0, y0, x1, y1 = box["box"]
        quad = [tuple(point) for point in box["quad"]]
        draw.line(quad + [quad[0]], fill=(*color, 245), width=3)
        draw.rectangle([x0, y0, x1, y1], outline=(*color, 95), width=1)
        text = box.get("text") or ""
        confidence = box.get("confidence")
        score = f" {confidence:.2f}" if isinstance(confidence, float) else ""
        label = f"P{box['index']:03d} {str(text)[:24]}{score}".strip()
        draw_label(draw, x0, max(0, y0 - 15), label, color, font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)


def valid_box(box: Any) -> bool:
    if not isinstance(box, list) or len(box) != 4:
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in box]
    except Exception:
        return False
    return x1 > x0 and y1 > y0


def draw_legend(draw: ImageDraw.ImageDraw, text: str, color: tuple[int, int, int], font: ImageFont.ImageFont) -> None:
    bbox = draw.textbbox((12, 12), text, font=font)
    draw.rectangle([bbox[0] - 6, bbox[1] - 4, bbox[2] + 6, bbox[3] + 4], fill=(255, 255, 255, 230))
    draw.text((12, 12), text, fill=color, font=font)


def draw_label(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    color: tuple[int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255, 220))
    draw.text((x, y), text, fill=color, font=font)


def font_default() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 13)
    except Exception:
        return ImageFont.load_default()


if __name__ == "__main__":
    main()
