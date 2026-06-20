from __future__ import annotations

from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont

from .metrics import mismatch_positions
from .normalization import is_numeric_token
from .schemas import DetectionBox, GroundTruthToken, RecognitionResult


MODEL_COLORS = {
    "svtrv2_b": (0, 114, 178),
    "paddleocr": (213, 94, 0),
    "existing": (0, 158, 115),
}
OK_COLOR = (0, 150, 80)
BAD_COLOR = (210, 40, 40)
NUMERIC_FILL = (255, 245, 180)


def render_model_overlay(
    image_path: Path,
    boxes: list[DetectionBox],
    results: Iterable[RecognitionResult],
    *,
    output_path: Path,
    ground_truth: list[GroundTruthToken] | None = None,
    numeric_only: bool = False,
    mismatches_only: bool = False,
    show_confidence: bool = True,
    show_boxes: bool = True,
) -> Path:
    image = Image.open(image_path).convert("RGB")
    overlay = image.copy()
    draw = ImageDraw.Draw(overlay, "RGBA")
    font = _font()
    result_by_crop = {result.crop_id: result for result in results}
    gt_by_id = {token.token_id: token for token in ground_truth or []}
    model_id = next(iter(result_by_crop.values())).model_id if result_by_crop else "unknown"
    color = MODEL_COLORS.get(model_id, (80, 80, 80))

    for box in boxes:
        result = result_by_crop.get(box.crop_id)
        gt = gt_by_id.get(box.crop_id)
        raw_text = result.raw_text if result else ""
        token_is_numeric = (gt.token_type == "number" if gt else False) or is_numeric_token(raw_text)
        correct = gt is not None and result is not None and gt.text == result.raw_text
        mismatch = gt is not None and result is not None and gt.text != result.raw_text
        if numeric_only and not token_is_numeric:
            continue
        if mismatches_only and not mismatch:
            continue

        x1, y1, x2, y2 = box.bbox
        line_color = OK_COLOR if correct else BAD_COLOR if mismatch else color
        if show_boxes:
            draw.rectangle([x1, y1, x2, y2], outline=(*line_color, 230), width=2)
            if token_is_numeric:
                draw.rectangle([x1, y1, x2, y2], fill=(*NUMERIC_FILL, 45))
        label = _label(box.crop_id, result, token_is_numeric, show_confidence)
        if mismatch and gt:
            positions = mismatch_positions(gt.text, raw_text)
            compact = ",".join(f"{item['index']}:{item['ground_truth']}->{item['prediction']}" for item in positions[:4])
            label = f"{label} ! {compact}"
        _draw_label(draw, (x1, max(0, y1 - 16)), label, font, line_color)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay.save(output_path)
    return output_path


def render_comparison_overlay(image_paths: list[Path], labels: list[str], output_path: Path) -> Path:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        raise ValueError("no overlay images supplied")
    font = _font()
    label_height = 24
    width = sum(image.width for image in images)
    height = max(image.height for image in images) + label_height
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    x = 0
    for label, image in zip(labels, images):
        draw.text((x + 6, 4), label, fill=(0, 0, 0), font=font)
        canvas.paste(image, (x, label_height))
        x += image.width
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)
    return output_path


def _label(crop_id: str, result: RecognitionResult | None, numeric: bool, show_confidence: bool) -> str:
    if result is None:
        return f"{crop_id} | no result"
    text = result.raw_text if result.raw_text else "<empty>"
    parts = [crop_id, text]
    if numeric:
        parts.append("#")
    if show_confidence and result.confidence is not None:
        parts.append(f"{result.confidence:.2f}")
    if result.error:
        parts.append("unavailable")
    return " | ".join(parts)


def _draw_label(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont, color: tuple[int, int, int]) -> None:
    x, y = xy
    bbox = draw.textbbox((x, y), text, font=font)
    draw.rectangle([bbox[0] - 2, bbox[1] - 1, bbox[2] + 2, bbox[3] + 1], fill=(255, 255, 255, 220))
    draw.text((x, y), text, fill=color, font=font)


def _font() -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("arial.ttf", 12)
    except Exception:
        return ImageFont.load_default()

