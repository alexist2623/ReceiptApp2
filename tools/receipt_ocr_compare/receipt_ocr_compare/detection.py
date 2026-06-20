from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from .schemas import DetectionBox, GroundTruthToken


class DetectorUnavailable(RuntimeError):
    pass


def load_ground_truth_jsonl(path: Path | None) -> dict[str, list[GroundTruthToken]]:
    if path is None or not path.exists():
        return {}
    by_image: dict[str, list[GroundTruthToken]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            image_name = str(payload["image"])
            tokens = []
            for token in payload.get("tokens", []):
                tokens.append(
                    GroundTruthToken(
                        image=image_name,
                        token_id=str(token.get("id") or token.get("token_id")),
                        bbox=[int(v) for v in token["bbox"]],
                        text=str(token.get("text", "")),
                        token_type=str(token.get("type", token.get("token_type", "unknown"))),
                    )
                )
            by_image[image_name] = tokens
    return by_image


def gt_tokens_for_image(gt_by_image: dict[str, list[GroundTruthToken]], image_path: Path) -> list[GroundTruthToken]:
    return gt_by_image.get(image_path.name) or gt_by_image.get(str(image_path)) or []


def detect_boxes(
    image_path: Path,
    *,
    detector: str,
    model_dir: Path,
    vendor_dir: Path,
    ground_truth: list[GroundTruthToken] | None = None,
) -> tuple[list[DetectionBox], dict[str, Any]]:
    selected = detector.lower()
    if selected == "ground_truth":
        if not ground_truth:
            raise DetectorUnavailable("ground_truth detector selected but no ground truth tokens were supplied")
        return _boxes_from_ground_truth(image_path, ground_truth), {"detector": "ground_truth", "available": True}
    if selected == "simple":
        return simple_detect_boxes(image_path), {"detector": "simple", "available": True}
    if selected == "existing":
        raise DetectorUnavailable("existing project detector is Android/ML Kit based and has no Python runner")
    if selected == "paddleocr":
        return paddle_detect_boxes(image_path, model_dir=model_dir)
    if selected == "auto":
        if ground_truth:
            return _boxes_from_ground_truth(image_path, ground_truth), {
                "detector": "ground_truth",
                "available": True,
                "reason": "ground truth was provided",
            }
        try:
            return paddle_detect_boxes(image_path, model_dir=model_dir)
        except DetectorUnavailable as exc:
            boxes = simple_detect_boxes(image_path)
            return boxes, {"detector": "simple", "available": True, "fallback_reason": str(exc)}
    raise DetectorUnavailable(f"unknown detector: {detector}")


def simple_detect_boxes(image_path: Path, *, threshold: int = 220, min_dark_pixels: int = 3) -> list[DetectionBox]:
    image = Image.open(image_path).convert("L")
    width, height = image.size
    pixels = image.load()

    dark_by_row = []
    for y in range(height):
        count = sum(1 for x in range(width) if pixels[x, y] < threshold)
        dark_by_row.append(count)

    row_spans = _group_indices([idx for idx, count in enumerate(dark_by_row) if count >= min_dark_pixels], gap=2)
    boxes: list[DetectionBox] = []
    token_index = 1
    for y1, y2 in row_spans:
        dark_by_col = []
        for x in range(width):
            count = sum(1 for y in range(y1, y2 + 1) if pixels[x, y] < threshold)
            dark_by_col.append(count)
        min_col_pixels = max(1, (y2 - y1 + 1) // 8)
        col_spans = _group_indices([idx for idx, count in enumerate(dark_by_col) if count >= min_col_pixels], gap=3)
        for x1, x2 in col_spans:
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            boxes.append(
                DetectionBox(
                    image=image_path.name,
                    crop_id=f"token_{token_index:04d}",
                    bbox=[max(0, x1 - 1), max(0, y1 - 1), min(width, x2 + 2), min(height, y2 + 2)],
                    detector="simple",
                    confidence=None,
                )
            )
            token_index += 1
    return boxes


def paddle_detect_boxes(image_path: Path, *, model_dir: Path) -> tuple[list[DetectionBox], dict[str, Any]]:
    local_det_dir = model_dir / "paddleocr" / "det"
    if not local_det_dir.exists() or not any(local_det_dir.iterdir()):
        raise DetectorUnavailable(f"PaddleOCR detector model directory is missing or empty: {local_det_dir}")
    try:
        from paddleocr import PaddleOCR  # type: ignore
    except Exception as exc:
        raise DetectorUnavailable(f"paddleocr package is not importable: {exc}") from exc
    try:
        engine = PaddleOCR(det_model_dir=str(local_det_dir), rec=False, use_angle_cls=False, show_log=False)
        raw = engine.ocr(str(image_path), det=True, rec=False, cls=False)
    except Exception as exc:
        raise DetectorUnavailable(f"PaddleOCR detector failed: {exc}") from exc

    boxes: list[DetectionBox] = []
    for idx, poly in enumerate(_iter_paddle_polys(raw), 1):
        xs = [int(point[0]) for point in poly]
        ys = [int(point[1]) for point in poly]
        boxes.append(
            DetectionBox(
                image=image_path.name,
                crop_id=f"token_{idx:04d}",
                bbox=[min(xs), min(ys), max(xs), max(ys)],
                detector="paddleocr",
                confidence=None,
            )
        )
    return boxes, {"detector": "paddleocr", "available": True, "model_dir": str(local_det_dir)}


def _boxes_from_ground_truth(image_path: Path, tokens: list[GroundTruthToken]) -> list[DetectionBox]:
    return [
        DetectionBox(
            image=image_path.name,
            crop_id=token.token_id,
            bbox=token.bbox,
            detector="ground_truth",
            confidence=1.0,
        )
        for token in tokens
    ]


def _group_indices(indices: list[int], *, gap: int) -> list[tuple[int, int]]:
    if not indices:
        return []
    spans: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx - prev <= gap + 1:
            prev = idx
            continue
        spans.append((start, prev))
        start = prev = idx
    spans.append((start, prev))
    return spans


def _iter_paddle_polys(raw: Any) -> list[list[list[float]]]:
    if not raw:
        return []
    if isinstance(raw, dict):
        polys = raw.get("dt_polys") or raw.get("boxes") or raw.get("polys") or []
        return [poly.tolist() if hasattr(poly, "tolist") else poly for poly in polys]
    if isinstance(raw, list):
        first = raw[0] if raw else []
        if isinstance(first, dict):
            return _iter_paddle_polys(first.get("res", first))
        if first and isinstance(first[0], (list, tuple)) and len(first[0]) == 4 and isinstance(first[0][0], (list, tuple)):
            return [item[0] if len(item) > 0 else item for item in first]
        if first and isinstance(first[0], (list, tuple)) and len(first[0]) == 2:
            return first
    return []

