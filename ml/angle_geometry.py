"""Angle and quadrilateral helpers for receipt OCR geometry.

The existing LayoutLMv3 pipeline is axis-aligned: it feeds ``bbox`` in the
0..1000 coordinate space. This module keeps that behavior intact while exposing
optional per-word angle features derived from OCR corner points. If a word has
no quadrilateral/angle information, callers receive an all-zero feature vector.
"""

from __future__ import annotations

import math
from typing import Any, Iterable

import torch


ANGLE_FEATURE_DIM = 9


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _point_from_obj(obj: Any) -> tuple[float, float] | None:
    if isinstance(obj, dict):
        x = _as_float(obj.get("x", obj.get("X")))
        y = _as_float(obj.get("y", obj.get("Y")))
        if x is None or y is None:
            return None
        return x, y
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        x = _as_float(obj[0])
        y = _as_float(obj[1])
        if x is None or y is None:
            return None
        return x, y
    return None


def parse_quad(value: Any) -> list[tuple[float, float]] | None:
    """Parse common OCR quadrilateral forms into four points.

    Supported inputs include flat ``[x1, y1, ..., x4, y4]`` lists, point-list
    forms, dictionaries with ``x1/y1..x4/y4``, and wrappers such as
    ``{"vertices": [...]}`` or ``{"cornerPoints": [...]}``.
    """

    if value is None:
        return None
    if isinstance(value, dict):
        for key in ("quad", "cornerPoints", "corner_points", "vertices", "points", "polygon"):
            if key in value:
                parsed = parse_quad(value.get(key))
                if parsed is not None:
                    return parsed
        if all(key in value for key in ("x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4")):
            points = []
            for idx in range(1, 5):
                point = _point_from_obj((value.get(f"x{idx}"), value.get(f"y{idx}")))
                if point is None:
                    return None
                points.append(point)
            return points
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 8 and not isinstance(value[0], (list, tuple, dict)):
            floats = [_as_float(item) for item in value]
            if any(item is None for item in floats):
                return None
            return [(floats[i], floats[i + 1]) for i in range(0, 8, 2)]  # type: ignore[index]
        points = [_point_from_obj(item) for item in value]
        points = [point for point in points if point is not None]
        if len(points) >= 4:
            return points[:4]
    return None


def quad_to_axis_aligned_box(value: Any) -> list[int] | None:
    """Convert a quad or rectangular box-like object into ``[x0, y0, x1, y1]``."""

    quad = parse_quad(value)
    if quad:
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]
    if isinstance(value, dict):
        if all(key in value for key in ("x0", "y0", "x1", "y1")):
            coords = [_as_float(value[key]) for key in ("x0", "y0", "x1", "y1")]
        elif all(key in value for key in ("left", "top", "right", "bottom")):
            coords = [_as_float(value[key]) for key in ("left", "top", "right", "bottom")]
        elif all(key in value for key in ("x", "y", "width", "height")):
            x, y, w, h = [_as_float(value[key]) for key in ("x", "y", "width", "height")]
            coords = None if None in (x, y, w, h) else [x, y, x + w, y + h]  # type: ignore[operator]
        else:
            coords = None
        if coords and not any(item is None for item in coords):
            x0, y0, x1, y1 = [int(round(float(item))) for item in coords]
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            return [x0, y0, x1, y1]
    if isinstance(value, (list, tuple)) and len(value) == 4:
        coords = [_as_float(item) for item in value]
        if not any(item is None for item in coords):
            x0, y0, x1, y1 = [int(round(float(item))) for item in coords]
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            return [x0, y0, x1, y1]
    return None


def clamp_box(box: Iterable[Any] | None, width: int, height: int) -> list[int] | None:
    if box is None or width <= 0 or height <= 0:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(item))) for item in box]
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def normalize_box_1000(box: Iterable[Any], width: int, height: int) -> list[int]:
    x0, y0, x1, y1 = [float(item) for item in box]
    values = [
        int(1000 * x0 / max(width, 1)),
        int(1000 * y0 / max(height, 1)),
        int(1000 * x1 / max(width, 1)),
        int(1000 * y1 / max(height, 1)),
    ]
    return [max(0, min(value, 1000)) for value in values]


def normalize_angle_deg(angle_deg: float | int | None) -> float | None:
    if angle_deg is None:
        return None
    value = float(angle_deg)
    while value <= -180.0:
        value += 360.0
    while value > 180.0:
        value -= 360.0
    return value


def _unit_vector(dx: float, dy: float) -> tuple[float, float] | None:
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None
    return dx / norm, dy / norm


def angle_deg_from_quad(value: Any) -> float | None:
    """Estimate text-line angle from OCR corner points.

    Assumes the common point order top-left, top-right, bottom-right,
    bottom-left. If that order is not reliable, the longest edge fallback still
    returns a stable orientation.
    """

    quad = parse_quad(value)
    if not quad or len(quad) < 4:
        return None
    p0, p1, p2, p3 = quad[:4]
    candidates = []
    for start, end in ((p0, p1), (p3, p2)):
        vec = _unit_vector(end[0] - start[0], end[1] - start[1])
        if vec is not None:
            candidates.append(vec)
    if candidates:
        dx = sum(item[0] for item in candidates)
        dy = sum(item[1] for item in candidates)
        if math.hypot(dx, dy) > 1e-6:
            return normalize_angle_deg(math.degrees(math.atan2(dy, dx)))

    longest = None
    longest_len = 0.0
    for start, end in ((p0, p1), (p1, p2), (p2, p3), (p3, p0)):
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        length = math.hypot(dx, dy)
        if length > longest_len:
            longest = (dx, dy)
            longest_len = length
    if longest is None or longest_len <= 1e-6:
        return None
    return normalize_angle_deg(math.degrees(math.atan2(longest[1], longest[0])))


def angle_deg_to_feature(
    angle_deg: float | int | None,
    box: Iterable[Any] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[float]:
    """Return a fixed-size feature vector; missing angles are all zeros."""

    angle = normalize_angle_deg(float(angle_deg)) if angle_deg is not None else None
    if angle is None:
        return [0.0] * ANGLE_FEATURE_DIM

    theta = math.radians(angle)
    width_norm = 0.0
    height_norm = 0.0
    aspect_log = 0.0
    area_norm = 0.0
    if box is not None and image_width and image_height:
        try:
            x0, y0, x1, y1 = [float(item) for item in box]
            w = max(0.0, x1 - x0)
            h = max(0.0, y1 - y0)
            width_norm = min(w / max(float(image_width), 1.0), 1.0)
            height_norm = min(h / max(float(image_height), 1.0), 1.0)
            aspect_log = max(-4.0, min(math.log((w + 1.0) / (h + 1.0)), 4.0)) / 4.0
            area_norm = min((w * h) / max(float(image_width * image_height), 1.0), 1.0)
        except (TypeError, ValueError):
            pass

    return [
        math.sin(theta),
        math.cos(theta),
        angle / 180.0,
        abs(angle) / 180.0,
        1.0,
        width_norm,
        height_norm,
        aspect_log,
        area_norm,
    ]


def relative_angle_deg(head_angle: float | None, dep_angle: float | None) -> float | None:
    if head_angle is None or dep_angle is None:
        return None
    return normalize_angle_deg(float(dep_angle) - float(head_angle))


def build_word_angle_features(
    word_payload: dict[str, Any] | None = None,
    *,
    box: Iterable[Any] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Build feature/debug pair for a single OCR word."""

    payload = word_payload or {}
    explicit_angle = payload.get("angle_deg", payload.get("angleDeg"))
    angle = normalize_angle_deg(explicit_angle) if explicit_angle is not None else None
    quad_source = None
    quad = None
    for key in ("quad", "cornerPoints", "corner_points", "vertices", "points", "polygon"):
        parsed = parse_quad(payload.get(key))
        if parsed:
            quad = parsed
            quad_source = key
            break
    if angle is None and quad is not None:
        angle = angle_deg_from_quad(quad)
    resolved_box = list(box) if box is not None else None
    if resolved_box is None:
        for key in ("box", "bbox", "bounding_box", "boundingBox", "rect"):
            resolved_box = quad_to_axis_aligned_box(payload.get(key))
            if resolved_box is not None:
                break
    feature = angle_deg_to_feature(angle, resolved_box, image_width, image_height)
    return feature, {
        "angle_deg": angle,
        "has_angle": bool(angle is not None),
        "quad_source": quad_source,
        "feature_dim": ANGLE_FEATURE_DIM,
    }


def build_angle_features_for_words(
    word_payloads: Iterable[dict[str, Any]] | None,
    *,
    boxes: Iterable[Iterable[Any]] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[list[list[float]], list[dict[str, Any]]]:
    payloads = list(word_payloads or [])
    box_list = list(boxes or [])
    features = []
    debug = []
    total = max(len(payloads), len(box_list))
    for idx in range(total):
        payload = payloads[idx] if idx < len(payloads) else {}
        box = box_list[idx] if idx < len(box_list) else None
        feature, row = build_word_angle_features(payload, box=box, image_width=image_width, image_height=image_height)
        features.append(feature)
        debug.append({"word_idx": idx, **row})
    return features, debug


def align_angle_features_to_tokens(
    encoding: Any,
    word_angle_features: Iterable[Iterable[float]] | None,
    *,
    batch_index: int = 0,
    first_subword_only: bool = False,
) -> torch.Tensor:
    """Align word-level angle features to a token sequence using word_ids."""

    word_features = [list(row) for row in (word_angle_features or [])]
    if not word_features:
        word_features = []
    seq_len = int(encoding["input_ids"].shape[1])
    output = torch.zeros((seq_len, ANGLE_FEATURE_DIM), dtype=torch.float32)
    try:
        word_ids = encoding.word_ids(batch_index=batch_index)
    except Exception as exc:  # pragma: no cover - tokenizer-version specific
        raise RuntimeError("BatchEncoding.word_ids is required for angle feature alignment.") from exc
    seen = set()
    attention = encoding.get("attention_mask")
    attention_row = attention[batch_index].tolist() if attention is not None else [1] * len(word_ids)
    for token_idx, word_idx in enumerate(word_ids):
        if token_idx >= seq_len or word_idx is None or not attention_row[token_idx]:
            continue
        if first_subword_only and int(word_idx) in seen:
            continue
        seen.add(int(word_idx))
        if int(word_idx) >= len(word_features):
            continue
        row = list(word_features[int(word_idx)])
        if len(row) != ANGLE_FEATURE_DIM:
            raise ValueError(f"angle feature dim must be {ANGLE_FEATURE_DIM}, got {len(row)}")
        output[token_idx] = torch.tensor(row, dtype=torch.float32)
    return output


def align_batch_angle_features_to_tokens(
    encoding: Any,
    batch_word_angle_features: Iterable[Iterable[Iterable[float]]] | None,
    *,
    first_subword_only: bool = False,
) -> torch.Tensor:
    rows = []
    features = list(batch_word_angle_features or [])
    batch_size = int(encoding["input_ids"].shape[0])
    for batch_index in range(batch_size):
        word_features = features[batch_index] if batch_index < len(features) else []
        rows.append(
            align_angle_features_to_tokens(
                encoding,
                word_features,
                batch_index=batch_index,
                first_subword_only=first_subword_only,
            )
        )
    return torch.stack(rows, dim=0)
