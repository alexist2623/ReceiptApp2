"""Angle, quadrilateral, and rotation helpers for receipt OCR geometry.

The base LayoutLMv3 pipeline still consumes axis-aligned ``bbox`` values in the
0..1000 coordinate space. This module adds optional per-word geometry features
that can preserve rotated OCR quadrilaterals without breaking no-angle data.
Missing angle/quad information is represented by zero vectors.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Iterator

import torch


ANGLE_FEATURE_DIM_BASIC = 9
RELATIVE_QUAD_FEATURE_DIM = 8
ANGLE_QUAD_FEATURE_DIM = 18
DEFAULT_MAX_ABS_ANGLE_DEG = 45.0

# Backward-compatible alias used by earlier scripts/checkpoints.
ANGLE_FEATURE_DIM = ANGLE_FEATURE_DIM_BASIC

ANGLE_ENCODING_MODES = {
    "none",
    "raw_scalar",
    "sincos",
    "sincos_scalar",
    "relative_quad",
    "angle_quad",
}
TODO_ANGLE_ENCODING_MODES = {"bucket_embedding", "spade_like"}


class AngleFeatureBatch(dict):
    """Dict result that still supports old ``features, debug = ...`` unpacking."""

    def __iter__(self) -> Iterator[Any]:
        yield self["angle_features"]
        yield self["word_angles"]


def angle_feature_dim_for_mode(mode: str | None) -> int:
    mode = normalize_angle_encoding_mode(mode)
    if mode == "none":
        return 0
    if mode == "raw_scalar":
        return 1
    if mode == "sincos":
        return 2
    if mode == "sincos_scalar":
        return ANGLE_FEATURE_DIM_BASIC
    if mode == "relative_quad":
        return RELATIVE_QUAD_FEATURE_DIM
    if mode == "angle_quad":
        return ANGLE_QUAD_FEATURE_DIM
    raise ValueError(f"Unsupported angle encoding mode: {mode}")


def normalize_angle_encoding_mode(mode: str | None) -> str:
    mode = (mode or "sincos_scalar").strip().lower()
    if mode in TODO_ANGLE_ENCODING_MODES:
        raise NotImplementedError(f"Angle encoding mode '{mode}' is reserved but not implemented yet.")
    if mode not in ANGLE_ENCODING_MODES:
        raise ValueError(f"Unsupported angle encoding mode: {mode}")
    return mode


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_point(obj: Any) -> tuple[float, float] | None:
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


def _box_from_dict(value: dict[str, Any]) -> list[int] | None:
    if all(key in value for key in ("x0", "y0", "x1", "y1")):
        coords = [_as_float(value[key]) for key in ("x0", "y0", "x1", "y1")]
    elif all(key in value for key in ("left", "top", "right", "bottom")):
        coords = [_as_float(value[key]) for key in ("left", "top", "right", "bottom")]
    elif all(key in value for key in ("x", "y", "width", "height")):
        x, y, w, h = [_as_float(value[key]) for key in ("x", "y", "width", "height")]
        coords = None if None in (x, y, w, h) else [x, y, x + w, y + h]  # type: ignore[operator]
    else:
        coords = None
    if coords is None or any(item is None for item in coords):
        return None
    x0, y0, x1, y1 = [int(round(float(item))) for item in coords]
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def parse_quad(value: Any) -> list[tuple[float, float]] | None:
    """Parse common OCR quadrilateral forms into four points."""

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
                point = parse_point((value.get(f"x{idx}"), value.get(f"y{idx}")))
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
            return [(float(floats[i]), float(floats[i + 1])) for i in range(0, 8, 2)]
        points = [parse_point(item) for item in value]
        points = [point for point in points if point is not None]
        if len(points) >= 4:
            return points[:4]
    return None


def box_to_quad(box: Iterable[Any] | None) -> list[tuple[float, float]] | None:
    if box is None:
        return None
    try:
        x0, y0, x1, y1 = [float(item) for item in box]
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def quad_to_axis_aligned_box(value: Any) -> list[int] | None:
    """Convert a quad or rectangular box-like object into ``[x0, y0, x1, y1]``."""

    quad = parse_quad(value)
    if quad:
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        return [int(round(min(xs))), int(round(min(ys))), int(round(max(xs))), int(round(max(ys)))]
    if isinstance(value, dict):
        return _box_from_dict(value)
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
    """Normalize an orientation to the half-open text angle range ``[-90, 90)``."""

    if angle_deg is None:
        return None
    value = ((float(angle_deg) + 90.0) % 180.0) - 90.0
    if math.isclose(value, 90.0):
        value = -90.0
    if math.isclose(value, 0.0, abs_tol=1e-12):
        value = 0.0
    return value


def _unit_vector(dx: float, dy: float) -> tuple[float, float] | None:
    norm = math.hypot(dx, dy)
    if norm <= 1e-6:
        return None
    return dx / norm, dy / norm


def angle_deg_from_quad(value: Any) -> float | None:
    """Estimate text-line angle from OCR corner points."""

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


def relative_angle_deg(head_angle: float | None, dep_angle: float | None) -> float | None:
    if head_angle is None or dep_angle is None:
        return None
    return normalize_angle_deg(float(dep_angle) - float(head_angle))


def polygon_area(points: Iterable[Iterable[Any]] | None) -> float:
    quad = parse_quad(points)
    if not quad:
        return 0.0
    area = 0.0
    for idx, (x0, y0) in enumerate(quad):
        x1, y1 = quad[(idx + 1) % len(quad)]
        area += x0 * y1 - x1 * y0
    return abs(area) / 2.0


def _clip_angle(angle: float | None, max_abs_angle_deg: float) -> float:
    if angle is None:
        return 0.0
    max_abs = max(float(max_abs_angle_deg), 1e-6)
    return max(-max_abs, min(float(angle), max_abs))


def _angle_base(angle: float | None, max_abs_angle_deg: float) -> list[float]:
    if angle is None:
        return [0.0, 0.0, 0.0, 0.0]
    theta = math.radians(angle)
    clipped = _clip_angle(angle, max_abs_angle_deg)
    max_abs = max(float(max_abs_angle_deg), 1e-6)
    return [math.sin(theta), math.cos(theta), clipped / max_abs, abs(clipped) / max_abs]


def relative_quad_features(
    quad: Any,
    box: Iterable[Any] | None = None,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[float]:
    """Return eight point offsets relative to the axis-aligned box center/size."""

    parsed_quad = parse_quad(quad)
    resolved_box = list(box) if box is not None else quad_to_axis_aligned_box(parsed_quad)
    if not parsed_quad or resolved_box is None:
        return [0.0] * RELATIVE_QUAD_FEATURE_DIM
    try:
        x0, y0, x1, y1 = [float(item) for item in resolved_box]
    except (TypeError, ValueError):
        return [0.0] * RELATIVE_QUAD_FEATURE_DIM
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    features = []
    for x, y in parsed_quad[:4]:
        features.extend([(x - cx) / width, (y - cy) / height])
    return [max(-2.0, min(float(value), 2.0)) for value in features[:RELATIVE_QUAD_FEATURE_DIM]]


def angle_deg_to_feature(
    angle_deg: float | int | None,
    box: Iterable[Any] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    *,
    quad: Any = None,
    relative_angle: float | None = None,
    mode: str | None = "sincos_scalar",
    max_abs_angle_deg: float = DEFAULT_MAX_ABS_ANGLE_DEG,
) -> list[float]:
    mode = normalize_angle_encoding_mode(mode)
    feature_dim = angle_feature_dim_for_mode(mode)
    if feature_dim == 0:
        return []
    angle = normalize_angle_deg(float(angle_deg)) if angle_deg is not None else None
    has_angle = 1.0 if angle is not None else 0.0
    if mode == "raw_scalar":
        return [_clip_angle(angle, max_abs_angle_deg) / max(max_abs_angle_deg, 1e-6) if angle is not None else 0.0]
    if mode == "sincos":
        return _angle_base(angle, max_abs_angle_deg)[:2] if angle is not None else [0.0, 0.0]

    resolved_box = list(box) if box is not None else quad_to_axis_aligned_box(quad)
    width_norm = 0.0
    height_norm = 0.0
    aspect_log = 0.0
    area_norm = 0.0
    if resolved_box is not None and image_width and image_height:
        try:
            x0, y0, x1, y1 = [float(item) for item in resolved_box]
            w = max(0.0, x1 - x0)
            h = max(0.0, y1 - y0)
            width_norm = min(w / max(float(image_width), 1.0), 1.0)
            height_norm = min(h / max(float(image_height), 1.0), 1.0)
            aspect_log = max(-4.0, min(math.log((w + 1.0) / (h + 1.0)), 4.0)) / 4.0
            area_norm = min((w * h) / max(float(image_width * image_height), 1.0), 1.0)
        except (TypeError, ValueError):
            pass

    if mode == "sincos_scalar":
        if angle is None:
            return [0.0] * ANGLE_FEATURE_DIM_BASIC
        return _angle_base(angle, max_abs_angle_deg) + [has_angle, width_norm, height_norm, aspect_log, area_norm]

    rel_quad = relative_quad_features(quad or box_to_quad(resolved_box), resolved_box)
    if mode == "relative_quad":
        return rel_quad

    if mode == "angle_quad":
        rel_angle_base = _angle_base(relative_angle, max_abs_angle_deg)[:3] if relative_angle is not None else [0.0, 0.0, 0.0]
        quad_area = polygon_area(quad) if quad is not None else 0.0
        if image_width and image_height:
            quad_area = min(quad_area / max(float(image_width * image_height), 1.0), 1.0)
        has_quad = 1.0 if parse_quad(quad) is not None else 0.0
        values = _angle_base(angle, max_abs_angle_deg) + rel_angle_base + rel_quad + [quad_area, has_angle, has_quad]
        if len(values) != ANGLE_QUAD_FEATURE_DIM:
            raise AssertionError(f"angle_quad feature dim mismatch: {len(values)}")
        return values

    raise ValueError(f"Unsupported angle encoding mode: {mode}")


def _extract_payload_quad(payload: dict[str, Any]) -> tuple[list[tuple[float, float]] | None, str | None]:
    for key in ("quad", "cornerPoints", "corner_points", "vertices", "points", "polygon"):
        parsed = parse_quad(payload.get(key))
        if parsed:
            return parsed, key
    return None, None


def build_word_angle_features(
    word_payload: dict[str, Any] | None = None,
    *,
    box: Iterable[Any] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    mode: str | None = "sincos_scalar",
    max_abs_angle_deg: float = DEFAULT_MAX_ABS_ANGLE_DEG,
    page_angle_deg: float | None = None,
) -> tuple[list[float], dict[str, Any]]:
    """Build feature/debug pair for a single OCR word."""

    payload = word_payload or {}
    explicit_angle = payload.get("angle_deg", payload.get("angleDeg", payload.get("angle")))
    angle = normalize_angle_deg(explicit_angle) if explicit_angle is not None else None
    quad, quad_source = _extract_payload_quad(payload)
    if angle is None and quad is not None:
        angle = angle_deg_from_quad(quad)
    resolved_box = list(box) if box is not None else None
    if resolved_box is None:
        for key in ("box", "bbox", "bounding_box", "boundingBox", "rect"):
            resolved_box = quad_to_axis_aligned_box(payload.get(key))
            if resolved_box is not None:
                break
    if quad is None and resolved_box is not None:
        quad = box_to_quad(resolved_box)
    relative = relative_angle_deg(page_angle_deg, angle) if page_angle_deg is not None else angle
    feature = angle_deg_to_feature(
        angle,
        resolved_box,
        image_width,
        image_height,
        quad=quad,
        relative_angle=relative,
        mode=mode,
        max_abs_angle_deg=max_abs_angle_deg,
    )
    return feature, {
        "angle_deg": angle,
        "relative_angle_deg": relative,
        "has_angle": bool(angle is not None),
        "has_quad": parse_quad(quad) is not None,
        "quad_source": quad_source,
        "feature_dim": len(feature),
        "angle_encoding_mode": normalize_angle_encoding_mode(mode),
    }


def build_angle_features_for_words(
    word_payloads: Iterable[dict[str, Any]] | None,
    *,
    boxes: Iterable[Iterable[Any]] | None = None,
    image_width: int | None = None,
    image_height: int | None = None,
    mode: str | None = "sincos_scalar",
    max_abs_angle_deg: float = DEFAULT_MAX_ABS_ANGLE_DEG,
    page_angle_deg: float | None = None,
) -> AngleFeatureBatch:
    payloads = list(word_payloads or [])
    box_list = list(boxes or [])
    features = []
    debug = []
    total = max(len(payloads), len(box_list))
    for idx in range(total):
        payload = payloads[idx] if idx < len(payloads) else {}
        box = box_list[idx] if idx < len(box_list) else None
        feature, row = build_word_angle_features(
            payload,
            box=box,
            image_width=image_width,
            image_height=image_height,
            mode=mode,
            max_abs_angle_deg=max_abs_angle_deg,
            page_angle_deg=page_angle_deg,
        )
        features.append(feature)
        debug.append({"word_idx": idx, **row})
    return AngleFeatureBatch(
        {
            "angle_features": features,
            "word_angles": debug,
            "feature_dim": angle_feature_dim_for_mode(mode),
            "angle_encoding_mode": normalize_angle_encoding_mode(mode),
            "num_words_with_angle": sum(1 for row in debug if row.get("has_angle")),
            "num_words_with_quad": sum(1 for row in debug if row.get("has_quad")),
        }
    )


def _word_ids_from_encoding(encoding: Any, batch_index: int) -> list[int | None]:
    try:
        word_ids = list(encoding.word_ids(batch_index=batch_index))
        if len(word_ids) == 1 and isinstance(word_ids[0], list):
            word_ids = list(word_ids[0])
        elif word_ids and all(isinstance(item, list) for item in word_ids):
            word_ids = list(word_ids[batch_index])
        return word_ids
    except Exception as exc:  # pragma: no cover - tokenizer-version specific
        raise RuntimeError("BatchEncoding.word_ids is required for angle feature alignment.") from exc


def align_angle_features_to_tokens(
    encoding_or_word_features: Any,
    word_angle_features_or_word_ids: Iterable[Any] | None = None,
    max_length: int | None = None,
    feature_dim: int | None = None,
    *,
    batch_index: int = 0,
    first_subword_only: bool = False,
) -> torch.Tensor:
    """Align word-level angle features to token positions.

    Supports both the old call form ``(encoding, word_features)`` and the new
    direct form ``(word_features, word_ids, max_length, feature_dim)``.
    """

    is_encoding = hasattr(encoding_or_word_features, "word_ids") and "input_ids" in encoding_or_word_features
    if is_encoding:
        encoding = encoding_or_word_features
        word_features = [list(row) for row in (word_angle_features_or_word_ids or [])]
        word_ids = _word_ids_from_encoding(encoding, batch_index)
        seq_len = int(encoding["input_ids"].shape[1] if max_length is None else max_length)
        attention = encoding.get("attention_mask")
        attention_row = attention[batch_index].tolist() if attention is not None else [1] * len(word_ids)
    else:
        word_features = [list(row) for row in (encoding_or_word_features or [])]
        word_ids = list(word_angle_features_or_word_ids or [])
        if len(word_ids) == 1 and isinstance(word_ids[0], list):
            word_ids = list(word_ids[0])
        elif word_ids and all(isinstance(item, list) for item in word_ids):
            word_ids = list(word_ids[0])
        seq_len = int(max_length if max_length is not None else len(word_ids))
        attention_row = [1] * len(word_ids)

    inferred_dim = feature_dim if feature_dim is not None else (len(word_features[0]) if word_features else ANGLE_FEATURE_DIM)
    output = torch.zeros((seq_len, int(inferred_dim)), dtype=torch.float32)
    seen = set()
    for token_idx, word_idx in enumerate(word_ids):
        if token_idx >= seq_len or word_idx is None:
            continue
        if token_idx < len(attention_row) and not attention_row[token_idx]:
            continue
        word_idx = int(word_idx)
        if first_subword_only and word_idx in seen:
            continue
        seen.add(word_idx)
        if word_idx >= len(word_features):
            continue
        row = list(word_features[word_idx])
        if len(row) != int(inferred_dim):
            raise ValueError(f"angle feature dim must be {inferred_dim}, got {len(row)}")
        output[token_idx] = torch.tensor(row, dtype=torch.float32)
    return output


def align_batch_angle_features_to_tokens(
    encoding: Any,
    batch_word_angle_features: Iterable[Iterable[Iterable[float]]] | None,
    *,
    first_subword_only: bool = False,
    feature_dim: int | None = None,
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
                feature_dim=feature_dim,
            )
        )
    return torch.stack(rows, dim=0)


def affine_matrix_for_rotation(width: int, height: int, angle_deg: float, *, expand: bool = True) -> dict[str, Any]:
    """Return point-transform metadata for PIL-style counter-clockwise rotation."""

    theta = math.radians(float(angle_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    cx = float(width) / 2.0
    cy = float(height) / 2.0

    def raw_transform(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        dx = x - cx
        dy = y - cy
        return cos_t * dx + sin_t * dy, -sin_t * dx + cos_t * dy

    corners = [(0.0, 0.0), (float(width), 0.0), (float(width), float(height)), (0.0, float(height))]
    rotated = [raw_transform(point) for point in corners]
    if expand:
        min_x = min(x for x, _ in rotated)
        min_y = min(y for _, y in rotated)
        max_x = max(x for x, _ in rotated)
        max_y = max(y for _, y in rotated)
        out_width = int(math.ceil(max_x - min_x))
        out_height = int(math.ceil(max_y - min_y))
        offset_x = -min_x
        offset_y = -min_y
    else:
        out_width = int(width)
        out_height = int(height)
        offset_x = float(width) / 2.0
        offset_y = float(height) / 2.0
    return {
        "angle_deg": float(angle_deg),
        "expand": bool(expand),
        "input_width": int(width),
        "input_height": int(height),
        "output_width": out_width,
        "output_height": out_height,
        "cos": cos_t,
        "sin": sin_t,
        "cx": cx,
        "cy": cy,
        "offset_x": offset_x,
        "offset_y": offset_y,
    }


def apply_affine_to_points(points: Iterable[Any], matrix: dict[str, Any]) -> list[tuple[float, float]]:
    out = []
    cos_t = float(matrix["cos"])
    sin_t = float(matrix["sin"])
    cx = float(matrix["cx"])
    cy = float(matrix["cy"])
    offset_x = float(matrix["offset_x"])
    offset_y = float(matrix["offset_y"])
    for item in points:
        point = parse_point(item)
        if point is None:
            continue
        x, y = point
        dx = x - cx
        dy = y - cy
        out.append((cos_t * dx + sin_t * dy + offset_x, -sin_t * dx + cos_t * dy + offset_y))
    return out


def rotate_points(points: Iterable[Any], width: int, height: int, angle_deg: float, *, expand: bool = True):
    matrix = affine_matrix_for_rotation(width, height, angle_deg, expand=expand)
    return apply_affine_to_points(points, matrix)


def rotate_image_and_quads(
    image,
    word_quads: Iterable[Any],
    angle_deg: float,
    *,
    fillcolor=(255, 255, 255),
    resample=None,
):
    """Rotate a PIL image and every word quad with the same affine transform."""

    from PIL import Image

    if resample is None:
        resample = Image.Resampling.BICUBIC if hasattr(Image, "Resampling") else Image.BICUBIC
    width, height = image.size
    matrix = affine_matrix_for_rotation(width, height, angle_deg, expand=True)
    rotated = image.rotate(float(angle_deg), resample=resample, expand=True, fillcolor=fillcolor)
    rotated_quads = []
    for quad in word_quads:
        parsed = parse_quad(quad)
        if parsed is None:
            rotated_quads.append(None)
            continue
        rotated_quads.append(apply_affine_to_points(parsed, matrix))
    matrix["actual_output_width"], matrix["actual_output_height"] = rotated.size
    return rotated, rotated_quads, matrix
