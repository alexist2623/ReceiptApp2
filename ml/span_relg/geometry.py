import math


def validate_box(box) -> bool:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return False
    try:
        x0, y0, x1, y1 = [float(v) for v in box]
    except (TypeError, ValueError):
        return False
    return x1 > x0 and y1 > y0


def clamp_box(box, width, height):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(v))) for v in box]
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def union_boxes(boxes):
    valid = [box for box in boxes if validate_box(box)]
    if not valid:
        return None
    return [
        int(min(box[0] for box in valid)),
        int(min(box[1] for box in valid)),
        int(max(box[2] for box in valid)),
        int(max(box[3] for box in valid)),
    ]


def normalize_box_1000(box, width, height):
    if not validate_box(box):
        return None
    x0, y0, x1, y1 = [float(v) for v in box]
    values = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]
    return [max(0, min(v, 1000)) for v in values]


def box1000_to_unit(box):
    return [max(0.0, min(float(v) / 1000.0, 1.0)) for v in box]


def box_center(box_unit):
    x0, y0, x1, y1 = box_unit
    return (x0 + x1) / 2.0, (y0 + y1) / 2.0


def box_wh(box_unit):
    x0, y0, x1, y1 = box_unit
    return max(0.0, x1 - x0), max(0.0, y1 - y0)


def _overlap_ratio(a0, a1, b0, b1):
    overlap = max(0.0, min(a1, b1) - max(a0, b0))
    denom = max(1e-6, min(max(0.0, a1 - a0), max(0.0, b1 - b0)))
    return overlap / denom


def pair_geometry_features(box_i_unit, box_j_unit):
    xi0, yi0, xi1, yi1 = [float(v) for v in box_i_unit]
    xj0, yj0, xj1, yj1 = [float(v) for v in box_j_unit]
    cxi, cyi = box_center([xi0, yi0, xi1, yi1])
    cxj, cyj = box_center([xj0, yj0, xj1, yj1])
    wi, hi = box_wh([xi0, yi0, xi1, yi1])
    wj, hj = box_wh([xj0, yj0, xj1, yj1])
    dx = cxj - cxi
    dy = cyj - cyi
    distance = math.sqrt(dx * dx + dy * dy)
    angle = math.atan2(dy, dx) if distance > 0 else 0.0
    x_gap = max(0.0, max(xi0, xj0) - min(xi1, xj1))
    y_gap = max(0.0, max(yi0, yj0) - min(yi1, yj1))
    area_i = wi * hi
    area_j = wj * hj
    avg_h = max(1e-6, (hi + hj) / 2.0)
    avg_w = max(1e-6, (wi + wj) / 2.0)
    same_row = 1.0 / (1.0 + abs(dy) / avg_h)
    same_col = 1.0 / (1.0 + abs(dx) / avg_w)
    return [
        cxi,
        cyi,
        wi,
        hi,
        cxj,
        cyj,
        wj,
        hj,
        dx,
        dy,
        abs(dx),
        abs(dy),
        distance,
        math.sin(angle),
        math.cos(angle),
        _overlap_ratio(xi0, xi1, xj0, xj1),
        _overlap_ratio(yi0, yi1, yj0, yj1),
        x_gap,
        y_gap,
        1.0 if cxj > cxi else 0.0,
        1.0 if cxj < cxi else 0.0,
        1.0 if cyj > cyi else 0.0,
        1.0 if cyj < cyi else 0.0,
        area_i,
        area_j,
        area_j / max(area_i, 1e-6),
        same_row,
        same_col,
    ]


def pair_geometry_dim():
    return len(pair_geometry_features([0.0, 0.0, 0.1, 0.1], [0.2, 0.0, 0.3, 0.1]))

