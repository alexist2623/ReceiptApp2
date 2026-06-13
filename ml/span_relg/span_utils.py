from .geometry import normalize_box_1000, union_boxes
from ml.receipt_schema import canonicalize_label, label_to_field, normalize_span_text


def make_span_text(words) -> str:
    return " ".join(str(word) for word in words)


def _raw_label_field(label):
    value = str(label or "O").strip()
    if not value or value == "O":
        return "O"
    if value.startswith(("B-", "I-")):
        value = value[2:]
    return value.upper().replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


def _new_span(span_id, field, raw_field, word_idx, prediction, raw_label, canonical_label):
    return {
        "span_id": span_id,
        "field": field,
        "raw_field": raw_field,
        "text": str(prediction.get("text", "")),
        "raw_text": str(prediction.get("text", "")),
        "normalized_text": normalize_span_text(field, str(prediction.get("text", ""))),
        "word_indices": [word_idx],
        "first_word_idx": word_idx,
        "box": prediction.get("box"),
        "normalized_box": prediction.get("normalized_box"),
        "confidence": float(prediction.get("confidence", 1.0)),
        "confidence_min": float(prediction.get("confidence", 1.0)),
        "category": prediction.get("category"),
        "group_id": prediction.get("group_id"),
        "sub_group_id": prediction.get("sub_group_id"),
        "row_id": prediction.get("row_id"),
        "line_id": prediction.get("line_id"),
        "raw_labels": [raw_label],
        "canonical_labels": [canonical_label],
        "warnings": [],
    }


def _finalize_span(span, predictions, image_width, image_height):
    words = [predictions[idx].get("text", "") for idx in span["word_indices"]]
    boxes = [predictions[idx].get("box") for idx in span["word_indices"]]
    norm_boxes = [predictions[idx].get("normalized_box") for idx in span["word_indices"]]
    confidences = [float(predictions[idx].get("confidence", 1.0)) for idx in span["word_indices"]]
    span["text"] = make_span_text(words)
    span["raw_text"] = span["text"]
    span["normalized_text"] = normalize_span_text(span["field"], span["text"])
    span["box"] = union_boxes(boxes)
    span["normalized_box"] = union_boxes(norm_boxes)
    if span["normalized_box"] is None and span["box"] is not None:
        span["normalized_box"] = normalize_box_1000(span["box"], image_width, image_height)
    span["confidence"] = sum(confidences) / len(confidences) if confidences else 0.0
    span["confidence_min"] = min(confidences) if confidences else 0.0
    return span


def bio_predictions_to_spans(predictions: list[dict], image_width, image_height, canonicalize=True) -> list[dict]:
    spans = []
    current = None
    next_id = 0
    for word_idx, prediction in enumerate(predictions):
        raw_label = (
            prediction.get("canonical_label")
            or prediction.get("label")
            or prediction.get("pred_label")
            or prediction.get("gt_label")
            or "O"
        )
        label = canonicalize_label(raw_label) if canonicalize else str(raw_label)
        field = label_to_field(label)
        raw_field = _raw_label_field(raw_label) if canonicalize else field
        if field == "O":
            if current is not None:
                spans.append(_finalize_span(current, predictions, image_width, image_height))
                current = None
            continue
        is_begin = str(label).startswith("B-")
        is_inside = str(label).startswith("I-")
        if is_begin or current is None or current["field"] != field:
            if current is not None:
                spans.append(_finalize_span(current, predictions, image_width, image_height))
            current = _new_span(next_id, field, raw_field, word_idx, prediction, str(raw_label), label)
            next_id += 1
            if is_inside and current is not None:
                current["warnings"].append(f"I-{field} started without previous span; treated as B-{field}.")
        else:
            current["word_indices"].append(word_idx)
            current.setdefault("raw_labels", []).append(str(raw_label))
            current.setdefault("canonical_labels", []).append(label)
    if current is not None:
        spans.append(_finalize_span(current, predictions, image_width, image_height))
    for span_id, span in enumerate(spans):
        span["span_id"] = span_id
    return spans


def span_pool_hidden(word_hidden, word_indices, mode="first"):
    if not word_indices:
        raise ValueError("word_indices must not be empty")
    if mode == "first":
        return word_hidden[int(word_indices[0])]
    if mode == "mean":
        return word_hidden[word_indices].mean(dim=0)
    raise ValueError(f"Unsupported span pooling mode: {mode}")
