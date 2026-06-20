"""Optional BIO boundary repair for receipt word labels."""

from __future__ import annotations

from collections import Counter

from ml.receipt_schema import PRICE_FIELDS, label_to_field


def _center_y(box):
    return (float(box[1]) + float(box[3])) / 2.0


def _height(box):
    return max(1.0, float(box[3]) - float(box[1]))


def _same_line(prev_idx, idx, boxes, line_ids, max_y_center_delta_ratio):
    if line_ids and prev_idx < len(line_ids) and idx < len(line_ids):
        if line_ids[prev_idx] is not None and line_ids[idx] is not None:
            return line_ids[prev_idx] == line_ids[idx]
    if not boxes or prev_idx >= len(boxes) or idx >= len(boxes):
        return prev_idx + 1 == idx
    prev_box = boxes[prev_idx]
    box = boxes[idx]
    allowed = max(_height(prev_box), _height(box)) * float(max_y_center_delta_ratio)
    return abs(_center_y(prev_box) - _center_y(box)) <= allowed


def _near_gap(prev_idx, idx, boxes, max_gap_ratio):
    if not boxes or prev_idx >= len(boxes) or idx >= len(boxes):
        return prev_idx + 1 == idx
    prev_box = boxes[prev_idx]
    box = boxes[idx]
    gap = float(box[0]) - float(prev_box[2])
    if gap < 0:
        return True
    allowed = max(_height(prev_box), _height(box)) * float(max_gap_ratio)
    return gap <= allowed


def _price_merge_allowed(prev_word, word):
    prev = str(prev_word or "").strip()
    curr = str(word or "").strip()
    currency_or_joiner = {"$", "Y", "EUR", "GBP", "-", "."}
    if prev in currency_or_joiner:
        return True
    return curr.replace(".", "", 1).replace(",", "").isdigit() and prev in currency_or_joiner


def repair_bio_boundaries(
    labels,
    words=None,
    boxes=None,
    line_ids=None,
    max_same_line_gap_ratio=3.0,
    max_y_center_delta_ratio=0.6,
):
    """Repair safe same-field ``B-X B-X`` fragments into ``B-X I-X``.

    Returns ``(repaired_labels, report)``. The repair is intentionally narrow:
    summary/payment heads and most price spans remain separate unless the split
    is a currency/joiner artifact.
    """

    repaired = list(labels)
    words = list(words or [""] * len(labels))
    boxes = list(boxes or [])
    repairs = []
    skipped = []
    counts = Counter()
    for idx in range(1, len(repaired)):
        label = str(repaired[idx] or "O")
        if not label.startswith("B-"):
            continue
        field = label_to_field(label)
        if field == "O":
            continue
        prev_field = label_to_field(repaired[idx - 1])
        if prev_field != field:
            continue
        if field in {"SUBTOTAL_NAME", "TAX_NAME", "TOTAL_NAME", "TIP_NAME", "PAYMENT_METHOD"}:
            skipped.append({"word_idx": idx, "field": field, "reason": "summary/payment boundary"})
            continue
        if field in PRICE_FIELDS and not _price_merge_allowed(words[idx - 1], words[idx]):
            skipped.append({"word_idx": idx, "field": field, "reason": "unsafe price merge"})
            continue
        if not _same_line(idx - 1, idx, boxes, line_ids, max_y_center_delta_ratio):
            skipped.append({"word_idx": idx, "field": field, "reason": "not same line"})
            continue
        if not _near_gap(idx - 1, idx, boxes, max_same_line_gap_ratio):
            skipped.append({"word_idx": idx, "field": field, "reason": "large horizontal gap"})
            continue
        before = repaired[idx]
        repaired[idx] = f"I-{field}"
        counts[field] += 1
        repairs.append({"word_idx": idx, "field": field, "before": before, "after": repaired[idx], "word": words[idx]})
    return repaired, {
        "num_repairs": len(repairs),
        "repairs_by_field": dict(counts),
        "examples": repairs[:50],
        "skipped_examples": skipped[:50],
        "num_skipped": len(skipped),
    }
