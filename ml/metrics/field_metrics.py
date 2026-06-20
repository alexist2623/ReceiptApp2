"""Field-aware BIO metrics for receipt token classification."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

from seqeval.metrics import f1_score, precision_score, recall_score

from ml.receipt_schema import canonicalize_field, get_canonical_fields, label_to_field


REPORT_FIELDS = [
    "STORE_NAME",
    "STORE_ADDRESS",
    "STORE_PHONE",
    "DATE",
    "TIME",
    "ITEM_NAME",
    "ITEM_CODE",
    "ITEM_QTY",
    "ITEM_UNIT_PRICE",
    "ITEM_PRICE",
    "ITEM_OPTION",
    "ITEM_TAX_FLAG",
    "SUBTOTAL_NAME",
    "SUBTOTAL_PRICE",
    "TAX_NAME",
    "TAX_PRICE",
    "TIP_NAME",
    "TIP_PRICE",
    "TOTAL_NAME",
    "TOTAL_PRICE",
    "PAYMENT_METHOD",
    "PAYMENT_INFO",
    "RECEIPT_ID",
    "STORE_ID",
]


def _prf(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def labels_to_spans(labels: list[str]) -> list[dict]:
    spans = []
    current = None
    for idx, label in enumerate(labels):
        field = label_to_field(label)
        if field == "O":
            if current is not None:
                current["end"] = idx - 1
                spans.append(current)
                current = None
            continue
        is_begin = str(label).startswith("B-")
        if current is None or is_begin or current["field"] != field:
            if current is not None:
                current["end"] = idx - 1
                spans.append(current)
            current = {"field": field, "start": idx, "end": idx}
        else:
            current["end"] = idx
    if current is not None:
        spans.append(current)
    return spans


def _span_key(span: dict) -> tuple:
    return (canonicalize_field(span["field"]), int(span["start"]), int(span["end"]))


def _overlaps(a: dict, b: dict) -> bool:
    if canonicalize_field(a["field"]) != canonicalize_field(b["field"]):
        return False
    return max(int(a["start"]), int(b["start"])) <= min(int(a["end"]), int(b["end"]))


def _field_span_metrics(true_sequences, pred_sequences, relaxed=False) -> dict:
    fields = list(dict.fromkeys(REPORT_FIELDS + get_canonical_fields()))
    totals = {field: {"tp": 0, "fp": 0, "fn": 0, "support_gold": 0, "support_pred": 0} for field in fields}
    for true_labels, pred_labels in zip(true_sequences, pred_sequences):
        gold = labels_to_spans(list(true_labels))
        pred = labels_to_spans(list(pred_labels))
        for span in gold:
            field = canonicalize_field(span["field"])
            totals.setdefault(field, {"tp": 0, "fp": 0, "fn": 0, "support_gold": 0, "support_pred": 0})
            totals[field]["support_gold"] += 1
        for span in pred:
            field = canonicalize_field(span["field"])
            totals.setdefault(field, {"tp": 0, "fp": 0, "fn": 0, "support_gold": 0, "support_pred": 0})
            totals[field]["support_pred"] += 1
        matched_gold = set()
        matched_pred = set()
        if relaxed:
            for pred_idx, pred_span in enumerate(pred):
                for gold_idx, gold_span in enumerate(gold):
                    if gold_idx in matched_gold:
                        continue
                    if _overlaps(pred_span, gold_span):
                        matched_pred.add(pred_idx)
                        matched_gold.add(gold_idx)
                        break
        else:
            gold_by_key = {_span_key(span): idx for idx, span in enumerate(gold)}
            for pred_idx, pred_span in enumerate(pred):
                key = _span_key(pred_span)
                if key in gold_by_key and gold_by_key[key] not in matched_gold:
                    matched_pred.add(pred_idx)
                    matched_gold.add(gold_by_key[key])
        for idx, span in enumerate(pred):
            field = canonicalize_field(span["field"])
            if idx in matched_pred:
                totals[field]["tp"] += 1
            else:
                totals[field]["fp"] += 1
        for idx, span in enumerate(gold):
            if idx not in matched_gold:
                totals[canonicalize_field(span["field"])]["fn"] += 1
    return {
        field: {**_prf(values["tp"], values["fp"], values["fn"]), "support_gold": values["support_gold"], "support_pred": values["support_pred"]}
        for field, values in totals.items()
        if values["support_gold"] or values["support_pred"] or field in REPORT_FIELDS
    }


def boundary_error_counts(true_sequences, pred_sequences) -> dict:
    counts = Counter()
    confusion = Counter()
    for true_labels, pred_labels in zip(true_sequences, pred_sequences):
        for idx, (gold, pred) in enumerate(zip(true_labels, pred_labels)):
            gold_field = label_to_field(gold)
            pred_field = label_to_field(pred)
            confusion[(gold_field, pred_field)] += 1
            if gold.startswith("I-") and pred == f"B-{gold_field}":
                counts["predicted_B_where_gold_I"] += 1
            if gold.startswith("B-") and pred == f"I-{gold_field}":
                counts["predicted_I_where_gold_B"] += 1
            if idx > 0 and pred.startswith("B-"):
                prev_field = label_to_field(pred_labels[idx - 1])
                if prev_field == pred_field and prev_field != "O":
                    counts["fragmented_same_field_B_after_B_or_I"] += 1
    return {
        "boundary_errors": dict(counts),
        "field_confusion_top": [
            {"gold": gold, "pred": pred, "count": count}
            for (gold, pred), count in confusion.most_common(50)
            if gold != pred
        ],
    }


def compute_field_metrics(true_sequences, pred_sequences, words=None, boxes=None, line_ids=None) -> dict:
    true_sequences = [list(seq) for seq in true_sequences]
    pred_sequences = [list(seq) for seq in pred_sequences]
    total = sum(len(seq) for seq in true_sequences)
    correct = sum(1 for true, pred in zip(true_sequences, pred_sequences) for t, p in zip(true, pred) if t == p)
    result = {
        "token_accuracy": correct / total if total else 0.0,
        "seqeval_precision": precision_score(true_sequences, pred_sequences, zero_division=0),
        "seqeval_recall": recall_score(true_sequences, pred_sequences, zero_division=0),
        "seqeval_f1": f1_score(true_sequences, pred_sequences, zero_division=0),
        "strict_span_metrics": _field_span_metrics(true_sequences, pred_sequences, relaxed=False),
        "relaxed_span_metrics": _field_span_metrics(true_sequences, pred_sequences, relaxed=True),
    }
    result.update(boundary_error_counts(true_sequences, pred_sequences))
    return result
