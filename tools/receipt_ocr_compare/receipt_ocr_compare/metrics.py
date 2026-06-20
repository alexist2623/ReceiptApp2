from __future__ import annotations

from collections import Counter, defaultdict
from itertools import zip_longest
from typing import Any, Iterable

from .normalization import (
    CONFUSION_GROUPS,
    currency_counter,
    decimal_count,
    digit_sequence,
    is_numeric_token,
    normalize_text,
    numeric_char_sequence,
    punctuation_counter,
    sign_counter,
    thousands_separator_count,
)
from .schemas import GroundTruthToken, RecognitionResult


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i]
        for j, cb in enumerate(b, 1):
            curr.append(
                min(
                    prev[j] + 1,
                    curr[j - 1] + 1,
                    prev[j - 1] + (0 if ca == cb else 1),
                )
            )
        prev = curr
    return prev[-1]


def cer(ground_truth: str, prediction: str) -> float:
    gt = normalize_text(ground_truth)
    pred = normalize_text(prediction)
    if not gt:
        return 0.0 if not pred else 1.0
    return levenshtein(gt, pred) / len(gt)


def character_accuracy(ground_truth: str, prediction: str) -> float:
    gt = normalize_text(ground_truth)
    pred = normalize_text(prediction)
    denom = max(len(gt), len(pred), 1)
    return max(0.0, 1.0 - levenshtein(gt, pred) / denom)


def exact_match(ground_truth: str, prediction: str) -> bool:
    return ground_truth == prediction


def normalized_exact_match(ground_truth: str, prediction: str) -> bool:
    return normalize_text(ground_truth) == normalize_text(prediction)


def digit_sequence_exact_match(ground_truth: str, prediction: str) -> bool:
    return digit_sequence(ground_truth) == digit_sequence(prediction)


def punctuation_mismatch_count(ground_truth: str, prediction: str) -> int:
    diff = punctuation_counter(ground_truth) - punctuation_counter(prediction)
    reverse = punctuation_counter(prediction) - punctuation_counter(ground_truth)
    return sum(diff.values()) + sum(reverse.values())


def _counter_mismatch(a: Counter[str], b: Counter[str]) -> int:
    return sum((a - b).values()) + sum((b - a).values())


def decimal_point_mismatch_count(ground_truth: str, prediction: str) -> int:
    return abs(decimal_count(ground_truth) - decimal_count(prediction))


def thousands_separator_mismatch_count(ground_truth: str, prediction: str) -> int:
    return abs(thousands_separator_count(ground_truth) - thousands_separator_count(prediction))


def sign_mismatch_count(ground_truth: str, prediction: str) -> int:
    return _counter_mismatch(sign_counter(ground_truth), sign_counter(prediction))


def currency_symbol_mismatch_count(ground_truth: str, prediction: str) -> int:
    return _counter_mismatch(currency_counter(ground_truth), currency_counter(prediction))


def confusion_pairs(ground_truth: str, prediction: str) -> Counter[str]:
    groups = {tuple(pair) for pair in CONFUSION_GROUPS}
    groups |= {(b, a) for a, b in groups}
    counts: Counter[str] = Counter()
    for gt_ch, pred_ch in zip_longest(normalize_text(ground_truth), normalize_text(prediction), fillvalue=""):
        if gt_ch == pred_ch:
            continue
        if (gt_ch, pred_ch) in groups:
            label = f"{gt_ch or 'missing'}/{pred_ch or 'missing'}"
            counts[label] += 1
    return counts


def mismatch_positions(ground_truth: str, prediction: str) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for idx, (gt_ch, pred_ch) in enumerate(zip_longest(ground_truth, prediction, fillvalue="")):
        if gt_ch != pred_ch:
            mismatches.append({"index": idx, "ground_truth": gt_ch, "prediction": pred_ch})
    return mismatches


def per_token_comparison(
    gt_tokens: Iterable[GroundTruthToken],
    results: Iterable[RecognitionResult],
) -> list[dict[str, Any]]:
    gt_by_id = {(token.image, token.token_id): token for token in gt_tokens}
    gt_by_token_id = {token.token_id: token for token in gt_tokens}
    rows: list[dict[str, Any]] = []
    for result in results:
        gt = gt_by_id.get((result.image or "", result.crop_id)) or gt_by_token_id.get(result.crop_id)
        if gt is None:
            rows.append(
                {
                    "model_id": result.model_id,
                    "image": result.image,
                    "crop_id": result.crop_id,
                    "has_ground_truth": False,
                    "raw_text": result.raw_text,
                    "normalized_text": result.normalized_text,
                    "error": result.error,
                }
            )
            continue
        pred = result.raw_text
        gt_text = gt.text
        rows.append(
            {
                "model_id": result.model_id,
                "image": result.image,
                "crop_id": result.crop_id,
                "token_type": gt.token_type,
                "ground_truth": gt_text,
                "raw_text": pred,
                "normalized_text": result.normalized_text,
                "raw_exact_match": exact_match(gt_text, pred),
                "normalized_exact_match": normalized_exact_match(gt_text, result.normalized_text),
                "digit_sequence_exact_match": digit_sequence_exact_match(gt_text, pred),
                "character_accuracy": character_accuracy(gt_text, pred),
                "cer": cer(gt_text, pred),
                "numeric_char_cer": cer(numeric_char_sequence(gt_text), numeric_char_sequence(pred)),
                "punctuation_mismatch_count": punctuation_mismatch_count(gt_text, pred),
                "decimal_point_mismatch_count": decimal_point_mismatch_count(gt_text, pred),
                "thousands_separator_mismatch_count": thousands_separator_mismatch_count(gt_text, pred),
                "sign_mismatch_count": sign_mismatch_count(gt_text, pred),
                "currency_symbol_mismatch_count": currency_symbol_mismatch_count(gt_text, pred),
                "is_numeric_token": gt.token_type == "number" or is_numeric_token(gt_text),
                "confidence": result.confidence,
                "latency_ms": result.latency_ms,
                "error": result.error,
                "mismatch_positions": mismatch_positions(gt_text, pred),
            }
        )
    return rows


def summarize_model_metrics(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("has_ground_truth") is False:
            continue
        grouped[row["model_id"]].append(row)

    summaries: list[dict[str, Any]] = []
    for model_id, model_rows in grouped.items():
        total = len(model_rows)
        if total == 0:
            continue
        numeric_rows = [row for row in model_rows if row.get("is_numeric_token")]
        numeric_total = len(numeric_rows)
        predicted_ok = [row for row in model_rows if not row.get("error")]
        avg_conf = _mean(row.get("confidence") for row in predicted_ok)
        avg_latency = _mean(row.get("latency_ms") for row in predicted_ok)
        summary = {
            "model_id": model_id,
            "token_exact_match_accuracy": _rate(row.get("raw_exact_match") for row in model_rows),
            "numeric_token_raw_exact_match_accuracy": _rate(row.get("raw_exact_match") for row in numeric_rows),
            "numeric_token_normalized_exact_match_accuracy": _rate(
                row.get("normalized_exact_match") for row in numeric_rows
            ),
            "digit_sequence_exact_match_accuracy": _rate(
                row.get("digit_sequence_exact_match") for row in model_rows
            ),
            "character_accuracy": _mean(row.get("character_accuracy") for row in model_rows),
            "cer": _mean(row.get("cer") for row in model_rows),
            "numeric_char_cer": _mean(row.get("numeric_char_cer") for row in numeric_rows),
            "punctuation_mismatch_count": sum(int(row.get("punctuation_mismatch_count") or 0) for row in model_rows),
            "decimal_point_mismatch_count": sum(int(row.get("decimal_point_mismatch_count") or 0) for row in model_rows),
            "thousands_separator_mismatch_count": sum(
                int(row.get("thousands_separator_mismatch_count") or 0) for row in model_rows
            ),
            "sign_mismatch_count": sum(int(row.get("sign_mismatch_count") or 0) for row in model_rows),
            "currency_symbol_mismatch_count": sum(
                int(row.get("currency_symbol_mismatch_count") or 0) for row in model_rows
            ),
            "token_precision": len(predicted_ok) / len(predicted_ok) if predicted_ok else 0.0,
            "token_recall": len(predicted_ok) / total,
            "token_f1": _f1(len(predicted_ok) / len(predicted_ok) if predicted_ok else 0.0, len(predicted_ok) / total),
            "average_confidence": avg_conf,
            "token_latency_ms": avg_latency,
            "token_count": total,
            "numeric_token_count": numeric_total,
        }
        summaries.append(summary)
    return sorted(
        summaries,
        key=lambda row: (
            -(row["numeric_token_raw_exact_match_accuracy"] or 0.0),
            row["numeric_char_cer"] if row["numeric_char_cer"] is not None else 999.0,
            -(row["token_exact_match_accuracy"] or 0.0),
            row["token_latency_ms"] if row["token_latency_ms"] is not None else 999999.0,
        ),
    )


def confusion_matrix_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for row in rows:
        gt = row.get("ground_truth")
        pred = row.get("raw_text")
        if gt is None or pred is None:
            continue
        for pair, count in confusion_pairs(str(gt), str(pred)).items():
            counts[(row["model_id"], pair)] += count
    return [{"model_id": model_id, "confusion_pair": pair, "count": count} for (model_id, pair), count in counts.items()]


def _mean(values: Iterable[float | int | None]) -> float | None:
    vals = [float(value) for value in values if value is not None]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _rate(values: Iterable[bool | None]) -> float | None:
    vals = [bool(value) for value in values if value is not None]
    if not vals:
        return None
    return sum(1 for value in vals if value) / len(vals)


def _f1(precision: float, recall: float) -> float:
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
