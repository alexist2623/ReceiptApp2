import re
from collections import Counter

from ml.receipt_schema import canonicalize_field, is_hard_negative_for_item_grouping
from ml.span_relg.schema import DEP_FIELDS, SUMMARY_DEP_FIELDS


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def binary_edge_metrics(labels, probs, threshold=0.5):
    preds = [1 if float(prob) >= threshold else 0 for prob in probs]
    labels = [int(label) for label in labels]
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)
    out = prf(tp, fp, fn)
    out["support"] = sum(labels)
    out["predicted"] = sum(preds)
    return out


def field_edge_metrics(samples, probs_by_sample, threshold=0.5, fields=None):
    if fields is None:
        fields = DEP_FIELDS
    totals = {field: {"tp": 0, "fp": 0, "fn": 0} for field in fields}
    for sample, probs in zip(samples, probs_by_sample):
        for label, prob, field in zip(sample["pair_labels"].tolist(), probs, sample["pair_fields"]):
            field = canonicalize_field(field)
            if field not in totals:
                continue
            pred = float(prob) >= threshold
            gold = int(label) == 1
            if gold and pred:
                totals[field]["tp"] += 1
            elif not gold and pred:
                totals[field]["fp"] += 1
            elif gold and not pred:
                totals[field]["fn"] += 1
    return {field: prf(**counts) for field, counts in totals.items()}


def _edge_sets(sample, probs, threshold=0.5, dep_field="MENU_PRICE"):
    dep_field = canonicalize_field(dep_field)
    gold = set()
    pred = set()
    for idx, meta in enumerate(sample.get("pair_meta", [])):
        if canonicalize_field(meta.get("dep_field")) != dep_field:
            continue
        key = (meta.get("head_span_id"), meta.get("dep_span_id"))
        if int(sample["pair_labels"][idx].item()) == 1:
            gold.add(key)
        if float(probs[idx]) >= threshold:
            pred.add(key)
    return gold, pred


def menu_price_pair_metrics(samples, probs_by_sample, threshold=0.5):
    return item_price_pair_metrics(samples, probs_by_sample, threshold)


def item_price_pair_metrics(samples, probs_by_sample, threshold=0.5):
    tp = fp = fn = 0
    for sample, probs in zip(samples, probs_by_sample):
        gold, pred = _edge_sets(sample, probs, threshold, "ITEM_PRICE")
        tp += len(gold & pred)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return prf(tp, fp, fn)


def fields_pair_metrics(samples, probs_by_sample, fields, threshold=0.5):
    fields = {canonicalize_field(field) for field in fields}
    tp = fp = fn = 0
    for sample, probs in zip(samples, probs_by_sample):
        for idx, meta in enumerate(sample.get("pair_meta", [])):
            if canonicalize_field(meta.get("dep_field")) not in fields:
                continue
            pred = float(probs[idx]) >= threshold
            gold = int(sample["pair_labels"][idx].item()) == 1
            if gold and pred:
                tp += 1
            elif not gold and pred:
                fp += 1
            elif gold and not pred:
                fn += 1
    return prf(tp, fp, fn)


def summary_amount_pair_metrics(samples, probs_by_sample, threshold=0.5):
    return fields_pair_metrics(samples, probs_by_sample, SUMMARY_DEP_FIELDS, threshold)


def normalized_text(value):
    return re.sub(r"\s+", " ", str(value).strip().lower())


def hard_negative_false_positives(samples, probs_by_sample, threshold=0.5):
    count = 0
    store_count = 0
    total_subtotal_count = 0
    examples = []
    for sample, probs in zip(samples, probs_by_sample):
        for idx, meta in enumerate(sample.get("pair_meta", [])):
            field = canonicalize_field(meta.get("dep_field", ""))
            gold = int(sample["pair_labels"][idx].item()) == 1
            if (not gold) and is_hard_negative_for_item_grouping(field) and float(probs[idx]) >= threshold:
                count += 1
                if field.startswith("STORE_"):
                    store_count += 1
                if field.startswith("TOTAL_") or field.startswith("SUBTOTAL_") or field in {"TOTAL_PRICE", "SUBTOTAL_PRICE", "TAX_PRICE"}:
                    total_subtotal_count += 1
                if len(examples) < 50:
                    examples.append({"data_id": sample.get("data_id"), **meta, "dep_field": field, "prob": float(probs[idx])})
    return count, store_count, total_subtotal_count, examples


def dependent_collision_count(samples, probs_by_sample, threshold=0.5):
    total = 0
    for sample, probs in zip(samples, probs_by_sample):
        selected = Counter()
        for idx, meta in enumerate(sample.get("pair_meta", [])):
            if float(probs[idx]) >= threshold:
                selected[meta.get("dep_span_id")] += 1
        total += sum(count - 1 for count in selected.values() if count > 1)
    return total


def aggregate_metrics(samples, probs_by_sample, threshold=0.5):
    labels = []
    probs = []
    for sample, sample_probs in zip(samples, probs_by_sample):
        labels.extend(sample["pair_labels"].tolist())
        probs.extend(sample_probs)
    hard_fp, store_fp, total_subtotal_fp, hard_examples = hard_negative_false_positives(samples, probs_by_sample, threshold)
    item_price = item_price_pair_metrics(samples, probs_by_sample, threshold)
    summary_amount = summary_amount_pair_metrics(samples, probs_by_sample, threshold)
    return {
        "edge": binary_edge_metrics(labels, probs, threshold),
        "field_edges": field_edge_metrics(samples, probs_by_sample, threshold),
        "item_price_pair": item_price,
        "menu_price_pair": item_price,
        "summary_amount_pair": summary_amount,
        "hard_negative_false_positive_count": hard_fp,
        "store_false_positive_count": store_fp,
        "total_subtotal_false_positive_count": total_subtotal_fp,
        "hard_negative_false_positive_examples": hard_examples,
        "dependent_collision_count": dependent_collision_count(samples, probs_by_sample, threshold),
    }
