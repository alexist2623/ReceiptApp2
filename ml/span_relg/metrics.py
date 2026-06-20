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
    tax_count = 0
    summary_count = 0
    payment_count = 0
    by_field = Counter()
    by_head_dep = Counter()
    item_to_field = Counter()
    wrong_price_field_count = 0
    examples = []
    for sample, probs in zip(samples, probs_by_sample):
        for idx, meta in enumerate(sample.get("pair_meta", [])):
            field = canonicalize_field(meta.get("dep_field", ""))
            head_field = canonicalize_field(meta.get("head_field", ""))
            gold = int(sample["pair_labels"][idx].item()) == 1
            if (not gold) and is_hard_negative_for_item_grouping(field) and float(probs[idx]) >= threshold:
                count += 1
                by_field[field] += 1
                by_head_dep[f"{head_field}->{field}"] += 1
                if field.startswith("STORE_"):
                    store_count += 1
                if field.startswith("TOTAL_") or field.startswith("SUBTOTAL_") or field in {"TOTAL_PRICE", "SUBTOTAL_PRICE", "TAX_PRICE"}:
                    total_subtotal_count += 1
                if field.startswith("TAX_") or field == "TAX_PRICE":
                    tax_count += 1
                if field.startswith(("TOTAL_", "SUBTOTAL_", "DISCOUNT_", "SERVICE_", "TIP_")) or field in {
                    "TOTAL_PRICE",
                    "SUBTOTAL_PRICE",
                    "DISCOUNT_PRICE",
                    "SERVICE_PRICE",
                    "TIP_PRICE",
                }:
                    summary_count += 1
                if field.startswith(("PAYMENT_", "CARD_", "CASH_", "CHANGE_")):
                    payment_count += 1
                if head_field == "ITEM_NAME":
                    item_to_field[field] += 1
                    if field in {"TOTAL_PRICE", "SUBTOTAL_PRICE", "TAX_PRICE", "TIP_PRICE", "PAYMENT_INFO"}:
                        wrong_price_field_count += 1
                if len(examples) < 50:
                    examples.append({"data_id": sample.get("data_id"), **meta, "dep_field": field, "prob": float(probs[idx])})
    return {
        "count": count,
        "store_count": store_count,
        "total_subtotal_count": total_subtotal_count,
        "tax_count": tax_count,
        "summary_count": summary_count,
        "payment_count": payment_count,
        "by_field": dict(by_field.most_common()),
        "by_head_dep": dict(by_head_dep.most_common()),
        "item_to_field": dict(item_to_field.most_common()),
        "wrong_price_field_count": wrong_price_field_count,
        "examples": examples,
    }


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
    hard_fp = hard_negative_false_positives(samples, probs_by_sample, threshold)
    item_price = item_price_pair_metrics(samples, probs_by_sample, threshold)
    summary_amount = summary_amount_pair_metrics(samples, probs_by_sample, threshold)
    return {
        "edge": binary_edge_metrics(labels, probs, threshold),
        "field_edges": field_edge_metrics(samples, probs_by_sample, threshold),
        "item_price_pair": item_price,
        "menu_price_pair": item_price,
        "summary_amount_pair": summary_amount,
        "hard_negative_false_positive_count": hard_fp["count"],
        "store_false_positive_count": hard_fp["store_count"],
        "total_subtotal_false_positive_count": hard_fp["total_subtotal_count"],
        "tax_false_positive_count": hard_fp["tax_count"],
        "summary_false_positive_count": hard_fp["summary_count"],
        "payment_false_positive_count": hard_fp["payment_count"],
        "hard_negative_false_positive_by_field": hard_fp["by_field"],
        "hard_negative_false_positive_by_head_dep": hard_fp["by_head_dep"],
        "item_to_total_price_false_positive_count": hard_fp["item_to_field"].get("TOTAL_PRICE", 0),
        "item_to_subtotal_price_false_positive_count": hard_fp["item_to_field"].get("SUBTOTAL_PRICE", 0),
        "item_to_tax_price_false_positive_count": hard_fp["item_to_field"].get("TAX_PRICE", 0),
        "item_to_tip_price_false_positive_count": hard_fp["item_to_field"].get("TIP_PRICE", 0),
        "item_to_payment_info_false_positive_count": hard_fp["item_to_field"].get("PAYMENT_INFO", 0),
        "wrong_price_field_count": hard_fp["wrong_price_field_count"],
        "hard_negative_false_positive_examples": hard_fp["examples"],
        "dependent_collision_count": dependent_collision_count(samples, probs_by_sample, threshold),
    }
