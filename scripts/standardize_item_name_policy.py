import argparse
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import canonicalize_field, label_to_field, normalize_span_text


POLICY_VERSION = "item_name_core_only_v2026_06_26"
RELATION_LIST_KEYS = ("relations", "item_relations", "summary_relations", "payment_relations", "rel_g_edges")
PRICE_FIELDS = {
    "ITEM_PRICE",
    "ITEM_UNIT_PRICE",
    "ITEM_DISCOUNT",
    "SUBTOTAL_PRICE",
    "TAX_PRICE",
    "DISCOUNT_PRICE",
    "SERVICE_PRICE",
    "TOTAL_PRICE",
    "CASH_PRICE",
    "CHANGE_PRICE",
    "CARD_PRICE",
    "TIP_PRICE",
}
OPTION_SPAN_PHRASES = {
    "TAKE OUT",
    "TAKEOUT",
    "TO GO",
    "TOGO",
    "DINE IN",
    "EAT IN",
    "STAY",
    "HERE",
    "NO ICE",
    "LESS ICE",
    "EXTRA ICE",
    "NO SUGAR",
    "LESS SUGAR",
    "EXTRA SHOT",
    "DECAF",
    "OAT MILK",
    "SOY MILK",
    "ALMOND MILK",
    "LARGE",
    "MEDIUM",
    "SMALL",
    "REGULAR",
}
TAX_FLAG_TOKENS = {"T", "TX", "TAX", "G", "GP", "GST", "PST", "HST", "F", "N"}
ITEM_CATEGORY_HEADER_TOKENS = {
    "BEVERAGE",
    "BEVERAGES",
    "DRINK",
    "DRINKS",
    "FOOD",
    "GROCERY",
    "GROCERIES",
    "MEAT",
    "PRODUCE",
    "DAIRY",
    "DELI",
    "BAKERY",
    "SNACK",
    "SNACKS",
}
ITEM_ETC_PLACEHOLDER_TOKENS = {
    "OPENFOOD",
    "OPENFOODS",
    "OPENITEM",
    "OPENITEMS",
    "MISCITEM",
    "MISCITEMS",
    "MISC",
    "COURSEWKDAY",
}
LOYALTY_OR_MEMBERSHIP_TOKENS = {
    "AIRMILES",
    "AIRMILE",
    "AIRMILESCOLLECTOR",
    "LOYALTY",
    "REWARDS",
    "REWARD",
    "MEMBER",
    "MEMBERSHIP",
}
SERVICE_NAME_TOKENS = {
    "COPERTO",
    "COVER",
    "COVERCHARGE",
    "SERVICECHARGE",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Standardize custom/WildReceipt labeled_v2_1 JSON files so ITEM_NAME "
            "contains only the core product/menu name."
        )
    )
    parser.add_argument(
        "--input_dir",
        action="append",
        default=[],
        help="Root directory to scan. Can be passed multiple times.",
    )
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--backup_dir", default="outputs/item_name_policy_backups")
    parser.add_argument("--report_out", default="outputs/item_name_policy_standardization_report.json")
    parser.add_argument("--max_files", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true", help="Only report planned changes.")
    parser.add_argument("--apply", action="store_true", help="Overwrite changed JSON files after backing them up.")
    parser.add_argument("--debug_examples", type=int, default=100)
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def label_for_field(field, begin=True):
    field = canonicalize_field(field)
    if field == "O":
        return "O"
    return f"{'B' if begin else 'I'}-{field}"


def word_index(word, fallback):
    for key in ("word_idx", "globalWordIndex"):
        value = word.get(key) if isinstance(word, dict) else None
        if value is not None:
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return int(fallback)


def parse_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def box_from_word(word):
    box = parse_box(word.get("box") if isinstance(word, dict) else None)
    if box is not None:
        return box
    points = word.get("cornerPoints") if isinstance(word, dict) else None
    if not isinstance(points, list) or not points:
        return None
    xs = []
    ys = []
    for point in points:
        if isinstance(point, list) and len(point) >= 2:
            xs.append(point[0])
            ys.append(point[1])
    if not xs or not ys:
        return None
    return parse_box([min(xs), min(ys), max(xs), max(ys)])


def union_boxes(boxes):
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def normalize_box(box, width, height):
    if box is None or not width or not height:
        return None
    return [
        max(0, min(1000, int(1000 * box[0] / width))),
        max(0, min(1000, int(1000 * box[1] / height))),
        max(0, min(1000, int(1000 * box[2] / width))),
        max(0, min(1000, int(1000 * box[3] / height))),
    ]


def word_field(word, labels, idx):
    if isinstance(word, dict) and word.get("field") is not None:
        return canonicalize_field(word.get("field"))
    if isinstance(word, dict) and word.get("label") is not None:
        return label_to_field(word.get("label"))
    if isinstance(labels, list) and idx < len(labels):
        return label_to_field(labels[idx])
    return "O"


def word_text(word):
    return str(word.get("text", "") if isinstance(word, dict) else "").strip()


def norm_text(text):
    return re.sub(r"\s+", " ", str(text or "").strip()).upper()


def compact_text(text):
    return re.sub(r"[^A-Z0-9]", "", norm_text(text))


def is_price_like(text):
    value = norm_text(text)
    if not value:
        return False
    value = value.replace("$", "").strip()
    value = re.sub(r"^(?:CAD|USD|PHP|AUD|NZD|RM|RS|R|P)\s*", "", value).strip()
    value = value.strip("()")
    if value.endswith(("+", "-")):
        value = value[:-1]
    if not re.search(r"[.,]\d{2}$", value):
        return False
    return bool(re.fullmatch(r"-?\d{1,6}(?:,\d{3})*(?:[.,]\d{2})", value))


def is_qty_like(text):
    value = norm_text(text)
    return bool(
        re.fullmatch(r"X\s*\d{1,3}(?:[.,]\d+)?", value)
        or re.fullmatch(r"\d{1,3}(?:[.,]\d+)?\s*X", value)
        or value in {"QTY", "QUANTITY"}
    )


def is_tax_flag(text):
    value = norm_text(text).strip("*")
    return value in TAX_FLAG_TOKENS


def is_item_category_header_like(text):
    compact = compact_text(text)
    if compact in ITEM_CATEGORY_HEADER_TOKENS:
        return True
    without_leading_code = re.sub(r"^\d+", "", compact)
    if without_leading_code in ITEM_CATEGORY_HEADER_TOKENS:
        return True
    # WildReceipt often appends tax/category suffixes to department headers, e.g. GROCERYFT2.
    return bool(
        re.fullmatch(r"(?:BEVERAGE|BEVERAGES|GROCERY|GROCERIES|MEAT|FOOD)(?:F|FT|TX|T)?\d{0,3}", compact)
        or re.fullmatch(r"(?:GROCERY|GROCERIES)(?:NONFOOD|HO|HOME)?", without_leading_code)
    )


def is_item_placeholder_like(text):
    compact = compact_text(text)
    return compact in ITEM_ETC_PLACEHOLDER_TOKENS or compact.startswith("OPENFOOD") or compact.startswith("OPENITEM")


def is_loyalty_or_membership_like(text):
    compact = compact_text(text)
    if compact in LOYALTY_OR_MEMBERSHIP_TOKENS:
        return True
    return compact.startswith("AIRMILES") or compact.startswith("BONUSAIRMILES")


def is_service_name_like(text):
    compact = compact_text(text)
    if compact in SERVICE_NAME_TOKENS:
        return True
    return bool(re.fullmatch(r"\d+X?COPERTO", compact))


def is_code_like(text):
    value = norm_text(text)
    if not value or is_price_like(value) or is_qty_like(value) or is_size_or_pack_like(value):
        return False
    if "@" in value:
        return False
    compact = re.sub(r"[^A-Z0-9]", "", value)
    if len(compact) < 4:
        return False
    has_digit = any(ch.isdigit() for ch in compact)
    if not has_digit:
        return False
    if compact.isdigit() and len(compact) >= 4:
        return True
    if re.fullmatch(r"\d{3,}[A-Z0-9]{0,8}", compact):
        return True
    if re.fullmatch(r"[A-Z]{1,5}\d{3,}[A-Z0-9]{0,8}", compact):
        return True
    if re.search(r"[/-]\d{4,}", value):
        return True
    return False


def is_strong_code_like_inside_item_name(text):
    value = norm_text(text)
    if not value or is_price_like(value) or is_qty_like(value) or is_size_or_pack_like(value):
        return False
    compact = re.sub(r"[^A-Z0-9]", "", value)
    if compact.isdigit() and len(compact) >= 4:
        return True
    if re.fullmatch(r"(?:TPD|SKU|PLU)[/-]?\d{3,}[A-Z0-9]*", value):
        return True
    if re.fullmatch(r"#?\d{4,}[A-Z]{0,4}", value):
        return True
    return False


def is_size_or_pack_like(text):
    value = norm_text(text).replace(" ", "")
    if not value:
        return False
    unit = r"(?:OZ|0Z|ML|M1|L|G|KG|LB|LBS|CT|PK|PKG|PACK|PC|PCS|EA)"
    if re.fullmatch(rf"\d+(?:[.,]\d+)?{unit}", value):
        return True
    if re.fullmatch(rf"\d+X\d+(?:[.,]\d+)?{unit}", value):
        return True
    return False


def image_size(payload):
    width = payload.get("image_width") or payload.get("width")
    height = payload.get("image_height") or payload.get("height")
    image = payload.get("image")
    if isinstance(image, dict):
        width = width or image.get("width")
        height = height or image.get("height")
    try:
        return int(width or 0), int(height or 0)
    except (TypeError, ValueError):
        return 0, 0


def assign_row_keys(words):
    explicit = []
    for idx, word in enumerate(words):
        if not isinstance(word, dict):
            explicit.append(None)
            continue
        value = word.get("line_id", word.get("lineId", word.get("row_id", word.get("rowId"))))
        explicit.append(value)
    if any(value is not None for value in explicit):
        return [f"line:{value}" if value is not None else f"idx:{idx}" for idx, value in enumerate(explicit)]

    rows = []
    entries = []
    heights = []
    for idx, word in enumerate(words):
        box = box_from_word(word) if isinstance(word, dict) else None
        if box is None:
            entries.append((idx, None, None))
            continue
        center_y = (box[1] + box[3]) / 2.0
        height = max(1, box[3] - box[1])
        heights.append(height)
        entries.append((idx, center_y, height))
    if not heights:
        return [f"idx:{idx}" for idx in range(len(words))]
    heights = sorted(heights)
    median_height = heights[len(heights) // 2]
    tolerance = max(8.0, median_height * 0.65)
    row_keys = [None] * len(words)
    for idx, center_y, _height in sorted([entry for entry in entries if entry[1] is not None], key=lambda item: item[1]):
        assigned = None
        for row_idx, row in enumerate(rows):
            if abs(center_y - row["center"]) <= tolerance:
                assigned = row_idx
                row["values"].append(center_y)
                row["center"] = sum(row["values"]) / len(row["values"])
                break
        if assigned is None:
            assigned = len(rows)
            rows.append({"center": center_y, "values": [center_y]})
        row_keys[idx] = f"row:{assigned:04d}"
    for idx, key in enumerate(row_keys):
        if key is None:
            row_keys[idx] = f"idx:{idx}"
    return row_keys


def original_item_name_spans(words, fields, row_keys):
    spans = []
    current = []
    current_row = None
    for idx, field in enumerate(fields):
        if field == "ITEM_NAME" and (not current or row_keys[idx] == current_row):
            current.append(idx)
            current_row = row_keys[idx]
            continue
        if current:
            spans.append(current)
            current = []
            current_row = None
        if field == "ITEM_NAME":
            current = [idx]
            current_row = row_keys[idx]
    if current:
        spans.append(current)
    return spans


def standardize_fields(words, original_fields, row_keys):
    new_fields = list(original_fields)
    changes = []
    row_to_indices = defaultdict(list)
    for idx, row_key in enumerate(row_keys):
        row_to_indices[row_key].append(idx)

    option_indices = set()
    for span in original_item_name_spans(words, original_fields, row_keys):
        span_text = norm_text(" ".join(word_text(words[idx]) for idx in span))
        if span_text in OPTION_SPAN_PHRASES:
            option_indices.update(span)

    for idx, (word, old_field) in enumerate(zip(words, original_fields)):
        text = word_text(word)
        new_field = old_field
        reason = None
        if old_field == "ITEM_NAME":
            if idx in option_indices:
                new_field = "ITEM_OPTION"
                reason = "item_name_span_is_option_phrase"
            elif is_service_name_like(text):
                new_field = "SERVICE_NAME"
                reason = "service_name_like_token_inside_item_name"
            elif is_loyalty_or_membership_like(text):
                new_field = "PAYMENT_INFO"
                reason = "loyalty_or_membership_token_inside_item_name"
            elif is_item_placeholder_like(text):
                new_field = "ITEM_ETC"
                reason = "placeholder_token_inside_item_name"
            elif is_item_category_header_like(text):
                new_field = "ITEM_CATEGORY"
                reason = "category_header_token_inside_item_name"
            elif is_size_or_pack_like(text):
                new_field = "ITEM_OPTION"
                reason = "size_or_pack_token_inside_item_name"
            elif is_price_like(text):
                new_field = "ITEM_PRICE"
                reason = "price_like_token_inside_item_name"
            elif is_qty_like(text):
                new_field = "ITEM_QTY"
                reason = "quantity_like_token_inside_item_name"
            elif is_tax_flag(text):
                new_field = "ITEM_TAX_FLAG"
                reason = "tax_flag_token_inside_item_name"
            elif is_strong_code_like_inside_item_name(text):
                new_field = "ITEM_CODE"
                reason = "code_like_token_inside_item_name"
        if new_field != old_field:
            new_fields[idx] = new_field
            changes.append(change_entry(idx, text, old_field, new_field, reason))

    # Row context pass: WildReceipt often leaves SKU/unit-price tokens as O even when the
    # row already contains ITEM_NAME + ITEM_PRICE. Keep this conservative.
    for row_key, indices in row_to_indices.items():
        row_fields = {new_fields[idx] for idx in indices}
        if "ITEM_NAME" not in row_fields or "ITEM_PRICE" not in row_fields:
            continue
        has_qty = "ITEM_QTY" in row_fields or any(is_qty_like(word_text(words[idx])) for idx in indices)
        for idx in indices:
            if new_fields[idx] != "O":
                continue
            text = word_text(words[idx])
            new_field = None
            reason = None
            if is_code_like(text):
                new_field = "ITEM_CODE"
                reason = "code_like_o_token_in_item_row"
            elif is_size_or_pack_like(text):
                new_field = "ITEM_OPTION"
                reason = "size_or_pack_o_token_in_item_row"
            elif is_qty_like(text):
                new_field = "ITEM_QTY"
                reason = "quantity_like_o_token_in_item_row"
            elif is_price_like(text) and has_qty:
                new_field = "ITEM_UNIT_PRICE"
                reason = "price_like_o_token_in_item_row_with_qty"
            if new_field:
                new_fields[idx] = new_field
                changes.append(change_entry(idx, text, "O", new_field, reason))
    return new_fields, changes


def change_entry(idx, text, old_field, new_field, reason):
    return {
        "word_idx": idx,
        "text": text,
        "old_field": old_field,
        "new_field": new_field,
        "old_label": label_for_field(old_field, begin=True),
        "new_label": label_for_field(new_field, begin=True),
        "reason": reason,
    }


def recompute_bio_labels(fields, row_keys):
    labels = []
    previous_field = "O"
    previous_row = None
    for field, row_key in zip(fields, row_keys):
        field = canonicalize_field(field)
        if field == "O":
            labels.append("O")
        else:
            begin = field != previous_field or row_key != previous_row
            labels.append(label_for_field(field, begin=begin))
        previous_field = field
        previous_row = row_key
    return labels


def build_spans(words, labels, row_keys, width, height):
    spans = []
    current = None
    for idx, (word, label, row_key) in enumerate(zip(words, labels, row_keys)):
        field = label_to_field(label)
        if field == "O":
            if current is not None:
                spans.append(finalize_span(current, words, width, height))
                current = None
            continue
        should_start = label.startswith("B-") or current is None or current["field"] != field or current["row_key"] != row_key
        if should_start:
            if current is not None:
                spans.append(finalize_span(current, words, width, height))
            current = {
                "field": field,
                "row_key": row_key,
                "word_indices": [idx],
                "raw_labels": [label],
                "first_word_idx": idx,
                "span_id": len(spans),
            }
        else:
            current["word_indices"].append(idx)
            current["raw_labels"].append(label)
    if current is not None:
        spans.append(finalize_span(current, words, width, height))
    for span_id, span in enumerate(spans):
        span["span_id"] = span_id
        span.pop("row_key", None)
    return spans


def finalize_span(span, words, width, height):
    indices = span["word_indices"]
    text = " ".join(word_text(words[idx]) for idx in indices).strip()
    boxes = [box_from_word(words[idx]) for idx in indices]
    box = union_boxes(boxes)
    span["text"] = text
    span["normalized_text"] = normalize_span_text(span["field"], text)
    span["box"] = box
    span["normalized_box"] = normalize_box(box, width, height)
    span["word_ids"] = [words[idx].get("wordId") for idx in indices if isinstance(words[idx], dict) and words[idx].get("wordId")]
    return span


def normalize_indices(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        try:
            return [int(value)]
        except ValueError:
            return []
    if isinstance(value, list):
        out = []
        for item in value:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                continue
        return out
    return []


def side_indices(relation, side):
    if side == "head":
        return normalize_indices(relation.get("head_word_indices") or relation.get("head_word_idx"))
    return normalize_indices(
        relation.get("tail_word_indices")
        or relation.get("dep_word_indices")
        or relation.get("dependent_word_indices")
        or relation.get("tail_word_idx")
        or relation.get("dep_word_idx")
        or relation.get("dependent_word_idx")
    )


def set_side_indices(relation, side, indices):
    if side == "head":
        relation["head_word_indices"] = indices
        relation["head_word_idx"] = indices[0] if len(indices) == 1 and "head_word_idx" in relation else relation.get("head_word_idx")
        return
    if "dep_word_indices" in relation or "dep_field" in relation or "dep_text" in relation or "head_span_id" in relation:
        relation["dep_word_indices"] = indices
        if "dep_word_idx" in relation:
            relation["dep_word_idx"] = indices[0] if len(indices) == 1 else relation.get("dep_word_idx")
    else:
        relation["tail_word_indices"] = indices
        if "tail_word_idx" in relation:
            relation["tail_word_idx"] = indices[0] if len(indices) == 1 else relation.get("tail_word_idx")


def side_field(relation, side):
    if side == "head":
        return canonicalize_field(relation.get("head_field"))
    return canonicalize_field(relation.get("tail_field") or relation.get("dep_field") or relation.get("dependent_field"))


def set_side_text(relation, side, text):
    if side == "head":
        relation["head_text"] = text
        return
    if "dep_text" in relation or "dep_field" in relation or "head_span_id" in relation:
        relation["dep_text"] = text
    else:
        relation["tail_text"] = text


def set_side_box(relation, side, box):
    if side == "head":
        relation["head_box"] = box
        return
    if "dep_field" in relation or "dep_text" in relation or "head_span_id" in relation:
        relation["dep_box"] = box
    else:
        relation["tail_box"] = box


def text_for_indices(words, indices):
    return " ".join(word_text(words[idx]) for idx in indices if 0 <= idx < len(words)).strip()


def box_for_indices(words, indices):
    return union_boxes([box_from_word(words[idx]) for idx in indices if 0 <= idx < len(words)])


def update_relations(payload, words, fields):
    relation_updates = Counter()
    relation_warnings = []
    for key in RELATION_LIST_KEYS:
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        kept_relations = []
        for ordinal, relation in enumerate(values):
            if not isinstance(relation, dict):
                continue
            relation_id = relation.get("relation_id") or relation.get("edge_id") or f"{key}_{ordinal:06d}"
            if "relation_id" not in relation and key != "rel_g_edges":
                relation["relation_id"] = relation_id
            if "edge_id" not in relation and key == "rel_g_edges":
                relation["edge_id"] = relation_id
            drop_relation = False
            for side in ("head", "tail"):
                indices = side_indices(relation, side)
                field = side_field(relation, side)
                if not indices:
                    relation_warnings.append({"relation_id": relation_id, "list": key, "side": side, "reason": "missing_indices"})
                    drop_relation = True
                    continue
                valid = [idx for idx in indices if 0 <= idx < len(words)]
                if len(valid) != len(indices):
                    relation_warnings.append({"relation_id": relation_id, "list": key, "side": side, "reason": "index_out_of_range"})
                if not valid:
                    drop_relation = True
                    continue
                matching = [idx for idx in valid if fields[idx] == field]
                if field != "O":
                    if not matching:
                        relation_warnings.append(
                            {
                                "relation_id": relation_id,
                                "list": key,
                                "side": side,
                                "reason": "no_indices_matching_declared_field_after_policy",
                                "declared_field": field,
                                "indices": valid,
                                "actual_fields": [fields[idx] for idx in valid],
                            }
                        )
                        drop_relation = True
                        continue
                    if matching != valid:
                        relation_updates["side_indices_filtered"] += 1
                        set_side_indices(relation, side, matching)
                        valid = matching
                text = text_for_indices(words, valid)
                box = box_for_indices(words, valid)
                set_side_text(relation, side, text)
                set_side_box(relation, side, box)
            relation_updates[f"{key}_seen"] += 1
            if drop_relation:
                relation_updates[f"{key}_removed_field_mismatch"] += 1
            else:
                kept_relations.append(relation)
        if len(kept_relations) != len(values):
            payload[key] = kept_relations
    return relation_updates, relation_warnings


def rebuild_word_relation_links(payload, words):
    for word in words:
        if not isinstance(word, dict):
            continue
        word["relation_ids_as_head"] = []
        word["relation_ids_as_tail"] = []
        word["rel_g_edge_ids_as_head"] = []
        word["rel_g_edge_ids_as_tail"] = []

    def add_links(indices, key, relation_id):
        for idx in indices:
            if 0 <= idx < len(words) and isinstance(words[idx], dict):
                values = words[idx].setdefault(key, [])
                if relation_id not in values:
                    values.append(relation_id)

    for key in ("relations", "item_relations", "summary_relations", "payment_relations"):
        values = payload.get(key)
        if not isinstance(values, list):
            continue
        for ordinal, relation in enumerate(values):
            if not isinstance(relation, dict):
                continue
            relation_id = relation.get("relation_id") or f"{key}_{ordinal:06d}"
            add_links(side_indices(relation, "head"), "relation_ids_as_head", relation_id)
            add_links(side_indices(relation, "tail"), "relation_ids_as_tail", relation_id)

    values = payload.get("rel_g_edges")
    if isinstance(values, list):
        for ordinal, relation in enumerate(values):
            if not isinstance(relation, dict):
                continue
            edge_id = relation.get("edge_id") or relation.get("relation_id") or f"rel_g_{ordinal:06d}"
            add_links(side_indices(relation, "head"), "rel_g_edge_ids_as_head", edge_id)
            add_links(side_indices(relation, "tail"), "rel_g_edge_ids_as_tail", edge_id)


def update_item_groups(payload, words, fields):
    groups = payload.get("item_groups")
    if not isinstance(groups, list):
        return Counter()
    updates = Counter()
    for group in groups:
        if not isinstance(group, dict):
            continue
        head_indices = normalize_indices(group.get("head_word_indices") or group.get("item_name_word_indices"))
        name_indices = [idx for idx in head_indices if 0 <= idx < len(fields) and fields[idx] == "ITEM_NAME"]
        if name_indices and name_indices != head_indices:
            group["head_word_indices"] = name_indices
            group["item_name_word_indices"] = name_indices
            group["head_text"] = text_for_indices(words, name_indices)
            group["expected_item_name"] = group["head_text"]
            group["description"] = group["head_text"]
            updates["item_group_head_filtered"] += 1
        fields_map = defaultdict(list)
        for idx, field in enumerate(fields):
            if field != "O":
                fields_map[field].append(idx)
        if isinstance(group.get("fields"), dict):
            for field in ("ITEM_NAME", "ITEM_CODE", "ITEM_SKU", "ITEM_QTY", "ITEM_UNIT_PRICE", "ITEM_PRICE", "ITEM_TAX_FLAG"):
                old_indices = normalize_indices(group["fields"].get(field))
                if not old_indices:
                    continue
                kept = [idx for idx in old_indices if 0 <= idx < len(fields) and fields[idx] == field]
                if kept != old_indices:
                    group["fields"][field] = kept
                    updates["item_group_field_filtered"] += 1
        linked_fields = group.get("linked_fields")
        if isinstance(linked_fields, dict):
            for field, indices in list(linked_fields.items()):
                canonical = canonicalize_field(field)
                old_indices = normalize_indices(indices)
                kept = [idx for idx in old_indices if 0 <= idx < len(fields) and fields[idx] == canonical]
                if kept != old_indices:
                    linked_fields[field] = kept
                    updates["item_group_linked_field_filtered"] += 1
        linked_texts = group.get("linked_texts")
        if isinstance(linked_texts, dict) and isinstance(group.get("linked_fields"), dict):
            for field, indices in group["linked_fields"].items():
                linked_texts[field] = text_for_indices(words, normalize_indices(indices))
    return updates


def update_counts(payload, words):
    labels = [word.get("label", "O") if isinstance(word, dict) else "O" for word in words]
    fields = [label_to_field(label) for label in labels]
    payload["labels"] = labels
    payload["label_counts"] = dict(Counter(labels))
    payload["field_counts"] = dict(Counter(fields))
    if isinstance(payload.get("relations"), list):
        payload["relation_counts"] = dict(Counter(str(rel.get("relation_type", "unknown")) for rel in payload["relations"] if isinstance(rel, dict)))


def cleanup_semantic_item_category_metadata(words, fields, payload):
    updates = Counter()
    word_keys = (
        "semantic_item_category",
        "semantic_item_category_confidence",
        "semantic_item_category_rule",
    )
    for idx, word in enumerate(words):
        if not isinstance(word, dict) or fields[idx] == "ITEM_NAME":
            continue
        removed = False
        for key in word_keys:
            if key in word:
                word.pop(key, None)
                removed = True
        if removed:
            updates["word_semantic_category_removed"] += 1

    for relation_key in RELATION_LIST_KEYS:
        relations = payload.get(relation_key)
        if not isinstance(relations, list):
            continue
        for relation in relations:
            if not isinstance(relation, dict):
                continue
            if canonicalize_field(relation.get("head_field")) == "ITEM_NAME":
                continue
            for key in (
                "head_semantic_item_category",
                "head_semantic_item_category_confidence",
                "head_semantic_item_category_rule",
            ):
                if key in relation:
                    relation.pop(key, None)
                    updates["relation_semantic_category_removed"] += 1
    return updates


def add_policy_metadata(payload, report):
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    policy = payload.get("label_policy")
    if not isinstance(policy, dict):
        policy = {}
    policy["item_name_policy"] = POLICY_VERSION
    policy["item_name_definition"] = "ITEM_NAME contains only the core product/menu name."
    policy["excluded_from_item_name"] = ["quantity", "price", "unit price", "tax flag", "SKU/code/PLU", "standalone option/modifier"]
    payload["label_policy"] = policy
    payload["item_name_policy_repair"] = {
        "policy_version": POLICY_VERSION,
        "updated_at": now,
        "changed_word_count": len(report["word_changes"]),
        "transition_counts": {f"{old}->{new}": count for (old, new), count in report["transition_counts"].items()},
    }


def process_payload(payload):
    words = payload.get("words")
    if not isinstance(words, list):
        raise ValueError("payload missing words list")
    labels = payload.get("labels")
    width, height = image_size(payload)
    row_keys = assign_row_keys(words)
    original_fields = [word_field(word, labels, idx) for idx, word in enumerate(words)]
    new_fields, word_changes = standardize_fields(words, original_fields, row_keys)
    new_labels = recompute_bio_labels(new_fields, row_keys)
    for idx, word in enumerate(words):
        if not isinstance(word, dict):
            continue
        word["field"] = new_fields[idx]
        word["label"] = new_labels[idx]
    payload["spans"] = build_spans(words, new_labels, row_keys, width, height)
    relation_updates, relation_warnings = update_relations(payload, words, new_fields)
    item_group_updates = update_item_groups(payload, words, new_fields)
    semantic_metadata_updates = cleanup_semantic_item_category_metadata(words, new_fields, payload)
    rebuild_word_relation_links(payload, words)
    update_counts(payload, words)
    transition_counts = Counter((change["old_field"], change["new_field"]) for change in word_changes)
    material_relation_updates = Counter(
        {key: value for key, value in relation_updates.items() if not str(key).endswith("_seen")}
    )
    report = {
        "changed": bool(word_changes or material_relation_updates or item_group_updates or semantic_metadata_updates),
        "word_changes": word_changes,
        "transition_counts": transition_counts,
        "relation_updates": dict(relation_updates),
        "relation_warnings": relation_warnings,
        "item_group_updates": dict(item_group_updates),
        "semantic_metadata_updates": dict(semantic_metadata_updates),
        "num_words": len(words),
        "num_spans": len(payload.get("spans") or []),
    }
    add_policy_metadata(payload, report)
    return payload, report


def collect_files(input_dirs, exclude_dir_name, max_files=None):
    paths = []
    for root in input_dirs:
        root_path = Path(root)
        if not root_path.exists():
            fail(f"input_dir not found: {root_path}")
        for path in sorted(root_path.rglob("*_labeled_v2_1.json")):
            if exclude_dir_name and exclude_dir_name in path.parts:
                continue
            paths.append(path)
    deduped = []
    seen = set()
    for path in paths:
        resolved = str(path.resolve())
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(path)
    if max_files is not None:
        deduped = deduped[: max(0, int(max_files))]
    return deduped


def backup_file(path, input_dirs, backup_dir):
    path = Path(path)
    backup_dir = Path(backup_dir)
    rel = None
    for root in input_dirs:
        root = Path(root)
        try:
            rel = path.resolve().relative_to(root.resolve())
            root_name = root.name or "root"
            break
        except ValueError:
            continue
    if rel is None:
        root_name = "absolute"
        rel = Path(str(path).lstrip("/").replace(":", ""))
    target = backup_dir / root_name / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    return target


def main():
    args = parse_args()
    if not args.input_dir:
        fail("Pass at least one --input_dir.")
    if args.apply and args.dry_run:
        fail("Use either --apply or --dry_run, not both.")
    if not args.apply and not args.dry_run:
        print("No mode passed; defaulting to --dry_run.")
        args.dry_run = True

    files = collect_files(args.input_dir, args.exclude_dir_name, args.max_files)
    report = {
        "policy_version": POLICY_VERSION,
        "mode": "apply" if args.apply else "dry_run",
        "input_dirs": args.input_dir,
        "exclude_dir_name": args.exclude_dir_name,
        "backup_dir": args.backup_dir if args.apply else None,
        "num_files": len(files),
        "num_changed_files": 0,
        "num_unchanged_files": 0,
        "num_failed_files": 0,
        "word_transition_counts": {},
        "reason_counts": {},
        "relation_update_counts": {},
        "item_group_update_counts": {},
        "semantic_metadata_update_counts": {},
        "changed_examples": [],
        "files": [],
        "failures": [],
    }
    global_transitions = Counter()
    reason_counts = Counter()
    relation_update_counts = Counter()
    item_group_update_counts = Counter()
    semantic_metadata_update_counts = Counter()
    for index, path in enumerate(files, start=1):
        try:
            original = load_json(path)
            updated, file_report = process_payload(json.loads(json.dumps(original, ensure_ascii=False)))
            changed = file_report["changed"]
            if changed:
                report["num_changed_files"] += 1
                if args.apply:
                    backup_path = backup_file(path, args.input_dir, args.backup_dir)
                    save_json(path, updated)
                else:
                    backup_path = None
                for key, value in file_report["transition_counts"].items():
                    global_transitions[f"{key[0]}->{key[1]}"] += value
                reason_counts.update(change["reason"] for change in file_report["word_changes"])
                relation_update_counts.update(file_report["relation_updates"])
                item_group_update_counts.update(file_report["item_group_updates"])
                semantic_metadata_update_counts.update(file_report["semantic_metadata_updates"])
                if len(report["changed_examples"]) < args.debug_examples:
                    report["changed_examples"].append(
                        {
                            "path": str(path),
                            "backup_path": str(backup_path) if backup_path else None,
                            "word_changes": file_report["word_changes"][:20],
                            "relation_updates": file_report["relation_updates"],
                            "item_group_updates": file_report["item_group_updates"],
                            "semantic_metadata_updates": file_report["semantic_metadata_updates"],
                            "relation_warnings": file_report["relation_warnings"][:20],
                        }
                    )
            else:
                report["num_unchanged_files"] += 1
            report["files"].append(
                {
                    "path": str(path),
                    "changed": changed,
                    "word_change_count": len(file_report["word_changes"]),
                    "transition_counts": {f"{a}->{b}": c for (a, b), c in file_report["transition_counts"].items()},
                    "relation_updates": file_report["relation_updates"],
                    "item_group_updates": file_report["item_group_updates"],
                    "semantic_metadata_updates": file_report["semantic_metadata_updates"],
                    "relation_warning_count": len(file_report["relation_warnings"]),
                }
            )
        except Exception as exc:
            report["num_failed_files"] += 1
            report["failures"].append({"path": str(path), "error": repr(exc)})
            print(f"WARNING: failed {path}: {exc}")
        if index % 250 == 0:
            print(f"processed {index}/{len(files)} files")

    report["word_transition_counts"] = dict(global_transitions)
    report["reason_counts"] = dict(reason_counts)
    report["relation_update_counts"] = dict(relation_update_counts)
    report["item_group_update_counts"] = dict(item_group_update_counts)
    report["semantic_metadata_update_counts"] = dict(semantic_metadata_update_counts)
    save_json(args.report_out, report)
    print(json.dumps({k: report[k] for k in (
        "mode",
        "num_files",
        "num_changed_files",
        "num_unchanged_files",
        "num_failed_files",
        "word_transition_counts",
        "reason_counts",
        "relation_update_counts",
        "item_group_update_counts",
        "semantic_metadata_update_counts",
    )}, ensure_ascii=False, indent=2))
    print(f"report path: {args.report_out}")
    if args.apply:
        print(f"backup dir: {args.backup_dir}")


if __name__ == "__main__":
    main()
