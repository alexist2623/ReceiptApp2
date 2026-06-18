"""Span-level rel-g schema backed by the canonical receipt schema."""

from ml.receipt_schema import (
    canonical_output_key,
    canonicalize_field,
    field_for_vocab,
)

ITEM_HEAD_FIELDS = ["ITEM_NAME"]

SUMMARY_HEAD_FIELDS = [
    "SUBTOTAL_NAME",
    "TAX_NAME",
    "TOTAL_NAME",
    "TIP_NAME",
]

HEAD_FIELDS = ITEM_HEAD_FIELDS + SUMMARY_HEAD_FIELDS

ITEM_DEP_FIELDS = [
    "ITEM_PRICE",
    "ITEM_QTY",
    "ITEM_UNIT_PRICE",
    "ITEM_CODE",
    "ITEM_SKU",
    "ITEM_DISCOUNT",
    "ITEM_OPTION",
    "ITEM_TAX_FLAG",
    "ITEM_ETC",
]

SUMMARY_DEP_FIELDS = [
    "SUBTOTAL_PRICE",
    "TAX_PRICE",
    "TAX_RATE",
    "TOTAL_PRICE",
    "TIP_PRICE",
]

DEP_FIELDS = ITEM_DEP_FIELDS + SUMMARY_DEP_FIELDS

HARD_NEGATIVE_FIELDS = [
    "STORE_NAME",
    "STORE_ADDRESS",
    "STORE_PHONE",
    "DATE",
    "TIME",
    "RECEIPT_ID",
    "SUBTOTAL_NAME",
    "SUBTOTAL_PRICE",
    "TAX_NAME",
    "TAX_RATE",
    "TAX_PRICE",
    "DISCOUNT_NAME",
    "DISCOUNT_PRICE",
    "SERVICE_NAME",
    "SERVICE_PRICE",
    "TOTAL_NAME",
    "TOTAL_PRICE",
    "CASH_NAME",
    "CASH_PRICE",
    "CHANGE_NAME",
    "CHANGE_PRICE",
    "CARD_NAME",
    "CARD_PRICE",
    "TIP_NAME",
    "TIP_PRICE",
    "PAYMENT_METHOD",
    "PAYMENT_CARD",
    "PAYMENT_AUTH_CODE",
    "PAYMENT_INFO",
    "APPROVAL_CODE",
    "TRANSACTION_ID",
]

CONTEXT_FIELD = "CONTEXT_TOKEN"


def _unique(values):
    seen = set()
    out = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


# Preserve the original field-id order as much as possible: item fields first,
# then document/summary/payment hard negatives. Head/dependent behavior is
# controlled by the sets above, not by the field-list position.
ALL_FIELDS = _unique(["O", CONTEXT_FIELD] + ITEM_HEAD_FIELDS + ITEM_DEP_FIELDS + HARD_NEGATIVE_FIELDS)


def normalize_category(category: str) -> str:
    if category is None:
        return "O"
    value = str(category).strip().lower()
    if value in {"", "o", "unknown", "none", "null"}:
        return "O"
    value = "_".join(value.split())
    value = value.replace("-", "_").replace("/", "_")
    return value


def category_to_field(category: str) -> str:
    normalized = normalize_category(category)
    if normalized == "O":
        return "O"
    normalized = normalized.replace("sub_total.", "subtotal.")
    return canonicalize_field(normalized.upper().replace(".", "_"))


def label_to_field(label: str) -> str:
    if label is None:
        return "O"
    value = str(label).strip()
    if value == "O" or not value:
        return "O"
    if value.startswith(("B-", "I-")):
        value = value[2:]
    if value.startswith("SUB_TOTAL_"):
        value = "SUBTOTAL_" + value[len("SUB_TOTAL_") :]
    return canonicalize_field(value)


def is_head_field(field: str) -> bool:
    return canonicalize_field(field) in set(HEAD_FIELDS)


def is_dependent_field(field: str) -> bool:
    return canonicalize_field(field) in set(DEP_FIELDS)


def is_hard_negative_field(field: str) -> bool:
    return canonicalize_field(field) in set(HARD_NEGATIVE_FIELDS)


def is_candidate_dep_field(field: str) -> bool:
    return is_dependent_field(field) or is_hard_negative_field(field)


def app_field_name(field: str) -> str:
    return canonical_output_key(field)


def canonical_pair_field(field: str) -> str:
    return canonicalize_field(field)
