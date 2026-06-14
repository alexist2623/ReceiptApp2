"""Span-level rel-g schema backed by the canonical receipt schema."""

from ml.receipt_schema import (
    PAYMENT_FIELDS,
    SUMMARY_FIELDS,
    canonical_output_key,
    canonicalize_field,
    field_for_vocab,
    is_hard_negative_for_item_grouping,
    is_item_dependent_field,
    is_item_head_field,
)

HEAD_FIELDS = ["ITEM_NAME"]

DEP_FIELDS = [
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

ALL_FIELDS = ["O", CONTEXT_FIELD] + HEAD_FIELDS + DEP_FIELDS + HARD_NEGATIVE_FIELDS


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
    return is_item_head_field(field)


def is_dependent_field(field: str) -> bool:
    return is_item_dependent_field(field)


def is_hard_negative_field(field: str) -> bool:
    return is_hard_negative_for_item_grouping(field)


def is_candidate_dep_field(field: str) -> bool:
    return is_dependent_field(field) or is_hard_negative_field(field)


def app_field_name(field: str) -> str:
    return canonical_output_key(field)


def canonical_pair_field(field: str) -> str:
    return canonicalize_field(field)
