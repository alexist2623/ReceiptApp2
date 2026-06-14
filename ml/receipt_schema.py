"""Canonical retail/general receipt field schema.

The pipeline keeps raw OCR tokens unchanged. BIO labels connect split tokens into
semantic spans, and span text is normalized only after span recovery.
"""

from __future__ import annotations

import re
from typing import Iterable


SCHEMA_VERSION = "receipt_labels_v2"

DOCUMENT_FIELDS = [
    "STORE_NAME",
    "STORE_ADDRESS",
    "STORE_PHONE",
    "STORE_ID",
    "RECEIPT_ID",
    "DATE",
    "TIME",
    "CASHIER",
    "REGISTER_ID",
]

ITEM_FIELDS = [
    "ITEM_NAME",
    "ITEM_CODE",
    "ITEM_SKU",
    "ITEM_QTY",
    "ITEM_UNIT_PRICE",
    "ITEM_PRICE",
    "ITEM_DISCOUNT",
    "ITEM_CATEGORY",
    "ITEM_OPTION",
    "ITEM_TAX_FLAG",
    "ITEM_ETC",
]

SUMMARY_FIELDS = [
    "SUBTOTAL_NAME",
    "SUBTOTAL_PRICE",
    "TAX_NAME",
    "TAX_RATE",
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
    "TAX_PRICE",
]

PAYMENT_FIELDS = [
    "PAYMENT_METHOD",
    "PAYMENT_CARD",
    "PAYMENT_AUTH_CODE",
    "PAYMENT_INFO",
    "APPROVAL_CODE",
    "TRANSACTION_ID",
]

OLD_TO_NEW_FIELD_ALIAS = {
    "MENU_NM": "ITEM_NAME",
    "MENU_SUB_NM": "ITEM_OPTION",
    "MENU_CNT": "ITEM_QTY",
    "MENU_NUM": "ITEM_CODE",
    "MENU_UNITPRICE": "ITEM_UNIT_PRICE",
    "MENU_PRICE": "ITEM_PRICE",
    "MENU_DISCOUNTPRICE": "ITEM_DISCOUNT",
    "MENU_ITEMSUBTOTAL": "ITEM_PRICE",
    "MENU_VATYN": "ITEM_TAX_FLAG",
    "MENU_ETC": "ITEM_ETC",
    "MENU_SUB_PRICE": "ITEM_PRICE",
    "MENU_SUB_CNT": "ITEM_QTY",
    "MENU_SUB_UNITPRICE": "ITEM_UNIT_PRICE",
    "MENU_SUB_ETC": "ITEM_ETC",
    "VOID_MENU_NM": "ITEM_NAME",
    "VOID_MENU_PRICE": "ITEM_PRICE",
    "TOTAL_TOTAL_PRICE": "TOTAL_PRICE",
    "TOTAL_CASHPRICE": "CASH_PRICE",
    "TOTAL_CHANGEPRICE": "CHANGE_PRICE",
    "TOTAL_CREDITCARDPRICE": "CARD_PRICE",
    "TOTAL_EMONEYPRICE": "CARD_PRICE",
    "TOTAL_TOTAL_ETC": "PAYMENT_INFO",
    "TOTAL_MENUTYPE_CNT": "PAYMENT_INFO",
    "TOTAL_MENUQTY_CNT": "PAYMENT_INFO",
    "SUBTOTAL_SUBTOTAL_PRICE": "SUBTOTAL_PRICE",
    "SUBTOTAL_TAX_PRICE": "TAX_PRICE",
    "SUBTOTAL_DISCOUNT_PRICE": "DISCOUNT_PRICE",
    "SUBTOTAL_SERVICE_PRICE": "SERVICE_PRICE",
    "SUBTOTAL_OTHERSVC_PRICE": "SERVICE_PRICE",
    "SUBTOTAL_ETC": "PAYMENT_INFO",
    "SUB_TOTAL_SUBTOTAL_PRICE": "SUBTOTAL_PRICE",
    "SUB_TOTAL_TAX_PRICE": "TAX_PRICE",
    "SUB_TOTAL_DISCOUNT_PRICE": "DISCOUNT_PRICE",
    "SUB_TOTAL_SERVICE_PRICE": "SERVICE_PRICE",
    "SUB_TOTAL_OTHERSVC_PRICE": "SERVICE_PRICE",
    "SUB_TOTAL_ETC": "PAYMENT_INFO",
}

NEW_TO_OLD_FIELD_ALIAS = {
    "ITEM_NAME": "MENU_NM",
    "ITEM_OPTION": "MENU_SUB_NM",
    "ITEM_QTY": "MENU_CNT",
    "ITEM_UNIT_PRICE": "MENU_UNITPRICE",
    "ITEM_PRICE": "MENU_PRICE",
    "ITEM_DISCOUNT": "MENU_DISCOUNTPRICE",
    "ITEM_ETC": "MENU_ETC",
    "TOTAL_PRICE": "TOTAL_TOTAL_PRICE",
    "CASH_PRICE": "TOTAL_CASHPRICE",
    "CHANGE_PRICE": "TOTAL_CHANGEPRICE",
    "CARD_PRICE": "TOTAL_CREDITCARDPRICE",
    "SUBTOTAL_PRICE": "SUBTOTAL_SUBTOTAL_PRICE",
    "TAX_PRICE": "SUBTOTAL_TAX_PRICE",
    "DISCOUNT_PRICE": "SUBTOTAL_DISCOUNT_PRICE",
    "SERVICE_PRICE": "SUBTOTAL_SERVICE_PRICE",
    "PAYMENT_INFO": "SUBTOTAL_ETC",
}

CANONICAL_FIELDS = DOCUMENT_FIELDS + ITEM_FIELDS + SUMMARY_FIELDS + PAYMENT_FIELDS
CANONICAL_FIELD_SET = set(CANONICAL_FIELDS) | {"O"}
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


def _clean_field(field: str | None) -> str:
    if field is None:
        return "O"
    value = str(field).strip()
    if not value:
        return "O"
    if value in {"O", "UNKNOWN"}:
        return value
    return value.upper().replace(".", "_").replace("-", "_").replace("/", "_").replace(" ", "_")


def get_canonical_fields() -> list[str]:
    return list(CANONICAL_FIELDS)


def get_bio_label_list() -> list[str]:
    labels = ["O"]
    for field in CANONICAL_FIELDS:
        labels.extend([f"B-{field}", f"I-{field}"])
    return labels


def build_label_maps():
    label_list = get_bio_label_list()
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {str(idx): label for label, idx in label2id.items()}
    return label_list, label2id, id2label


def canonicalize_field(field: str | None, use_alias: bool = True) -> str:
    value = _clean_field(field)
    if value == "O":
        return "O"
    if value in CANONICAL_FIELD_SET:
        return value
    if use_alias and value in OLD_TO_NEW_FIELD_ALIAS:
        return OLD_TO_NEW_FIELD_ALIAS[value]
    return value


def label_to_field(label: str | None) -> str:
    if label is None:
        return "O"
    value = str(label).strip()
    if not value or value == "O":
        return "O"
    if value.startswith(("B-", "I-")):
        value = value[2:]
    return canonicalize_field(value)


def canonicalize_label(label: str | None, use_alias: bool = True) -> str:
    if label is None:
        return "O"
    value = str(label).strip()
    if not value or value == "O":
        return "O"
    prefix = ""
    field = value
    if value.startswith(("B-", "I-")):
        prefix, field = value[:2], value[2:]
    canonical = canonicalize_field(field, use_alias=use_alias)
    if canonical == "O":
        return "O"
    return f"{prefix or 'B-'}{canonical}"


def old_label_to_new_label(label: str) -> str:
    return canonicalize_label(label, use_alias=True)


def new_label_to_old_alias(label: str) -> str:
    if label == "O":
        return "O"
    prefix = ""
    field = label
    if label.startswith(("B-", "I-")):
        prefix, field = label[:2], label[2:]
    old = NEW_TO_OLD_FIELD_ALIAS.get(canonicalize_field(field, use_alias=False), field)
    return f"{prefix or 'B-'}{old}"


def field_to_bio(field: str) -> tuple[str, str]:
    canonical = canonicalize_field(field)
    if canonical == "O":
        return "O", "O"
    return f"B-{canonical}", f"I-{canonical}"


def is_item_head_field(field: str | None) -> bool:
    return canonicalize_field(field) == "ITEM_NAME"


def is_item_dependent_field(field: str | None) -> bool:
    return canonicalize_field(field) in {
        "ITEM_PRICE",
        "ITEM_QTY",
        "ITEM_UNIT_PRICE",
        "ITEM_CODE",
        "ITEM_SKU",
        "ITEM_DISCOUNT",
        "ITEM_OPTION",
        "ITEM_TAX_FLAG",
        "ITEM_ETC",
    }


def is_summary_field(field: str | None) -> bool:
    return canonicalize_field(field) in set(SUMMARY_FIELDS)


def is_document_field(field: str | None) -> bool:
    return canonicalize_field(field) in set(DOCUMENT_FIELDS)


def is_payment_field(field: str | None) -> bool:
    return canonicalize_field(field) in set(PAYMENT_FIELDS)


def is_hard_negative_for_item_grouping(field: str | None) -> bool:
    canonical = canonicalize_field(field)
    return canonical == "O" or is_summary_field(canonical) or is_document_field(canonical) or is_payment_field(canonical)


def is_relg_candidate_dep_field(field: str | None) -> bool:
    return is_item_dependent_field(field) or is_hard_negative_for_item_grouping(field)


def canonical_output_key(field: str | None) -> str:
    mapping = {
        "STORE_NAME": "store_name",
        "STORE_ADDRESS": "store_address",
        "STORE_PHONE": "store_phone",
        "STORE_ID": "store_id",
        "RECEIPT_ID": "receipt_id",
        "DATE": "date",
        "TIME": "time",
        "CASHIER": "cashier",
        "REGISTER_ID": "register_id",
        "ITEM_NAME": "item_name",
        "ITEM_CODE": "item_code",
        "ITEM_SKU": "item_sku",
        "ITEM_QTY": "quantity",
        "ITEM_UNIT_PRICE": "unit_price",
        "ITEM_PRICE": "price",
        "ITEM_DISCOUNT": "discount",
        "ITEM_CATEGORY": "item_category",
        "ITEM_OPTION": "options",
        "ITEM_TAX_FLAG": "tax_flag",
        "ITEM_ETC": "item_etc",
        "SUBTOTAL_NAME": "subtotal_name",
        "SUBTOTAL_PRICE": "subtotal_price",
        "TAX_NAME": "tax_name",
        "TAX_RATE": "tax_rate",
        "TAX_PRICE": "tax_price",
        "DISCOUNT_NAME": "discount_name",
        "DISCOUNT_PRICE": "discount_price",
        "SERVICE_NAME": "service_name",
        "SERVICE_PRICE": "service_price",
        "TOTAL_NAME": "total_name",
        "TOTAL_PRICE": "total_price",
        "CASH_NAME": "cash_name",
        "CASH_PRICE": "cash_price",
        "CHANGE_NAME": "change_name",
        "CHANGE_PRICE": "change_price",
        "CARD_NAME": "card_name",
        "CARD_PRICE": "card_price",
        "TIP_NAME": "tip_name",
        "TIP_PRICE": "tip_price",
        "PAYMENT_METHOD": "payment_method",
        "PAYMENT_CARD": "payment_card",
        "PAYMENT_AUTH_CODE": "payment_auth_code",
        "PAYMENT_INFO": "payment_info",
        "APPROVAL_CODE": "approval_code",
        "TRANSACTION_ID": "transaction_id",
    }
    return mapping.get(canonicalize_field(field), str(field or "unknown").lower())


def field_for_vocab(field: str | None, vocab: dict | Iterable[str]) -> str | None:
    if isinstance(vocab, dict):
        keys = set(vocab.keys())
    else:
        keys = set(vocab)
    canonical = canonicalize_field(field)
    if canonical in keys:
        return canonical
    old = NEW_TO_OLD_FIELD_ALIAS.get(canonical)
    if old in keys:
        return old
    raw = _clean_field(field)
    if raw in keys:
        return raw
    return None


def normalize_span_text(field: str | None, text: str) -> str:
    value = re.sub(r"\s+", " ", str(text or "").strip())
    if canonicalize_field(field) not in PRICE_FIELDS:
        return value
    value = re.sub(r"\s*([.$,])\s*", r"\1", value)
    value = re.sub(r"\s+", "", value)
    return value
