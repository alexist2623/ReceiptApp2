HEAD_FIELDS = ["MENU_NM"]

DEP_FIELDS = [
    "MENU_PRICE",
    "MENU_CNT",
    "MENU_UNITPRICE",
    "MENU_DISCOUNTPRICE",
    "MENU_ITEMSUBTOTAL",
    "MENU_SUB_NM",
    "MENU_SUB_PRICE",
    "MENU_SUB_CNT",
    "MENU_SUB_UNITPRICE",
    "MENU_SUB_ETC",
    "MENU_ETC",
]

HARD_NEGATIVE_FIELDS = [
    "TOTAL_TOTAL_PRICE",
    "TOTAL_CASHPRICE",
    "TOTAL_CHANGEPRICE",
    "TOTAL_CREDITCARDPRICE",
    "TOTAL_EMONEYPRICE",
    "SUBTOTAL_SUBTOTAL_PRICE",
    "SUBTOTAL_TAX_PRICE",
    "SUBTOTAL_DISCOUNT_PRICE",
    "SUBTOTAL_SERVICE_PRICE",
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
    return normalized.upper().replace(".", "_")


def label_to_field(label: str) -> str:
    if label is None:
        return "O"
    value = str(label).strip()
    if value == "O" or not value:
        return "O"
    if value.startswith("B-") or value.startswith("I-"):
        value = value[2:]
    if value.startswith("SUB_TOTAL_"):
        value = "SUBTOTAL_" + value[len("SUB_TOTAL_") :]
    return value


def is_head_field(field: str) -> bool:
    return field in HEAD_FIELDS


def is_dependent_field(field: str) -> bool:
    return field in DEP_FIELDS


def is_hard_negative_field(field: str) -> bool:
    return field in HARD_NEGATIVE_FIELDS


def is_candidate_dep_field(field: str) -> bool:
    return is_dependent_field(field) or is_hard_negative_field(field)


def app_field_name(field: str) -> str:
    mapping = {
        "MENU_NM": "menu_name",
        "MENU_PRICE": "price",
        "MENU_CNT": "count",
        "MENU_UNITPRICE": "unit_price",
        "MENU_DISCOUNTPRICE": "discount_price",
        "MENU_ITEMSUBTOTAL": "item_subtotal",
        "MENU_SUB_NM": "sub_names",
        "MENU_SUB_PRICE": "sub_prices",
        "MENU_SUB_CNT": "sub_counts",
        "MENU_SUB_UNITPRICE": "sub_unit_prices",
        "MENU_SUB_ETC": "sub_etc",
        "MENU_ETC": "etc",
        "TOTAL_TOTAL_PRICE": "total.total_price",
        "SUBTOTAL_SUBTOTAL_PRICE": "subtotal.subtotal_price",
    }
    return mapping.get(field, str(field).lower())

