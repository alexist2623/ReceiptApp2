# Receipt Schema V2

CORD labels such as `MENU_NM` and `MENU_PRICE` were useful for restaurant
receipts, but retail/general receipts need broader fields. Schema v2 uses
canonical fields such as `STORE_NAME`, `ITEM_NAME`, `ITEM_PRICE`, `ITEM_CODE`,
`TOTAL_PRICE`, and `PAYMENT_INFO`.

## Canonical Fields

Document-level fields:

`STORE_NAME`, `STORE_ADDRESS`, `STORE_PHONE`, `STORE_ID`, `RECEIPT_ID`, `DATE`,
`TIME`, `CASHIER`, `REGISTER_ID`

Item-level fields:

`ITEM_NAME`, `ITEM_CODE`, `ITEM_SKU`, `ITEM_QTY`, `ITEM_UNIT_PRICE`,
`ITEM_PRICE`, `ITEM_DISCOUNT`, `ITEM_CATEGORY`, `ITEM_OPTION`, `ITEM_TAX_FLAG`,
`ITEM_ETC`

Summary/payment fields:

`SUBTOTAL_PRICE`, `TAX_PRICE`, `DISCOUNT_PRICE`, `SERVICE_PRICE`, `TOTAL_PRICE`,
`CASH_PRICE`, `CHANGE_PRICE`, `CARD_PRICE`, `TIP_PRICE`, `PAYMENT_METHOD`,
`PAYMENT_CARD`, `PAYMENT_AUTH_CODE`, `PAYMENT_INFO`, `APPROVAL_CODE`,
`TRANSACTION_ID`

## Old CORD Aliases

Old labels remain readable. Examples:

- `MENU_NM` -> `ITEM_NAME`
- `MENU_PRICE` -> `ITEM_PRICE`
- `MENU_CNT` -> `ITEM_QTY`
- `MENU_UNITPRICE` -> `ITEM_UNIT_PRICE`
- `MENU_DISCOUNTPRICE` -> `ITEM_DISCOUNT`
- `TOTAL_TOTAL_PRICE` -> `TOTAL_PRICE`
- `SUBTOTAL_SUBTOTAL_PRICE` -> `SUBTOTAL_PRICE`
- `SUBTOTAL_TAX_PRICE` -> `TAX_PRICE`

This lets old CORD checkpoints emit `B-MENU_NM` while the new grouping pipeline
treats it as `B-ITEM_NAME`.

## Raw OCR Token Policy

Raw OCR tokens are never merged in the OCR step. Tokens like `$`, `16.99`, `.`,
and `99` remain exactly as OCR produced them. BIO labels connect them into spans,
then span-level normalization creates display/output text.

Example:

```text
Raw OCR     Label
WINNERS     B-STORE_NAME
SKINCARE    B-ITEM_NAME
&           I-ITEM_NAME
MAK         I-ITEM_NAME
$           B-ITEM_PRICE
16.99       I-ITEM_PRICE
Total       O
$           B-TOTAL_PRICE
44.77       I-TOTAL_PRICE
```

Recovered spans:

```text
STORE_NAME = "WINNERS"
ITEM_NAME = "SKINCARE & MAK"
ITEM_PRICE = "$16.99"
TOTAL_PRICE = "$44.77"
```

## Rel-G Grouping

Item grouping uses `ITEM_NAME` as the only item head. Candidate item dependents:

`ITEM_PRICE`, `ITEM_QTY`, `ITEM_UNIT_PRICE`, `ITEM_CODE`, `ITEM_SKU`,
`ITEM_DISCOUNT`, `ITEM_OPTION`, `ITEM_TAX_FLAG`, `ITEM_ETC`

`STORE_NAME`, `STORE_ADDRESS`, `TOTAL_PRICE`, `SUBTOTAL_PRICE`, `TAX_PRICE`, and
`PAYMENT_INFO` are not item grouping targets. They are document/summary/payment
fields and hard negatives for item grouping.

Correct relation:

```text
ITEM_NAME("SKINCARE & MAK") -> ITEM_PRICE("$16.99")
```

Negative relation:

```text
ITEM_NAME("SKINCARE & MAK") -> TOTAL_PRICE("$44.77")
```

## Current Scope

This migration updates schema/code/docs/tests only. It does not run LayoutLMv3
fine-tuning, rel-g training, OCR, or checkpoint modification. Once user-labeled
OCR JSONL data is ready, export `schemas/receipt_labels_v2.json`, fine-tune
LayoutLMv3 with schema v2 labels, rebuild rel-g cache, and train a new rel-g
checkpoint.
