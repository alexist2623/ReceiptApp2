# Schema Migration Audit

This audit records old CORD/menu-centric schema usages found while expanding the
pipeline to the retail/general receipt schema v2.

| file | old usage | action |
| --- | --- | --- |
| `ml/span_relg/schema.py` | `HEAD_FIELDS = ["MENU_NM"]`, `MENU_PRICE`, `MENU_CNT`, `MENU_UNITPRICE`, hard negatives such as `TOTAL_TOTAL_PRICE` | Replaced with canonical `ITEM_NAME`, `ITEM_PRICE`, `ITEM_QTY`, `ITEM_UNIT_PRICE`, document/summary/payment hard negatives. Old labels are accepted through aliases. |
| `ml/span_relg/span_utils.py` | BIO labels were converted directly to old fields | Added canonical label conversion. Raw OCR tokens are not merged; BIO spans carry `raw_text` and price-normalized `normalized_text`. |
| `ml/span_relg/feature_cache.py` | Candidate heads/deps were determined from old fields | Candidate generation now canonicalizes span fields. Field ids can resolve canonical fields into old checkpoint vocab aliases. |
| `ml/span_relg/decode.py` | Output centered on `menu_name`, `price`, `count`, `unit_price` | Output now uses `item_name`, `price`, `quantity`, `unit_price`, `item_code`, plus `store`, `subtotal`, `total`, `payment`; `menu_name` remains as a compatibility alias. |
| `ml/span_relg/metrics.py` | `menu_price_pair` metrics only | Added `item_price_pair` metrics and kept `menu_price_pair` aliases. Added store and total/subtotal false-positive counts. |
| `ml/span_relg/visualization.py` | User overlay text said `Menu -> Price`; `MENU_NM` got special color | Updated wording to `Item -> Price`, supports both `ITEM_NAME` and `MENU_NM`. |
| `scripts/infer_user_ocr_json.py` | Prediction JSON wrote only raw `label` | Now writes `label`, `canonical_label`, `field`, and `canonical_field`. |
| `scripts/batch_infer_user_ocr_json.py` | Batch prediction JSON wrote only raw `label` | Now writes `label`, `canonical_label`, `field`, and `canonical_field`. |
| `scripts/infer_user_span_relg.py` | Rel-g grouping expected old span fields and dropped unknown fields by direct vocab membership | Now reads `canonical_label` first and resolves fields against old/new rel-g vocabs with aliases. |
| `scripts/build_span_relg_dataset.py` | Cache metadata listed `MENU_NM` as candidate head | Metadata now advertises `ITEM_NAME` and item-dependent fields for future cache builds. Existing caches are not rebuilt by this migration. |
| `scripts/eval_span_relg.py` | Summary keys include `menu_price_pair_*`, `MENU_CNT`, `MENU_UNITPRICE` | Backward-compatible metrics remain; canonical `item_price_pair_*` is provided by `ml/span_relg/metrics.py`. |
| `scripts/eval_predicted_span_relg_e2e.py` | Report text and span metrics mention `MENU_NM`/`MENU_PRICE` | Existing report remains compatible with old checkpoints; canonical metrics are available through aliasing. |
| `scripts/test_exported_receipt_zip.py` | Mapping CSV/HTML used `menu_name` wording | Existing compatibility keys remain. Decoder now also emits `item_name`; future UI can prefer `item_name`. |
| `scripts/train_span_relg.py` | Best metric defaults to `menu_price_pair_f1` | Training is not run in this migration. Existing checkpoints remain compatible; future training can switch to `item_price_pair_f1`. |
| `docs/*.md`, `ml/README_STEP*.md` | Several docs mention `MENU_NM`, `MENU_PRICE`, `TOTAL_TOTAL_PRICE` | Added schema v2 documentation and OCR token policy. Older step docs still describe historical CORD stages. |

Key migration rule: raw OCR tokens are preserved. Split tokens such as `$`,
`16.99`, `.`, and `99` are connected by BIO labels and normalized only after
span recovery.
