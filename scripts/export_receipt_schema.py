import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import (
    OLD_TO_NEW_FIELD_ALIAS,
    SCHEMA_VERSION,
    build_label_maps,
    get_canonical_fields,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Export receipt label schema v2 JSON.")
    parser.add_argument("--out", default="schemas/receipt_labels_v2.json")
    return parser.parse_args()


def main():
    args = parse_args()
    label_list, label2id, id2label = build_label_maps()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "canonical_fields": get_canonical_fields(),
        "label_list": label_list,
        "label2id": label2id,
        "id2label": id2label,
        "old_to_new_field_alias": OLD_TO_NEW_FIELD_ALIAS,
        "notes": [
            "Raw OCR tokens are not merged.",
            "Split price tokens should be labeled with BIO, e.g. '$' B-ITEM_PRICE and '16.99' I-ITEM_PRICE.",
            "STORE_NAME is document-level and excluded from item rel-g grouping.",
            "TOTAL_PRICE and SUBTOTAL_PRICE are summary fields and hard negatives for item grouping.",
        ],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"schema JSON path: {out}")
    print(f"num_fields: {len(payload['canonical_fields'])}")
    print(f"num_labels: {len(label_list)}")


if __name__ == "__main__":
    main()
