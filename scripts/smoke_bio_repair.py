import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.bio_repair import repair_bio_boundaries


def main():
    words = ["SKINCARE", "&", "MAK", "$", "16.99", "TOTAL", "$", "22.99"]
    labels = [
        "B-ITEM_NAME",
        "B-ITEM_NAME",
        "B-ITEM_NAME",
        "B-ITEM_PRICE",
        "I-ITEM_PRICE",
        "B-TOTAL_NAME",
        "B-TOTAL_PRICE",
        "I-TOTAL_PRICE",
    ]
    boxes = [
        [0, 0, 70, 20],
        [75, 0, 85, 20],
        [90, 0, 140, 20],
        [200, 0, 210, 20],
        [212, 0, 260, 20],
        [0, 60, 70, 80],
        [200, 60, 210, 80],
        [212, 60, 260, 80],
    ]
    repaired, report = repair_bio_boundaries(words=words, labels=labels, boxes=boxes)
    expected = [
        "B-ITEM_NAME",
        "I-ITEM_NAME",
        "I-ITEM_NAME",
        "B-ITEM_PRICE",
        "I-ITEM_PRICE",
        "B-TOTAL_NAME",
        "B-TOTAL_PRICE",
        "I-TOTAL_PRICE",
    ]
    result = {"input": labels, "repaired": repaired, "expected": expected, "report": report}
    print(json.dumps(result, indent=2))
    if repaired != expected:
        raise SystemExit("BIO repair smoke failed")


if __name__ == "__main__":
    main()
