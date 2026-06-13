import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.span_relg.decode import decode_edges_to_items
from ml.span_relg.geometry import normalize_box_1000
from ml.span_relg.span_utils import bio_predictions_to_spans


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test schema v2 span merge and decoder with mock predictions.")
    parser.add_argument("--input", default="sample_data/mock_prediction_schema_v2.json")
    parser.add_argument("--out", default="outputs/schema_v2_mock/mock_grouped.json")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def make_node(span):
    return {
        "node_id": span["span_id"],
        "node_kind": "SPAN",
        "field": span["field"],
        "raw_field": span.get("raw_field"),
        "text": span["text"],
        "normalized_text": span["normalized_text"],
        "word_indices": span["word_indices"],
        "first_word_idx": span["first_word_idx"],
        "box": span["box"],
        "normalized_box": span["normalized_box"],
        "span_id": span["span_id"],
        "confidence": span["confidence"],
    }


def assert_equal(name, actual, expected):
    if actual != expected:
        raise AssertionError(f"{name}: expected {expected!r}, got {actual!r}")


def main():
    args = parse_args()
    payload = load_json(args.input)
    width = int(payload["image_width"])
    height = int(payload["image_height"])
    predictions = []
    for item in payload["predictions"]:
        entry = dict(item)
        entry["normalized_box"] = normalize_box_1000(entry["box"], width, height)
        predictions.append(entry)
    spans = bio_predictions_to_spans(predictions, width, height)
    nodes = [make_node(span) for span in spans]
    by_field = {span["field"]: span for span in spans}
    head = by_field["ITEM_NAME"]
    price = by_field["ITEM_PRICE"]
    sample = {
        "nodes": nodes,
        "candidate_pairs": [[head["span_id"], price["span_id"]]],
        "pair_meta": [
            {
                "head_node_id": head["span_id"],
                "dep_node_id": price["span_id"],
                "head_span_id": head["span_id"],
                "dep_span_id": price["span_id"],
                "head_field": "ITEM_NAME",
                "dep_field": "ITEM_PRICE",
                "head_text": head["text"],
                "dep_text": price["text"],
                "label": 1,
            }
        ],
    }
    grouped = decode_edges_to_items(sample, [0.99], threshold=0.5)
    assert_equal("STORE_NAME", by_field["STORE_NAME"]["normalized_text"], "WINNERS")
    assert_equal("ITEM_NAME", by_field["ITEM_NAME"]["normalized_text"], "SKINCARE & MAK")
    assert_equal("ITEM_PRICE", by_field["ITEM_PRICE"]["normalized_text"], "$16.99")
    assert_equal("TOTAL_PRICE", by_field["TOTAL_PRICE"]["normalized_text"], "$44.77")
    assert_equal("item_count", len(grouped["items"]), 1)
    assert grouped["store"]["store_name"] is not None, "store.store_name is missing"
    assert grouped["total"]["total_price"] is not None, "total.total_price is missing"
    assert grouped["items"][0]["price"]["normalized_text"] == "$16.99", "item price not decoded"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"spans": spans, "grouped": grouped}, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.debug:
        for span in spans:
            print(f"{span['span_id']}: {span['field']} text={span['text']!r} norm={span['normalized_text']!r}")
        print(json.dumps(grouped, ensure_ascii=False, indent=2)[:2000])
    print(f"mock grouped JSON path: {out}")
    print("schema v2 mock decode passed.")


if __name__ == "__main__":
    main()
