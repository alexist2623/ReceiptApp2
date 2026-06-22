import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.angle_geometry import ANGLE_QUAD_FEATURE_DIM, build_angle_features_for_words
from ml.layoutlmv3_angle_inputs import build_sample_angle_features


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test angle-aware LayoutLMv3 sample feature preparation.")
    parser.add_argument("--angle_encoding_mode", default="angle_quad")
    return parser.parse_args()


def main():
    args = parse_args()
    sample = {
        "id": "angle_smoke",
        "width": 200,
        "height": 120,
        "words": ["MILK", "$2.99"],
        "boxes": [[20, 20, 80, 42], [120, 22, 170, 44]],
        "normalized_boxes": [[100, 166, 400, 350], [600, 183, 850, 366]],
        "word_payloads": [
            {"text": "MILK", "quad": [20, 20, 80, 28, 78, 42, 18, 34]},
            {"text": "$2.99", "quad": [120, 22, 170, 20, 172, 44, 122, 46]},
        ],
    }
    result = build_angle_features_for_words(
        sample["word_payloads"],
        boxes=sample["boxes"],
        image_width=sample["width"],
        image_height=sample["height"],
        mode=args.angle_encoding_mode,
    )
    if args.angle_encoding_mode == "angle_quad" and result["feature_dim"] != ANGLE_QUAD_FEATURE_DIM:
        raise RuntimeError(f"expected angle_quad dim {ANGLE_QUAD_FEATURE_DIM}, got {result['feature_dim']}")
    sample = build_sample_angle_features(
        sample,
        angle_encoding_mode=args.angle_encoding_mode,
        angle_feature_dim=result["feature_dim"],
    )
    if len(sample["angle_features"]) != len(sample["words"]):
        raise RuntimeError("angle feature count does not match word count")
    print(
        json.dumps(
            {
                "sample_id": sample["id"],
                "angle_encoding_mode": sample["angle_encoding_mode"],
                "angle_feature_dim": sample["angle_feature_dim"],
                "num_words_with_angle": sample["num_words_with_angle"],
                "first_feature": sample["angle_features"][0],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("Angle-aware LayoutLMv3 feature smoke passed.")


if __name__ == "__main__":
    main()
