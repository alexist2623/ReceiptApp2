import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Create a sample user OCR JSON file.")
    parser.add_argument("--out", default="sample_data/sample_user_ocr.json")
    return parser.parse_args()


def main():
    args = parse_args()
    payload = {
        "image_width": 800,
        "image_height": 1200,
        "words": [
            {"text": "volcano", "box": [80, 120, 210, 155]},
            {"text": "iced", "box": [220, 120, 300, 155]},
            {"text": "coffee", "box": [80, 160, 220, 195]},
            {"text": "4,000", "box": [580, 120, 690, 155]},
            {"text": "TOTAL", "box": [80, 980, 220, 1020]},
            {"text": "4,000", "box": [580, 980, 690, 1020]},
        ],
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    print(f"Saved sample OCR JSON: {out_path}")


if __name__ == "__main__":
    main()
