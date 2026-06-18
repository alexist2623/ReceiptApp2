import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.overlay_labeled_relations import overlay_relations


def parse_args():
    parser = argparse.ArgumentParser(description="Batch overlay hand-labeled receipt relations.")
    parser.add_argument("--input_dir", required=True, help="Directory containing *_receipt_ocr folders.")
    parser.add_argument("--out_dir", default="outputs/relation_overlay")
    parser.add_argument(
        "--relation_source",
        choices=["all", "relations", "item_relations", "summary_relations", "payment_relations", "rel_g_edges"],
        default="all",
    )
    parser.add_argument("--coordinate_mode", choices=["strict", "auto-scale"], default="strict")
    parser.add_argument("--font_size", type=int, default=20)
    parser.add_argument("--line_width", type=int, default=4)
    parser.add_argument(
        "--show_relation_labels",
        action="store_true",
        help="Draw relation text such as ITEM_NAME -> ITEM_PRICE. Hidden by default.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def find_pairs(input_dir):
    root = Path(input_dir)
    folders = sorted([p for p in root.rglob("*_receipt_ocr") if p.is_dir()])
    pairs = []
    for folder in folders:
        capture_id = folder.name[: -len("_receipt_ocr")]
        image = folder / f"{capture_id}.jpg"
        label_json = folder / f"{capture_id}_labeled_v2_1.json"
        if image.exists() and label_json.exists():
            pairs.append((capture_id, image, label_json))
    return pairs


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: input_dir not found: {input_dir}", file=sys.stderr)
        raise SystemExit(1)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = find_pairs(input_dir)
    results = []
    failures = []
    for capture_id, image, label_json in pairs:
        out_path = out_dir / f"{capture_id}_relations_overlay.png"
        summary_path = out_dir / f"{capture_id}_relations_summary.json"
        try:
            summary = overlay_relations(
                image_path=image,
                label_json_path=label_json,
                out_path=out_path,
                summary_path=summary_path,
                relation_source=args.relation_source,
                coordinate_mode=args.coordinate_mode,
                font_size=args.font_size,
                line_width=args.line_width,
                show_relation_labels=args.show_relation_labels,
                debug=args.debug,
            )
            results.append(
                {
                    "capture_id": capture_id,
                    "image": str(image),
                    "label_json": str(label_json),
                    "overlay": str(out_path),
                    "summary": str(summary_path),
                    "relation_count": summary["relation_count"],
                    "skipped_relation_count": summary["skipped_relation_count"],
                    "status": "passed",
                }
            )
        except SystemExit as exc:
            failures.append({"capture_id": capture_id, "status": "failed", "error": str(exc)})
        except Exception as exc:
            failures.append({"capture_id": capture_id, "status": "failed", "error": str(exc)})

    batch_summary = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "relation_source": args.relation_source,
        "coordinate_mode": args.coordinate_mode,
        "num_pairs": len(pairs),
        "num_passed": len(results),
        "num_failed": len(failures),
        "results": results,
        "failures": failures,
    }
    batch_summary_path = out_dir / "batch_relations_summary.json"
    save_json(batch_summary_path, batch_summary)
    print(f"input_dir: {input_dir}")
    print(f"num_pairs: {len(pairs)}")
    print(f"num_passed: {len(results)}")
    print(f"num_failed: {len(failures)}")
    print(f"batch summary JSON path: {batch_summary_path}")
    if failures:
        raise SystemExit(1)
    print("Batch labeled relation overlay passed.")


if __name__ == "__main__":
    main()
