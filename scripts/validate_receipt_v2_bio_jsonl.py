import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.angle_geometry import parse_quad
from ml.receipt_schema import get_bio_label_list


def parse_args():
    parser = argparse.ArgumentParser(description="Validate ReceiptApp2 BIO JSONL files before training.")
    parser.add_argument("--bio_dir", required=True)
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--require_angle", action="store_true")
    parser.add_argument("--max_examples", type=int, default=20)
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if line.strip():
                yield line_no, json.loads(line)


def validate_record(record, valid_labels, require_angle):
    errors = []
    warnings = []
    words = record.get("words") or []
    boxes = record.get("boxes") or []
    norm = record.get("normalized_boxes") or []
    labels = record.get("labels") or []
    if not words:
        errors.append("empty words")
    lengths = {"words": len(words), "boxes": len(boxes), "normalized_boxes": len(norm), "labels": len(labels)}
    if len(set(lengths.values())) != 1:
        errors.append(f"length mismatch: {lengths}")
    for idx, label in enumerate(labels):
        if label not in valid_labels and label != "IGNORE":
            errors.append(f"unknown label at {idx}: {label}")
            break
    for idx, box in enumerate(boxes[: len(words)]):
        try:
            x0, y0, x1, y1 = [int(v) for v in box]
        except Exception:
            errors.append(f"invalid box at {idx}: {box}")
            break
        if x1 <= x0 or y1 <= y0:
            errors.append(f"zero-area box at {idx}: {box}")
            break
    for idx, box in enumerate(norm[: len(words)]):
        try:
            values = [int(v) for v in box]
        except Exception:
            errors.append(f"invalid normalized_box at {idx}: {box}")
            break
        if len(values) != 4 or any(v < 0 or v > 1000 for v in values):
            errors.append(f"normalized_box out of 0..1000 at {idx}: {box}")
            break
    payloads = record.get("word_payloads") or []
    if require_angle:
        if len(payloads) != len(words):
            errors.append(f"word_payloads length mismatch: {len(payloads)} vs {len(words)}")
        else:
            missing_quad = [idx for idx, payload in enumerate(payloads) if parse_quad(payload.get("quad")) is None]
            missing_angle = [idx for idx, payload in enumerate(payloads) if payload.get("angle_deg") is None]
            if missing_quad:
                errors.append(f"missing quad count={len(missing_quad)} first={missing_quad[:10]}")
            if missing_angle:
                warnings.append(f"missing angle count={len(missing_angle)} first={missing_angle[:10]}")
    return errors, warnings


def main():
    args = parse_args()
    bio_dir = Path(args.bio_dir)
    if not bio_dir.exists():
        raise SystemExit(f"bio_dir not found: {bio_dir}")
    valid_labels = set(get_bio_label_list())
    report = {
        "bio_dir": str(bio_dir),
        "require_angle": bool(args.require_angle),
        "splits": {},
        "error_examples": [],
        "warning_examples": [],
    }
    total_errors = 0
    total_warnings = 0
    label_counts = Counter()
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        path = bio_dir / f"{split}.jsonl"
        if not path.exists():
            report["splits"][split] = {"exists": False}
            total_errors += 1
            continue
        split_stats = {"exists": True, "records": 0, "errors": 0, "warnings": 0}
        for line_no, record in load_jsonl(path):
            split_stats["records"] += 1
            label_counts.update(record.get("labels") or [])
            errors, warnings = validate_record(record, valid_labels, args.require_angle)
            if errors:
                split_stats["errors"] += len(errors)
                total_errors += len(errors)
                if len(report["error_examples"]) < args.max_examples:
                    report["error_examples"].append({"split": split, "line": line_no, "id": record.get("id"), "errors": errors})
            if warnings:
                split_stats["warnings"] += len(warnings)
                total_warnings += len(warnings)
                if len(report["warning_examples"]) < args.max_examples:
                    report["warning_examples"].append({"split": split, "line": line_no, "id": record.get("id"), "warnings": warnings})
        report["splits"][split] = split_stats
    report["label_counts_top50"] = dict(label_counts.most_common(50))
    report["total_errors"] = total_errors
    report["total_warnings"] = total_warnings
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if total_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
