import argparse
import json
import re
import shutil
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

from datasets import load_from_disk
from PIL import Image
from tqdm import tqdm


KNOWN_CATEGORIES = {
    "menu.nm",
    "menu.num",
    "menu.unitprice",
    "menu.cnt",
    "menu.discountprice",
    "menu.price",
    "menu.itemsubtotal",
    "menu.vatyn",
    "menu.etc",
    "menu.sub_nm",
    "menu.sub_unitprice",
    "menu.sub_cnt",
    "menu.sub_price",
    "menu.sub_etc",
    "void_menu.nm",
    "void_menu.price",
    "subtotal.subtotal_price",
    "subtotal.discount_price",
    "subtotal.service_price",
    "subtotal.othersvc_price",
    "subtotal.tax_price",
    "subtotal.etc",
    "total.total_price",
    "total.total_etc",
    "total.cashprice",
    "total.changeprice",
    "total.creditcardprice",
    "total.emoneyprice",
    "total.menutype_cnt",
    "total.menuqty_cnt",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert CORD-v2 valid_line categories to word-level BIO labels."
    )
    parser.add_argument(
        "--data_dir",
        default="../receipt_training_data2",
        help="CORD-v2 dataset directory readable by datasets.load_from_disk.",
    )
    parser.add_argument(
        "--out_dir",
        default="processed_data/cord_bio",
        help="Directory for converted BIO JSONL files.",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to process. If omitted, all splits are processed.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Process only this many samples per split.",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Inspect dataset/category structure without writing converted files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite out_dir if it already exists.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Raise errors for unknown categories, invalid boxes, and missing words.",
    )
    parser.add_argument(
        "--keep_empty_text",
        action="store_true",
        help="Keep empty text words. By default they are skipped.",
    )
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def warn_or_fail(message, strict):
    if strict:
        fail(message)
    print(f"WARNING: {message}")


def load_dataset_or_fail(data_dir):
    path = Path(data_dir)
    if not path.exists():
        fail(f"CORD-v2 dataset not found at {data_dir}. Run download step first.")
    return load_from_disk(str(path))


def print_splits(dataset):
    print("dataset splits:")
    for split in dataset.keys():
        print(f"  {split}: {len(dataset[split])}")


def ensure_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")
    if isinstance(image, bytes):
        return Image.open(BytesIO(image)).convert("RGB")
    if isinstance(image, dict):
        if image.get("path"):
            return Image.open(image["path"]).convert("RGB")
        if image.get("bytes"):
            return Image.open(BytesIO(image["bytes"])).convert("RGB")
    if hasattr(image, "convert"):
        converted = image.convert("RGB")
        if isinstance(converted, Image.Image):
            return converted
    raise TypeError(f"Unsupported image type: {type(image)}")


def parse_ground_truth(raw_ground_truth):
    if isinstance(raw_ground_truth, dict):
        return raw_ground_truth
    if isinstance(raw_ground_truth, bytes):
        raw_ground_truth = raw_ground_truth.decode("utf-8")
    if isinstance(raw_ground_truth, str):
        try:
            return json.loads(raw_ground_truth)
        except json.JSONDecodeError as exc:
            print("Failed to parse sample['ground_truth']. First 500 chars:")
            print(raw_ground_truth[:500])
            raise ValueError("sample['ground_truth'] is not valid JSON") from exc
    raise TypeError(f"Unsupported ground_truth type: {type(raw_ground_truth)}")


def normalize_category(category):
    if category is None:
        return "O"
    normalized = str(category).strip()
    if normalized.lower() in {"", "o", "unknown", "none", "null", "nan"}:
        return "O"
    normalized = normalized.lower()
    normalized = normalized.replace("-", "_").replace("/", "_")
    normalized = re.sub(r"\s+", "_", normalized)
    return normalized


def category_to_label_base(category):
    normalized = normalize_category(category)
    if normalized == "O":
        return "O"
    return normalized.upper().replace(".", "_")


def to_int(value):
    return int(round(float(value)))


def points_to_box(values):
    xs = [to_int(value) for value in values[0::2]]
    ys = [to_int(value) for value in values[1::2]]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def quad_to_box(quad_or_box):
    if quad_or_box is None:
        return None
    if isinstance(quad_or_box, dict):
        if {"x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"}.issubset(quad_or_box):
            return points_to_box(
                [
                    quad_or_box["x1"],
                    quad_or_box["y1"],
                    quad_or_box["x2"],
                    quad_or_box["y2"],
                    quad_or_box["x3"],
                    quad_or_box["y3"],
                    quad_or_box["x4"],
                    quad_or_box["y4"],
                ]
            )
        if {"x0", "y0", "x1", "y1"}.issubset(quad_or_box):
            return points_to_box(
                [quad_or_box["x0"], quad_or_box["y0"], quad_or_box["x1"], quad_or_box["y1"]]
            )
        if {"left", "top", "right", "bottom"}.issubset(quad_or_box):
            return points_to_box(
                [
                    quad_or_box["left"],
                    quad_or_box["top"],
                    quad_or_box["right"],
                    quad_or_box["bottom"],
                ]
            )
        for nested_key in ("quad", "box", "bbox"):
            if nested_key in quad_or_box:
                return quad_to_box(quad_or_box[nested_key])
    if isinstance(quad_or_box, (list, tuple)):
        if len(quad_or_box) == 4:
            x0, y0, x1, y1 = [to_int(value) for value in quad_or_box]
            if x1 < x0:
                x0, x1 = x1, x0
            if y1 < y0:
                y0, y1 = y1, y0
            return [x0, y0, x1, y1]
        if len(quad_or_box) == 8:
            return points_to_box(quad_or_box)
    return None


def clamp_box(box, width, height):
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def normalize_box(box, width, height):
    x0, y0, x1, y1 = box
    normalized = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]
    return [max(0, min(value, 1000)) for value in normalized]


def get_word_text(word):
    return word.get("text") or word.get("value") or word.get("word") or ""


def get_word_quad(word):
    for key in ("quad", "box", "bbox"):
        if key in word:
            return word.get(key)
    coordinate_keys = {"x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4"}
    box_keys = {"x0", "y0", "x1", "y1"}
    ltrb_keys = {"left", "top", "right", "bottom"}
    word_keys = set(word.keys())
    if coordinate_keys.issubset(word_keys) or box_keys.issubset(word_keys) or ltrb_keys.issubset(word_keys):
        return word
    return None


def inspect_ground_truth(ground_truth):
    print(f"ground_truth top-level keys: {list(ground_truth.keys())}")
    gt_parse = ground_truth.get("gt_parse")
    if isinstance(gt_parse, dict):
        print(f"gt_parse keys: {list(gt_parse.keys())}")
    meta = ground_truth.get("meta")
    if isinstance(meta, dict):
        print(
            "meta: "
            f"keys={list(meta.keys())}, "
            f"version={meta.get('version')}, split={meta.get('split')}, image_id={meta.get('image_id')}"
        )
    valid_line = ground_truth.get("valid_line")
    if isinstance(valid_line, list):
        print(f"valid_line count: {len(valid_line)}")
        print("first 2 valid_line raw structures:")
        print(json.dumps(valid_line[:2], ensure_ascii=False, indent=2)[:4000])
    else:
        print("valid_line not found")


def update_stats_for_line(stats, category_norm):
    stats["line_category_counts"][category_norm] += 1
    if category_norm != "O" and category_norm not in KNOWN_CATEGORIES:
        stats["unknown_categories"].add(category_norm)


def convert_sample(sample, split, index, args, stats):
    image = ensure_pil_rgb(sample["image"])
    width, height = image.size
    ground_truth = parse_ground_truth(sample["ground_truth"])
    valid_lines = ground_truth.get("valid_line")
    sample_id = f"{split}_{index:06d}"

    if not isinstance(valid_lines, list):
        message = f"{sample_id}: valid_line not found"
        if args.strict:
            fail(message)
        stats["skipped_samples"].append({"id": sample_id, "reason": "valid_line not found"})
        return None

    record = {
        "id": sample_id,
        "split": split,
        "index": index,
        "image_size": {"width": width, "height": height},
        "words": [],
        "boxes": [],
        "normalized_boxes": [],
        "labels": [],
        "categories": [],
        "line_ids": [],
        "word_indices": [],
        "group_ids": [],
        "sub_group_ids": [],
        "row_ids": [],
        "line_texts": [],
        "image_ref": None,
    }

    for line_idx, line in enumerate(valid_lines):
        if not isinstance(line, dict):
            warn_or_fail(f"{sample_id}: non-dict line at {line_idx}: {type(line)}", args.strict)
            continue

        line_text_raw = line.get("text", "")
        category_raw = line.get("category", "O")
        category_norm = normalize_category(category_raw)
        label_base = category_to_label_base(category_norm)
        group_id = line.get("group_id", None)
        sub_group_id = line.get("sub_group_id", None)
        line_row_id = line.get("row_id", None)
        words_raw = line.get("words", [])

        update_stats_for_line(stats, category_norm)
        if category_norm != "O" and category_norm not in KNOWN_CATEGORIES and args.strict:
            fail(f"{sample_id}: unknown category '{category_norm}' at line {line_idx}")

        if not isinstance(words_raw, list):
            warn_or_fail(f"{sample_id}: line {line_idx} has non-list words: {type(words_raw)}", args.strict)
            continue

        pending_entries = []
        for word_idx, word in enumerate(words_raw):
            if not isinstance(word, dict):
                warn_or_fail(
                    f"{sample_id}: non-dict word at line {line_idx}, word {word_idx}: {type(word)}",
                    args.strict,
                )
                continue

            text = str(get_word_text(word))
            if not args.keep_empty_text:
                text = text.strip()
                if not text:
                    continue

            raw_box = quad_to_box(get_word_quad(word))
            if raw_box is None:
                issue = {
                    "id": sample_id,
                    "line_idx": line_idx,
                    "word_idx": word_idx,
                    "reason": "missing or unsupported box",
                }
                stats["invalid_boxes"].append(issue)
                warn_or_fail(f"{sample_id}: missing/unsupported box at line {line_idx}, word {word_idx}", args.strict)
                continue

            pixel_box = clamp_box(raw_box, width, height)
            if pixel_box is None:
                issue = {
                    "id": sample_id,
                    "line_idx": line_idx,
                    "word_idx": word_idx,
                    "raw_box": raw_box,
                    "reason": "invalid after clamp",
                }
                stats["invalid_boxes"].append(issue)
                warn_or_fail(f"{sample_id}: invalid box at line {line_idx}, word {word_idx}: {raw_box}", args.strict)
                continue

            pending_entries.append(
                {
                    "word": text,
                    "box": pixel_box,
                    "normalized_box": normalize_box(pixel_box, width, height),
                    "category": category_norm,
                    "word_idx": word_idx,
                    "row_id": line_row_id if line_row_id is not None else word.get("row_id", None),
                }
            )

        if not pending_entries:
            continue

        line_text = line_text_raw or " ".join(entry["word"] for entry in pending_entries).strip()
        for valid_word_idx, entry in enumerate(pending_entries):
            if label_base == "O":
                label = "O"
            else:
                prefix = "B" if valid_word_idx == 0 else "I"
                label = f"{prefix}-{label_base}"

            record["words"].append(entry["word"])
            record["boxes"].append(entry["box"])
            record["normalized_boxes"].append(entry["normalized_box"])
            record["labels"].append(label)
            record["categories"].append(entry["category"])
            record["line_ids"].append(line_idx)
            record["word_indices"].append(entry["word_idx"])
            record["group_ids"].append(group_id)
            record["sub_group_ids"].append(sub_group_id)
            record["row_ids"].append(entry["row_id"])
            record["line_texts"].append(line_text)

            stats["category_counts"][entry["category"]] += 1
            stats["word_label_counts"][label] += 1
            if entry["category"] != "O":
                stats["category_to_label_base"][entry["category"]] = category_to_label_base(entry["category"])

    lengths = {
        "words": len(record["words"]),
        "boxes": len(record["boxes"]),
        "normalized_boxes": len(record["normalized_boxes"]),
        "labels": len(record["labels"]),
        "categories": len(record["categories"]),
        "line_ids": len(record["line_ids"]),
        "word_indices": len(record["word_indices"]),
        "group_ids": len(record["group_ids"]),
        "sub_group_ids": len(record["sub_group_ids"]),
        "row_ids": len(record["row_ids"]),
        "line_texts": len(record["line_texts"]),
    }
    if len(set(lengths.values())) != 1:
        message = f"{sample_id}: field length mismatch: {lengths}"
        if args.strict:
            fail(message)
        stats["skipped_samples"].append({"id": sample_id, "reason": message})
        return None

    if not record["words"]:
        message = f"{sample_id}: converted record has no words"
        if args.strict:
            fail(message)
        stats["skipped_samples"].append({"id": sample_id, "reason": "no words"})
        return None

    return record, len(valid_lines)


def new_stats():
    return {
        "category_counts": Counter(),
        "line_category_counts": Counter(),
        "word_label_counts": Counter(),
        "unknown_categories": set(),
        "skipped_samples": [],
        "invalid_boxes": [],
        "category_to_label_base": {},
    }


def get_splits_to_process(dataset, split):
    if split is not None:
        if split not in dataset:
            available = ", ".join(dataset.keys())
            fail(f"Split '{split}' not found. Available splits: {available}")
        return [split]
    return list(dataset.keys())


def collect_category_counts(dataset, splits):
    counts = Counter()
    for split in splits:
        for sample in tqdm(dataset[split], desc=f"inspect categories {split}"):
            ground_truth = parse_ground_truth(sample["ground_truth"])
            valid_lines = ground_truth.get("valid_line", [])
            if not isinstance(valid_lines, list):
                continue
            for line in valid_lines:
                if isinstance(line, dict):
                    counts[normalize_category(line.get("category", "O"))] += 1
    return counts


def print_preview(record, limit=30):
    print(f"preview id: {record['id']}")
    print(f"image_size: {record['image_size']}")
    print(f"num_words: {len(record['words'])}")
    print(f"first {min(limit, len(record['words']))} converted words:")
    for idx in range(min(limit, len(record["words"]))):
        print(
            f"[{idx}] word={record['words'][idx]!r} "
            f"box={record['boxes'][idx]} "
            f"norm={record['normalized_boxes'][idx]} "
            f"cat={record['categories'][idx]} "
            f"label={record['labels'][idx]} "
            f"line={record['line_ids'][idx]} "
            f"group={record['group_ids'][idx]} "
            f"sub_group={record['sub_group_ids'][idx]} "
            f"row={record['row_ids'][idx]}"
        )


def inspect_only(dataset, splits, args):
    print("inspect_only mode: no files will be written")
    first_split = splits[0]
    first_sample = dataset[first_split][0]
    print(f"first inspected split/index: {first_split}/0")
    print(f"first sample keys: {list(first_sample.keys())}")
    image = ensure_pil_rgb(first_sample["image"])
    print(f"first sample image size: width={image.size[0]}, height={image.size[1]}")
    ground_truth = parse_ground_truth(first_sample["ground_truth"])
    inspect_ground_truth(ground_truth)

    category_counts = collect_category_counts(dataset, splits)
    unknown = sorted(category for category in category_counts if category != "O" and category not in KNOWN_CATEGORIES)
    print("category list:")
    print(json.dumps(sorted(category_counts), ensure_ascii=False, indent=2))
    print("unknown categories compared to hard-coded list:")
    print(json.dumps(unknown, ensure_ascii=False, indent=2))
    print("top 30 category counts:")
    for category, count in category_counts.most_common(30):
        print(f"  {category}: {count}")

    stats = new_stats()
    converted = convert_sample(first_sample, first_split, 0, args, stats)
    if converted is None:
        print("first sample conversion preview unavailable")
        return
    record, _ = converted
    print_preview(record, limit=30)


def build_label_payload(category_to_label_base):
    bases = sorted(set(category_to_label_base.values()))
    label_list = ["O"]
    for base in bases:
        if base == "O":
            continue
        label_list.append(f"B-{base}")
        label_list.append(f"I-{base}")
    label2id = {label: idx for idx, label in enumerate(label_list)}
    id2label = {str(idx): label for label, idx in label2id.items()}
    return {
        "label_list": label_list,
        "label2id": label2id,
        "id2label": id2label,
        "category_to_label_base": dict(sorted(category_to_label_base.items())),
    }


def sanity_check(out_dir, splits, label_list):
    label_set = set(label_list)
    warnings = []
    total_labels = Counter()
    total_categories = Counter()
    total_words = 0
    total_o = 0

    print("sanity check:")
    for split in splits:
        path = out_dir / f"{split}.jsonl"
        split_samples = 0
        split_words = 0
        split_lines = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                record = json.loads(line)
                split_samples += 1
                field_lengths = {
                    key: len(record[key])
                    for key in (
                        "words",
                        "boxes",
                        "normalized_boxes",
                        "labels",
                        "categories",
                        "line_ids",
                        "word_indices",
                        "group_ids",
                        "sub_group_ids",
                        "row_ids",
                        "line_texts",
                    )
                }
                if len(set(field_lengths.values())) != 1:
                    warnings.append(f"{path}:{line_number}: length mismatch {field_lengths}")
                if not record["words"]:
                    warnings.append(f"{path}:{line_number}: zero words")

                previous_by_line = {}
                for idx, label in enumerate(record["labels"]):
                    total_labels[label] += 1
                    total_categories[record["categories"][idx]] += 1
                    split_words += 1
                    total_words += 1
                    if label == "O":
                        total_o += 1
                    if label not in label_set:
                        warnings.append(f"{path}:{line_number}: label not in label_list: {label}")
                    if label != "O" and not (label.startswith("B-") or label.startswith("I-")):
                        warnings.append(f"{path}:{line_number}: malformed label: {label}")
                    if label == "O" and (label.startswith("B-") or label.startswith("I-")):
                        warnings.append(f"{path}:{line_number}: O label has BIO prefix")

                    normalized_box = record["normalized_boxes"][idx]
                    if any(value < 0 or value > 1000 for value in normalized_box):
                        warnings.append(f"{path}:{line_number}: normalized box out of range: {normalized_box}")

                    line_id = record["line_ids"][idx]
                    if label.startswith("I-"):
                        base = label[2:]
                        prev_label = previous_by_line.get(line_id)
                        if prev_label not in {f"B-{base}", f"I-{base}"}:
                            warnings.append(
                                f"{path}:{line_number}: I-{base} at token {idx} does not follow same line BIO"
                            )
                    previous_by_line[line_id] = label

                split_lines += len(set(record["line_ids"]))

        print(f"  {split}: samples={split_samples}, words={split_words}, lines={split_lines}")

    o_ratio = total_o / total_words if total_words else 0.0
    print(f"O ratio: {o_ratio:.6f}")
    print("top 20 label counts:")
    for label, count in total_labels.most_common(20):
        print(f"  {label}: {count}")
    print("top 20 category counts:")
    for category, count in total_categories.most_common(20):
        print(f"  {category}: {count}")
    if warnings:
        print("sanity warnings:")
        for warning in warnings[:100]:
            print(f"  {warning}")
    else:
        print("sanity check passed: no warnings")
    return warnings


def prepare_out_dir(out_dir, overwrite):
    if out_dir.exists():
        if not overwrite:
            fail(f"Output directory already exists: {out_dir}. Use --overwrite to replace it.")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)


def write_json(path, payload):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def convert_dataset(dataset, splits, args):
    out_dir = Path(args.out_dir)
    prepare_out_dir(out_dir, args.overwrite)

    stats = new_stats()
    summary = {
        "data_dir": args.data_dir,
        "out_dir": args.out_dir,
        "splits": {},
        "num_labels": 0,
        "label_list": [],
        "notes": [
            "This is word-level BIO data converted from CORD-v2 valid_line categories.",
            "No LayoutLMv3 forward/fine-tuning was performed in this step.",
            "Boxes are stored both as pixel boxes and 0-1000 normalized boxes.",
        ],
    }

    first_records = {}
    for split in splits:
        split_dataset = dataset[split]
        input_count = len(split_dataset)
        limit = input_count if args.max_samples is None else min(args.max_samples, input_count)
        written = 0
        split_words = 0
        split_lines = 0
        split_path = out_dir / f"{split}.jsonl"

        with split_path.open("w", encoding="utf-8") as handle:
            for index in tqdm(range(limit), desc=f"convert {split}"):
                converted = convert_sample(split_dataset[index], split, index, args, stats)
                if converted is None:
                    continue
                record, num_lines = converted
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
                split_words += len(record["words"])
                split_lines += num_lines
                first_records.setdefault(split, record)

        skipped_for_split = [
            item for item in stats["skipped_samples"] if item.get("id", "").startswith(f"{split}_")
        ]
        summary["splits"][split] = {
            "num_input_samples": input_count,
            "num_processed_samples": limit,
            "num_written_samples": written,
            "num_skipped_samples": len(skipped_for_split),
            "num_words": split_words,
            "num_lines": split_lines,
        }

    label_payload = build_label_payload(stats["category_to_label_base"])
    summary["num_labels"] = len(label_payload["label_list"])
    summary["label_list"] = label_payload["label_list"]

    category_stats = {
        "category_counts": dict(stats["category_counts"].most_common()),
        "line_category_counts": dict(stats["line_category_counts"].most_common()),
        "word_label_counts": dict(stats["word_label_counts"].most_common()),
        "unknown_categories": sorted(stats["unknown_categories"]),
        "skipped_samples": stats["skipped_samples"],
        "invalid_boxes": stats["invalid_boxes"],
    }

    write_json(out_dir / "labels.json", label_payload)
    write_json(out_dir / "category_stats.json", category_stats)
    write_json(out_dir / "summary.json", summary)

    warnings = sanity_check(out_dir, splits, label_payload["label_list"])
    if warnings:
        print(f"sanity check completed with {len(warnings)} warnings")
    else:
        print("sanity check passed")

    print("debug preview for each split:")
    for split in splits:
        if split in first_records:
            print_preview(first_records[split], limit=30)

    print("written files:")
    for path in [
        out_dir / "labels.json",
        out_dir / "category_stats.json",
        out_dir / "summary.json",
        *[out_dir / f"{split}.jsonl" for split in splits],
    ]:
        print(f"  {path}")


def main():
    args = parse_args()
    print(f"WSL/conda Python path: {sys.executable}")
    print(f"dataset path: {args.data_dir}")
    dataset = load_dataset_or_fail(args.data_dir)
    print(f"dataset keys: {list(dataset.keys())}")
    print_splits(dataset)
    splits = get_splits_to_process(dataset, args.split)
    print(f"splits to process: {splits}")
    if args.max_samples is not None:
        print(f"max_samples per split: {args.max_samples}")

    first_split = splits[0]
    first_sample = dataset[first_split][0]
    print(f"first sample keys: {list(first_sample.keys())}")
    inspect_ground_truth(parse_ground_truth(first_sample["ground_truth"]))

    if args.inspect_only:
        inspect_only(dataset, splits, args)
        return

    convert_dataset(dataset, splits, args)


if __name__ == "__main__":
    main()
