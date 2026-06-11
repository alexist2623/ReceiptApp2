import argparse
import json
import sys
from hashlib import md5
from io import BytesIO
from pathlib import Path

from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(
        description="Draw CORD-v2 ground-truth boxes on sample receipt images."
    )
    parser.add_argument(
        "--data_dir",
        default="../receipt_training_data2",
        help="CORD-v2 dataset directory readable by datasets.load_from_disk.",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to inspect. Validated against dataset.keys().",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Sample index to overlay.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="If provided, save this many consecutive samples starting at index.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/cord_gt_overlay",
        help="Directory for overlay images and debug JSON files.",
    )
    parser.add_argument(
        "--mode",
        default="both",
        choices=("word", "line", "both"),
        help="Draw word boxes, line union boxes, or both.",
    )
    parser.add_argument(
        "--show_text",
        action="store_true",
        help="Append OCR text to overlay labels.",
    )
    parser.add_argument(
        "--max_text_len",
        type=int,
        default=30,
        help="Maximum text length shown in overlay labels.",
    )
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_dataset_or_fail(data_dir):
    path = Path(data_dir)
    if not path.exists():
        fail(f"CORD-v2 dataset not found at {data_dir}. Run download step first.")
    return load_from_disk(str(path))


def print_splits(dataset):
    lengths = {split: len(dataset[split]) for split in dataset.keys()}
    print("Available splits and lengths:")
    for split, length in lengths.items():
        print(f"  {split}: {length}")


def ensure_pil_image(image):
    if isinstance(image, Image.Image):
        return image

    if isinstance(image, (str, Path)):
        return Image.open(image)

    if isinstance(image, bytes):
        return Image.open(BytesIO(image))

    if isinstance(image, dict):
        image_path = image.get("path")
        image_bytes = image.get("bytes")
        if image_path:
            return Image.open(image_path)
        if image_bytes:
            return Image.open(BytesIO(image_bytes))

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


def to_int(value):
    return int(round(float(value)))


def normalize_box(values):
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
            values = [
                quad_or_box["x1"],
                quad_or_box["y1"],
                quad_or_box["x2"],
                quad_or_box["y2"],
                quad_or_box["x3"],
                quad_or_box["y3"],
                quad_or_box["x4"],
                quad_or_box["y4"],
            ]
            return normalize_box(values)

        if {"x0", "y0", "x1", "y1"}.issubset(quad_or_box):
            return normalize_box(
                [quad_or_box["x0"], quad_or_box["y0"], quad_or_box["x1"], quad_or_box["y1"]]
            )

        if {"left", "top", "right", "bottom"}.issubset(quad_or_box):
            return normalize_box(
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
            return normalize_box(quad_or_box)

    return None


def clamp_box(box, image_width, image_height):
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, image_width - 1))
    x1 = max(0, min(x1, image_width - 1))
    y0 = max(0, min(y0, image_height - 1))
    y1 = max(0, min(y1, image_height - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


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


def truncate_text(text, max_text_len):
    text = str(text or "")
    if len(text) <= max_text_len:
        return text
    if max_text_len <= 3:
        return text[:max_text_len]
    return text[: max_text_len - 3] + "..."


def category_color(category):
    digest = md5(str(category).encode("utf-8")).digest()
    return tuple(80 + channel % 156 for channel in digest[:3])


def draw_rectangle(draw, box, color, width):
    try:
        draw.rectangle(box, outline=color, width=width)
    except TypeError:
        x0, y0, x1, y1 = box
        for offset in range(width):
            draw.rectangle(
                [x0 - offset, y0 - offset, x1 + offset, y1 + offset],
                outline=color,
            )


def draw_label(draw, label, anchor_box, color, font):
    x0, y0, x1, _ = anchor_box
    try:
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
    except AttributeError:
        text_width, text_height = draw.textsize(label, font=font)

    label_x = x0
    label_y = y0 - text_height - 3 if y0 - text_height - 3 >= 0 else y0 + 2
    label_bg = [
        label_x,
        label_y,
        min(label_x + text_width + 4, max(x1 + text_width, label_x + text_width + 4)),
        label_y + text_height + 2,
    ]
    draw.rectangle(label_bg, fill=(255, 255, 255))
    draw.text((label_x + 2, label_y + 1), label, fill=color, font=font)


def line_text_from_words(words):
    return " ".join(word["text"] for word in words if word.get("text")).strip()


def inspect_ground_truth(ground_truth):
    keys = list(ground_truth.keys()) if isinstance(ground_truth, dict) else []
    print(f"ground_truth top-level keys: {keys}")

    gt_parse = ground_truth.get("gt_parse") if isinstance(ground_truth, dict) else None
    if isinstance(gt_parse, dict):
        print(f"gt_parse keys: {list(gt_parse.keys())}")
    elif gt_parse is not None:
        print(f"gt_parse type: {type(gt_parse)}")

    meta = ground_truth.get("meta") if isinstance(ground_truth, dict) else None
    if meta is not None:
        print(f"meta: {meta}")

    valid_line = ground_truth.get("valid_line") if isinstance(ground_truth, dict) else None
    if not isinstance(valid_line, list):
        print("ground_truth structure:")
        print(json.dumps(ground_truth, ensure_ascii=False, indent=2)[:4000])
        fail("valid_line not found")

    print(f"valid_line count: {len(valid_line)}")
    return valid_line


def parse_lines(valid_lines, image_width, image_height):
    parsed_lines = []
    total_word_boxes = 0
    total_line_boxes = 0

    for line_idx, line in enumerate(valid_lines):
        if not isinstance(line, dict):
            print(f"WARNING: skipping non-dict line at index {line_idx}: {type(line)}")
            continue

        words_raw = line.get("words", [])
        if not isinstance(words_raw, list):
            print(f"WARNING: line {line_idx} has non-list words: {type(words_raw)}")
            words_raw = []

        category = line.get("category", "UNKNOWN")
        group_id = line.get("group_id", None)
        sub_group_id = line.get("sub_group_id", None)
        row_id = line.get("row_id", None)
        parsed_words = []

        for word_idx, word in enumerate(words_raw):
            if not isinstance(word, dict):
                print(f"WARNING: skipping non-dict word at line {line_idx}, word {word_idx}: {type(word)}")
                continue

            if row_id is None:
                row_id = word.get("row_id", None)

            raw_box = quad_to_box(get_word_quad(word))
            if raw_box is None:
                print(f"WARNING: missing/unsupported box at line {line_idx}, word {word_idx}")
                continue

            box = clamp_box(raw_box, image_width, image_height)
            if box is None:
                print(f"WARNING: skipped empty box at line {line_idx}, word {word_idx}: {raw_box}")
                continue

            parsed_words.append(
                {
                    "word_idx": word_idx,
                    "text": get_word_text(word),
                    "box": box,
                }
            )
            total_word_boxes += 1

        line_box = None
        if parsed_words:
            line_box = [
                min(word["box"][0] for word in parsed_words),
                min(word["box"][1] for word in parsed_words),
                max(word["box"][2] for word in parsed_words),
                max(word["box"][3] for word in parsed_words),
            ]
            total_line_boxes += 1

        line_text = line.get("text", "") or line_text_from_words(parsed_words)
        parsed_lines.append(
            {
                "line_idx": line_idx,
                "text": line_text,
                "category": category,
                "group_id": group_id,
                "sub_group_id": sub_group_id,
                "row_id": row_id,
                "line_box": line_box,
                "words": parsed_words,
            }
        )

    return parsed_lines, total_word_boxes, total_line_boxes


def make_line_label(line, show_text, max_text_len):
    label = (
        f"L{line['line_idx']} {line['category']} "
        f"g={line['group_id']} sg={line['sub_group_id']} r={line['row_id']}"
    )
    if show_text:
        label += f": {truncate_text(line['text'], max_text_len)}"
    return label


def make_word_label(line, word, show_text, max_text_len):
    label = f"W{line['line_idx']}.{word['word_idx']} {line['category']}"
    if show_text:
        label += f": {truncate_text(word['text'], max_text_len)}"
    return label


def draw_overlay(image, lines, mode, show_text, max_text_len):
    overlay = image.convert("RGB")
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    if mode in ("line", "both"):
        for line in lines:
            if not line["line_box"]:
                continue
            color = category_color(line["category"])
            draw_rectangle(draw, line["line_box"], color, width=4)
            draw_label(draw, make_line_label(line, show_text, max_text_len), line["line_box"], color, font)

    if mode in ("word", "both"):
        for line in lines:
            color = category_color(line["category"])
            for word in line["words"]:
                draw_rectangle(draw, word["box"], color, width=1)
                draw_label(draw, make_word_label(line, word, show_text, max_text_len), word["box"], color, font)

    return overlay


def process_sample(dataset_split, split, index, out_dir, mode, show_text, max_text_len):
    sample = dataset_split[index]
    print("")
    print(f"Selected split/index: {split}/{index}")
    print(f"sample keys: {list(sample.keys())}")

    image = ensure_pil_image(sample["image"])
    image_width, image_height = image.size
    print(f"image size: width={image_width}, height={image_height}")

    try:
        ground_truth = parse_ground_truth(sample["ground_truth"])
    except Exception as exc:
        fail(str(exc))

    valid_lines = inspect_ground_truth(ground_truth)
    lines, total_word_boxes, total_line_boxes = parse_lines(valid_lines, image_width, image_height)

    print(f"total word boxes: {total_word_boxes}")
    print(f"total line boxes: {total_line_boxes}")

    split_out_dir = Path(out_dir) / split
    split_out_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = split_out_dir / f"{split}_{index:06d}_{mode}.png"
    debug_path = split_out_dir / f"{split}_{index:06d}_debug.json"

    overlay = draw_overlay(image, lines, mode, show_text, max_text_len)
    overlay.save(overlay_path)

    debug_payload = {
        "split": split,
        "index": index,
        "image_width": image_width,
        "image_height": image_height,
        "ground_truth_keys": list(ground_truth.keys()),
        "num_valid_lines": len(valid_lines),
        "num_word_boxes": total_word_boxes,
        "num_line_boxes": total_line_boxes,
        "lines": lines,
    }
    with debug_path.open("w", encoding="utf-8") as handle:
        json.dump(debug_payload, handle, ensure_ascii=False, indent=2)

    print(f"saved overlay path: {overlay_path}")
    print(f"saved debug json path: {debug_path}")
    return {
        "index": index,
        "overlay_path": overlay_path,
        "debug_path": debug_path,
        "num_valid_lines": len(valid_lines),
        "num_word_boxes": total_word_boxes,
        "num_line_boxes": total_line_boxes,
    }


def main():
    args = parse_args()
    print(f"dataset path: {args.data_dir}")
    dataset = load_dataset_or_fail(args.data_dir)
    print_splits(dataset)

    if args.split not in dataset:
        available = ", ".join(dataset.keys())
        fail(f"Split '{args.split}' not found. Available splits: {available}")

    split_dataset = dataset[args.split]
    split_length = len(split_dataset)
    if args.index < 0 or args.index >= split_length:
        fail(f"Index {args.index} is out of range for split '{args.split}' with length {split_length}.")

    if args.num_samples is None:
        sample_indices = [args.index]
    else:
        if args.num_samples <= 0:
            fail("--num_samples must be greater than 0.")
        end_index = args.index + args.num_samples
        if end_index > split_length:
            fail(
                f"Requested samples [{args.index}, {end_index}) exceed split '{args.split}' "
                f"length {split_length}."
            )
        sample_indices = list(range(args.index, end_index))

    summaries = []
    for sample_index in sample_indices:
        summaries.append(
            process_sample(
                dataset_split=split_dataset,
                split=args.split,
                index=sample_index,
                out_dir=args.out_dir,
                mode=args.mode,
                show_text=args.show_text,
                max_text_len=args.max_text_len,
            )
        )

    print("")
    print("Completed overlay generation:")
    for summary in summaries:
        print(
            f"  {args.split}/{summary['index']}: "
            f"valid_lines={summary['num_valid_lines']}, "
            f"word_boxes={summary['num_word_boxes']}, "
            f"line_boxes={summary['num_line_boxes']}"
        )


if __name__ == "__main__":
    main()
