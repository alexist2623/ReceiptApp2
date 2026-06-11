import argparse
import json
import sys
from io import BytesIO
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image
from transformers import AutoModel, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a LayoutLMv3-base forward smoke test with CORD-v2 GT words and boxes."
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
        help="Sample index for the forward smoke test.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="microsoft/layoutlmv3-base",
        help="Hugging Face model id or local model directory.",
    )
    parser.add_argument(
        "--local_model_dir",
        default="models/layoutlmv3-base",
        help="Directory used by --save_local_model.",
    )
    parser.add_argument(
        "--save_local_model",
        action="store_true",
        help="Save processor/model with save_pretrained to local_model_dir.",
    )
    parser.add_argument(
        "--local_files_only",
        action="store_true",
        help="Load processor/model from local files only.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "cuda"),
        help="Device selection. auto uses cuda when available, otherwise cpu.",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length for the processor encoding.",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/layoutlmv3_smoke",
        help="Directory for smoke test debug JSON files.",
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
    print("available splits and lengths:")
    for split, length in lengths.items():
        print(f"  {split}: {length}")


def ensure_pil_rgb(image):
    if isinstance(image, Image.Image):
        return image.convert("RGB")

    if isinstance(image, (str, Path)):
        return Image.open(image).convert("RGB")

    if isinstance(image, bytes):
        return Image.open(BytesIO(image)).convert("RGB")

    if isinstance(image, dict):
        image_path = image.get("path")
        image_bytes = image.get("bytes")
        if image_path:
            return Image.open(image_path).convert("RGB")
        if image_bytes:
            return Image.open(BytesIO(image_bytes)).convert("RGB")

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


def quad_points_to_box(values):
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
            return quad_points_to_box(
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
            return quad_points_to_box(
                [quad_or_box["x0"], quad_or_box["y0"], quad_or_box["x1"], quad_or_box["y1"]]
            )

        if {"left", "top", "right", "bottom"}.issubset(quad_or_box):
            return quad_points_to_box(
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
            return quad_points_to_box(quad_or_box)

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


def extract_words_and_boxes(ground_truth, width, height):
    print(f"ground_truth top-level keys: {list(ground_truth.keys())}")
    valid_lines = ground_truth.get("valid_line")
    if not isinstance(valid_lines, list):
        fail("valid_line not found in ground_truth.")

    print(f"valid_line count: {len(valid_lines)}")
    words = []
    pixel_boxes = []
    normalized_boxes = []

    for line_idx, line in enumerate(valid_lines):
        if not isinstance(line, dict):
            print(f"WARNING: skipping non-dict line at index {line_idx}: {type(line)}")
            continue

        line_words = line.get("words", [])
        if not isinstance(line_words, list):
            print(f"WARNING: line {line_idx} has non-list words: {type(line_words)}")
            continue

        for word_idx, word in enumerate(line_words):
            if not isinstance(word, dict):
                print(f"WARNING: skipping non-dict word at line {line_idx}, word {word_idx}: {type(word)}")
                continue

            text = str(get_word_text(word)).strip()
            if not text:
                print(f"WARNING: skipped empty text at line {line_idx}, word {word_idx}")
                continue

            raw_box = quad_to_box(get_word_quad(word))
            if raw_box is None:
                print(f"WARNING: missing/unsupported box at line {line_idx}, word {word_idx}")
                continue

            pixel_box = clamp_box(raw_box, width, height)
            if pixel_box is None:
                print(f"WARNING: skipped invalid box at line {line_idx}, word {word_idx}: {raw_box}")
                continue

            words.append(text)
            pixel_boxes.append(pixel_box)
            normalized_boxes.append(normalize_box(pixel_box, width, height))

    if not words:
        fail("No OCR words with valid boxes were extracted from valid_line.")

    if not (len(words) == len(pixel_boxes) == len(normalized_boxes)):
        fail(
            "words, pixel_boxes, and normalized_boxes length mismatch: "
            f"{len(words)}, {len(pixel_boxes)}, {len(normalized_boxes)}"
        )

    print(f"extracted word count: {len(words)}")
    print(f"pixel box count: {len(pixel_boxes)}")
    print(f"normalized box count: {len(normalized_boxes)}")
    print("first 10 words + normalized boxes:")
    for word, box in list(zip(words, normalized_boxes))[:10]:
        print(f"  {word}: {box}")
    return words, pixel_boxes, normalized_boxes, len(valid_lines)


def resolve_model_path(args):
    local_dir = Path(args.local_model_dir)
    if args.local_files_only and local_dir.exists():
        return str(local_dir)
    return args.model_name_or_path


def select_device(device_arg):
    cuda_available = torch.cuda.is_available()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not cuda_available:
            fail("--device cuda was requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if cuda_available else "cpu")


def tensor_shape(value):
    if hasattr(value, "shape"):
        return list(value.shape)
    return None


def save_local_model(processor, model, local_model_dir):
    local_dir = Path(local_model_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    processor.save_pretrained(str(local_dir))
    model.save_pretrained(str(local_dir))

    print(f"saved local model dir: {local_dir}")
    print("local model files:")
    for path in sorted(local_dir.iterdir()):
        if path.is_file():
            print(f"  {path.name}")


def main():
    args = parse_args()
    print(f"WSL/conda Python path: {sys.executable}")
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

    print(f"selected split/index: {args.split}/{args.index}")
    sample = split_dataset[args.index]
    print(f"sample keys: {list(sample.keys())}")

    image = ensure_pil_rgb(sample["image"])
    image_width, image_height = image.size
    print(f"image size: width={image_width}, height={image_height}")

    try:
        ground_truth = parse_ground_truth(sample["ground_truth"])
    except Exception as exc:
        fail(str(exc))

    words, pixel_boxes, normalized_boxes, valid_line_count = extract_words_and_boxes(
        ground_truth, image_width, image_height
    )

    model_path = resolve_model_path(args)
    print(f"model path: {model_path}")
    print("processor apply_ocr=False")

    processor = AutoProcessor.from_pretrained(
        model_path,
        apply_ocr=False,
        local_files_only=args.local_files_only,
    )
    model = AutoModel.from_pretrained(
        model_path,
        local_files_only=args.local_files_only,
    )

    if args.save_local_model:
        save_local_model(processor, model, args.local_model_dir)

    encoding = processor(
        image,
        words,
        boxes=normalized_boxes,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=args.max_length,
    )

    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"cuda device name: {torch.cuda.get_device_name(0)}")

    device = select_device(args.device)
    print(f"selected device: {device}")

    encoding_shapes = {}
    print("encoding shapes:")
    for key, value in encoding.items():
        shape = tensor_shape(value)
        encoding_shapes[key] = shape
        print(f"  {key}: {shape}")

    encoding = {key: value.to(device) for key, value in encoding.items()}
    model.to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(**encoding)

    print(f"outputs type: {type(outputs)}")
    if hasattr(outputs, "keys"):
        print(f"outputs keys: {list(outputs.keys())}")
    else:
        print(f"outputs attributes: {[name for name in dir(outputs) if not name.startswith('_')][:20]}")

    last_hidden_shape = list(outputs.last_hidden_state.shape)
    print(f"last_hidden_state shape: {last_hidden_shape}")
    pooler_shape = None
    if getattr(outputs, "pooler_output", None) is not None:
        pooler_shape = list(outputs.pooler_output.shape)
        print(f"pooler_output shape: {pooler_shape}")

    split_out_dir = Path(args.out_dir) / args.split
    split_out_dir.mkdir(parents=True, exist_ok=True)
    debug_path = split_out_dir / f"{args.split}_{args.index:06d}_smoke.json"
    debug_payload = {
        "split": args.split,
        "index": args.index,
        "image_width": image_width,
        "image_height": image_height,
        "num_words": len(words),
        "first_words": words[:10],
        "first_normalized_boxes": normalized_boxes[:10],
        "encoding_shapes": encoding_shapes,
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "device": str(device),
        "model_path": model_path,
        "last_hidden_state_shape": last_hidden_shape,
    }
    if pooler_shape is not None:
        debug_payload["pooler_output_shape"] = pooler_shape

    with debug_path.open("w", encoding="utf-8") as handle:
        json.dump(debug_payload, handle, ensure_ascii=False, indent=2)
    print(f"saved debug JSON path: {debug_path}")
    print("LayoutLMv3-base forward smoke test passed.")


if __name__ == "__main__":
    main()
