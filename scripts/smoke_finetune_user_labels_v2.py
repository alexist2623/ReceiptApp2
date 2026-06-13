import argparse
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import torch
from PIL import Image, ImageOps
from transformers import AutoModelForTokenClassification, AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import canonicalize_label


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smoke-test LayoutLMv3 fine-tuning with one or more user labeled receipt JSON files."
    )
    parser.add_argument("--image", required=True, help="Receipt image path for a single labeled JSON.")
    parser.add_argument("--label_json", required=True, help="Labeled receipt JSON path.")
    parser.add_argument("--label_schema", default="schemas/receipt_labels_v2.json")
    parser.add_argument("--model_name_or_path", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--out_dir", default="outputs/user_label_finetune_smoke")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def select_device(device):
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_image(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def parse_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def clamp_box(box, width, height):
    if box is None:
        return None
    x0, y0, x1, y1 = box
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def normalize_box(box, width, height):
    x0, y0, x1, y1 = box
    values = [
        int(1000 * x0 / width),
        int(1000 * y0 / height),
        int(1000 * x1 / width),
        int(1000 * y1 / height),
    ]
    return [max(0, min(value, 1000)) for value in values]


def load_label_schema(path):
    payload = load_json(path)
    label_list = list(payload["label_list"])
    label2id = {str(label): int(idx) for label, idx in payload["label2id"].items()}
    id2label = {int(idx): str(label) for idx, label in payload["id2label"].items()}
    if label2id.get("O") != 0:
        fail("label schema invalid: label2id['O'] must be 0.")
    return payload, label_list, label2id, id2label


def load_labeled_sample(image_path, label_json_path, label2id):
    image = load_image(image_path)
    width, height = image.size
    payload = load_json(label_json_path)
    words_payload = payload.get("words")
    if not isinstance(words_payload, list):
        fail("label_json must contain a words list.")
    top_labels = payload.get("labels")
    if top_labels is not None and len(top_labels) != len(words_payload):
        fail(f"words length {len(words_payload)} != labels length {len(top_labels)}")

    json_width = payload.get("image_width") or payload.get("width")
    json_height = payload.get("image_height") or payload.get("height")
    warnings = []
    if json_width is not None and json_height is not None and (int(json_width) != width or int(json_height) != height):
        warnings.append(f"image size mismatch: json={json_width}x{json_height}, actual={width}x{height}")

    words = []
    boxes = []
    normalized_boxes = []
    labels = []
    skipped = []
    unknown_labels = []
    for idx, item in enumerate(words_payload):
        if not isinstance(item, dict):
            skipped.append({"word_idx": idx, "reason": "word is not an object"})
            continue
        text = str(item.get("text", "")).strip()
        label = item.get("label")
        if label is None and top_labels is not None:
            label = top_labels[idx]
        canonical = canonicalize_label(label)
        box = clamp_box(parse_box(item.get("box")), width, height)
        if not text:
            skipped.append({"word_idx": idx, "reason": "empty text"})
            continue
        if box is None:
            skipped.append({"word_idx": idx, "text": text, "reason": "invalid box", "box": item.get("box")})
            continue
        if canonical not in label2id:
            unknown_labels.append({"word_idx": idx, "text": text, "label": label, "canonical_label": canonical})
            continue
        words.append(text)
        boxes.append(box)
        normalized_boxes.append(normalize_box(box, width, height))
        labels.append(canonical)

    if unknown_labels:
        first = unknown_labels[:20]
        fail(f"Unknown labels for schema {first}. Export/update schemas/receipt_labels_v2.json first.")
    if not words:
        fail("No valid labeled words found.")

    return {
        "image": image,
        "width": width,
        "height": height,
        "words": words,
        "boxes": boxes,
        "normalized_boxes": normalized_boxes,
        "labels": labels,
        "label_ids": [label2id[label] for label in labels],
        "warnings": warnings,
        "skipped": skipped,
    }


def encoding_with_labels(processor, sample, max_length):
    encoding = processor(
        sample["image"],
        sample["words"],
        boxes=sample["normalized_boxes"],
        word_labels=sample["label_ids"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if "labels" not in encoding or encoding["labels"].shape != encoding["input_ids"].shape:
        fail("Processor did not produce token-level labels with the expected input_ids shape.")
    return encoding


def token_label_preview(processor, encoding, sample, id2label, limit=80):
    word_ids = encoding.word_ids(batch_index=0)
    input_ids = encoding["input_ids"][0].tolist()
    labels = encoding["labels"][0].tolist()
    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids)
    rows = []
    for token_idx, (token, word_idx, label_id) in enumerate(zip(tokens, word_ids, labels)):
        if len(rows) >= limit:
            break
        label = id2label.get(label_id, "IGN") if label_id != -100 else "IGN"
        rows.append(
            {
                "token_idx": token_idx,
                "token": token,
                "word_idx": word_idx,
                "word": sample["words"][word_idx] if word_idx is not None and word_idx < len(sample["words"]) else None,
                "label": label,
            }
        )
    return rows


def main():
    args = parse_args()
    image_path = Path(args.image)
    label_json_path = Path(args.label_json)
    schema_path = Path(args.label_schema)
    model_path = Path(args.model_name_or_path)
    if not image_path.exists():
        fail(f"image not found: {image_path}")
    if not label_json_path.exists():
        fail(f"label_json not found: {label_json_path}")
    if not schema_path.exists():
        fail(f"label_schema not found: {schema_path}")
    if args.local_files_only and not model_path.exists():
        fail(f"model checkpoint not found: {model_path}")

    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    device = select_device(args.device)
    print(f"selected_device: {device}")
    if torch.cuda.is_available():
        print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")

    schema_payload, label_list, label2id, id2label = load_label_schema(schema_path)
    sample = load_labeled_sample(image_path, label_json_path, label2id)
    print(f"image_size: {sample['width']}x{sample['height']}")
    print(f"num_words: {len(sample['words'])}")
    print(f"num_labels: {len(label_list)}")
    print(f"label_counts: {dict(Counter(sample['labels']))}")

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        apply_ocr=False,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label={idx: label for idx, label in id2label.items()},
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    model.train()

    encoding = encoding_with_labels(processor, sample, args.max_length)
    encoding_shapes = {key: list(value.shape) for key, value in encoding.items() if hasattr(value, "shape")}
    non_ignored_labels = int((encoding["labels"] != -100).sum().item())
    print(f"encoding_shapes: {encoding_shapes}")
    print(f"non_ignored_token_labels: {non_ignored_labels}")
    if non_ignored_labels == 0:
        fail("No non-ignored token labels were produced.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses = []
    batch = {
        key: value.to(device)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids", "labels"}
    }
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            fail(f"Invalid loss at step {step}: {loss}")
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu().item())
        losses.append(value)
        print(f"step {step}/{args.steps} loss={value:.6f}")

    model.eval()
    with torch.no_grad():
        eval_outputs = model(**batch)
        eval_loss = float(eval_outputs.loss.detach().cpu().item())
    print(f"eval_loss_after_steps: {eval_loss:.6f}")

    preview = token_label_preview(processor, encoding, sample, id2label)
    report = {
        "image": str(image_path),
        "label_json": str(label_json_path),
        "label_schema": str(schema_path),
        "schema_version": schema_payload.get("schema_version"),
        "model_name_or_path": args.model_name_or_path,
        "local_files_only": args.local_files_only,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "num_words": len(sample["words"]),
        "num_labels": len(label_list),
        "label_counts": dict(Counter(sample["labels"])),
        "encoding_shapes": encoding_shapes,
        "non_ignored_token_labels": non_ignored_labels,
        "train_losses": losses,
        "eval_loss_after_steps": eval_loss,
        "warnings": sample["warnings"],
        "skipped": sample["skipped"],
        "token_label_preview": preview,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"{label_json_path.stem}_smoke_report.json"
    save_json(report_path, report)
    if args.debug:
        print("token label preview:")
        for row in preview[:80]:
            print(row)
    print(f"smoke report path: {report_path}")
    print("User labeled receipt fine-tune smoke run passed.")


if __name__ == "__main__":
    main()
