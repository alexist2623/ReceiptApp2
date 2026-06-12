import argparse
import json
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LayoutLMv3 token classification inference and draw CORD GT/pred overlays."
    )
    parser.add_argument("--bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--checkpoint", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--save_overlay_limit", type=int, default=50)
    parser.add_argument("--out_dir", default="outputs/cord_test_pred_overlay")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--show_text",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show word text in the overlay labels.",
    )
    parser.add_argument("--max_text_len", type=int, default=25)
    parser.add_argument("--hide_o", action="store_true")
    parser.add_argument("--draw_correct_only", action="store_true")
    parser.add_argument("--draw_errors_only", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_path(path, message):
    if not Path(path).exists():
        fail(message)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


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


def load_labels(bio_dir):
    labels_path = Path(bio_dir) / "labels.json"
    require_path(labels_path, f"{labels_path} not found. Run step 3 first.")
    with labels_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    label_list = payload["label_list"]
    label2id = {str(label): int(idx) for label, idx in payload["label2id"].items()}
    id2label = {int(idx): str(label) for idx, label in payload["id2label"].items()}

    if "O" not in label_list:
        fail("labels.json is invalid: 'O' is missing from label_list.")
    if label2id.get("O") != 0:
        fail("labels.json is invalid: label2id['O'] must be 0.")

    print(f"labels.json num_labels: {len(label_list)}")
    return label_list, label2id, id2label, payload


def validate_record(record, line_number, label2id, source_path):
    required = ("id", "split", "index", "words", "normalized_boxes", "labels")
    missing = [key for key in required if key not in record]
    if missing:
        fail(f"{source_path} line {line_number} is missing required fields: {missing}")

    lengths = {
        "words": len(record["words"]),
        "normalized_boxes": len(record["normalized_boxes"]),
        "labels": len(record["labels"]),
    }
    if len(set(lengths.values())) != 1:
        fail(f"{source_path} line {line_number} has length mismatch: {lengths}")

    for box in record["normalized_boxes"]:
        if len(box) != 4 or any(int(value) < 0 or int(value) > 1000 for value in box):
            fail(f"{source_path} line {line_number} has invalid normalized box: {box}")

    unknown_labels = [label for label in record["labels"] if label not in label2id]
    if unknown_labels:
        fail(
            f"{source_path} line {line_number} has labels not present in labels.json: "
            f"{unknown_labels[:5]}"
        )


def load_bio_records(bio_dir, split, label2id):
    jsonl_path = Path(bio_dir) / f"{split}.jsonl"
    require_path(jsonl_path, f"{jsonl_path} not found. Run step 3 first.")

    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_record(record, line_number, label2id, jsonl_path)
            records.append(record)
    if not records:
        fail(f"No records loaded from {jsonl_path}.")
    return records, jsonl_path


def select_records(records, args):
    if args.all:
        return records
    if args.num_samples is not None:
        if args.start_index < 0 or args.start_index >= len(records):
            fail(f"start_index {args.start_index} out of range for split length {len(records)}")
        return records[args.start_index : args.start_index + args.num_samples]

    matching = [record for record in records if int(record["index"]) == args.index]
    if matching:
        return [matching[0]]
    if 0 <= args.index < len(records):
        return [records[args.index]]
    fail(f"index {args.index} out of range for split length {len(records)}")


def load_raw_dataset(raw_data_dir, split):
    require_path(raw_data_dir, f"{raw_data_dir} not found. Run CORD-v2 download step first.")
    raw_dataset = load_from_disk(str(raw_data_dir))
    if split not in raw_dataset:
        fail(f"Raw dataset split '{split}' not found. Available splits: {list(raw_dataset.keys())}")
    return raw_dataset


def select_device(device_arg):
    cuda_available = torch.cuda.is_available()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not cuda_available:
            fail("--device cuda requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if cuda_available else "cpu")


def compare_model_labels(model, label_list, label2id, id2label):
    print(f"loaded model num_labels: {model.config.num_labels}")
    if model.config.num_labels != len(label_list):
        print(
            f"WARNING: model num_labels={model.config.num_labels} differs from "
            f"labels.json num_labels={len(label_list)}"
        )

    config_id2label = getattr(model.config, "id2label", {}) or {}
    mismatches = []
    for idx, label in id2label.items():
        config_label = config_id2label.get(idx, config_id2label.get(str(idx)))
        if config_label is not None and config_label != label:
            mismatches.append((idx, config_label, label))
    if mismatches:
        print(f"WARNING: checkpoint id2label differs from labels.json. First mismatches: {mismatches[:5]}")

    model.config.id2label = id2label
    model.config.label2id = label2id


def fallback_align_labels(encoding, word_label_ids):
    word_ids = encoding.word_ids(batch_index=0)
    token_labels = []
    previous_word_idx = None
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or encoding["attention_mask"][0, token_idx].item() == 0:
            token_labels.append(-100)
        elif word_idx != previous_word_idx:
            token_labels.append(int(word_label_ids[word_idx]))
        else:
            token_labels.append(-100)
        previous_word_idx = word_idx
    return torch.tensor([token_labels], dtype=torch.long)


def normalized_to_pixel_box(box, width, height):
    x0, y0, x1, y1 = [int(v) for v in box]
    pixel_box = [
        int(round(x0 * width / 1000)),
        int(round(y0 * height / 1000)),
        int(round(x1 * width / 1000)),
        int(round(y1 * height / 1000)),
    ]
    return clamp_box(pixel_box, width, height)


def clamp_box(box, width, height):
    x0, y0, x1, y1 = [int(v) for v in box]
    x0 = max(0, min(x0, width - 1))
    x1 = max(0, min(x1, width - 1))
    y0 = max(0, min(y0, height - 1))
    y1 = max(0, min(y1, height - 1))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def get_pixel_boxes(record, image):
    width, height = image.size
    if record.get("boxes"):
        return [clamp_box(box, width, height) for box in record["boxes"]]
    return [normalized_to_pixel_box(box, width, height) for box in record["normalized_boxes"]]


def predict_one_sample(record, image, processor, model, device, label2id, id2label, max_length):
    words = [str(word) for word in record["words"]]
    normalized_boxes = [[int(v) for v in box] for box in record["normalized_boxes"]]
    gt_labels = [str(label) for label in record["labels"]]
    word_label_ids = [label2id[label] for label in gt_labels]

    encoding = processor(
        image,
        words,
        boxes=normalized_boxes,
        word_labels=word_label_ids,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if "labels" not in encoding or list(encoding["labels"].shape) != list(encoding["input_ids"].shape):
        print("WARNING: processor did not return valid token labels; using fallback alignment.")
        encoding["labels"] = fallback_align_labels(encoding, word_label_ids)
    else:
        labels = encoding["labels"].clone()
        labels[encoding["attention_mask"] == 0] = -100
        encoding["labels"] = labels

    try:
        word_ids = encoding.word_ids(batch_index=0)
    except Exception as exc:
        fail(f"BatchEncoding.word_ids is required for word-level prediction restore: {exc}")

    model_inputs = {
        key: value.to(device)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"}
    }
    with torch.no_grad():
        outputs = model(**model_inputs)
        logits = outputs.logits.detach().cpu()[0]
        probs = torch.softmax(logits, dim=-1)
        pred_ids = logits.argmax(dim=-1)

    labels_tensor = encoding["labels"][0].detach().cpu()
    attention_mask = encoding["attention_mask"][0].detach().cpu()
    tokens = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())

    word_pred_labels = [None] * len(words)
    word_confidences = [None] * len(words)
    token_true_labels = []
    token_pred_labels = []
    token_debug = []

    for token_idx, word_idx in enumerate(word_ids):
        pred_id = int(pred_ids[token_idx].item())
        pred_label = id2label[pred_id]
        confidence = float(probs[token_idx, pred_id].item())
        label_id = int(labels_tensor[token_idx].item())
        gt_label = "IGN" if label_id == -100 else id2label[label_id]
        word_text = None if word_idx is None or word_idx >= len(words) else words[word_idx]

        if label_id != -100:
            token_true_labels.append(gt_label)
            token_pred_labels.append(pred_label)

        token_debug.append(
            {
                "token_idx": token_idx,
                "token": tokens[token_idx],
                "word_idx": word_idx,
                "word": word_text,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "confidence": confidence,
            }
        )

        if word_idx is None or word_idx >= len(words) or attention_mask[token_idx].item() == 0:
            continue
        if word_pred_labels[word_idx] is None:
            word_pred_labels[word_idx] = pred_label
            word_confidences[word_idx] = confidence

    missing_words = [idx for idx, label in enumerate(word_pred_labels) if label is None]
    if missing_words:
        print(f"ERROR: word-level predictions missing for word indices: {missing_words[:20]}", file=sys.stderr)
        print(f"word count={len(words)} max_length={max_length}", file=sys.stderr)
        print(f"first 40 word_ids={word_ids[:40]}", file=sys.stderr)
        fail("Could not restore word-level predictions for every word. Increase max_length or inspect truncation.")

    if len(word_pred_labels) != len(words) or len(gt_labels) != len(words):
        fail(
            f"Prediction length mismatch: words={len(words)} gt={len(gt_labels)} "
            f"pred={len(word_pred_labels)}"
        )

    pixel_boxes = get_pixel_boxes(record, image)
    correct = [gt == pred for gt, pred in zip(gt_labels, word_pred_labels)]
    return {
        "record_id": record["id"],
        "split": record["split"],
        "index": record["index"],
        "words": words,
        "boxes": pixel_boxes,
        "normalized_boxes": normalized_boxes,
        "gt_labels": gt_labels,
        "pred_labels": word_pred_labels,
        "confidences": word_confidences,
        "correct": correct,
        "token_level": {
            "true_labels": token_true_labels,
            "pred_labels": token_pred_labels,
        },
        "token_debug": token_debug,
        "labels_non_ignored": len(token_true_labels),
    }


def text_size(draw, text, font):
    try:
        bbox = draw.multiline_textbbox((0, 0), text, font=font)
        return bbox[2] - bbox[0], bbox[3] - bbox[1]
    except AttributeError:
        return draw.multiline_textsize(text, font=font)


def draw_rectangle(draw, box, color, width=2):
    try:
        draw.rectangle(box, outline=color, width=width)
    except TypeError:
        x0, y0, x1, y1 = box
        for offset in range(width):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)


def label_color(gt_label, pred_label):
    if gt_label == pred_label and gt_label != "O":
        return (20, 150, 80)
    if gt_label == "O" and pred_label == "O":
        return (130, 130, 130)
    if gt_label == "O" and pred_label != "O":
        return (220, 120, 20)
    return (210, 45, 45)


def should_draw_word(gt_label, pred_label, correct, args):
    if args.hide_o and gt_label == "O" and pred_label == "O":
        return False
    if args.draw_correct_only and not correct:
        return False
    if args.draw_errors_only and correct:
        return False
    return True


def draw_label(draw, box, lines, color, font, image_size):
    width, height = image_size
    text = "\n".join(lines)
    text_w, text_h = text_size(draw, text, font)
    x0, y0, _, y1 = box
    x = max(0, min(x0, width - text_w - 6))
    y = y0 - text_h - 6
    if y < 0:
        y = y1 + 2
    y = max(0, min(y, height - text_h - 6))
    draw.rectangle([x, y, x + text_w + 5, y + text_h + 5], fill=(255, 255, 255))
    draw.multiline_text((x + 2, y + 2), text, fill=color, font=font)


def save_overlay(image, prediction, record, args, output_path):
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()

    for idx, word in enumerate(prediction["words"]):
        gt_label = prediction["gt_labels"][idx]
        pred_label = prediction["pred_labels"][idx]
        correct = prediction["correct"][idx]
        if not should_draw_word(gt_label, pred_label, correct, args):
            continue
        confidence = prediction["confidences"][idx]
        box = prediction["boxes"][idx]
        color = label_color(gt_label, pred_label)
        draw_rectangle(draw, box, color, width=3 if not correct else 2)
        lines = [f"GT={gt_label}", f"P={pred_label} {confidence:.2f}"]
        if args.show_text:
            word_display = word[: args.max_text_len]
            lines.append(word_display)
        draw_label(draw, box, lines, color, font, output.size)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def optional_list_value(record, key, idx):
    values = record.get(key)
    if isinstance(values, list) and idx < len(values):
        return values[idx]
    return None


def build_compare_payload(record, image, prediction):
    width, height = image.size
    total_words = len(prediction["words"])
    correct_count = sum(prediction["correct"])
    non_o_indices = [idx for idx, label in enumerate(prediction["gt_labels"]) if label != "O"]
    non_o_correct = sum(1 for idx in non_o_indices if prediction["correct"][idx])
    word_results = []

    for idx, word in enumerate(prediction["words"]):
        word_results.append(
            {
                "word_idx": idx,
                "word": word,
                "box": prediction["boxes"][idx],
                "normalized_box": prediction["normalized_boxes"][idx],
                "gt_label": prediction["gt_labels"][idx],
                "pred_label": prediction["pred_labels"][idx],
                "confidence": prediction["confidences"][idx],
                "correct": prediction["correct"][idx],
                "category": optional_list_value(record, "categories", idx),
                "line_id": optional_list_value(record, "line_ids", idx),
                "group_id": optional_list_value(record, "group_ids", idx),
                "sub_group_id": optional_list_value(record, "sub_group_ids", idx),
                "row_id": optional_list_value(record, "row_ids", idx),
            }
        )

    return {
        "id": record["id"],
        "split": record["split"],
        "index": record["index"],
        "image_width": width,
        "image_height": height,
        "num_words": total_words,
        "word_accuracy": correct_count / total_words if total_words else 0.0,
        "non_o_word_accuracy": non_o_correct / len(non_o_indices) if non_o_indices else 0.0,
        "word_results": word_results,
    }


def compute_metrics(predictions):
    word_correct = 0
    word_total = 0
    non_o_correct = 0
    non_o_total = 0
    token_correct = 0
    token_total = 0
    true_sequences = []
    pred_sequences = []
    confusion = Counter()
    pred_label_counts = Counter()
    gt_label_counts = Counter()

    for prediction in predictions:
        gt_labels = prediction["gt_labels"]
        pred_labels = prediction["pred_labels"]
        true_sequences.append(gt_labels)
        pred_sequences.append(pred_labels)
        for gt_label, pred_label in zip(gt_labels, pred_labels):
            word_total += 1
            gt_label_counts[gt_label] += 1
            pred_label_counts[pred_label] += 1
            confusion[(gt_label, pred_label)] += 1
            if gt_label == pred_label:
                word_correct += 1
            if gt_label != "O":
                non_o_total += 1
                if gt_label == pred_label:
                    non_o_correct += 1

        token_true = prediction["token_level"]["true_labels"]
        token_pred = prediction["token_level"]["pred_labels"]
        for gt_label, pred_label in zip(token_true, token_pred):
            token_total += 1
            if gt_label == pred_label:
                token_correct += 1

    metrics = {
        "num_words": word_total,
        "num_non_o_words": non_o_total,
        "word_accuracy": word_correct / word_total if word_total else 0.0,
        "non_o_word_accuracy": non_o_correct / non_o_total if non_o_total else 0.0,
        "token_accuracy": token_correct / token_total if token_total else 0.0,
        "seqeval_precision": float(precision_score(true_sequences, pred_sequences)) if true_sequences else 0.0,
        "seqeval_recall": float(recall_score(true_sequences, pred_sequences)) if true_sequences else 0.0,
        "seqeval_f1": float(f1_score(true_sequences, pred_sequences)) if true_sequences else 0.0,
        "gt_label_counts": dict(gt_label_counts),
        "pred_label_counts": dict(pred_label_counts),
        "confusion": confusion,
        "true_sequences": true_sequences,
        "pred_sequences": pred_sequences,
    }
    return metrics


def write_report_files(args, metrics, num_samples, num_overlay_saved, device, cuda_available):
    out_dir = Path(args.out_dir)
    summary = {
        "checkpoint": args.checkpoint,
        "split": args.split,
        "num_samples": num_samples,
        "num_words": metrics["num_words"],
        "num_non_o_words": metrics["num_non_o_words"],
        "word_accuracy": metrics["word_accuracy"],
        "non_o_word_accuracy": metrics["non_o_word_accuracy"],
        "token_accuracy": metrics["token_accuracy"],
        "seqeval_precision": metrics["seqeval_precision"],
        "seqeval_recall": metrics["seqeval_recall"],
        "seqeval_f1": metrics["seqeval_f1"],
        "num_overlay_saved": num_overlay_saved,
        "device": str(device),
        "cuda_available": cuda_available,
    }
    save_json(out_dir / "metrics_summary.json", summary)

    report = classification_report(
        metrics["true_sequences"],
        metrics["pred_sequences"],
        digits=4,
    )
    (out_dir / "seqeval_report.txt").parent.mkdir(parents=True, exist_ok=True)
    with (out_dir / "seqeval_report.txt").open("w", encoding="utf-8") as handle:
        handle.write(report)

    confusion_top = [
        {"gt": gt, "pred": pred, "count": count}
        for (gt, pred), count in metrics["confusion"].most_common(50)
    ]
    save_json(out_dir / "confusion_top.json", confusion_top)

    run_config = {
        "bio_dir": args.bio_dir,
        "raw_data_dir": args.raw_data_dir,
        "checkpoint": args.checkpoint,
        "split": args.split,
        "index": args.index,
        "start_index": args.start_index,
        "num_samples": args.num_samples,
        "all": args.all,
        "save_overlay_limit": args.save_overlay_limit,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "device": args.device,
        "local_files_only": args.local_files_only,
        "hide_o": args.hide_o,
        "draw_correct_only": args.draw_correct_only,
        "draw_errors_only": args.draw_errors_only,
    }
    save_json(out_dir / "run_config.json", run_config)
    return summary, confusion_top


def print_debug_info(first_record, first_prediction, metrics):
    print("first sample first 50 words/boxes/GT labels:")
    for idx, word in enumerate(first_prediction["words"][:50]):
        print(
            f"  [{idx}] word={word!r} box={first_prediction['boxes'][idx]} "
            f"gt={first_prediction['gt_labels'][idx]}"
        )
    print("first 80 token alignment:")
    for item in first_prediction["token_debug"][:80]:
        print(
            f"  token_idx={item['token_idx']} token={item['token']!r} "
            f"word_idx={item['word_idx']} word={item['word']!r} "
            f"gt={item['gt_label']} pred={item['pred_label']} conf={item['confidence']:.4f}"
        )
    print(f"labels != -100 token count: {first_prediction['labels_non_ignored']}")
    total_words = metrics["num_words"]
    o_ratio = metrics["gt_label_counts"].get("O", 0) / total_words if total_words else 0.0
    print(f"O label ratio: {o_ratio:.6f}")
    print("prediction label distribution top 30:")
    for label, count in Counter(metrics["pred_label_counts"]).most_common(30):
        print(f"  {label}: {count}")
    print("error examples top 30:")
    errors = []
    for idx, (gt, pred) in enumerate(zip(first_prediction["gt_labels"], first_prediction["pred_labels"])):
        if gt != pred:
            errors.append((idx, first_prediction["words"][idx], gt, pred, first_prediction["confidences"][idx]))
    for idx, word, gt, pred, conf in errors[:30]:
        print(f"  [{idx}] word={word!r} gt={gt} pred={pred} conf={conf:.4f}")


def main():
    args = parse_args()

    print(f"WSL/conda Python path: {sys.executable}")
    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"cuda device name: {torch.cuda.get_device_name(0)}")
    device = select_device(args.device)
    print(f"selected device: {device}")

    split_jsonl = Path(args.bio_dir) / f"{args.split}.jsonl"
    require_path(split_jsonl, f"{split_jsonl} not found. Run step 3 first.")
    require_path(Path(args.bio_dir) / "labels.json", f"{args.bio_dir}/labels.json not found. Run step 3 first.")
    require_path(args.raw_data_dir, f"{args.raw_data_dir} not found. Run CORD-v2 download step first.")
    require_path(args.checkpoint, f"{args.checkpoint} not found. Run step 5 full fine-tuning first.")

    label_list, label2id, id2label, _labels_payload = load_labels(args.bio_dir)
    records, bio_split_path = load_bio_records(args.bio_dir, args.split, label2id)
    selected_records = select_records(records, args)
    if not selected_records:
        fail("No selected records to process.")

    print(f"checkpoint path: {args.checkpoint}")
    print(f"dataset path: {args.raw_data_dir}")
    print(f"bio split file path: {bio_split_path}")

    raw_dataset = load_raw_dataset(args.raw_data_dir, args.split)
    print(f"raw dataset splits and lengths: { {split: len(raw_dataset[split]) for split in raw_dataset.keys()} }")
    print(f"processing split: {args.split}")
    print(f"selected sample count: {len(selected_records)}")
    print(f"first sample id: {selected_records[0]['id']}")
    print(f"first sample word count: {len(selected_records[0]['words'])}")
    print(f"first 20 GT labels: {selected_records[0]['labels'][:20]}")

    print("processor apply_ocr=False")
    processor = AutoProcessor.from_pretrained(
        args.checkpoint,
        apply_ocr=False,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.checkpoint,
        local_files_only=args.local_files_only,
    )
    compare_model_labels(model, label_list, label2id, id2label)
    model.to(device)
    model.eval()

    out_dir = Path(args.out_dir)
    split_out_dir = out_dir / args.split
    split_out_dir.mkdir(parents=True, exist_ok=True)

    predictions = []
    overlay_paths = []
    compare_paths = []
    first_overlay_path = None
    first_compare_path = None
    overlay_limit = len(selected_records) if not args.all else args.save_overlay_limit

    for record_idx, record in enumerate(tqdm(selected_records, desc=f"inference {args.split}")):
        raw_index = int(record["index"])
        if raw_index < 0 or raw_index >= len(raw_dataset[args.split]):
            fail(f"{record['id']}: raw dataset index {raw_index} out of range for split length {len(raw_dataset[args.split])}")
        image = ensure_pil_rgb(raw_dataset[args.split][raw_index]["image"])
        prediction = predict_one_sample(
            record=record,
            image=image,
            processor=processor,
            model=model,
            device=device,
            label2id=label2id,
            id2label=id2label,
            max_length=args.max_length,
        )
        predictions.append(prediction)

        compare_payload = build_compare_payload(record, image, prediction)
        compare_path = split_out_dir / f"{args.split}_{raw_index:06d}_compare.json"
        save_json(compare_path, compare_payload)
        compare_paths.append(str(compare_path))
        if first_compare_path is None:
            first_compare_path = compare_path

        if record_idx < overlay_limit:
            overlay_path = split_out_dir / f"{args.split}_{raw_index:06d}_pred.png"
            save_overlay(image, prediction, record, args, overlay_path)
            overlay_paths.append(str(overlay_path))
            if first_overlay_path is None:
                first_overlay_path = overlay_path

    metrics = compute_metrics(predictions)
    metrics_summary, confusion_top = write_report_files(
        args=args,
        metrics=metrics,
        num_samples=len(selected_records),
        num_overlay_saved=len(overlay_paths),
        device=device,
        cuda_available=cuda_available,
    )

    first_prediction = predictions[0]
    print(f"first 20 pred labels after inference: {first_prediction['pred_labels'][:20]}")
    if first_overlay_path:
        print(f"output overlay path: {first_overlay_path}")
    if first_compare_path:
        print(f"output compare JSON path: {first_compare_path}")
    print(f"word accuracy: {metrics_summary['word_accuracy']:.6f}")
    print(f"non-O word accuracy: {metrics_summary['non_o_word_accuracy']:.6f}")
    print(f"token accuracy: {metrics_summary['token_accuracy']:.6f}")
    print(f"seqeval precision: {metrics_summary['seqeval_precision']:.6f}")
    print(f"seqeval recall: {metrics_summary['seqeval_recall']:.6f}")
    print(f"seqeval F1: {metrics_summary['seqeval_f1']:.6f}")
    print(f"metrics_summary path: {out_dir / 'metrics_summary.json'}")
    print(f"seqeval_report path: {out_dir / 'seqeval_report.txt'}")
    print(f"confusion_top path: {out_dir / 'confusion_top.json'}")

    if args.debug:
        print_debug_info(selected_records[0], first_prediction, metrics)
        print("confusion top 30:")
        for item in confusion_top[:30]:
            print(f"  GT={item['gt']} P={item['pred']} count={item['count']}")

    print("CORD test prediction overlay step passed.")


if __name__ == "__main__":
    main()
