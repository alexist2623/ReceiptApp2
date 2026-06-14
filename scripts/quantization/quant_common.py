import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image, ImageOps
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import canonicalize_label


KEY_FIELDS = [
    "ITEM_NAME",
    "ITEM_PRICE",
    "TOTAL_PRICE",
    "SUBTOTAL_PRICE",
    "TAX_PRICE",
]


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def file_size_mb(path):
    path = Path(path)
    return path.stat().st_size / (1024 * 1024) if path.exists() else None


def onnx_total_size_mb(path):
    path = Path(path)
    total = path.stat().st_size if path.exists() else 0
    external = Path(str(path) + ".data")
    if external.exists():
        total += external.stat().st_size
    return total / (1024 * 1024) if total else None


def load_labels(checkpoint):
    checkpoint = Path(checkpoint)
    label_path = checkpoint / "labels.json"
    if label_path.exists():
        payload = json.loads(label_path.read_text(encoding="utf-8"))
        label_list = payload.get("label_list")
        label2id = payload.get("label2id")
        id2label = payload.get("id2label")
    else:
        config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        raw_id2label = config.get("id2label", {})
        id2label = {str(k): v for k, v in raw_id2label.items()}
        label_list = [id2label[str(i)] for i in range(len(id2label))]
        label2id = {label: i for i, label in enumerate(label_list)}
        payload = {"label_list": label_list, "label2id": label2id, "id2label": id2label}
    if not label_list or not label2id:
        fail(f"Could not load labels from {checkpoint}")
    id2label_int = {int(k): v for k, v in id2label.items()}
    return payload, label_list, label2id, id2label_int


def load_cord_jsonl(cord_bio_dir, split, max_samples=None):
    path = Path(cord_bio_dir) / f"{split}.jsonl"
    if not path.exists():
        fail(f"CORD BIO split not found: {path}")
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def load_raw_dataset(cord_raw_data_dir):
    path = Path(cord_raw_data_dir)
    if not path.exists():
        fail(f"CORD raw dataset not found: {path}")
    return load_from_disk(str(path))


def image_to_rgb(image):
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    return Image.open(image).convert("RGB")


def canonicalize_sequence(labels):
    return [canonicalize_label(label) for label in labels]


def sanitize_box(box):
    return [max(0, min(int(v), 1000)) for v in box]


def prepare_cord_sample(record, raw_dataset, label2id, split=None):
    sample_split = split or record.get("split")
    index = int(record["index"])
    raw = raw_dataset[sample_split][index]
    image = image_to_rgb(raw["image"])
    width, height = image.size
    words = [str(word) for word in record["words"]]
    normalized_boxes = [sanitize_box(box) for box in record["normalized_boxes"]]
    raw_labels = [str(label) for label in record["labels"]]
    canonical_labels = canonicalize_sequence(raw_labels)
    keep_words = []
    keep_boxes = []
    keep_labels = []
    skipped = 0
    for word, box, raw_label, canonical_label in zip(words, normalized_boxes, raw_labels, canonical_labels):
        if not word.strip():
            skipped += 1
            continue
        if raw_label not in label2id:
            raise ValueError(f"Label {raw_label!r} is not in checkpoint label map.")
        keep_words.append(word)
        keep_boxes.append(box)
        keep_labels.append((raw_label, canonical_label))
    return {
        "id": record.get("id", f"{sample_split}_{index:06d}"),
        "source": "cord",
        "split": sample_split,
        "index": index,
        "image": image,
        "image_size": [width, height],
        "words": keep_words,
        "normalized_boxes": keep_boxes,
        "gold_labels": [label for label, _ in keep_labels],
        "canonical_gold_labels": [label for _, label in keep_labels],
        "label_ids": [label2id[label] for label, _ in keep_labels],
        "skipped_words": skipped,
    }


def encode_sample(processor, sample, max_length=512, include_labels=True):
    kwargs = {
        "images": sample["image"],
        "text": sample["words"],
        "boxes": sample["normalized_boxes"],
        "padding": "max_length",
        "truncation": True,
        "max_length": max_length,
        "return_tensors": "pt",
    }
    if include_labels:
        kwargs["word_labels"] = sample["label_ids"]
    return processor(**kwargs)


def first_token_word_predictions(logits, encoding, id2label, num_words, attention_mask=None):
    word_ids = encoding.word_ids(batch_index=0)
    probs = torch.softmax(torch.as_tensor(logits), dim=-1)
    pred_ids = torch.argmax(probs, dim=-1)
    word_pred = [None] * num_words
    word_conf = [None] * num_words
    token_pred_labels = []
    token_word_ids = []
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if attention_mask is not None and int(attention_mask[token_idx]) == 0:
            continue
        if word_idx >= num_words:
            continue
        pred_id = int(pred_ids[token_idx].item())
        token_pred_labels.append(id2label[pred_id])
        token_word_ids.append(int(word_idx))
        if word_pred[word_idx] is None:
            word_pred[word_idx] = id2label[pred_id]
            word_conf[word_idx] = float(probs[token_idx, pred_id].item())
    for idx in range(num_words):
        if word_pred[idx] is None:
            word_pred[idx] = "O"
            word_conf[idx] = 0.0
    return word_pred, word_conf, token_pred_labels, token_word_ids


def labels_from_encoding(labels_tensor, id2label):
    labels = []
    for value in labels_tensor.tolist():
        if int(value) == -100:
            continue
        labels.append(id2label[int(value)])
    return labels


def field_from_label(label):
    if label == "O":
        return "O"
    if label.startswith(("B-", "I-")):
        return label[2:]
    return label


def prf(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def compute_field_metrics(true_sequences, pred_sequences):
    fields = sorted(
        {
            field_from_label(label)
            for seq in true_sequences + pred_sequences
            for label in seq
            if field_from_label(label) != "O"
        }
    )
    metrics = {}
    for field in fields:
        tp = fp = fn = 0
        for gold_seq, pred_seq in zip(true_sequences, pred_sequences):
            for gold, pred in zip(gold_seq, pred_seq):
                gold_field = field_from_label(gold)
                pred_field = field_from_label(pred)
                gold_hit = gold_field == field
                pred_hit = pred_field == field
                if gold_hit and pred_hit:
                    tp += 1
                elif not gold_hit and pred_hit:
                    fp += 1
                elif gold_hit and not pred_hit:
                    fn += 1
        metrics[field] = prf(tp, fp, fn)
    for field in KEY_FIELDS:
        metrics.setdefault(field, prf(0, 0, 0))
    return metrics


def confusion_top(true_sequences, pred_sequences, limit=50):
    counts = Counter()
    for gold_seq, pred_seq in zip(true_sequences, pred_sequences):
        for gold, pred in zip(gold_seq, pred_seq):
            if gold != pred:
                counts[(gold, pred)] += 1
    return [{"gold": gold, "pred": pred, "count": count} for (gold, pred), count in counts.most_common(limit)]


def label_distribution(sequences):
    counts = Counter(label for seq in sequences for label in seq)
    return dict(counts)


def aggregate_token_metrics(true_sequences, pred_sequences, num_tokens, correct_tokens):
    return {
        "seqeval_precision": precision_score(true_sequences, pred_sequences, zero_division=0) if true_sequences else 0.0,
        "seqeval_recall": recall_score(true_sequences, pred_sequences, zero_division=0) if true_sequences else 0.0,
        "seqeval_f1": f1_score(true_sequences, pred_sequences, zero_division=0) if true_sequences else 0.0,
        "token_accuracy": correct_tokens / num_tokens if num_tokens else 0.0,
        "num_tokens": num_tokens,
        "field_metrics": compute_field_metrics(true_sequences, pred_sequences),
        "classification_report": classification_report(true_sequences, pred_sequences, digits=4, zero_division=0)
        if true_sequences
        else "",
    }


def numpy_inputs_from_encoding(encoding):
    inputs = {}
    for key in ("input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"):
        if key in encoding:
            inputs[key] = encoding[key].detach().cpu().numpy()
    return inputs


def torch_inputs_from_encoding(encoding, device):
    return {
        key: value.to(device)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids", "labels"}
    }


def provider_list(provider):
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def latency_stats(values):
    if not values:
        return {"avg_ms": None, "p50_ms": None, "p95_ms": None, "count": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "avg_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "count": int(len(values)),
    }


def timed_call(fn, warmup=0, repeat=1):
    for _ in range(max(0, warmup)):
        fn()
    times = []
    result = None
    for _ in range(max(1, repeat)):
        start = time.perf_counter()
        result = fn()
        times.append((time.perf_counter() - start) * 1000.0)
    return result, times


def finite_report(outputs):
    report = {}
    for name, array in outputs.items():
        arr = np.asarray(array)
        report[name] = {
            "shape": list(arr.shape),
            "has_nan": bool(np.isnan(arr).any()),
            "has_inf": bool(np.isinf(arr).any()),
        }
    return report


def safe_float(value):
    if value is None:
        return None
    try:
        value = float(value)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value
