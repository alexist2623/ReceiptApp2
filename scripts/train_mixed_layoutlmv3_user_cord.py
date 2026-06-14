import argparse
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from PIL import Image, ImageOps
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoProcessor, get_linear_schedule_with_warmup

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import canonicalize_label
from scripts.smoke_finetune_user_labels_v2 import (
    clamp_box,
    load_label_schema,
    load_labeled_sample,
    normalize_box,
    parse_box,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 on mixed CORD BIO + non-Temp user labels.")
    parser.add_argument("--cord_bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--cord_raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--user_input_dir", required=True)
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--label_schema", default="schemas/receipt_labels_v2.json")
    parser.add_argument("--model_name_or_path", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_dir", default="models/layoutlmv3-mixed-cord-user-non-temp")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_cord_train_samples", type=int, default=None)
    parser.add_argument("--max_cord_eval_samples", type=int, default=None)
    parser.add_argument("--user_validation_count", type=int, default=3)
    parser.add_argument("--user_repeat", type=int, default=10)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_path", default=None)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


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


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def limit_records(records, max_samples):
    if max_samples is None:
        return records
    return records[: max(0, int(max_samples))]


def collect_user_pairs(input_dir, exclude_dir_name):
    input_dir = Path(input_dir)
    if not input_dir.exists():
        fail(f"user_input_dir not found: {input_dir}")
    all_labels = sorted(input_dir.rglob("*_labeled_v2_1.json"))
    excluded = [path for path in all_labels if exclude_dir_name and exclude_dir_name in path.parts]
    labels = [path for path in all_labels if path not in excluded]
    pairs = []
    for label_path in labels:
        capture_id = label_path.name.replace("_labeled_v2_1.json", "")
        image_path = label_path.with_name(f"{capture_id}.jpg")
        if not image_path.exists():
            fail(f"image not found for {label_path}: {image_path}")
        pairs.append({"source": "user", "id": capture_id, "image": str(image_path), "label_json": str(label_path)})
    if len(pairs) < 2:
        fail("Need at least two non-Temp user samples.")
    return pairs, excluded


def split_user_pairs(pairs, validation_count):
    pairs = list(sorted(pairs, key=lambda item: item["id"]))
    validation_count = max(1, min(int(validation_count), len(pairs) - 1))
    return pairs[:-validation_count], pairs[-validation_count:]


def load_cord_records(cord_bio_dir, split, max_samples):
    path = Path(cord_bio_dir) / f"{split}.jsonl"
    if not path.exists():
        fail(f"CORD BIO split not found: {path}")
    records = limit_records(load_jsonl(path), max_samples)
    for record in records:
        record["source"] = "cord"
    return records


def image_to_rgb(image):
    if isinstance(image, Image.Image):
        return ImageOps.exif_transpose(image).convert("RGB")
    return Image.open(image).convert("RGB")


def canonicalize_label_list(labels, label2id):
    out = []
    unknown = []
    for label in labels:
        canonical = canonicalize_label(label)
        if canonical not in label2id:
            unknown.append({"label": label, "canonical_label": canonical})
        out.append(canonical)
    if unknown:
        fail(f"Unknown labels after canonicalization: {unknown[:20]}")
    return out


class MixedReceiptDataset(Dataset):
    def __init__(self, records, label2id, raw_dataset=None):
        self.records = list(records)
        self.label2id = label2id
        self.raw_dataset = raw_dataset

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        if record.get("source") == "user":
            sample = load_labeled_sample(record["image"], record["label_json"], self.label2id)
            sample["id"] = record["id"]
            sample["source"] = "user"
            return sample
        split = record["split"]
        raw = self.raw_dataset[split][int(record["index"])]
        image = image_to_rgb(raw["image"])
        width, height = image.size
        words = [str(word) for word in record["words"]]
        labels = canonicalize_label_list(record["labels"], self.label2id)
        normalized_boxes = [[int(v) for v in box] for box in record["normalized_boxes"]]
        boxes = record.get("boxes")
        if boxes is None:
            boxes = [
                [
                    int(round(box[0] * width / 1000)),
                    int(round(box[1] * height / 1000)),
                    int(round(box[2] * width / 1000)),
                    int(round(box[3] * height / 1000)),
                ]
                for box in normalized_boxes
            ]
        boxes = [clamp_box(parse_box(box), width, height) for box in boxes]
        keep_words = []
        keep_boxes = []
        keep_normalized = []
        keep_labels = []
        for word, box, norm, label in zip(words, boxes, normalized_boxes, labels):
            if not str(word).strip() or box is None:
                continue
            keep_words.append(str(word))
            keep_boxes.append(box)
            keep_normalized.append([max(0, min(int(v), 1000)) for v in norm])
            keep_labels.append(label)
        if not keep_words:
            fail(f"CORD record has no valid words: {record.get('id')}")
        return {
            "id": record["id"],
            "source": "cord",
            "image": image,
            "width": width,
            "height": height,
            "words": keep_words,
            "boxes": keep_boxes,
            "normalized_boxes": keep_normalized,
            "labels": keep_labels,
            "label_ids": [self.label2id[label] for label in keep_labels],
            "warnings": [],
            "skipped": [],
        }


def collate_fn(processor, max_length):
    def collate(samples):
        encoding = processor(
            [sample["image"] for sample in samples],
            [sample["words"] for sample in samples],
            boxes=[sample["normalized_boxes"] for sample in samples],
            word_labels=[sample["label_ids"] for sample in samples],
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        if "labels" not in encoding or encoding["labels"].shape != encoding["input_ids"].shape:
            fail("Processor did not produce labels shaped like input_ids.")
        encoding["record_ids"] = [sample["id"] for sample in samples]
        encoding["sources"] = [sample["source"] for sample in samples]
        return encoding

    return collate


def move_batch(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids", "labels"}
    }


def evaluate(model, loader, device, id2label):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    true_sequences = []
    pred_sequences = []
    source_counts = Counter()
    with torch.no_grad():
        for batch in loader:
            model_batch = move_batch(batch, device)
            outputs = model(**model_batch)
            labels = model_batch["labels"]
            mask = labels != -100
            token_count = int(mask.sum().item())
            if token_count == 0:
                continue
            total_loss += float(outputs.loss.detach().cpu().item()) * token_count
            total_tokens += token_count
            preds = outputs.logits.argmax(dim=-1)
            correct += int(((preds == labels) & mask).sum().item())
            source_counts.update(batch.get("sources", []))
            for sample_idx in range(labels.shape[0]):
                sample_true = []
                sample_pred = []
                for label_id, pred_id, keep in zip(labels[sample_idx].tolist(), preds[sample_idx].tolist(), mask[sample_idx].tolist()):
                    if not keep:
                        continue
                    sample_true.append(id2label[int(label_id)])
                    sample_pred.append(id2label[int(pred_id)])
                if sample_true:
                    true_sequences.append(sample_true)
                    pred_sequences.append(sample_pred)
    metrics = {
        "eval_loss": total_loss / total_tokens if total_tokens else None,
        "token_accuracy": correct / total_tokens if total_tokens else 0.0,
        "seqeval_precision": precision_score(true_sequences, pred_sequences, zero_division=0) if true_sequences else 0.0,
        "seqeval_recall": recall_score(true_sequences, pred_sequences, zero_division=0) if true_sequences else 0.0,
        "seqeval_f1": f1_score(true_sequences, pred_sequences, zero_division=0) if true_sequences else 0.0,
        "num_eval_tokens": total_tokens,
        "source_counts": dict(source_counts),
    }
    model.train()
    return metrics


def plot_history(path, history, best_epoch):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: could not import matplotlib; skipped plot: {exc}")
        return None
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["eval_loss"] for row in history]
    f1 = [row["seqeval_f1"] for row in history]
    fig, loss_ax = plt.subplots(figsize=(11, 6))
    loss_ax.plot(epochs, train_loss, label="train loss", color="#2f6fed", linewidth=2)
    loss_ax.plot(epochs, val_loss, label="validation loss", color="#e07a2f", linewidth=2)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.25)
    f1_ax = loss_ax.twinx()
    f1_ax.plot(epochs, f1, label="validation F1", color="#24945a", linewidth=2)
    f1_ax.set_ylabel("F1")
    f1_ax.set_ylim(0.0, 1.05)
    if best_epoch is not None:
        loss_ax.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.5, alpha=0.8)
    lines, labels = loss_ax.get_legend_handles_labels()
    f1_lines, f1_labels = f1_ax.get_legend_handles_labels()
    loss_ax.legend(lines + f1_lines, labels + f1_labels, loc="upper right")
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def save_model_bundle(path, model, processor, labels_payload):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    processor.save_pretrained(path)
    save_json(path / "labels.json", labels_payload)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite_output_dir:
            fail(f"{output_dir} already exists. Use --overwrite_output_dir to replace it.")
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_payload, label_list, label2id, id2label = load_label_schema(args.label_schema)
    cord_train = load_cord_records(args.cord_bio_dir, "train", args.max_cord_train_samples)
    cord_eval = load_cord_records(args.cord_bio_dir, "validation", args.max_cord_eval_samples)
    user_pairs, excluded = collect_user_pairs(args.user_input_dir, args.exclude_dir_name)
    user_train, user_eval = split_user_pairs(user_pairs, args.user_validation_count)
    train_records = list(cord_train) + user_train * max(1, int(args.user_repeat))
    eval_records = list(cord_eval) + user_eval

    print(f"cord train samples: {len(cord_train)}")
    print(f"cord validation samples: {len(cord_eval)}")
    print(f"user train samples: {len(user_train)} repeated x{args.user_repeat}")
    print(f"user validation samples: {len(user_eval)}")
    print(f"excluded Temp labels: {len(excluded)}")
    print(f"mixed train records per epoch: {len(train_records)}")
    print(f"mixed validation records: {len(eval_records)}")
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    device = select_device(args.device)
    print(f"selected device: {device}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    print("Loading CORD raw dataset for images...")
    raw_dataset = load_from_disk(args.cord_raw_data_dir)
    processor = AutoProcessor.from_pretrained(args.model_name_or_path, apply_ocr=False, local_files_only=args.local_files_only)
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label={idx: label for idx, label in id2label.items()},
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
    )
    model.to(device)

    train_loader = DataLoader(
        MixedReceiptDataset(train_records, label2id, raw_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn(processor, args.max_length),
    )
    eval_loader = DataLoader(
        MixedReceiptDataset(eval_records, label2id, raw_dataset),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn(processor, args.max_length),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_update_steps = math.ceil(len(train_loader) / args.gradient_accumulation_steps) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=max(1, int(0.1 * total_update_steps)),
        num_training_steps=max(1, total_update_steps),
    )
    use_fp16 = args.fp16 and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)
    history = []
    best_epoch = None
    best_f1 = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        source_counts = Counter()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"mixed layout epoch {epoch:03d}/{args.epochs:03d}", unit="batch"), start=1):
            source_counts.update(batch.get("sources", []))
            model_batch = move_batch(batch, device)
            token_count = int((model_batch["labels"] != -100).sum().item())
            with torch.cuda.amp.autocast(enabled=use_fp16):
                outputs = model(**model_batch)
                loss = outputs.loss / args.gradient_accumulation_steps
            scaler.scale(loss).backward()
            running_loss += float(outputs.loss.detach().cpu().item()) * token_count
            running_tokens += token_count
            if step % args.gradient_accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        train_loss = running_loss / running_tokens if running_tokens else None
        eval_metrics = evaluate(model, eval_loader, device, id2label)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_metrics["eval_loss"],
            "token_accuracy": eval_metrics["token_accuracy"],
            "seqeval_precision": eval_metrics["seqeval_precision"],
            "seqeval_recall": eval_metrics["seqeval_recall"],
            "seqeval_f1": eval_metrics["seqeval_f1"],
            "lr": scheduler.get_last_lr()[0],
            "train_source_counts": dict(source_counts),
            "best_so_far": False,
        }
        if row["seqeval_f1"] > best_f1:
            best_f1 = row["seqeval_f1"]
            best_epoch = epoch
            best_metrics = dict(eval_metrics)
            row["best_so_far"] = True
            save_model_bundle(output_dir / "best", model, processor, labels_payload)
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={eval_metrics['eval_loss']:.6f} | val_f1={eval_metrics['seqeval_f1']:.6f} | "
            f"val_acc={eval_metrics['token_accuracy']:.6f} | best_epoch={best_epoch}"
        )

    final_metrics = evaluate(model, eval_loader, device, id2label)
    save_model_bundle(output_dir / "last", model, processor, labels_payload)
    saved_plot = plot_history(Path(args.plot_path) if args.plot_path else output_dir / "training_curve.png", history, best_epoch)
    save_json(
        output_dir / "training_config.json",
        {
            "cord_bio_dir": args.cord_bio_dir,
            "cord_raw_data_dir": args.cord_raw_data_dir,
            "user_input_dir": args.user_input_dir,
            "exclude_dir_name": args.exclude_dir_name,
            "excluded_temp_count": len(excluded),
            "model_name_or_path": args.model_name_or_path,
            "output_dir": str(output_dir),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "eval_batch_size": args.eval_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "max_cord_train_samples": args.max_cord_train_samples,
            "max_cord_eval_samples": args.max_cord_eval_samples,
            "user_repeat": args.user_repeat,
            "cord_train_samples": len(cord_train),
            "cord_validation_samples": len(cord_eval),
            "user_train_samples": len(user_train),
            "user_validation_samples": len(user_eval),
            "mixed_train_records_per_epoch": len(train_records),
            "mixed_validation_records": len(eval_records),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "fp16": use_fp16,
        },
    )
    save_json(output_dir / "training_history.json", {"history": history})
    save_json(output_dir / "best_metrics.json", {"best_epoch": best_epoch, "best_f1": best_f1, "metrics": best_metrics})
    save_json(output_dir / "final_metrics.json", final_metrics)
    print(f"best_epoch: {best_epoch}")
    print(f"best validation F1: {best_f1:.6f}")
    print(f"training_history path: {output_dir / 'training_history.json'}")
    print(f"training_curve path: {saved_plot}")
    print(f"best checkpoint: {output_dir / 'best'}")
    print(f"last checkpoint: {output_dir / 'last'}")
    print("Mixed CORD + user LayoutLMv3 fine-tuning passed.")


if __name__ == "__main__":
    main()
