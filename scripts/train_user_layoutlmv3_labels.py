import argparse
import json
import math
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoProcessor, get_linear_schedule_with_warmup

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.layoutlmv3_training import encode_layoutlmv3_with_ignore
from scripts.smoke_finetune_user_labels_v2 import load_label_schema, load_labeled_sample


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 on user labeled receipt JSON files.")
    parser.add_argument("--input_dir", required=True, nargs="+", help="One or more directories containing *_receipt_ocr folders.")
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--label_schema", default="schemas/receipt_labels_v2.json")
    parser.add_argument("--model_name_or_path", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_dir", default="models/layoutlmv3-user-labels-non-temp")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--validation_ratio", type=float, default=0.2)
    parser.add_argument("--validation_count", type=int, default=None)
    parser.add_argument("--split_by_parent", action="store_true", help="Split by parent capture id to avoid original/augmented leakage.")
    parser.add_argument("--validation_parent_count", type=int, default=None)
    parser.add_argument("--validation_augmented", action="store_true", help="Include augmented records in validation. Default keeps validation on originals only.")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
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


def path_is_excluded(path, exclude_dir_name):
    if not exclude_dir_name:
        return False
    needle = exclude_dir_name.lower()
    return any(needle in part.lower() for part in Path(path).parts)


def read_parent_capture_id(label_path, capture_id):
    try:
        with Path(label_path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        payload = {}
    augmentation = payload.get("augmentation") if isinstance(payload, dict) else None
    if isinstance(augmentation, dict) and augmentation.get("parent_capture_id"):
        return str(augmentation["parent_capture_id"]), True
    for key in ("parent_capture_id", "parentCaptureId"):
        if isinstance(payload, dict) and payload.get(key):
            return str(payload[key]), True
    match = re.match(r"(.+?)_aug_\d{3}_[0-9a-fA-F]+$", capture_id)
    if match:
        return match.group(1), True
    return capture_id, False


def collect_label_pairs(input_dirs, exclude_dir_name):
    input_dirs = [Path(path) for path in input_dirs]
    for input_dir in input_dirs:
        if not input_dir.exists():
            fail(f"input_dir not found: {input_dir}")
    all_labels = []
    for input_dir in input_dirs:
        all_labels.extend(sorted(input_dir.rglob("*_labeled_v2_1.json")))
    excluded = [path for path in all_labels if path_is_excluded(path, exclude_dir_name)]
    labels = [path for path in all_labels if path not in excluded]
    pairs = []
    missing_images = []
    for label_path in labels:
        capture_id = label_path.name.replace("_labeled_v2_1.json", "")
        image_path = label_path.with_name(f"{capture_id}.jpg")
        if not image_path.exists():
            missing_images.append(str(image_path))
            continue
        parent_id, is_augmented = read_parent_capture_id(label_path, capture_id)
        pairs.append(
            {
                "id": capture_id,
                "parent_id": parent_id,
                "is_augmented": is_augmented,
                "image": str(image_path),
                "label_json": str(label_path),
                "source_dir": str(next((root for root in input_dirs if root in label_path.parents), label_path.parent)),
            }
        )
    if missing_images:
        fail(f"Missing images for labeled JSON files: {missing_images[:10]}")
    if len(pairs) < 2:
        fail("Need at least two non-Temp labeled samples for train/validation split.")
    return pairs, excluded


def split_pairs(
    pairs,
    validation_ratio,
    validation_count,
    split_by_parent=False,
    validation_parent_count=None,
    validation_augmented=False,
):
    pairs = list(sorted(pairs, key=lambda item: item["id"]))
    if split_by_parent:
        parent_ids = sorted({item.get("parent_id") or item["id"] for item in pairs})
        if len(parent_ids) < 2:
            fail("Need at least two parent capture ids for parent-aware split.")
        if validation_parent_count is None:
            validation_parent_count = validation_count
        if validation_parent_count is None:
            validation_parent_count = max(1, int(round(len(parent_ids) * validation_ratio)))
        validation_parent_count = max(1, min(int(validation_parent_count), len(parent_ids) - 1))
        validation_parents = set(parent_ids[-validation_parent_count:])
        train_records = [item for item in pairs if (item.get("parent_id") or item["id"]) not in validation_parents]
        validation_records = [
            item
            for item in pairs
            if (item.get("parent_id") or item["id"]) in validation_parents
            and (validation_augmented or not item.get("is_augmented"))
        ]
        if not validation_records:
            validation_records = [item for item in pairs if (item.get("parent_id") or item["id"]) in validation_parents]
        return train_records, validation_records
    if validation_count is None:
        validation_count = max(1, int(round(len(pairs) * validation_ratio)))
    validation_count = max(1, min(int(validation_count), len(pairs) - 1))
    return pairs[:-validation_count], pairs[-validation_count:]


def limit_samples(records, max_samples):
    if max_samples is None:
        return records
    return list(records)[: max(0, int(max_samples))]


class UserReceiptLabelDataset(Dataset):
    def __init__(self, records, label2id):
        self.records = list(records)
        self.label2id = label2id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        sample = load_labeled_sample(record["image"], record["label_json"], self.label2id)
        sample["id"] = record["id"]
        sample["image_path"] = record["image"]
        sample["label_json"] = record["label_json"]
        return sample


def collate_fn(processor, max_length):
    def collate(samples):
        encoding = encode_layoutlmv3_with_ignore(processor, samples, max_length)
        encoding["record_ids"] = [sample["id"] for sample in samples]
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
    pred_counts = Counter()
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
            for sample_idx in range(labels.shape[0]):
                sample_true = []
                sample_pred = []
                for label_id, pred_id, keep in zip(labels[sample_idx].tolist(), preds[sample_idx].tolist(), mask[sample_idx].tolist()):
                    if not keep:
                        continue
                    true_label = id2label[int(label_id)]
                    pred_label = id2label[int(pred_id)]
                    sample_true.append(true_label)
                    sample_pred.append(pred_label)
                    pred_counts[pred_label] += 1
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
        "prediction_label_counts": dict(pred_counts),
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
    if args.device == "auto" and not torch.cuda.is_available():
        print("WARNING: CUDA is not available. This user-label fine-tune will run on CPU.")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite_output_dir:
            fail(f"{output_dir} already exists. Use --overwrite_output_dir to replace it.")
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_payload, label_list, label2id, id2label = load_label_schema(args.label_schema)
    records, excluded = collect_label_pairs(args.input_dir, args.exclude_dir_name)
    train_records, validation_records = split_pairs(
        records,
        args.validation_ratio,
        args.validation_count,
        split_by_parent=args.split_by_parent,
        validation_parent_count=args.validation_parent_count,
        validation_augmented=args.validation_augmented,
    )
    train_records = limit_samples(train_records, args.max_train_samples)
    validation_records = limit_samples(validation_records, args.max_eval_samples)
    print(f"input_dir: {args.input_dir}")
    print(f"all non-Temp records: {len(records)}")
    print(f"excluded Temp labels: {len(excluded)}")
    print(f"parent ids: {len({record.get('parent_id') or record['id'] for record in records})}")
    print(f"augmented records: {sum(1 for record in records if record.get('is_augmented'))}")
    print(f"train samples: {len(train_records)}")
    print(f"validation samples: {len(validation_records)}")
    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    device = select_device(args.device)
    print(f"selected device: {device}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

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

    train_dataset = UserReceiptLabelDataset(train_records, label2id)
    eval_dataset = UserReceiptLabelDataset(validation_records, label2id)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn(processor, args.max_length),
    )
    eval_loader = DataLoader(
        eval_dataset,
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
    if args.fp16 and not use_fp16:
        print("WARNING: --fp16 requested without CUDA; disabled fp16.")

    history = []
    best_epoch = None
    best_f1 = -1.0
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(tqdm(train_loader, desc=f"layout epoch {epoch:03d}/{args.epochs:03d}", unit="batch"), start=1):
            model_batch = move_batch(batch, device)
            labels = model_batch["labels"]
            token_count = int((labels != -100).sum().item())
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
    plot_path = Path(args.plot_path) if args.plot_path else output_dir / "training_curve.png"
    saved_plot = plot_history(plot_path, history, best_epoch)
    config = {
        "input_dir": args.input_dir,
        "exclude_dir_name": args.exclude_dir_name,
        "excluded_temp_count": len(excluded),
        "split_by_parent": args.split_by_parent,
        "validation_parent_count": args.validation_parent_count,
        "validation_augmented": args.validation_augmented,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "model_name_or_path": args.model_name_or_path,
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "fp16": use_fp16,
        "seed": args.seed,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "train_records": train_records,
        "validation_records": validation_records,
    }
    save_json(output_dir / "training_config.json", config)
    save_json(output_dir / "training_history.json", {"history": history})
    save_json(output_dir / "best_metrics.json", {"best_epoch": best_epoch, "best_f1": best_f1, "metrics": best_metrics})
    save_json(output_dir / "final_metrics.json", final_metrics)
    print(f"best_epoch: {best_epoch}")
    print(f"best validation F1: {best_f1:.6f}")
    print(f"training_history path: {output_dir / 'training_history.json'}")
    print(f"training_curve path: {saved_plot}")
    print(f"best checkpoint: {output_dir / 'best'}")
    print(f"last checkpoint: {output_dir / 'last'}")
    print("User LayoutLMv3 fine-tuning passed.")


if __name__ == "__main__":
    main()
