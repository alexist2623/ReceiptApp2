import argparse
import json
import math
import shutil
import sys
from collections import Counter, defaultdict
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

from ml.layoutlmv3_training import encode_layoutlmv3_with_ignore, is_ignore_label, labels_to_ids_with_ignore
from ml.metrics.field_metrics import compute_field_metrics
from ml.receipt_schema import canonicalize_label
from scripts.smoke_finetune_user_labels_v2 import clamp_box, load_label_schema, load_labeled_sample, normalize_box, parse_box


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune LayoutLMv3 on mixed CORD, WildReceipt, and hand-labeled receipt data.")
    parser.add_argument("--cord_bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--cord_raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--wild_bio_dir", default=None)
    parser.add_argument("--wildreceipt_bio_dir", default=None)
    parser.add_argument(
        "--user_input_dir",
        action="append",
        default=None,
        help="Directory containing *_receipt_ocr folders. Can be passed multiple times.",
    )
    parser.add_argument("--extra_user_jsonl", nargs="*", default=None)
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--label_schema", default="schemas/receipt_labels_v2.json")
    parser.add_argument("--model_name_or_path", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_dir", default="models/layoutlmv3-mixed-public-user")
    parser.add_argument("--sources", default="cord,wild,user", help="Comma-separated source list: cord,wild,user.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--cord_repeat", type=int, default=1)
    parser.add_argument("--wild_repeat", type=int, default=1)
    parser.add_argument("--user_repeat", type=int, default=10)
    parser.add_argument("--max_cord_train_samples", type=int, default=None)
    parser.add_argument("--max_cord_eval_samples", type=int, default=None)
    parser.add_argument("--max_wild_train_samples", type=int, default=None)
    parser.add_argument("--max_wild_eval_samples", type=int, default=None)
    parser.add_argument("--max_public_eval_samples", type=int, default=None)
    parser.add_argument("--max_user_train_samples", type=int, default=None)
    parser.add_argument("--user_validation_count", type=int, default=3)
    parser.add_argument("--user_test_count", type=int, default=3)
    parser.add_argument("--custom_train_ratio", type=float, default=None)
    parser.add_argument("--split_by_capture_id", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plot_path", default=None)
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--save_eval_predictions", action="store_true")
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


def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def select_device(device):
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA requested but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_records(records, max_samples):
    if max_samples is None:
        return list(records)
    return list(records)[: max(0, int(max_samples))]


def image_to_rgb(value):
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    return ImageOps.exif_transpose(Image.open(value)).convert("RGB")


def canonicalize_labels(labels, label2id):
    out = []
    unknown = []
    for label in labels:
        if is_ignore_label(label):
            out.append("IGNORE")
            continue
        canonical = canonicalize_label(label)
        if canonical not in label2id:
            unknown.append({"label": label, "canonical_label": canonical})
        out.append(canonical)
    if unknown:
        fail(f"Unknown labels after canonicalization: {unknown[:20]}")
    return out


def infer_capture_group(capture_id):
    value = str(capture_id)
    for marker in ("_aug_", "_augment_", "_copy_", "_variant_"):
        if marker in value:
            return value.split(marker, 1)[0]
    return value


def collect_user_pairs(input_dir, exclude_dir_name, max_samples):
    if not input_dir:
        return [], []
    input_dirs = input_dir if isinstance(input_dir, list) else [input_dir]
    all_labels = []
    for root in input_dirs:
        root_path = Path(root)
        if not root_path.exists():
            fail(f"user_input_dir not found: {root_path}")
        all_labels.extend(sorted(root_path.rglob("*_labeled_v2_1.json")))
    excluded = [path for path in all_labels if exclude_dir_name and exclude_dir_name in path.parts]
    labels = [path for path in all_labels if path not in excluded]
    pairs = []
    for label_path in labels:
        capture_id = label_path.name.replace("_labeled_v2_1.json", "")
        image_path = label_path.with_name(f"{capture_id}.jpg")
        if not image_path.exists():
            continue
        pairs.append({"source": "user", "id": capture_id, "group_id": infer_capture_group(capture_id), "image": str(image_path), "label_json": str(label_path)})
    return limit_records(pairs, max_samples), excluded


def split_user_pairs(pairs, validation_count, test_count, split_by_capture_id=True):
    if not pairs:
        return [], [], []
    if len(pairs) == 1:
        return pairs, pairs, []
    pairs = list(sorted(pairs, key=lambda item: item["id"]))
    if not split_by_capture_id:
        test_count = max(0, min(int(test_count), len(pairs) - 2))
        validation_count = max(1, min(int(validation_count), len(pairs) - test_count - 1))
        train_end = len(pairs) - validation_count - test_count
        return pairs[:train_end], pairs[train_end : train_end + validation_count], pairs[train_end + validation_count :]
    groups = []
    seen = set()
    for pair in pairs:
        group = pair.get("group_id") or pair["id"]
        if group not in seen:
            seen.add(group)
            groups.append(group)
    if len(groups) < 3:
        return pairs, pairs[-1:], []
    test_count = max(0, min(int(test_count), len(groups) - 2))
    validation_count = max(1, min(int(validation_count), len(groups) - test_count - 1))
    train_groups = set(groups[: len(groups) - validation_count - test_count])
    val_groups = set(groups[len(groups) - validation_count - test_count : len(groups) - test_count])
    test_groups = set(groups[len(groups) - test_count :]) if test_count else set()
    train = [pair for pair in pairs if pair.get("group_id") in train_groups]
    validation = [pair for pair in pairs if pair.get("group_id") in val_groups]
    test = [pair for pair in pairs if pair.get("group_id") in test_groups]
    return train, validation, test


def load_cord_records(cord_bio_dir, split, max_samples):
    path = Path(cord_bio_dir) / f"{split}.jsonl"
    if not path.exists():
        fail(f"CORD BIO split not found: {path}")
    rows = limit_records(load_jsonl(path), max_samples)
    for row in rows:
        row["source"] = "cord"
    return rows


def load_public_bio_records(bio_dir, split, source, max_samples):
    if not bio_dir:
        return []
    path = Path(bio_dir) / f"{split}.jsonl"
    if not path.exists():
        print(f"WARNING: {source} split not found, skipped: {path}")
        return []
    rows = limit_records(load_jsonl(path), max_samples)
    for row in rows:
        row["source"] = source
    return rows


def load_extra_jsonl_records(paths, max_samples=None):
    rows = []
    for path in paths or []:
        loaded = load_jsonl(path)
        for row in loaded:
            row.setdefault("source", "user_jsonl")
            row.setdefault("id", row.get("receipt_id") or f"user_jsonl_{len(rows):06d}")
        rows.extend(loaded)
    return limit_records(rows, max_samples)


def box_from_norm(norm, width, height):
    return [
        int(round(norm[0] * width / 1000)),
        int(round(norm[1] * height / 1000)),
        int(round(norm[2] * width / 1000)),
        int(round(norm[3] * height / 1000)),
    ]


class MixedPublicUserDataset(Dataset):
    def __init__(self, records, label2id, cord_raw_dataset=None):
        self.records = list(records)
        self.label2id = label2id
        self.cord_raw_dataset = cord_raw_dataset

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        source = record.get("source", "unknown")
        if source == "user":
            sample = load_labeled_sample(record["image"], record["label_json"], self.label2id)
            sample["id"] = record["id"]
            sample["source"] = source
            return sample

        if source == "cord":
            if self.cord_raw_dataset is None:
                raise RuntimeError("CORD raw dataset is required for CORD records.")
            raw = self.cord_raw_dataset[record["split"]][int(record["index"])]
            image = image_to_rgb(raw["image"])
            width, height = image.size
        else:
            image_path = record.get("image") or record.get("image_path")
            if not image_path:
                raise ValueError(f"public BIO record has no image path: {record.get('id')}")
            image = image_to_rgb(image_path)
            width, height = image.size

        words = [str(word) for word in record["words"]]
        labels = canonicalize_labels(record["labels"], self.label2id)
        normalized_boxes = [[max(0, min(int(v), 1000)) for v in box] for box in record["normalized_boxes"]]
        boxes = record.get("boxes") or [box_from_norm(box, width, height) for box in normalized_boxes]
        keep_words = []
        keep_boxes = []
        keep_normalized = []
        keep_labels = []
        for word, box, norm, label in zip(words, boxes, normalized_boxes, labels):
            box = clamp_box(parse_box(box), width, height)
            if not str(word).strip() or box is None:
                continue
            keep_words.append(str(word))
            keep_boxes.append(box)
            keep_normalized.append(norm if norm else normalize_box(box, width, height))
            keep_labels.append(label)
        if not keep_words:
            raise ValueError(f"record has no valid words: {record.get('id')}")
        label_ids, ignored = labels_to_ids_with_ignore(keep_labels, self.label2id)
        return {
            "id": record.get("id", f"{source}_{idx:06d}"),
            "source": source,
            "image": image,
            "width": width,
            "height": height,
            "words": keep_words,
            "boxes": keep_boxes,
            "normalized_boxes": keep_normalized,
            "labels": keep_labels,
            "label_ids": label_ids,
            "ignore_word_indices": ignored,
        }


def collate_fn(processor, max_length):
    def collate(samples):
        encoding = encode_layoutlmv3_with_ignore(processor, samples, max_length)
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


def _metric_from_sequences(true_sequences, pred_sequences):
    return {
        "seqeval_precision": precision_score(true_sequences, pred_sequences, zero_division=0) if true_sequences else 0.0,
        "seqeval_recall": recall_score(true_sequences, pred_sequences, zero_division=0) if true_sequences else 0.0,
        "seqeval_f1": f1_score(true_sequences, pred_sequences, zero_division=0) if true_sequences else 0.0,
        "field_metrics": compute_field_metrics(true_sequences, pred_sequences),
    }


def evaluate(model, loader, device, id2label):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    correct = 0
    true_sequences = []
    pred_sequences = []
    per_source_true = defaultdict(list)
    per_source_pred = defaultdict(list)
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
            sources = batch.get("sources", ["unknown"] * labels.shape[0])
            source_counts.update(sources)
            for sample_idx in range(labels.shape[0]):
                sample_true = []
                sample_pred = []
                for label_id, pred_id, keep in zip(labels[sample_idx].tolist(), preds[sample_idx].tolist(), mask[sample_idx].tolist()):
                    if not keep:
                        continue
                    sample_true.append(id2label[int(label_id)])
                    sample_pred.append(id2label[int(pred_id)])
                if sample_true:
                    source = sources[sample_idx]
                    true_sequences.append(sample_true)
                    pred_sequences.append(sample_pred)
                    per_source_true[source].append(sample_true)
                    per_source_pred[source].append(sample_pred)
    metrics = {
        "eval_loss": total_loss / total_tokens if total_tokens else None,
        "token_accuracy": correct / total_tokens if total_tokens else 0.0,
        "num_eval_tokens": total_tokens,
        "source_counts": dict(source_counts),
        **_metric_from_sequences(true_sequences, pred_sequences),
    }
    metrics["source_metrics"] = {
        source: _metric_from_sequences(per_source_true[source], per_source_pred[source])
        for source in sorted(per_source_true)
    }
    model.train()
    return metrics


def save_eval_predictions(path, model, loader, device, id2label):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    with path.open("w", encoding="utf-8") as handle, torch.no_grad():
        for batch in loader:
            model_batch = move_batch(batch, device)
            outputs = model(**model_batch)
            labels = model_batch["labels"]
            mask = labels != -100
            preds = outputs.logits.argmax(dim=-1)
            for sample_idx, record_id in enumerate(batch.get("record_ids", [])):
                true_labels = []
                pred_labels = []
                for label_id, pred_id, keep in zip(labels[sample_idx].tolist(), preds[sample_idx].tolist(), mask[sample_idx].tolist()):
                    if not keep:
                        continue
                    true_labels.append(id2label[int(label_id)])
                    pred_labels.append(id2label[int(pred_id)])
                handle.write(
                    json.dumps(
                        {
                            "id": record_id,
                            "source": batch.get("sources", ["unknown"])[sample_idx],
                            "true_labels": true_labels,
                            "pred_labels": pred_labels,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    model.train()


def plot_history(path, history, best_epoch):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: could not import matplotlib; skipped plot: {exc}")
        return None
    epochs = [row["epoch"] for row in history]
    fig, loss_ax = plt.subplots(figsize=(11, 6))
    loss_ax.plot(epochs, [row["train_loss"] for row in history], label="train loss", color="#2f6fed", linewidth=2)
    loss_ax.plot(epochs, [row["eval_loss"] for row in history], label="validation loss", color="#e07a2f", linewidth=2)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.25)
    f1_ax = loss_ax.twinx()
    f1_ax.plot(epochs, [row["seqeval_f1"] for row in history], label="validation F1", color="#24945a", linewidth=2)
    f1_ax.set_ylabel("F1")
    f1_ax.set_ylim(0.0, 1.05)
    if best_epoch is not None:
        loss_ax.axvline(best_epoch, color="#555", linestyle="--", linewidth=1.2)
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


def repeat_records(records, repeat):
    return list(records) * max(0, int(repeat))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    enabled_sources = {source.strip().lower() for source in args.sources.split(",") if source.strip()}
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite_output_dir:
            fail(f"{output_dir} already exists. Use --overwrite_output_dir to replace it.")
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_payload, label_list, label2id, id2label = load_label_schema(args.label_schema)
    cord_raw_dataset = None
    cord_train = cord_eval = []
    if "cord" in enabled_sources:
        cord_train = load_cord_records(args.cord_bio_dir, "train", args.max_cord_train_samples)
        cord_eval = load_cord_records(args.cord_bio_dir, "validation", args.max_public_eval_samples or args.max_cord_eval_samples)
        cord_raw_dataset = load_from_disk(args.cord_raw_data_dir)
    wild_train = wild_eval = []
    wild_bio_dir = args.wildreceipt_bio_dir or args.wild_bio_dir
    if "wild" in enabled_sources and wild_bio_dir:
        wild_train = load_public_bio_records(wild_bio_dir, "train", "wildreceipt", args.max_wild_train_samples)
        wild_eval = load_public_bio_records(wild_bio_dir, "validation", "wildreceipt", args.max_public_eval_samples or args.max_wild_eval_samples)
    user_train = user_eval = user_test = []
    excluded = []
    if "user" in enabled_sources and args.user_input_dir:
        user_pairs, excluded = collect_user_pairs(args.user_input_dir, args.exclude_dir_name, args.max_user_train_samples)
        user_train, user_eval, user_test = split_user_pairs(user_pairs, args.user_validation_count, args.user_test_count, args.split_by_capture_id)
    extra_user_records = load_extra_jsonl_records(args.extra_user_jsonl)
    if extra_user_records:
        user_train.extend(extra_user_records)

    train_records = (
        repeat_records(cord_train, args.cord_repeat)
        + repeat_records(wild_train, args.wild_repeat)
        + repeat_records(user_train, args.user_repeat)
    )
    eval_records = list(cord_eval) + list(wild_eval) + list(user_eval)
    custom_test_records = list(user_test)
    if not train_records or not eval_records:
        fail(f"No train/eval records. train={len(train_records)} eval={len(eval_records)} sources={sorted(enabled_sources)}")

    print(f"enabled_sources: {sorted(enabled_sources)}")
    print(f"cord train/eval: {len(cord_train)}/{len(cord_eval)} repeat={args.cord_repeat}")
    print(f"wild train/eval: {len(wild_train)}/{len(wild_eval)} repeat={args.wild_repeat}")
    print(f"user train/eval/test: {len(user_train)}/{len(user_eval)}/{len(user_test)} repeat={args.user_repeat}")
    print(f"excluded Temp user labels: {len(excluded)}")
    print(f"mixed train records per epoch: {len(train_records)}")
    print(f"mixed eval records: {len(eval_records)}")
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
        id2label={int(idx): label for idx, label in id2label.items()},
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
    )
    model.to(device)
    train_loader = DataLoader(
        MixedPublicUserDataset(train_records, label2id, cord_raw_dataset),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn(processor, args.max_length),
    )
    eval_loader = DataLoader(
        MixedPublicUserDataset(eval_records, label2id, cord_raw_dataset),
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn(processor, args.max_length),
    )
    custom_test_loader = None
    if custom_test_records:
        custom_test_loader = DataLoader(
            MixedPublicUserDataset(custom_test_records, label2id, cord_raw_dataset),
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
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_tokens = 0
        train_source_counts = Counter()
        for step, batch in enumerate(tqdm(train_loader, desc=f"mixed public/user epoch {epoch:03d}/{args.epochs:03d}", unit="batch"), start=1):
            train_source_counts.update(batch.get("sources", []))
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
            "train_source_counts": dict(train_source_counts),
            "eval_source_counts": eval_metrics["source_counts"],
            "source_metrics": eval_metrics["source_metrics"],
            "best_so_far": False,
        }
        custom_val_f1 = row["source_metrics"].get("user", {}).get("seqeval_f1")
        selection_f1 = custom_val_f1 if custom_val_f1 is not None else row["seqeval_f1"]
        row["selection_metric"] = selection_f1
        row["selection_metric_name"] = "custom_validation_seqeval_f1" if custom_val_f1 is not None else "validation_seqeval_f1"
        if selection_f1 > best_f1:
            best_f1 = selection_f1
            best_epoch = epoch
            best_metrics = eval_metrics
            row["best_so_far"] = True
            save_model_bundle(output_dir / "best", model, processor, labels_payload)
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={eval_metrics['eval_loss']:.6f} | val_f1={eval_metrics['seqeval_f1']:.6f} | "
            f"val_acc={eval_metrics['token_accuracy']:.6f} | selection={selection_f1:.6f} | best_epoch={best_epoch}"
        )

    final_metrics = evaluate(model, eval_loader, device, id2label)
    custom_test_metrics = evaluate(model, custom_test_loader, device, id2label) if custom_test_loader is not None else None
    save_model_bundle(output_dir / "last", model, processor, labels_payload)
    plot_path = Path(args.plot_path) if args.plot_path else output_dir / "training_curve.png"
    saved_plot = plot_history(plot_path, history, best_epoch)
    save_json(output_dir / "training_config.json", {**vars(args), "enabled_sources": sorted(enabled_sources), "device": str(device), "cuda_available": torch.cuda.is_available(), "fp16": use_fp16})
    save_json(output_dir / "training_history.json", {"history": history})
    save_json(output_dir / "best_metrics.json", {"best_epoch": best_epoch, "best_f1": best_f1, "metrics": best_metrics})
    save_json(output_dir / "final_metrics.json", final_metrics)
    save_json(output_dir / "field_metrics_validation.json", final_metrics.get("field_metrics", {}))
    save_json(output_dir / "field_metrics_custom_validation.json", final_metrics.get("source_metrics", {}).get("user", {}).get("field_metrics", {}))
    if custom_test_metrics is not None:
        save_json(output_dir / "field_metrics_custom_test.json", custom_test_metrics.get("field_metrics", {}))
        save_json(output_dir / "custom_test_metrics.json", custom_test_metrics)
    if args.save_eval_predictions:
        save_eval_predictions(output_dir / "eval_predictions.jsonl", model, eval_loader, device, id2label)
        if custom_test_loader is not None:
            save_eval_predictions(output_dir / "custom_test_predictions.jsonl", model, custom_test_loader, device, id2label)
    write_jsonl(output_dir / "train_manifest.jsonl", train_records)
    write_jsonl(output_dir / "eval_manifest.jsonl", eval_records)
    save_json(
        output_dir / "split_manifest.json",
        {
            "splits": {
                "train": [{"id": row.get("id"), "source": row.get("source"), "group_id": row.get("group_id")} for row in user_train if row.get("source") == "user"],
                "validation": [{"id": row.get("id"), "source": row.get("source"), "group_id": row.get("group_id")} for row in user_eval if row.get("source") == "user"],
                "test": [{"id": row.get("id"), "source": row.get("source"), "group_id": row.get("group_id")} for row in custom_test_records if row.get("source") == "user"],
            },
            "public_record_counts": {
                "cord_train": len(cord_train),
                "cord_validation": len(cord_eval),
                "wild_train": len(wild_train),
                "wild_validation": len(wild_eval),
                "extra_user_jsonl_train": len(extra_user_records),
            },
            "notes": ["User/custom samples are split by capture group when --split_by_capture_id is enabled."],
        },
    )
    print(f"best_epoch: {best_epoch}")
    print(f"best validation F1: {best_f1:.6f}")
    print(f"training_curve path: {saved_plot}")
    print(f"best checkpoint: {output_dir / 'best'}")
    print(f"last checkpoint: {output_dir / 'last'}")
    print("Mixed public + user LayoutLMv3 fine-tuning passed.")


if __name__ == "__main__":
    main()
