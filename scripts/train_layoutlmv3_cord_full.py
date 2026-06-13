import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from seqeval.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForTokenClassification,
    AutoProcessor,
    get_linear_schedule_with_warmup,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune LayoutLMv3 token classification on full CORD-v2 BIO data."
    )
    parser.add_argument("--bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--label_schema", default=None, help="Optional labels schema JSON, e.g. schemas/receipt_labels_v2.json.")
    parser.add_argument("--user_labeled_jsonl", default=None, help="Reserved for future schema-v2 user labeled JSONL fine-tuning.")
    parser.add_argument("--init_from_checkpoint", default=None, help="Reserved for initializing from an existing fine-tuned checkpoint.")
    parser.add_argument("--copy_old_classifier_rows", action="store_true", help="Reserved for alias-based classifier row initialization.")
    parser.add_argument("--model_name_or_path", default="microsoft/layoutlmv3-base")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_dir", default="models/layoutlmv3-cord-full")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=None)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--log_every_steps", type=int, default=20)
    parser.add_argument(
        "--eval_every_epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--overwrite_output_dir", action="store_true")
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument("--num_preview_samples", type=int, default=20)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def require_path(path, message):
    if not Path(path).exists():
        fail(message)


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


def load_labels(bio_dir, label_schema=None):
    labels_path = Path(label_schema) if label_schema else Path(bio_dir) / "labels.json"
    require_path(labels_path, f"{labels_path} not found. Run step 3 first or export schema v2.")
    with labels_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    label_list = payload["label_list"]
    label2id = {str(label): int(idx) for label, idx in payload["label2id"].items()}
    id2label = {int(idx): str(label) for idx, label in payload["id2label"].items()}

    if "O" not in label_list:
        fail("labels.json is invalid: 'O' is missing from label_list.")
    if label2id.get("O") != 0:
        fail("labels.json is invalid: label2id['O'] must be 0.")

    print(f"num_labels: {len(label_list)}")
    if len(label_list) <= 50:
        print(f"label_list: {label_list}")
    else:
        print(f"label_list first 50 / total {len(label_list)}: {label_list[:50]}")
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
        if len(box) != 4 or any(value < 0 or value > 1000 for value in box):
            fail(f"{source_path} line {line_number} has invalid normalized box: {box}")

    unknown_labels = [label for label in record["labels"] if label not in label2id]
    if unknown_labels:
        fail(
            f"{source_path} line {line_number} has labels not present in labels.json: "
            f"{unknown_labels[:5]}"
        )


def load_bio_records(bio_dir, split, label2id, max_samples=None):
    jsonl_path = Path(bio_dir) / f"{split}.jsonl"
    if split == "train":
        message = f"{jsonl_path} not found. Run step 3 first."
    else:
        message = f"{jsonl_path} not found. Run step 3 first."
    require_path(jsonl_path, message)

    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_record(record, line_number, label2id, jsonl_path)
            records.append(record)

    if max_samples is not None:
        records = records[:max_samples]
    if not records:
        fail(f"No records loaded from {jsonl_path}.")
    return records


def summarize_label_distribution(records, split):
    label_counts = Counter(label for record in records for label in record["labels"])
    total = sum(label_counts.values())
    o_ratio = label_counts.get("O", 0) / total if total else 0.0
    print(f"{split} samples: {len(records)}")
    print(f"{split} words: {total}")
    print(f"{split} O ratio: {o_ratio:.6f}")
    print(f"{split} label distribution top 30:")
    for label, count in label_counts.most_common(30):
        print(f"  {label}: {count}")
    return label_counts, o_ratio


def load_raw_dataset(raw_data_dir):
    require_path(raw_data_dir, f"{raw_data_dir} not found. Run CORD-v2 download step first.")
    raw_dataset = load_from_disk(str(raw_data_dir))
    for split in ("train", "validation"):
        if split not in raw_dataset:
            fail(f"Raw dataset split '{split}' not found. Available splits: {list(raw_dataset.keys())}")
    return raw_dataset


class CordBioTokenClassificationDataset(Dataset):
    def __init__(self, records, raw_dataset, label2id):
        self.records = records
        self.raw_dataset = raw_dataset
        self.label2id = label2id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        split = record["split"]
        raw_index = int(record["index"])
        raw_split = self.raw_dataset[split]
        if raw_index < 0 or raw_index >= len(raw_split):
            fail(f"{record['id']}: raw dataset index {raw_index} out of range for split length {len(raw_split)}")

        image = ensure_pil_rgb(raw_split[raw_index]["image"])
        word_labels = [self.label2id[label] for label in record["labels"]]
        return {
            "image": image,
            "words": [str(word) for word in record["words"]],
            "boxes": [[int(v) for v in box] for box in record["normalized_boxes"]],
            "word_labels": word_labels,
            "record_id": record["id"],
            "split": split,
            "index": raw_index,
            "metadata": record,
        }


class LayoutLMv3Collator:
    def __init__(self, processor, max_length, id2label, debug=False):
        self.processor = processor
        self.max_length = max_length
        self.id2label = id2label
        self.debug = debug
        self._printed_shapes = False
        self._printed_alignment = False

    def __call__(self, samples):
        images = [sample["image"] for sample in samples]
        words = [sample["words"] for sample in samples]
        boxes = [sample["boxes"] for sample in samples]
        word_labels = [sample["word_labels"] for sample in samples]

        encoding, alignment_mode = self._encode_with_labels(images, words, boxes, word_labels)
        labels = encoding["labels"]
        word_ids_batch = [encoding.word_ids(batch_index=i) for i in range(len(samples))]

        if self.debug and not self._printed_shapes:
            print("first batch encoding shapes:")
            for key, value in encoding.items():
                if hasattr(value, "shape"):
                    print(f"  {key}: {list(value.shape)}")
            non_ignored = int((labels != -100).sum().item())
            print(f"labels non -100 token count: {non_ignored}")
            print(f"label alignment mode: {alignment_mode}")
            self._printed_shapes = True

        if self.debug and not self._printed_alignment:
            self.print_alignment_preview(encoding, word_ids_batch[0], samples[0], limit=50)
            self._printed_alignment = True

        model_inputs = {
            key: value
            for key, value in encoding.items()
            if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids", "labels"}
        }
        return {
            "model_inputs": model_inputs,
            "records": [sample["metadata"] for sample in samples],
            "word_ids": word_ids_batch,
        }

    def _encode_with_labels(self, images, words, boxes, word_labels):
        try:
            encoding = self.processor(
                images,
                words,
                boxes=boxes,
                word_labels=word_labels,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            if "labels" in encoding and list(encoding["labels"].shape) == list(encoding["input_ids"].shape):
                labels = encoding["labels"].clone()
                labels[encoding["attention_mask"] == 0] = -100
                encoding["labels"] = labels
                return encoding, "processor_word_labels"
            print("WARNING: processor word_labels did not return valid token labels; using fallback alignment.")
        except Exception as exc:
            print(f"WARNING: processor word_labels path failed; using fallback alignment. Reason: {exc}")

        encoding = self.processor(
            images,
            words,
            boxes=boxes,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = self._fallback_align_labels(encoding, word_labels)
        encoding["labels"] = labels
        return encoding, "fallback_word_ids_first_subword"

    def _fallback_align_labels(self, encoding, word_labels):
        batch_labels = []
        for batch_idx, labels_for_words in enumerate(word_labels):
            word_ids = encoding.word_ids(batch_index=batch_idx)
            previous_word_idx = None
            token_labels = []
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None or encoding["attention_mask"][batch_idx, token_idx].item() == 0:
                    token_labels.append(-100)
                elif word_idx != previous_word_idx:
                    token_labels.append(int(labels_for_words[word_idx]))
                else:
                    token_labels.append(-100)
                previous_word_idx = word_idx
            batch_labels.append(token_labels)
        return torch.tensor(batch_labels, dtype=torch.long)

    def print_alignment_preview(self, encoding, word_ids, sample, limit=50):
        print("first sample token/word/label alignment preview:")
        input_ids = encoding["input_ids"][0].tolist()
        labels = encoding["labels"][0].tolist()
        tokens = self.processor.tokenizer.convert_ids_to_tokens(input_ids)
        for token_idx, (token, word_idx, label_id) in enumerate(zip(tokens, word_ids, labels)):
            if token_idx >= limit:
                break
            word_text = None if word_idx is None else sample["words"][word_idx]
            label = "IGN" if label_id == -100 else self.id2label[int(label_id)]
            print(
                f"  token_idx={token_idx} token={token!r} word_idx={word_idx} "
                f"word={word_text!r} gold_label={label}"
            )


def select_device(device_arg):
    cuda_available = torch.cuda.is_available()
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda":
        if not cuda_available:
            fail("--device cuda requested, but torch.cuda.is_available() is False.")
        return torch.device("cuda")
    return torch.device("cuda" if cuda_available else "cpu")


def move_model_inputs_to_device(batch, device):
    return {key: value.to(device) for key, value in batch["model_inputs"].items()}


def safe_seqeval_metric(metric_fn, true_sequences, pred_sequences):
    try:
        return float(metric_fn(true_sequences, pred_sequences)) if true_sequences else 0.0
    except Exception as exc:
        print(f"WARNING: seqeval metric failed: {exc}")
        return 0.0


def evaluate(model, eval_loader, device, id2label, processor=None, collect_preview=False, preview_limit=20):
    model.eval()
    losses = []
    total_correct = 0
    total_count = 0
    true_sequences = []
    pred_sequences = []
    label_counts = Counter()
    label_correct = Counter()
    pred_label_counts = Counter()
    previews = []

    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="validation", leave=False):
            model_inputs = move_model_inputs_to_device(batch, device)
            outputs = model(**model_inputs)
            losses.append(float(outputs.loss.detach().cpu().item()))
            preds = outputs.logits.argmax(dim=-1).detach().cpu()
            labels = batch["model_inputs"]["labels"].detach().cpu()
            input_ids = batch["model_inputs"]["input_ids"].detach().cpu()

            for row_idx, (pred_row, label_row) in enumerate(zip(preds, labels)):
                true_seq = []
                pred_seq = []
                preview_tokens = []
                tokens = None
                if collect_preview and processor is not None and len(previews) < preview_limit:
                    tokens = processor.tokenizer.convert_ids_to_tokens(input_ids[row_idx].tolist())

                for token_idx, (pred_id, label_id) in enumerate(zip(pred_row.tolist(), label_row.tolist())):
                    if label_id == -100:
                        continue
                    gt_label = id2label[int(label_id)]
                    pred_label = id2label[int(pred_id)]
                    total_count += 1
                    label_counts[gt_label] += 1
                    pred_label_counts[pred_label] += 1
                    if pred_id == label_id:
                        total_correct += 1
                        label_correct[gt_label] += 1
                    true_seq.append(gt_label)
                    pred_seq.append(pred_label)
                    if tokens is not None:
                        preview_tokens.append(
                            {
                                "token": tokens[token_idx],
                                "gt_label": gt_label,
                                "pred_label": pred_label,
                                "correct": pred_id == label_id,
                            }
                        )

                true_sequences.append(true_seq)
                pred_sequences.append(pred_seq)
                if collect_preview and preview_tokens and len(previews) < preview_limit:
                    record = batch["records"][row_idx]
                    previews.append(
                        {
                            "id": record["id"],
                            "split": record["split"],
                            "index": record["index"],
                            "tokens": preview_tokens,
                        }
                    )

    token_accuracy = total_correct / total_count if total_count else 0.0
    metrics = {
        "eval_loss": float(np.mean(losses)) if losses else math.nan,
        "token_accuracy": token_accuracy,
        "seqeval_precision": safe_seqeval_metric(precision_score, true_sequences, pred_sequences),
        "seqeval_recall": safe_seqeval_metric(recall_score, true_sequences, pred_sequences),
        "seqeval_f1": safe_seqeval_metric(f1_score, true_sequences, pred_sequences),
        "num_eval_tokens": total_count,
        "label_counts": dict(label_counts),
        "pred_label_counts": dict(pred_label_counts),
        "label_accuracy": {
            label: label_correct[label] / count
            for label, count in label_counts.items()
            if count > 0
        },
        "pred_o_ratio": pred_label_counts.get("O", 0) / total_count if total_count else 0.0,
    }
    return metrics, previews


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def save_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def save_model_bundle(model, processor, labels_payload, path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(path))
    processor.save_pretrained(str(path))
    save_json(path / "labels.json", labels_payload)


def prepare_output_dir(args):
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if args.resume_from_checkpoint:
            print(f"output_dir exists and resume_from_checkpoint is set: {output_dir}")
        elif args.overwrite_output_dir:
            print(f"Deleting existing output_dir before training: {output_dir.resolve()}")
            shutil.rmtree(output_dir)
        else:
            fail(f"{output_dir} already exists. Use --overwrite_output_dir or --resume_from_checkpoint.")
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def print_debug_samples(train_records, validation_records):
    print("train sample ids preview:")
    for record in train_records[:10]:
        print(f"  {record['id']} words={len(record['words'])}")
    print("validation sample ids preview:")
    for record in validation_records[:10]:
        print(f"  {record['id']} words={len(record['words'])}")
    first = train_records[0]
    print(f"first train sample: {first['id']} word_count={len(first['words'])}")
    print("first train sample first 50 words/labels:")
    for idx, (word, label) in enumerate(zip(first["words"][:50], first["labels"][:50])):
        print(f"  [{idx}] word={word!r} label={label}")


def build_dataloaders(args, train_dataset, eval_dataset, processor, id2label):
    eval_batch_size = args.eval_batch_size or args.batch_size
    train_collator = LayoutLMv3Collator(
        processor=processor,
        max_length=args.max_length,
        id2label=id2label,
        debug=args.debug,
    )
    eval_collator = LayoutLMv3Collator(
        processor=processor,
        max_length=args.max_length,
        id2label=id2label,
        debug=False,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=train_collator,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=eval_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=eval_collator,
    )
    return train_loader, eval_loader


def create_optimizer_and_scheduler(args, model, train_loader):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    update_steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_update_steps = update_steps_per_epoch * args.epochs
    warmup_steps = int(0.1 * total_update_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )
    print(f"train batches per epoch: {len(train_loader)}")
    print(f"gradient accumulation steps: {args.gradient_accumulation_steps}")
    print(f"total update steps: {total_update_steps}")
    print(f"warmup steps: {warmup_steps}")
    return optimizer, scheduler, total_update_steps, warmup_steps


def train_loop(
    args,
    model,
    processor,
    train_loader,
    eval_loader,
    device,
    id2label,
    labels_payload,
    output_dir,
):
    optimizer, scheduler, total_update_steps, warmup_steps = create_optimizer_and_scheduler(
        args, model, train_loader
    )
    use_fp16 = args.fp16 and device.type == "cuda"
    if args.fp16 and device.type != "cuda":
        print("WARNING: --fp16 was requested without CUDA; disabling mixed precision.")
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    history = []
    best_metrics = None
    best_epoch = None
    initial_train_loss = None
    global_update_step = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch:03d}/{args.epochs:03d}")
        epoch_losses = []
        accum_steps_since_update = 0

        for step, batch in enumerate(progress, start=1):
            model_inputs = move_model_inputs_to_device(batch, device)
            try:
                with torch.cuda.amp.autocast(enabled=use_fp16):
                    outputs = model(**model_inputs)
                    loss = outputs.loss
                    loss_to_backprop = loss / args.gradient_accumulation_steps
            except torch.cuda.OutOfMemoryError:
                print("CUDA out of memory during forward pass.", file=sys.stderr)
                print("Try --batch_size 1, --gradient_accumulation_steps 8, --fp16, --max_length 384 or 256.", file=sys.stderr)
                raise

            raw_loss = float(loss.detach().cpu().item())
            if initial_train_loss is None:
                initial_train_loss = raw_loss
            epoch_losses.append(raw_loss)

            if args.debug and epoch == 1 and step == 1:
                print(f"first batch loss: {raw_loss:.6f}")

            if use_fp16:
                scaler.scale(loss_to_backprop).backward()
            else:
                loss_to_backprop.backward()
            accum_steps_since_update += 1

            if accum_steps_since_update == args.gradient_accumulation_steps or step == len(train_loader):
                if use_fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                accum_steps_since_update = 0
                global_update_step += 1

            if args.log_every_steps > 0 and step % args.log_every_steps == 0:
                current_lr = scheduler.get_last_lr()[0]
                recent_loss = float(np.mean(epoch_losses[-args.log_every_steps:]))
                print(
                    f"epoch={epoch:03d} step={step:05d}/{len(train_loader):05d} "
                    f"loss={recent_loss:.6f} lr={current_lr:.8f}"
                )

        train_loss = float(np.mean(epoch_losses)) if epoch_losses else math.nan
        eval_metrics, previews = evaluate(
            model,
            eval_loader,
            device,
            id2label,
            processor=processor,
            collect_preview=True,
            preview_limit=args.num_preview_samples,
        )
        current_lr = scheduler.get_last_lr()[0]
        best_so_far = False
        if best_metrics is None:
            best_so_far = True
        else:
            current_f1 = eval_metrics["seqeval_f1"]
            best_f1 = best_metrics["seqeval_f1"]
            if current_f1 > best_f1:
                best_so_far = True
            elif current_f1 == best_f1 and eval_metrics["eval_loss"] < best_metrics["eval_loss"]:
                best_so_far = True

        epoch_record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_metrics["eval_loss"],
            "token_accuracy": eval_metrics["token_accuracy"],
            "seqeval_precision": eval_metrics["seqeval_precision"],
            "seqeval_recall": eval_metrics["seqeval_recall"],
            "seqeval_f1": eval_metrics["seqeval_f1"],
            "pred_o_ratio": eval_metrics["pred_o_ratio"],
            "lr": current_lr,
            "best_so_far": best_so_far,
        }
        history.append(epoch_record)

        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | train_loss={train_loss:.6f} | "
            f"val_loss={eval_metrics['eval_loss']:.6f} | "
            f"val_acc={eval_metrics['token_accuracy']:.6f} | "
            f"val_precision={eval_metrics['seqeval_precision']:.6f} | "
            f"val_recall={eval_metrics['seqeval_recall']:.6f} | "
            f"val_f1={eval_metrics['seqeval_f1']:.6f} | pred_o_ratio={eval_metrics['pred_o_ratio']:.6f}"
        )

        if args.debug and epoch == 1 and previews:
            print("first epoch validation prediction preview:")
            for item in previews[0]["tokens"][:50]:
                print(
                    f"  token={item['token']!r} gt={item['gt_label']} "
                    f"pred={item['pred_label']} correct={item['correct']}"
                )

        if best_so_far:
            best_metrics = dict(eval_metrics)
            best_epoch = epoch
            best_metrics["best_epoch"] = best_epoch
            save_model_bundle(model, processor, labels_payload, output_dir / "best")
            save_json(output_dir / "best_metrics.json", best_metrics)
            save_jsonl("outputs/layoutlmv3_cord_full_eval/validation_preview_epoch_best.jsonl", previews)
            print(f"new best checkpoint saved: {output_dir / 'best'}")

        if args.save_every_epoch:
            epoch_dir = output_dir / "checkpoints" / f"epoch_{epoch:03d}"
            save_model_bundle(model, processor, labels_payload, epoch_dir)

        save_json(output_dir / "training_history.json", {"history": history})

    final_metrics, last_previews = evaluate(
        model,
        eval_loader,
        device,
        id2label,
        processor=processor,
        collect_preview=True,
        preview_limit=args.num_preview_samples,
    )
    save_model_bundle(model, processor, labels_payload, output_dir / "last")
    save_jsonl("outputs/layoutlmv3_cord_full_eval/validation_preview_last.jsonl", last_previews)
    final_payload = {
        "epoch": args.epochs,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": history[-1]["train_loss"] if history else None,
        "final_eval_metrics": final_metrics,
        "best_epoch": best_epoch,
        "best_metrics": best_metrics,
        "total_update_steps": total_update_steps,
        "warmup_steps": warmup_steps,
    }
    save_json(output_dir / "final_metrics.json", final_payload)
    return initial_train_loss, history, best_epoch, best_metrics, final_payload


def load_model_and_processor(args, label_list, label2id, id2label):
    load_path = args.resume_from_checkpoint or args.model_name_or_path
    if args.resume_from_checkpoint:
        print(
            "Resuming model weights from checkpoint. Optimizer and scheduler state are not restored: "
            f"{args.resume_from_checkpoint}"
        )
    if "cord10" in str(load_path).lower() or "overfit" in str(load_path).lower():
        fail("Do not start full training from the cord10 overfit checkpoint. Use a base checkpoint.")

    print(f"model path: {load_path}")
    print("processor apply_ocr=False")
    processor = AutoProcessor.from_pretrained(
        load_path,
        apply_ocr=False,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        load_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
    )
    return model, processor, load_path


def save_training_config(
    args,
    output_dir,
    num_labels,
    train_records,
    validation_records,
    device,
    cuda_available,
    load_path,
):
    config = {
        "bio_dir": args.bio_dir,
        "raw_data_dir": args.raw_data_dir,
        "model_name_or_path": args.model_name_or_path,
        "actual_model_load_path": str(load_path),
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size or args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "max_length": args.max_length,
        "fp16": args.fp16 and device.type == "cuda",
        "seed": args.seed,
        "num_labels": num_labels,
        "num_train_samples": len(train_records),
        "num_eval_samples": len(validation_records),
        "device": str(device),
        "cuda_available": cuda_available,
    }
    save_json(output_dir / "training_config.json", config)
    return config


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"WSL/conda Python path: {sys.executable}")
    print(f"bio_dir: {args.bio_dir}")
    print(f"raw_data_dir: {args.raw_data_dir}")

    require_path(Path(args.bio_dir) / "train.jsonl", f"{args.bio_dir}/train.jsonl not found. Run step 3 first.")
    require_path(Path(args.bio_dir) / "validation.jsonl", f"{args.bio_dir}/validation.jsonl not found. Run step 3 first.")
    require_path(Path(args.bio_dir) / "labels.json", f"{args.bio_dir}/labels.json not found. Run step 3 first.")
    require_path(args.raw_data_dir, f"{args.raw_data_dir} not found. Run CORD-v2 download step first.")

    label_list, label2id, id2label, labels_payload = load_labels(args.bio_dir, args.label_schema)
    train_records = load_bio_records(args.bio_dir, "train", label2id, args.max_train_samples)
    validation_records = load_bio_records(args.bio_dir, "validation", label2id, args.max_eval_samples)

    train_label_counts, train_o_ratio = summarize_label_distribution(train_records, "train")
    validation_label_counts, validation_o_ratio = summarize_label_distribution(validation_records, "validation")

    if args.debug:
        print_debug_samples(train_records, validation_records)

    raw_dataset = load_raw_dataset(args.raw_data_dir)
    print(f"raw dataset splits: {list(raw_dataset.keys())}")
    print(f"raw train length: {len(raw_dataset['train'])}")
    print(f"raw validation length: {len(raw_dataset['validation'])}")

    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"cuda device name: {torch.cuda.get_device_name(0)}")
    device = select_device(args.device)
    print(f"selected device: {device}")

    if device.type != "cuda" and args.max_train_samples is None:
        fail("CUDA is not available; refusing to start full CORD fine-tuning on CPU. Use a GPU or a debug dry-run.")

    output_dir = prepare_output_dir(args)
    model, processor, load_path = load_model_and_processor(args, label_list, label2id, id2label)

    train_dataset = CordBioTokenClassificationDataset(train_records, raw_dataset, label2id)
    eval_dataset = CordBioTokenClassificationDataset(validation_records, raw_dataset, label2id)
    train_loader, eval_loader = build_dataloaders(args, train_dataset, eval_dataset, processor, id2label)

    training_config = save_training_config(
        args,
        output_dir,
        len(label_list),
        train_records,
        validation_records,
        device,
        cuda_available,
        load_path,
    )
    training_config["train_o_ratio"] = train_o_ratio
    training_config["validation_o_ratio"] = validation_o_ratio
    training_config["train_label_distribution_top_50"] = dict(train_label_counts.most_common(50))
    training_config["validation_label_distribution_top_50"] = dict(validation_label_counts.most_common(50))
    save_json(output_dir / "training_config.json", training_config)

    model.to(device)
    try:
        initial_train_loss, history, best_epoch, best_metrics, final_payload = train_loop(
            args=args,
            model=model,
            processor=processor,
            train_loader=train_loader,
            eval_loader=eval_loader,
            device=device,
            id2label=id2label,
            labels_payload=labels_payload,
            output_dir=output_dir,
        )
    except torch.cuda.OutOfMemoryError:
        print("CUDA out of memory.", file=sys.stderr)
        print("Recommended adjustments:", file=sys.stderr)
        print("  --batch_size 1", file=sys.stderr)
        print("  --gradient_accumulation_steps 8", file=sys.stderr)
        print("  --fp16", file=sys.stderr)
        print("  --max_length 384 or 256", file=sys.stderr)
        print("  stop other GPU processes", file=sys.stderr)
        raise

    if best_metrics:
        print(f"best validation F1: {best_metrics['seqeval_f1']:.6f}")
        print(f"best validation precision: {best_metrics['seqeval_precision']:.6f}")
        print(f"best validation recall: {best_metrics['seqeval_recall']:.6f}")
        print(f"best validation token accuracy: {best_metrics['token_accuracy']:.6f}")
        print(f"best epoch: {best_epoch}")
        print(f"best checkpoint path: {output_dir / 'best'}")
    print(f"last checkpoint path: {output_dir / 'last'}")
    print(f"training_history.json path: {output_dir / 'training_history.json'}")
    print(f"best_metrics.json path: {output_dir / 'best_metrics.json'}")
    print("validation preview best path: outputs/layoutlmv3_cord_full_eval/validation_preview_epoch_best.jsonl")
    print("validation preview last path: outputs/layoutlmv3_cord_full_eval/validation_preview_last.jsonl")

    pred_o_ratio = best_metrics.get("pred_o_ratio", 0.0) if best_metrics else 0.0
    if pred_o_ratio > 0.95:
        print("WARNING: possible O label collapse detected.")
    else:
        print("O label collapse: no obvious collapse detected.")
    print("LayoutLMv3 CORD full fine-tuning finished.")


if __name__ == "__main__":
    main()
