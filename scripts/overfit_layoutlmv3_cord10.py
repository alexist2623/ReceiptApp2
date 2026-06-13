import argparse
import json
import math
import random
import shutil
import sys
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image, ImageDraw, ImageFont
from seqeval.metrics import f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoProcessor


def parse_args():
    parser = argparse.ArgumentParser(
        description="Overfit LayoutLMv3 token classification on 10 CORD-v2 BIO samples."
    )
    parser.add_argument("--bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--label_schema", default=None, help="Optional labels schema JSON, e.g. schemas/receipt_labels_v2.json.")
    parser.add_argument("--user_labeled_jsonl", default=None, help="Reserved for future schema-v2 user labeled JSONL overfit tests.")
    parser.add_argument("--init_from_checkpoint", default=None, help="Reserved for initializing from an existing fine-tuned checkpoint.")
    parser.add_argument("--copy_old_classifier_rows", action="store_true", help="Reserved for alias-based classifier row initialization.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--model_name_or_path", default="microsoft/layoutlmv3-base")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--output_dir", default="models/layoutlmv3-cord10-overfit")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument(
        "--save_overlay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save prediction overlay images after training.",
    )
    parser.add_argument("--out_overlay_dir", default="outputs/cord10_overfit_overlay")
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


def load_bio_records(bio_dir, split, start_index, num_samples, label2id):
    jsonl_path = Path(bio_dir) / f"{split}.jsonl"
    require_path(jsonl_path, f"{jsonl_path} not found. Run step 3 first.")

    records = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            validate_record(record, line_number, label2id)
            records.append(record)

    end_index = start_index + num_samples
    selected = records[start_index:end_index]
    if len(selected) < num_samples:
        print(
            f"WARNING: requested {num_samples} samples from {jsonl_path}, "
            f"but only {len(selected)} are available from start_index={start_index}."
        )
    if not selected:
        fail("No BIO records selected for overfit test.")
    return selected


def validate_record(record, line_number, label2id):
    required = ("id", "split", "index", "words", "normalized_boxes", "labels")
    missing = [key for key in required if key not in record]
    if missing:
        fail(f"BIO JSONL line {line_number} is missing required fields: {missing}")

    lengths = {
        "words": len(record["words"]),
        "normalized_boxes": len(record["normalized_boxes"]),
        "labels": len(record["labels"]),
    }
    if len(set(lengths.values())) != 1:
        fail(f"BIO JSONL line {line_number} has length mismatch: {lengths}")

    for box in record["normalized_boxes"]:
        if len(box) != 4 or any(value < 0 or value > 1000 for value in box):
            fail(f"BIO JSONL line {line_number} has invalid normalized box: {box}")

    unknown_labels = [label for label in record["labels"] if label not in label2id]
    if unknown_labels:
        fail(f"BIO JSONL line {line_number} has labels not present in labels.json: {unknown_labels[:5]}")


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


def load_raw_dataset(raw_data_dir, split):
    require_path(raw_data_dir, f"{raw_data_dir} not found. Run CORD-v2 download step first.")
    raw_dataset = load_from_disk(str(raw_data_dir))
    if split not in raw_dataset:
        fail(f"Raw dataset split '{split}' not found. Available splits: {list(raw_dataset.keys())}")
    return raw_dataset


class CordBioOverfitDataset(Dataset):
    def __init__(self, records, raw_split, label2id):
        self.records = records
        self.raw_split = raw_split
        self.label2id = label2id

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        raw_index = int(record["index"])
        if raw_index < 0 or raw_index >= len(self.raw_split):
            fail(f"{record['id']}: raw dataset index {raw_index} out of range for split length {len(self.raw_split)}")
        image = ensure_pil_rgb(self.raw_split[raw_index]["image"])
        word_labels = [self.label2id[label] for label in record["labels"]]
        return {
            "image": image,
            "words": [str(word) for word in record["words"]],
            "boxes": [[int(v) for v in box] for box in record["normalized_boxes"]],
            "word_labels": word_labels,
            "record": record,
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
            "records": [sample["record"] for sample in samples],
            "word_ids": word_ids_batch,
            "words": words,
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


def compute_metrics(model, dataloader, device, id2label):
    model.eval()
    total_correct = 0
    total_count = 0
    true_sequences = []
    pred_sequences = []
    losses = []

    with torch.no_grad():
        for batch in dataloader:
            model_inputs = move_model_inputs_to_device(batch, device)
            outputs = model(**model_inputs)
            losses.append(float(outputs.loss.detach().cpu().item()))
            preds = outputs.logits.argmax(dim=-1).detach().cpu()
            labels = batch["model_inputs"]["labels"].detach().cpu()

            for pred_row, label_row in zip(preds, labels):
                true_seq = []
                pred_seq = []
                for pred_id, label_id in zip(pred_row.tolist(), label_row.tolist()):
                    if label_id == -100:
                        continue
                    total_count += 1
                    if pred_id == label_id:
                        total_correct += 1
                    true_seq.append(id2label[int(label_id)])
                    pred_seq.append(id2label[int(pred_id)])
                true_sequences.append(true_seq)
                pred_sequences.append(pred_seq)

    token_acc = total_correct / total_count if total_count else 0.0
    seq_f1 = f1_score(true_sequences, pred_sequences) if true_sequences else 0.0
    avg_loss = float(np.mean(losses)) if losses else math.nan
    model.train()
    return avg_loss, token_acc, seq_f1


def train(model, dataloader, device, args, id2label):
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    history = []
    initial_loss = None

    model.train()
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        progress = tqdm(dataloader, desc=f"epoch {epoch:03d}/{args.epochs:03d}", leave=False)
        running_loss = 0.0
        step_count = 0

        for step, batch in enumerate(progress, start=1):
            model_inputs = move_model_inputs_to_device(batch, device)
            outputs = model(**model_inputs)
            loss = outputs.loss / args.gradient_accumulation_steps
            loss.backward()

            raw_loss = float(outputs.loss.detach().cpu().item())
            if initial_loss is None:
                initial_loss = raw_loss
            running_loss += raw_loss
            step_count += 1

            if args.debug and epoch == 1 and step == 1:
                print(f"first epoch first batch loss: {raw_loss:.6f}")

            if step % args.gradient_accumulation_steps == 0 or step == len(dataloader):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        train_loss, token_acc, seq_f1 = compute_metrics(model, dataloader, device, id2label)
        epoch_summary = {
            "epoch": epoch,
            "loss": train_loss,
            "token_acc": token_acc,
            "seqeval_f1": seq_f1,
        }
        history.append(epoch_summary)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"loss={train_loss:.6f} | token_acc={token_acc:.6f} | seqeval_f1={seq_f1:.6f}"
        )

    return initial_loss, history


def save_checkpoint(model, processor, output_dir, labels_payload, metrics):
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(str(output_path))
    processor.save_pretrained(str(output_path))
    with (output_path / "labels.json").open("w", encoding="utf-8") as handle:
        json.dump(labels_payload, handle, ensure_ascii=False, indent=2)
    with (output_path / "training_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(f"saved checkpoint path: {output_path}")
    print(f"saved training metrics path: {output_path / 'training_metrics.json'}")


def normalized_to_pixel_box(box, width, height):
    x0, y0, x1, y1 = box
    return [
        int(round(x0 * width / 1000)),
        int(round(y0 * height / 1000)),
        int(round(x1 * width / 1000)),
        int(round(y1 * height / 1000)),
    ]


def draw_rectangle(draw, box, color, width=2):
    try:
        draw.rectangle(box, outline=color, width=width)
    except TypeError:
        x0, y0, x1, y1 = box
        for offset in range(width):
            draw.rectangle([x0 - offset, y0 - offset, x1 + offset, y1 + offset], outline=color)


def draw_label(draw, xy, lines, color, font):
    x, y = xy
    text = "\n".join(lines)
    try:
        bbox = draw.multiline_textbbox((x, y), text, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
    except AttributeError:
        w, h = draw.multiline_textsize(text, font=font)
    draw.rectangle([x, y, x + w + 4, y + h + 4], fill=(255, 255, 255))
    draw.multiline_text((x + 2, y + 2), text, fill=color, font=font)


def predict_one_sample(model, processor, sample, device, args, id2label):
    encoding = processor(
        sample["image"],
        sample["words"],
        boxes=sample["boxes"],
        padding="max_length",
        truncation=True,
        max_length=args.max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    model_inputs = {
        key: value.to(device)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"}
    }
    model.eval()
    with torch.no_grad():
        logits = model(**model_inputs).logits.detach().cpu()[0]
        probs = torch.softmax(logits, dim=-1)

    first_token_for_word = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if encoding["attention_mask"][0, token_idx].item() == 0:
            continue
        first_token_for_word.setdefault(word_idx, token_idx)

    results = []
    for word_idx, word in enumerate(sample["words"]):
        token_idx = first_token_for_word.get(word_idx)
        if token_idx is None:
            pred_label = "O"
            confidence = 0.0
        else:
            pred_id = int(torch.argmax(probs[token_idx]).item())
            pred_label = id2label[pred_id]
            confidence = float(probs[token_idx, pred_id].item())
        gt_label = sample["record"]["labels"][word_idx]
        width, height = sample["image"].size
        if sample["record"].get("boxes"):
            pixel_box = sample["record"]["boxes"][word_idx]
        else:
            pixel_box = normalized_to_pixel_box(sample["boxes"][word_idx], width, height)
        results.append(
            {
                "word_idx": word_idx,
                "word": word,
                "box": pixel_box,
                "gt_label": gt_label,
                "pred_label": pred_label,
                "confidence": confidence,
                "correct": pred_label == gt_label,
            }
        )
    return results


def save_prediction_overlays(model, processor, dataset, device, args, id2label):
    out_dir = Path(args.out_overlay_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    overlay_paths = []
    compare_paths = []
    first_preview = []

    for sample in dataset:
        record = sample["record"]
        image = sample["image"].copy().convert("RGB")
        draw = ImageDraw.Draw(image)
        results = predict_one_sample(model, processor, sample, device, args, id2label)
        correct = sum(1 for item in results if item["correct"])
        word_accuracy = correct / len(results) if results else 0.0

        for item in results:
            if item["gt_label"] == "O":
                color = (120, 120, 120)
            elif item["correct"]:
                color = (30, 150, 60)
            else:
                color = (210, 45, 45)
            draw_rectangle(draw, item["box"], color, width=2)
            x0, y0, _, _ = item["box"]
            label_y = max(0, y0 - 34)
            draw_label(
                draw,
                (x0, label_y),
                [
                    f"GT={item['gt_label']}",
                    f"P={item['pred_label']} {item['confidence']:.2f}",
                    f"word={item['word'][:20]}",
                ],
                color,
                font,
            )

        overlay_path = out_dir / f"{record['id']}_pred.png"
        compare_path = out_dir / f"{record['id']}_compare.json"
        image.save(overlay_path)
        compare_payload = {
            "id": record["id"],
            "split": record["split"],
            "index": record["index"],
            "num_words": len(results),
            "word_results": results,
            "word_accuracy": word_accuracy,
        }
        with compare_path.open("w", encoding="utf-8") as handle:
            json.dump(compare_payload, handle, ensure_ascii=False, indent=2)

        overlay_paths.append(str(overlay_path))
        compare_paths.append(str(compare_path))
        if not first_preview:
            first_preview = results[:30]

    print("saved overlay paths:")
    for path in overlay_paths:
        print(f"  {path}")
    print("saved compare JSON paths:")
    for path in compare_paths:
        print(f"  {path}")
    return overlay_paths, compare_paths, first_preview


def print_dataset_debug(records):
    print("used sample ids:")
    for record in records:
        print(f"  {record['id']} words={len(record['words'])}")
    label_counts = Counter(label for record in records for label in record["labels"])
    print("label distribution for selected samples:")
    for label, count in label_counts.most_common(30):
        print(f"  {label}: {count}")


def main():
    args = parse_args()
    set_seed(args.seed)

    print(f"WSL/conda Python path: {sys.executable}")
    print(f"bio_dir: {args.bio_dir}")
    print(f"raw_data_dir: {args.raw_data_dir}")

    bio_jsonl = Path(args.bio_dir) / f"{args.split}.jsonl"
    require_path(bio_jsonl, f"{bio_jsonl} not found. Run step 3 first.")
    require_path(Path(args.bio_dir) / "labels.json", f"{args.bio_dir}/labels.json not found. Run step 3 first.")
    require_path(args.raw_data_dir, f"{args.raw_data_dir} not found. Run CORD-v2 download step first.")

    label_list, label2id, id2label, labels_payload = load_labels(args.bio_dir, args.label_schema)
    records = load_bio_records(args.bio_dir, args.split, args.start_index, args.num_samples, label2id)
    raw_dataset = load_raw_dataset(args.raw_data_dir, args.split)
    raw_split = raw_dataset[args.split]

    print(f"selected split: {args.split}")
    print(f"selected sample count: {len(records)}")
    if args.debug:
        print_dataset_debug(records)

    print(f"model path: {args.model_name_or_path}")
    print("processor apply_ocr=False")
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        apply_ocr=False,
        local_files_only=args.local_files_only,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
        local_files_only=args.local_files_only,
    )

    print(f"torch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"torch.cuda.is_available(): {cuda_available}")
    if cuda_available:
        print(f"cuda device name: {torch.cuda.get_device_name(0)}")
    device = select_device(args.device)
    print(f"selected device: {device}")

    dataset = CordBioOverfitDataset(records, raw_split, label2id)
    collator = LayoutLMv3Collator(processor, args.max_length, id2label, debug=args.debug)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
    )
    eval_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=LayoutLMv3Collator(processor, args.max_length, id2label, debug=False),
    )

    model.to(device)
    initial_loss, history = train(model, dataloader, device, args, id2label)
    final_loss, final_token_acc, final_seqeval_f1 = compute_metrics(model, eval_dataloader, device, id2label)

    print(
        "Final metrics | "
        f"initial_loss={initial_loss:.6f} | final_loss={final_loss:.6f} | "
        f"final_token_acc={final_token_acc:.6f} | final_seqeval_f1={final_seqeval_f1:.6f}"
    )

    if final_loss < initial_loss and (final_token_acc > 0.80 or final_seqeval_f1 > 0.70):
        sanity_status = "passed"
        print("Overfit sanity check passed")
    else:
        sanity_status = "warning"
        print("Overfit sanity check warning: metrics did not improve enough")

    metrics_payload = {
        "num_samples": len(records),
        "sample_ids": [record["id"] for record in records],
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_token_acc": final_token_acc,
        "final_seqeval_f1": final_seqeval_f1,
        "sanity_status": sanity_status,
        "history": history,
    }
    save_checkpoint(model, processor, args.output_dir, labels_payload, metrics_payload)

    overlay_paths = []
    compare_paths = []
    first_preview = []
    if args.save_overlay:
        overlay_paths, compare_paths, first_preview = save_prediction_overlays(
            model, processor, dataset, device, args, id2label
        )
        if args.debug:
            print("first sample word-level GT vs pred preview:")
            for item in first_preview[:50]:
                print(
                    f"  [{item['word_idx']}] word={item['word']!r} "
                    f"gt={item['gt_label']} pred={item['pred_label']} "
                    f"conf={item['confidence']:.4f} correct={item['correct']}"
                )

    run_summary = {
        "sample_ids": [record["id"] for record in records],
        "num_labels": len(label_list),
        "checkpoint_dir": args.output_dir,
        "training_metrics_path": str(Path(args.output_dir) / "training_metrics.json"),
        "overlay_paths": overlay_paths,
        "compare_paths": compare_paths,
        "first_sample_preview": first_preview[:30],
    }
    with (Path(args.output_dir) / "run_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(run_summary, handle, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
