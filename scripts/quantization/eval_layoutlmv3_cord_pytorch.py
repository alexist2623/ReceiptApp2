import argparse
import json
import sys
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForTokenClassification, AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.quantization.quant_common import (
    aggregate_token_metrics,
    canonicalize_sequence,
    confusion_top,
    encode_sample,
    fail,
    first_token_word_predictions,
    label_distribution,
    labels_from_encoding,
    load_cord_jsonl,
    load_labels,
    load_raw_dataset,
    prepare_cord_sample,
    save_json,
    timed_call,
    torch_inputs_from_encoding,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CORD-only LayoutLMv3 PyTorch FP32 baseline.")
    parser.add_argument("--checkpoint", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--cord_bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--cord_raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--out_dir", default="outputs/quantization/cord_baseline_pytorch_fp32")
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--latency_repeat", type=int, default=1)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def select_device(name):
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    if args.local_files_only and not checkpoint.exists():
        fail(f"checkpoint not found: {checkpoint}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_payload, label_list, label2id, id2label = load_labels(checkpoint)
    raw_dataset = load_raw_dataset(args.cord_raw_data_dir)
    records = load_cord_jsonl(args.cord_bio_dir, args.split, args.max_samples)
    device = select_device(args.device)
    processor = AutoProcessor.from_pretrained(str(checkpoint), apply_ocr=False, local_files_only=args.local_files_only)
    model = AutoModelForTokenClassification.from_pretrained(str(checkpoint), local_files_only=args.local_files_only)
    model.to(device)
    model.eval()

    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"selected_device: {device}")
    print(f"checkpoint: {checkpoint}")
    print(f"split: {args.split} records={len(records)}")

    true_sequences = []
    pred_sequences = []
    predictions = []
    num_tokens = 0
    correct_tokens = 0
    total_loss = 0.0
    latencies = []
    for record in tqdm(records, desc=f"pytorch eval {args.split}", unit="sample"):
        sample = prepare_cord_sample(record, raw_dataset, label2id, args.split)
        encoding = encode_sample(processor, sample, args.max_length, include_labels=True)
        model_inputs = torch_inputs_from_encoding(encoding, device)

        def run_model():
            with torch.no_grad():
                return model(**model_inputs, output_hidden_states=True, return_dict=True)

        outputs, times = timed_call(run_model, warmup=args.warmup, repeat=args.latency_repeat)
        latencies.extend(times)
        labels = model_inputs["labels"]
        mask = labels != -100
        token_count = int(mask.sum().item())
        preds = outputs.logits.argmax(dim=-1)
        correct = int(((preds == labels) & mask).sum().item())
        num_tokens += token_count
        correct_tokens += correct
        total_loss += float(outputs.loss.detach().cpu().item()) * token_count

        true_token_labels_raw = labels_from_encoding(labels[0].detach().cpu(), id2label)
        pred_token_labels_raw = []
        for label_id, pred_id, keep in zip(labels[0].detach().cpu().tolist(), preds[0].detach().cpu().tolist(), mask[0].detach().cpu().tolist()):
            if keep:
                pred_token_labels_raw.append(id2label[int(pred_id)])
        true_token_labels = canonicalize_sequence(true_token_labels_raw)
        pred_token_labels = canonicalize_sequence(pred_token_labels_raw)
        true_sequences.append(true_token_labels)
        pred_sequences.append(pred_token_labels)

        word_pred_raw, conf, _, _ = first_token_word_predictions(
            outputs.logits[0].detach().cpu(),
            encoding,
            id2label,
            len(sample["words"]),
            attention_mask=encoding["attention_mask"][0].detach().cpu().tolist(),
        )
        predictions.append(
            {
                "id": sample["id"],
                "source": "cord",
                "split": args.split,
                "index": sample["index"],
                "words": sample["words"],
                "gold_labels": sample["gold_labels"],
                "pred_labels": word_pred_raw,
                "canonical_gold_labels": sample["canonical_gold_labels"],
                "canonical_pred_labels": canonicalize_sequence(word_pred_raw),
                "confidences": conf,
                "image_size": sample["image_size"],
            }
        )
        if args.debug and len(predictions) <= 1:
            print(json.dumps(predictions[-1], ensure_ascii=False)[:2000])

    metrics = aggregate_token_metrics(true_sequences, pred_sequences, num_tokens, correct_tokens)
    metrics.update(
        {
            "checkpoint": str(checkpoint),
            "split": args.split,
            "max_samples": args.max_samples,
            "num_samples": len(records),
            "eval_loss": total_loss / num_tokens if num_tokens else None,
            "label_list_size": len(label_list),
            "latency_ms": {
                "avg": sum(latencies) / len(latencies) if latencies else None,
                "count": len(latencies),
                "min": min(latencies) if latencies else None,
                "max": max(latencies) if latencies else None,
            },
            "user_data_used": False,
            "source": "CORD-only",
        }
    )
    save_json(out_dir / f"metrics_{args.split}.json", metrics)
    write_jsonl(out_dir / f"predictions_{args.split}.jsonl", predictions)
    save_json(
        out_dir / f"label_distribution_{args.split}.json",
        {
            "gold": label_distribution(true_sequences),
            "pred": label_distribution(pred_sequences),
        },
    )
    save_json(out_dir / f"confusion_top_{args.split}.json", confusion_top(true_sequences, pred_sequences))
    save_json(out_dir / "labels.json", labels_payload)
    print(f"num_samples: {len(records)}")
    print(f"num_tokens: {num_tokens}")
    print(f"seqeval_f1: {metrics['seqeval_f1']:.6f}")
    for field in ("ITEM_NAME", "ITEM_PRICE", "TOTAL_PRICE", "SUBTOTAL_PRICE", "TAX_PRICE"):
        print(f"{field}_f1: {metrics['field_metrics'].get(field, {}).get('f1', 0.0):.6f}")
    print(f"metrics path: {out_dir / f'metrics_{args.split}.json'}")
    print("PyTorch FP32 CORD baseline evaluation passed.")


if __name__ == "__main__":
    main()
