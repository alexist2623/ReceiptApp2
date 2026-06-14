import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from tqdm import tqdm
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.quantization.quant_common import (
    aggregate_token_metrics,
    canonicalize_sequence,
    confusion_top,
    encode_sample,
    fail,
    file_size_mb,
    first_token_word_predictions,
    label_distribution,
    labels_from_encoding,
    latency_stats,
    load_cord_jsonl,
    load_labels,
    load_raw_dataset,
    numpy_inputs_from_encoding,
    onnx_total_size_mb,
    prepare_cord_sample,
    provider_list,
    save_json,
    timed_call,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate CORD-only LayoutLMv3 ONNX model.")
    parser.add_argument("--onnx_model", required=True)
    parser.add_argument("--checkpoint_for_processor", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--cord_bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--cord_raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--onnx_provider", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--latency_repeat", type=int, default=1)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--out_dir", default="outputs/quantization/cord_onnx_eval")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    onnx_model = Path(args.onnx_model)
    checkpoint = Path(args.checkpoint_for_processor)
    if not onnx_model.exists():
        fail(f"ONNX model not found: {onnx_model}")
    if args.local_files_only and not checkpoint.exists():
        fail(f"checkpoint_for_processor not found: {checkpoint}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels_payload, label_list, label2id, id2label = load_labels(checkpoint)
    raw_dataset = load_raw_dataset(args.cord_raw_data_dir)
    records = load_cord_jsonl(args.cord_bio_dir, args.split, args.max_samples)
    processor = AutoProcessor.from_pretrained(str(checkpoint), apply_ocr=False, local_files_only=args.local_files_only)
    session = ort.InferenceSession(str(onnx_model), providers=provider_list(args.onnx_provider))
    output_names = [output.name for output in session.get_outputs()]
    print(f"onnx_model: {onnx_model}")
    print(f"providers: {session.get_providers()}")
    print(f"split: {args.split} records={len(records)}")

    true_sequences = []
    pred_sequences = []
    predictions = []
    num_tokens = 0
    correct_tokens = 0
    latencies = []
    for record in tqdm(records, desc=f"onnx eval {args.split}", unit="sample"):
        sample = prepare_cord_sample(record, raw_dataset, label2id, args.split)
        encoding = encode_sample(processor, sample, args.max_length, include_labels=True)
        inputs = numpy_inputs_from_encoding(encoding)

        def run_onnx():
            return session.run(None, inputs)

        ort_outputs, times = timed_call(run_onnx, warmup=args.warmup if not latencies else 0, repeat=args.latency_repeat)
        latencies.extend(times)
        outputs = dict(zip(output_names, ort_outputs))
        logits = outputs["logits"][0]
        labels = encoding["labels"][0]
        mask = labels != -100
        pred_ids = np.argmax(logits, axis=-1)
        gold_raw = labels_from_encoding(labels, id2label)
        pred_raw = []
        for gold_id, pred_id, keep in zip(labels.tolist(), pred_ids.tolist(), mask.tolist()):
            if not keep:
                continue
            pred_label = id2label[int(pred_id)]
            pred_raw.append(pred_label)
            correct_tokens += int(pred_label == id2label[int(gold_id)])
        num_tokens += int(mask.sum().item())
        gold = canonicalize_sequence(gold_raw)
        pred = canonicalize_sequence(pred_raw)
        true_sequences.append(gold)
        pred_sequences.append(pred)
        word_pred_raw, conf, _, _ = first_token_word_predictions(
            torch.from_numpy(logits),
            encoding,
            id2label,
            len(sample["words"]),
            attention_mask=encoding["attention_mask"][0].tolist(),
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
    metrics = aggregate_token_metrics(true_sequences, pred_sequences, num_tokens, correct_tokens)
    metrics.update(
        {
            "onnx_model": str(onnx_model),
            "checkpoint_for_processor": str(checkpoint),
            "split": args.split,
            "max_samples": args.max_samples,
            "num_samples": len(records),
            "label_list_size": len(label_list),
            "model_size_mb": onnx_total_size_mb(onnx_model),
            "model_main_file_size_mb": file_size_mb(onnx_model),
            "latency": latency_stats(latencies),
            "providers": session.get_providers(),
            "user_data_used": False,
            "source": "CORD-only",
        }
    )
    save_json(out_dir / f"metrics_{args.split}.json", metrics)
    write_jsonl(out_dir / f"predictions_{args.split}.jsonl", predictions)
    save_json(out_dir / f"label_distribution_{args.split}.json", {"gold": label_distribution(true_sequences), "pred": label_distribution(pred_sequences)})
    save_json(out_dir / f"field_metrics_{args.split}.json", metrics["field_metrics"])
    save_json(out_dir / f"latency_{args.split}.json", metrics["latency"])
    save_json(out_dir / f"confusion_top_{args.split}.json", confusion_top(true_sequences, pred_sequences))
    save_json(out_dir / "labels.json", labels_payload)
    print(f"num_samples: {len(records)}")
    print(f"num_tokens: {num_tokens}")
    print(f"seqeval_f1: {metrics['seqeval_f1']:.6f}")
    print(f"avg_latency_ms: {metrics['latency']['avg_ms']}")
    print(f"metrics path: {out_dir / f'metrics_{args.split}.json'}")
    print("ONNX CORD evaluation passed.")


if __name__ == "__main__":
    main()
