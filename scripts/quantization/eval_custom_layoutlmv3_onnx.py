import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.quantization.custom_quant_common import (
    encode_custom_sample,
    load_custom_records,
    make_onnx_session,
    prepare_custom_sample,
)
from scripts.quantization.quant_common import (
    aggregate_token_metrics,
    canonicalize_sequence,
    confusion_top,
    fail,
    file_size_mb,
    first_token_word_predictions,
    label_distribution,
    labels_from_encoding,
    latency_stats,
    load_labels,
    numpy_inputs_from_encoding,
    onnx_total_size_mb,
    save_json,
    timed_call,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate custom receipt LayoutLMv3 ONNX model.")
    parser.add_argument("--onnx_model", required=True)
    parser.add_argument("--checkpoint_for_processor", default="models/layoutlmv3-item-policy-mixed-100/best")
    parser.add_argument("--input_dir", required=True, nargs="+")
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--split_manifest", default=None)
    parser.add_argument("--split", default="all")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--onnx_provider", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--latency_repeat", type=int, default=1)
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--out_dir", default="outputs/quantization/custom_onnx_eval")
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
    records, excluded = load_custom_records(args.input_dir, args.exclude_dir_name, args.split_manifest, args.split, args.max_samples)
    processor = AutoProcessor.from_pretrained(str(checkpoint), apply_ocr=False, local_files_only=args.local_files_only)
    session = make_onnx_session(onnx_model, args.onnx_provider)
    output_names = [output.name for output in session.get_outputs()]
    print(f"onnx_model: {onnx_model}")
    print(f"providers: {session.get_providers()}")
    print(f"records: {len(records)} excluded_temp={len(excluded)}")

    true_sequences = []
    pred_sequences = []
    word_true_sequences = []
    word_pred_sequences = []
    predictions = []
    num_tokens = 0
    correct_tokens = 0
    word_total = 0
    word_correct = 0
    latencies = []
    failures = []
    for record in tqdm(records, desc="custom onnx eval", unit="sample"):
        try:
            sample = prepare_custom_sample(record, label2id)
            encoding = encode_custom_sample(processor, sample, args.max_length, include_labels=True)
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
            word_gold = canonicalize_sequence(sample["gold_labels"])
            word_pred = canonicalize_sequence(word_pred_raw)
            word_true_sequences.append(word_gold)
            word_pred_sequences.append(word_pred)
            for gold_label, pred_label in zip(word_gold, word_pred):
                word_total += 1
                word_correct += int(gold_label == pred_label)
            predictions.append(
                {
                    "id": sample["id"],
                    "source": "custom",
                    "words": sample["words"],
                    "gold_labels": sample["gold_labels"],
                    "pred_labels": word_pred_raw,
                    "canonical_gold_labels": word_gold,
                    "canonical_pred_labels": word_pred,
                    "confidences": conf,
                    "image_size": sample["image_size"],
                    "label_json": record["label_json"],
                    "image": record["image"],
                }
            )
        except Exception as exc:
            failures.append({"id": record.get("id"), "error": repr(exc)})
            print(f"WARNING: skipped {record.get('id')}: {exc}")
    if not true_sequences:
        fail("No samples were evaluated successfully.")
    metrics = aggregate_token_metrics(true_sequences, pred_sequences, num_tokens, correct_tokens)
    word_metrics = aggregate_token_metrics(word_true_sequences, word_pred_sequences, word_total, word_correct)
    metrics.update(
        {
            "onnx_model": str(onnx_model),
            "checkpoint_for_processor": str(checkpoint),
            "split": args.split,
            "max_samples": args.max_samples,
            "num_samples": len(predictions),
            "failed_samples": len(failures),
            "excluded_temp_count": len(excluded),
            "label_list_size": len(label_list),
            "model_size_mb": onnx_total_size_mb(onnx_model),
            "model_main_file_size_mb": file_size_mb(onnx_model),
            "latency": latency_stats(latencies),
            "providers": session.get_providers(),
            "user_data_used": True,
            "source": "custom-labeled-receipts",
            "word_level": word_metrics,
            "word_label_accuracy": word_correct / word_total if word_total else 0.0,
        }
    )
    save_json(out_dir / "metrics_all.json", metrics)
    save_json(out_dir / "metrics_summary.json", metrics)
    write_jsonl(out_dir / "predictions_all.jsonl", predictions)
    save_json(out_dir / "label_distribution_all.json", {"gold": label_distribution(true_sequences), "pred": label_distribution(pred_sequences)})
    save_json(out_dir / "word_label_distribution_all.json", {"gold": label_distribution(word_true_sequences), "pred": label_distribution(word_pred_sequences)})
    save_json(out_dir / "field_metrics_all.json", metrics["field_metrics"])
    save_json(out_dir / "latency_all.json", metrics["latency"])
    save_json(out_dir / "confusion_top_all.json", confusion_top(true_sequences, pred_sequences))
    save_json(out_dir / "failures.json", failures)
    save_json(out_dir / "labels.json", labels_payload)
    print(f"num_samples: {len(predictions)}")
    print(f"num_tokens: {num_tokens}")
    print(f"seqeval_f1: {metrics['seqeval_f1']:.6f}")
    print(f"word_label_accuracy: {metrics['word_label_accuracy']:.6f}")
    print(f"avg_latency_ms: {metrics['latency']['avg_ms']}")
    print(f"metrics path: {out_dir / 'metrics_all.json'}")
    print("Custom ONNX evaluation passed.")


if __name__ == "__main__":
    main()
