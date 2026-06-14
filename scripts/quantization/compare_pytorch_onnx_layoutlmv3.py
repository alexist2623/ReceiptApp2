import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
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
    labels_from_encoding,
    load_cord_jsonl,
    load_labels,
    load_raw_dataset,
    numpy_inputs_from_encoding,
    prepare_cord_sample,
    provider_list,
    save_json,
    torch_inputs_from_encoding,
    write_jsonl,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare PyTorch FP32 and ONNX FP32 LayoutLMv3 outputs.")
    parser.add_argument("--pytorch_checkpoint", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--onnx_model", default="models/layoutlmv3-cord-onnx/fp32/model.onnx")
    parser.add_argument("--cord_bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--cord_raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=50)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="cuda", choices=("cpu", "cuda"))
    parser.add_argument("--onnx_provider", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--out_dir", default="outputs/quantization/cord_onnx_fp32")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def select_device(name):
    if name == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cpu")


def update_diff(stats, name, diff):
    diff = np.asarray(diff)
    stats.setdefault(name, {"max": 0.0, "sum": 0.0, "count": 0})
    stats[name]["max"] = max(stats[name]["max"], float(np.max(diff)))
    stats[name]["sum"] += float(np.sum(diff))
    stats[name]["count"] += int(diff.size)


def mean_diff(stats, name):
    item = stats.get(name, {})
    return item.get("sum", 0.0) / item.get("count", 1)


def main():
    args = parse_args()
    checkpoint = Path(args.pytorch_checkpoint)
    onnx_model = Path(args.onnx_model)
    if args.local_files_only and not checkpoint.exists():
        fail(f"PyTorch checkpoint not found: {checkpoint}")
    if not onnx_model.exists():
        fail(f"ONNX model not found: {onnx_model}. Run export step first.")
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
    session = ort.InferenceSession(str(onnx_model), providers=provider_list(args.onnx_provider))
    print(f"PyTorch checkpoint: {checkpoint}")
    print(f"ONNX model: {onnx_model}")
    print(f"ONNX providers: {session.get_providers()}")
    print(f"split: {args.split} max_samples={args.max_samples}")

    torch_true_sequences = []
    torch_pred_sequences = []
    onnx_true_sequences = []
    onnx_pred_sequences = []
    num_gold_tokens = 0
    torch_correct = 0
    onnx_correct = 0
    token_argmax_same = 0
    token_argmax_total = 0
    word_argmax_same = 0
    word_argmax_total = 0
    diff_stats = {}
    per_sample = []
    mismatch_rows = []

    for record in tqdm(records, desc=f"parity {args.split}", unit="sample"):
        sample = prepare_cord_sample(record, raw_dataset, label2id, args.split)
        encoding = encode_sample(processor, sample, args.max_length, include_labels=True)
        model_inputs = torch_inputs_from_encoding(encoding, device)
        with torch.no_grad():
            torch_outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
        onnx_outputs = session.run(None, numpy_inputs_from_encoding(encoding))
        output_names = [output.name for output in session.get_outputs()]
        ort_by_name = dict(zip(output_names, onnx_outputs))
        torch_logits = torch_outputs.logits[0].detach().cpu().numpy()
        torch_hidden = torch_outputs.hidden_states[-1][0].detach().cpu().numpy()
        onnx_logits = ort_by_name["logits"][0]
        onnx_hidden = ort_by_name["last_hidden_state"][0]
        update_diff(diff_stats, "logits", np.abs(onnx_logits - torch_logits))
        update_diff(diff_stats, "hidden", np.abs(onnx_hidden - torch_hidden))
        labels = model_inputs["labels"][0].detach().cpu()
        mask = labels != -100
        gold_raw = labels_from_encoding(labels, id2label)
        torch_pred_raw = []
        onnx_pred_raw = []
        torch_ids = np.argmax(torch_logits, axis=-1)
        onnx_ids = np.argmax(onnx_logits, axis=-1)
        for label_id, torch_id, onnx_id, keep in zip(labels.tolist(), torch_ids.tolist(), onnx_ids.tolist(), mask.tolist()):
            if not keep:
                continue
            gold_label = id2label[int(label_id)]
            torch_label = id2label[int(torch_id)]
            onnx_label = id2label[int(onnx_id)]
            torch_pred_raw.append(torch_label)
            onnx_pred_raw.append(onnx_label)
            torch_correct += int(torch_label == gold_label)
            onnx_correct += int(onnx_label == gold_label)
            token_argmax_same += int(torch_label == onnx_label)
            token_argmax_total += 1
        num_gold_tokens += int(mask.sum().item())
        gold = canonicalize_sequence(gold_raw)
        torch_pred = canonicalize_sequence(torch_pred_raw)
        onnx_pred = canonicalize_sequence(onnx_pred_raw)
        torch_true_sequences.append(gold)
        torch_pred_sequences.append(torch_pred)
        onnx_true_sequences.append(gold)
        onnx_pred_sequences.append(onnx_pred)

        torch_word, _, _, _ = first_token_word_predictions(
            torch.from_numpy(torch_logits),
            encoding,
            id2label,
            len(sample["words"]),
            attention_mask=encoding["attention_mask"][0].tolist(),
        )
        onnx_word, _, _, _ = first_token_word_predictions(
            torch.from_numpy(onnx_logits),
            encoding,
            id2label,
            len(sample["words"]),
            attention_mask=encoding["attention_mask"][0].tolist(),
        )
        for idx, (left, right) in enumerate(zip(torch_word, onnx_word)):
            same = left == right
            word_argmax_same += int(same)
            word_argmax_total += 1
            if not same and len(mismatch_rows) < 200:
                mismatch_rows.append(
                    {
                        "id": sample["id"],
                        "index": sample["index"],
                        "word_idx": idx,
                        "word": sample["words"][idx],
                        "pytorch_label": left,
                        "onnx_label": right,
                        "gold_label": sample["gold_labels"][idx],
                    }
                )
        per_sample.append(
            {
                "id": sample["id"],
                "split": args.split,
                "index": sample["index"],
                "num_words": len(sample["words"]),
                "token_argmax_agreement": token_argmax_same / token_argmax_total if token_argmax_total else 0.0,
                "word_argmax_agreement": word_argmax_same / word_argmax_total if word_argmax_total else 0.0,
                "logits_max_abs_diff": float(np.max(np.abs(onnx_logits - torch_logits))),
                "logits_mean_abs_diff": float(np.mean(np.abs(onnx_logits - torch_logits))),
                "hidden_max_abs_diff": float(np.max(np.abs(onnx_hidden - torch_hidden))),
                "hidden_mean_abs_diff": float(np.mean(np.abs(onnx_hidden - torch_hidden))),
            }
        )

    torch_metrics = aggregate_token_metrics(torch_true_sequences, torch_pred_sequences, num_gold_tokens, torch_correct)
    onnx_metrics = aggregate_token_metrics(onnx_true_sequences, onnx_pred_sequences, num_gold_tokens, onnx_correct)
    f1_drop = torch_metrics["seqeval_f1"] - onnx_metrics["seqeval_f1"]
    report = {
        "pytorch_checkpoint": str(checkpoint),
        "onnx_model": str(onnx_model),
        "split": args.split,
        "max_samples": args.max_samples,
        "num_samples": len(records),
        "num_tokens": num_gold_tokens,
        "pytorch_seqeval_f1": torch_metrics["seqeval_f1"],
        "onnx_seqeval_f1": onnx_metrics["seqeval_f1"],
        "f1_drop": f1_drop,
        "overall_token_argmax_agreement": token_argmax_same / token_argmax_total if token_argmax_total else 0.0,
        "word_level_argmax_agreement": word_argmax_same / word_argmax_total if word_argmax_total else 0.0,
        "logits_max_abs_diff": diff_stats["logits"]["max"],
        "logits_mean_abs_diff": mean_diff(diff_stats, "logits"),
        "hidden_max_abs_diff": diff_stats["hidden"]["max"],
        "hidden_mean_abs_diff": mean_diff(diff_stats, "hidden"),
        "providers": session.get_providers(),
    }
    report["passed"] = (
        report["overall_token_argmax_agreement"] >= 0.99
        and report["word_level_argmax_agreement"] >= 0.99
        and abs(report["f1_drop"]) <= 0.001
    )
    report["pytorch_metrics"] = torch_metrics
    report["onnx_metrics"] = onnx_metrics
    save_json(out_dir / f"parity_report_{args.split}.json", report)
    write_jsonl(out_dir / f"per_sample_diff_{args.split}.jsonl", per_sample)
    write_jsonl(out_dir / f"mismatch_examples_{args.split}.jsonl", mismatch_rows)
    save_json(out_dir / "labels.json", labels_payload)
    save_json(out_dir / f"confusion_top_{args.split}.json", confusion_top(onnx_true_sequences, onnx_pred_sequences))
    print(f"PyTorch F1: {report['pytorch_seqeval_f1']:.6f}")
    print(f"ONNX F1: {report['onnx_seqeval_f1']:.6f}")
    print(f"F1 drop: {report['f1_drop']:.6f}")
    print(f"token agreement: {report['overall_token_argmax_agreement']:.6f}")
    print(f"word agreement: {report['word_level_argmax_agreement']:.6f}")
    print(f"hidden mean abs diff: {report['hidden_mean_abs_diff']:.8f}")
    print(f"passed: {report['passed']}")
    print(f"report path: {out_dir / f'parity_report_{args.split}.json'}")


if __name__ == "__main__":
    main()
