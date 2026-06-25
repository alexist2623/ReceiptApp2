import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import onnxruntime as ort
import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.span_relg.cord_spans import make_gold_spans_from_cord
from ml.span_relg.feature_cache import build_cache_sample
from ml.span_relg.schema import ALL_FIELDS
from ml.receipt_schema import field_for_vocab
from scripts.quantization.quant_common import fail, numpy_inputs_from_encoding, provider_list, save_json


def parse_args():
    parser = argparse.ArgumentParser(description="Build CORD rel-g cache using ONNX last_hidden_state features.")
    parser.add_argument("--raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--onnx_model", required=True)
    parser.add_argument("--checkpoint_for_processor", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--schema_source_dir", default="processed_data/span_relg")
    parser.add_argument("--relg_checkpoint", default="models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best")
    parser.add_argument("--out_dir", default="processed_data/span_relg_cord_onnx_int8_dynamic")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--onnx_provider", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--device", default="cpu", help="Accepted for CLI compatibility; ONNX provider controls execution.")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--include_context_tokens", default="all", choices=("all", "o_only", "none"))
    parser.add_argument("--span_pooling", default="first", choices=("first", "mean"))
    parser.add_argument("--group_key_strategy", default="group", choices=("group", "group_sub", "group_row"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def prepare_out_dir(path, overwrite):
    out_dir = Path(path)
    if out_dir.exists():
        if not overwrite:
            fail(f"{out_dir} already exists. Use --overwrite to rebuild it.")
        print(f"Removing existing output directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_reference_schema(schema_source_dir, relg_checkpoint):
    candidates = [
        Path(schema_source_dir) / "schema.json",
        Path(relg_checkpoint) / "schema.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            schema = json.load(handle)
        field_list = schema.get("field_list") or list(schema.get("field2id", {}).keys())
        field2id = schema.get("field2id") or {field: idx for idx, field in enumerate(field_list)}
        kind2id = schema.get("kind2id") or {"SPAN": 0, "TOKEN": 1}
        if not field_list or not field2id:
            continue
        return {
            "source": str(path),
            "field_list": list(field_list),
            "field2id": {str(key): int(value) for key, value in field2id.items()},
            "kind2id": {str(key): int(value) for key, value in kind2id.items()},
            "schema": schema,
        }
    field2id = {field: idx for idx, field in enumerate(ALL_FIELDS)}
    return {
        "source": "fallback:ml.span_relg.schema.ALL_FIELDS",
        "field_list": list(ALL_FIELDS),
        "field2id": field2id,
        "kind2id": {"SPAN": 0, "TOKEN": 1},
        "schema": {},
    }


def compute_word_hidden_onnx(image, words, normalized_boxes, processor, session, max_length=512):
    encoding = processor(
        image,
        words,
        boxes=normalized_boxes,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    output_names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(output_names, session.run(None, numpy_inputs_from_encoding(encoding))))
    hidden = torch.from_numpy(outputs["last_hidden_state"][0]).float()
    first_token_for_word = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if int(encoding["attention_mask"][0, token_idx].item()) == 0:
            continue
        first_token_for_word.setdefault(int(word_idx), token_idx)
    missing = [idx for idx in range(len(words)) if idx not in first_token_for_word]
    if missing:
        raise ValueError(f"Missing ONNX word hidden states for {len(missing)} words; first missing={missing[:20]}")
    token_indices = [first_token_for_word[idx] for idx in range(len(words))]
    word_hidden = torch.stack([hidden[token_idx] for token_idx in token_indices], dim=0)
    return {
        "word_hidden": word_hidden,
        "word_token_indices": token_indices,
        "encoding_shapes": {key: list(value.shape) for key, value in encoding.items() if hasattr(value, "shape")},
        "onnx_output_shapes": {key: list(value.shape) for key, value in outputs.items()},
    }


def filter_spans_for_field_vocab(sample_info, field2id):
    kept = []
    dropped = []
    for span in sample_info.get("spans", []):
        if field_for_vocab(span.get("field"), field2id) is None:
            dropped.append(
                {
                    "field": span.get("field"),
                    "raw_field": span.get("raw_field"),
                    "text": span.get("text"),
                    "word_indices": span.get("word_indices"),
                }
            )
            continue
        kept.append(span)
    sample_info = dict(sample_info)
    sample_info["spans"] = kept
    return sample_info, dropped


def main():
    args = parse_args()
    raw_data_dir = Path(args.raw_data_dir)
    onnx_model = Path(args.onnx_model)
    checkpoint = Path(args.checkpoint_for_processor)
    if not raw_data_dir.exists():
        fail(f"CORD raw dataset not found: {raw_data_dir}")
    if not onnx_model.exists():
        fail(f"ONNX model not found: {onnx_model}")
    if args.local_files_only and not checkpoint.exists():
        fail(f"checkpoint_for_processor not found: {checkpoint}")
    out_dir = prepare_out_dir(args.out_dir, args.overwrite)
    dataset = load_from_disk(str(raw_data_dir))
    if args.split not in dataset:
        fail(f"split {args.split!r} not found. Available: {list(dataset.keys())}")
    processor = AutoProcessor.from_pretrained(str(checkpoint), apply_ocr=False, local_files_only=args.local_files_only)
    session = ort.InferenceSession(str(onnx_model), providers=provider_list(args.onnx_provider))
    reference_schema = load_reference_schema(args.schema_source_dir, args.relg_checkpoint)
    field2id = reference_schema["field2id"]
    kind2id = reference_schema["kind2id"]
    field_list = reference_schema["field_list"]
    print(f"Reference rel-g schema source: {reference_schema['source']}")
    print(f"Reference rel-g num fields: {len(field_list)}")
    split_dir = out_dir / args.split
    split_dir.mkdir(parents=True, exist_ok=True)
    limit = min(len(dataset[args.split]), args.max_samples) if args.max_samples is not None else len(dataset[args.split])
    records = []
    skipped = []
    counters = Counter()
    field_counts = Counter()
    pair_field_counts = Counter()
    hidden_dim = None
    for index in tqdm(range(limit), desc=f"onnx relg cache {args.split}", unit="sample"):
        sample_id = f"{args.split}_{index:06d}"
        try:
            sample_info = make_gold_spans_from_cord(dataset[args.split][index], group_key_strategy=args.group_key_strategy)
            sample_info, dropped_spans = filter_spans_for_field_vocab(sample_info, field2id)
            counters["dropped_unsupported_spans"] += len(dropped_spans)
            word_features = compute_word_hidden_onnx(
                sample_info["image"],
                sample_info["words"],
                sample_info["normalized_boxes"],
                processor,
                session,
                max_length=args.max_length,
            )
            cache = build_cache_sample(
                sample_id,
                args.split,
                index,
                sample_info,
                word_features["word_hidden"],
                field2id,
                kind2id,
                include_context_tokens=args.include_context_tokens,
                span_pooling=args.span_pooling,
            )
            cache["word_token_indices"] = word_features["word_token_indices"]
            cache["encoding_shapes"] = word_features["encoding_shapes"]
            cache["onnx_output_shapes"] = word_features["onnx_output_shapes"]
            if cache["candidate_pairs"].numel() == 0:
                raise ValueError("No candidate rel-g pairs found.")
            if dropped_spans:
                cache["dropped_unsupported_spans"] = dropped_spans
            sample_path = split_dir / f"{sample_id}.pt"
            torch.save(cache, sample_path)
            records.append({"id": sample_id, "split": args.split, "index": index, "path": str(sample_path)})
            counters["input_samples"] += 1
            counters["written_samples"] += 1
            counters["nodes"] += len(cache["nodes"])
            counters["candidate_pairs"] += int(cache["pair_labels"].numel())
            counters["positive_pairs"] += int(cache["pair_labels"].sum().item())
            counters["negative_pairs"] += int(cache["pair_labels"].numel() - cache["pair_labels"].sum().item())
            for node in cache["nodes"]:
                if node.get("node_kind") == "SPAN":
                    field_counts[node.get("field")] += 1
            pair_field_counts.update(cache.get("pair_fields", []))
            if hidden_dim is None:
                hidden_dim = int(cache["node_hidden"].shape[-1])
        except Exception as exc:
            skipped.append({"id": sample_id, "index": index, "error": repr(exc)})
            counters["input_samples"] += 1
            counters["skipped_samples"] += 1
            print(f"WARNING: skipped {sample_id}: {exc}")
    manifest = {"splits": {args.split: records}, "records": records}
    loaded_schema = dict(reference_schema.get("schema") or {})
    candidate_dep_fields = loaded_schema.get(
        "candidate_dep_fields",
        ["ITEM_PRICE", "ITEM_QTY", "ITEM_UNIT_PRICE", "ITEM_CODE", "ITEM_SKU", "ITEM_DISCOUNT", "ITEM_OPTION", "ITEM_TAX_FLAG", "ITEM_ETC"],
    )
    schema = {
        **loaded_schema,
        "field_list": field_list,
        "field2id": field2id,
        "kind2id": kind2id,
        "hidden_dim": hidden_dim,
        "candidate_head_fields": loaded_schema.get("candidate_head_fields", ["ITEM_NAME"]),
        "candidate_dep_fields": candidate_dep_fields,
        "reference_schema_source": reference_schema["source"],
        "notes": list(loaded_schema.get("notes", []))
        + ["Built from ONNX last_hidden_state features.", "CORD-only data; no user data used."],
    }
    summary = {
        "raw_data_dir": str(raw_data_dir),
        "onnx_model": str(onnx_model),
        "checkpoint_for_processor": str(checkpoint),
        "out_dir": str(out_dir),
        "split": args.split,
        "max_samples": args.max_samples,
        "providers": session.get_providers(),
        "reference_schema_source": reference_schema["source"],
        "num_reference_fields": len(field_list),
        "splits": {args.split: dict(counters)},
        "skipped_samples": skipped,
        "field_counts": dict(field_counts),
        "pair_field_counts": dict(pair_field_counts),
        "num_cached_samples": len(records),
        "hidden_dim": hidden_dim,
        "user_data_used": False,
        "source": "CORD-only",
    }
    save_json(out_dir / "schema.json", schema)
    save_json(out_dir / "manifest.json", manifest)
    save_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:5000])
    print(f"schema path: {out_dir / 'schema.json'}")
    print(f"manifest path: {out_dir / 'manifest.json'}")
    print(f"summary path: {out_dir / 'summary.json'}")
    print("ONNX hidden span rel-g cache build passed.")


if __name__ == "__main__":
    main()
