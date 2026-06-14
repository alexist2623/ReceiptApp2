import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.quantization.quant_common import (
    encode_sample,
    fail,
    first_token_word_predictions,
    load_cord_jsonl,
    load_labels,
    load_raw_dataset,
    numpy_inputs_from_encoding,
    prepare_cord_sample,
    provider_list,
    save_json,
)


TOKENIZER_FILE_NAMES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export Android parity fixtures for CORD-only LayoutLMv3 INT8 ONNX.")
    parser.add_argument("--checkpoint_for_processor", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--onnx_model", default="models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx")
    parser.add_argument("--cord_bio_dir", default="processed_data/cord_bio")
    parser.add_argument("--cord_raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--out_dir", default="fixtures/layoutlmv3_cord_int8_android")
    parser.add_argument("--splits", default="validation,test", help="Comma-separated split names.")
    parser.add_argument("--samples_per_split", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--onnx_provider", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tokenizer_hashes(*roots):
    hashes = {}
    for root in roots:
        root = Path(root)
        for name in TOKENIZER_FILE_NAMES:
            path = root / name
            if path.exists() and str(path) not in hashes:
                hashes[str(path)] = sha256_file(path)
    return hashes


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def prepare_fixture_dir(path, overwrite):
    path = Path(path)
    if path.exists() and overwrite:
        for child in path.iterdir():
            if child.is_dir():
                import shutil

                shutil.rmtree(child)
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)
    return path


def session_io(session):
    return {
        "inputs": [
            {"name": item.name, "shape": [str(dim) for dim in item.shape], "type": item.type}
            for item in session.get_inputs()
        ],
        "outputs": [
            {"name": item.name, "shape": [str(dim) for dim in item.shape], "type": item.type}
            for item in session.get_outputs()
        ],
    }


def active_token_predictions(logits, attention_mask, id2label):
    pred_ids = np.argmax(logits, axis=-1).astype(np.int64)
    rows = []
    for token_idx, (pred_id, keep) in enumerate(zip(pred_ids.tolist(), attention_mask.tolist())):
        if int(keep) == 0:
            continue
        rows.append({"token_idx": token_idx, "label_id": int(pred_id), "label": id2label[int(pred_id)]})
    return rows


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint_for_processor)
    onnx_model = Path(args.onnx_model)
    if args.local_files_only and not checkpoint.exists():
        fail(f"checkpoint_for_processor not found: {checkpoint}")
    if not onnx_model.exists():
        fail(f"ONNX model not found: {onnx_model}. Run dynamic INT8 quantization first.")
    out_dir = prepare_fixture_dir(args.out_dir, args.overwrite)
    labels_payload, label_list, label2id, id2label = load_labels(checkpoint)
    raw_dataset = load_raw_dataset(args.cord_raw_data_dir)
    processor = AutoProcessor.from_pretrained(str(checkpoint), apply_ocr=False, local_files_only=args.local_files_only)
    session = ort.InferenceSession(str(onnx_model), providers=provider_list(args.onnx_provider))
    output_names = [output.name for output in session.get_outputs()]
    model_hash = sha256_file(onnx_model)
    token_hashes = tokenizer_hashes(checkpoint, onnx_model.parent)
    split_names = [part.strip() for part in args.splits.split(",") if part.strip()]
    fixture_rows = []
    for split in split_names:
        records = load_cord_jsonl(args.cord_bio_dir, split, args.samples_per_split)
        for record in records:
            sample = prepare_cord_sample(record, raw_dataset, label2id, split)
            fixture_dir = out_dir / sample["id"]
            if fixture_dir.exists() and not args.overwrite:
                fail(f"Fixture already exists: {fixture_dir}. Use --overwrite to replace.")
            fixture_dir.mkdir(parents=True, exist_ok=True)
            encoding = encode_sample(processor, sample, args.max_length, include_labels=False)
            inputs = numpy_inputs_from_encoding(encoding)
            outputs = dict(zip(output_names, session.run(None, inputs)))
            logits = outputs["logits"][0]
            last_hidden_state = outputs["last_hidden_state"][0]
            word_labels, word_confidences, _, _ = first_token_word_predictions(
                torch.from_numpy(logits),
                encoding,
                id2label,
                len(sample["words"]),
                attention_mask=encoding["attention_mask"][0].tolist(),
            )
            word_ids = encoding.word_ids(batch_index=0)
            first_token_indices = {}
            for token_idx, word_idx in enumerate(word_ids):
                if word_idx is None:
                    continue
                if int(encoding["attention_mask"][0, token_idx].item()) == 0:
                    continue
                first_token_indices.setdefault(int(word_idx), int(token_idx))
            expected_rows = []
            for word_idx, (word, label, confidence) in enumerate(zip(sample["words"], word_labels, word_confidences)):
                token_idx = first_token_indices.get(word_idx)
                label_id = label2id.get(label)
                expected_rows.append(
                    {
                        "word_idx": word_idx,
                        "word": word,
                        "first_token_idx": token_idx,
                        "label": label,
                        "label_id": label_id,
                        "confidence": confidence,
                    }
                )
            np.save(fixture_dir / "input_ids.npy", inputs["input_ids"])
            np.save(fixture_dir / "attention_mask.npy", inputs["attention_mask"])
            np.save(fixture_dir / "bbox.npy", inputs["bbox"])
            np.save(fixture_dir / "pixel_values.npy", inputs["pixel_values"])
            np.save(fixture_dir / "logits.npy", outputs["logits"])
            np.save(fixture_dir / "last_hidden_state.npy", outputs["last_hidden_state"])
            write_json(fixture_dir / "expected_word_labels.json", expected_rows)
            write_json(fixture_dir / "words.json", sample["words"])
            write_json(fixture_dir / "boxes.json", record.get("boxes", []))
            write_json(fixture_dir / "normalized_boxes.json", sample["normalized_boxes"])
            metadata = {
                "sample_id": sample["id"],
                "split": sample["split"],
                "index": sample["index"],
                "model_path": str(onnx_model),
                "model_sha256": model_hash,
                "tokenizer_files_sha256": token_hashes,
                "image_size": {"width": sample["image_size"][0], "height": sample["image_size"][1]},
                "max_length": args.max_length,
                "label_list": label_list,
                "expected_top_labels": expected_rows[:30],
                "active_token_predictions": active_token_predictions(logits, inputs["attention_mask"][0], id2label),
                "onnx_runtime_provider": session.get_providers(),
                "onnx_io": session_io(session),
                "tensor_files": {
                    "input_ids": "input_ids.npy",
                    "attention_mask": "attention_mask.npy",
                    "bbox": "bbox.npy",
                    "pixel_values": "pixel_values.npy",
                    "logits": "logits.npy",
                    "last_hidden_state": "last_hidden_state.npy",
                },
                "json_files": {
                    "words": "words.json",
                    "boxes": "boxes.json",
                    "normalized_boxes": "normalized_boxes.json",
                    "expected_word_labels": "expected_word_labels.json",
                },
            }
            write_json(fixture_dir / "metadata.json", metadata)
            fixture_rows.append(
                {
                    "sample_id": sample["id"],
                    "split": split,
                    "index": sample["index"],
                    "path": str(fixture_dir),
                    "num_words": len(sample["words"]),
                    "image_size": sample["image_size"],
                }
            )
            print(f"Wrote fixture: {fixture_dir}")
    manifest = {
        "fixture_count": len(fixture_rows),
        "model_path": str(onnx_model),
        "model_sha256": model_hash,
        "checkpoint_for_processor": str(checkpoint),
        "tokenizer_files_sha256": token_hashes,
        "splits": split_names,
        "samples_per_split": args.samples_per_split,
        "max_length": args.max_length,
        "label_list": label_list,
        "onnx_runtime_provider": session.get_providers(),
        "fixtures": fixture_rows,
    }
    write_json(out_dir / "manifest.json", manifest)
    save_json(out_dir / "labels.json", labels_payload)
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:5000])
    print(f"manifest path: {out_dir / 'manifest.json'}")
    print("Android LayoutLMv3 parity fixture export passed.")


if __name__ == "__main__":
    main()
