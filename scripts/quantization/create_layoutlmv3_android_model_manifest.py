import argparse
import hashlib
import json
import sys
from pathlib import Path

import onnxruntime as ort

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from scripts.quantization.quant_common import fail, load_labels, onnx_total_size_mb, save_json


TOKENIZER_FILE_NAMES = [
    "tokenizer.json",
    "tokenizer_config.json",
    "processor_config.json",
    "vocab.json",
    "merges.txt",
    "special_tokens_map.json",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Create Android model artifact manifest for LayoutLMv3 INT8 ONNX.")
    parser.add_argument("--model", default="models/layoutlmv3-cord-onnx/int8_dynamic/model.onnx")
    parser.add_argument("--checkpoint_for_processor", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--labels", default=None, help="Defaults to <model_dir>/labels.json, then checkpoint labels.json.")
    parser.add_argument("--out", default="artifacts/layoutlmv3_cord_int8_dynamic_manifest.json")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--image_size", type=int, default=224)
    return parser.parse_args()


def sha256_file(path):
    path = Path(path)
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_tokenizer_files(*roots):
    rows = []
    seen = set()
    for root in roots:
        root = Path(root)
        for name in TOKENIZER_FILE_NAMES:
            path = root / name
            key = str(path.resolve()) if path.exists() else str(path)
            if not path.exists() or key in seen:
                continue
            seen.add(key)
            rows.append({"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size})
    return rows


def io_rows(items):
    return [{"name": item.name, "shape": [str(dim) for dim in item.shape], "type": item.type} for item in items]


def labels_path(model_dir, checkpoint, explicit):
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend([Path(model_dir) / "labels.json", Path(checkpoint) / "labels.json"])
    for path in candidates:
        if path.exists():
            return path
    fail(f"labels.json not found. Checked: {', '.join(str(path) for path in candidates)}")


def main():
    args = parse_args()
    model = Path(args.model)
    checkpoint = Path(args.checkpoint_for_processor)
    if not model.exists():
        fail(f"model not found: {model}")
    if not checkpoint.exists():
        fail(f"checkpoint_for_processor not found: {checkpoint}")
    label_path = labels_path(model.parent, checkpoint, args.labels)
    labels_payload, label_list, _, _ = load_labels(label_path.parent)
    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    manifest = {
        "model_variant": "layoutlmv3-cord-int8-dynamic",
        "model_path": str(model),
        "model_sha256": sha256_file(model),
        "model_size_mb": onnx_total_size_mb(model),
        "labels_json_path": str(label_path),
        "labels_json_sha256": sha256_file(label_path),
        "label_count": len(label_list),
        "label_list": label_list,
        "tokenizer_files": collect_tokenizer_files(model.parent, checkpoint),
        "expected_inputs": io_rows(session.get_inputs()),
        "expected_outputs": io_rows(session.get_outputs()),
        "android_input_contract": {
            "input_ids": {"shape": [1, args.max_length], "dtype": "int64"},
            "attention_mask": {"shape": [1, args.max_length], "dtype": "int64"},
            "bbox": {"shape": [1, args.max_length, 4], "dtype": "int64", "coordinate_policy": "0..1000"},
            "pixel_values": {"shape": [1, 3, args.image_size, args.image_size], "dtype": "float32"},
        },
        "android_output_contract": {
            "logits": {"shape": [1, args.max_length, len(label_list)], "dtype": "float32"},
            "last_hidden_state": {"shape": ["1", "text+visual_tokens", 768], "dtype": "float32"},
        },
        "max_length": args.max_length,
        "image_input_size": args.image_size,
        "apply_ocr": False,
        "coordinate_policy": "OCR pixel boxes must be converted to LayoutLMv3 0..1000 token bboxes.",
        "runtime_gate_order": [
            "Phase A: precomputed tensor ONNX smoke",
            "Phase B: Android preprocessing parity",
            "Phase C: OCR to label overlay MVP",
            "Phase D: rel-g integration later",
        ],
        "labels_payload_keys": sorted(labels_payload.keys()),
    }
    save_json(args.out, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2)[:5000])
    print(f"manifest path: {args.out}")
    print("Android model artifact manifest creation passed.")


if __name__ == "__main__":
    main()
