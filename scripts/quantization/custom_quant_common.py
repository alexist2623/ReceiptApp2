import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.receipt_schema import canonicalize_label
from scripts.build_user_span_relg_dataset import collect_label_pairs, make_sample_info
from scripts.quantization.quant_common import fail, numpy_inputs_from_encoding, provider_list


def load_custom_records(input_dir, exclude_dir_name="Temp", split_manifest=None, split="all", max_samples=None):
    records, excluded = collect_label_pairs(input_dir, exclude_dir_name)
    if split_manifest:
        allowed = _load_manifest_ids(split_manifest, split)
        records = [record for record in records if record["id"] in allowed]
    records = records[:max_samples] if max_samples is not None else records
    if not records:
        fail("No custom labeled records found.")
    return records, excluded


def _load_manifest_ids(path, split):
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    splits = payload.get("splits", payload) if isinstance(payload, dict) else {}
    if split == "all":
        out = set()
        if isinstance(splits, dict):
            for values in splits.values():
                if isinstance(values, dict) and "records" in values:
                    values = values["records"]
                if isinstance(values, dict):
                    values = values.values()
                for item in values if isinstance(values, list) else []:
                    out.add(str(item.get("id") if isinstance(item, dict) else item))
        return out
    values = splits.get(split) if isinstance(splits, dict) else []
    if isinstance(values, dict) and "records" in values:
        values = values["records"]
    if isinstance(values, dict):
        values = values.values()
    return {str(item.get("id") if isinstance(item, dict) else item) for item in values}


def prepare_custom_sample(record, label2id, repair_labels=False):
    sample_info = make_sample_info(record, repair_labels=repair_labels)
    words = []
    boxes = []
    normalized_boxes = []
    gold_labels = []
    label_ids = []
    for prediction in sample_info["predictions"]:
        label = canonicalize_label(prediction.get("label", "O"))
        if label not in label2id:
            fail(f"Label {label!r} from {record['label_json']} is not in checkpoint label map.")
        words.append(prediction["text"])
        boxes.append(prediction["box"])
        normalized_boxes.append(prediction["normalized_box"])
        gold_labels.append(label)
        label_ids.append(label2id[label])
    if not words:
        fail(f"No valid OCR words in {record['label_json']}")
    return {
        "id": record["id"],
        "source": "custom",
        "split": "custom",
        "index": None,
        "image": sample_info["image"],
        "image_size": [sample_info["width"], sample_info["height"]],
        "width": sample_info["width"],
        "height": sample_info["height"],
        "words": words,
        "boxes": boxes,
        "normalized_boxes": normalized_boxes,
        "gold_labels": gold_labels,
        "canonical_gold_labels": list(gold_labels),
        "label_ids": label_ids,
        "record": record,
        "sample_info": sample_info,
    }


def encode_custom_sample(processor, sample, max_length=512, include_labels=True):
    kwargs = {
        "images": sample["image"],
        "text": sample["words"],
        "boxes": sample["normalized_boxes"],
        "padding": "max_length",
        "truncation": True,
        "max_length": max_length,
        "return_tensors": "pt",
    }
    if include_labels:
        kwargs["word_labels"] = sample["label_ids"]
    return processor(**kwargs)


def make_onnx_session(onnx_model, provider="cpu"):
    providers = provider_list(provider)
    try:
        return ort.InferenceSession(str(onnx_model), providers=providers)
    except Exception:
        if provider == "cuda":
            print("WARNING: CUDAExecutionProvider failed; falling back to CPUExecutionProvider.")
            return ort.InferenceSession(str(onnx_model), providers=["CPUExecutionProvider"])
        raise


def run_onnx_layout_prediction(image, words, boxes, processor, session, id2label, max_length=512):
    encoding = processor(
        image,
        words,
        boxes=boxes,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    outputs = session.run(None, numpy_inputs_from_encoding(encoding))
    output_names = [output.name for output in session.get_outputs()]
    by_name = dict(zip(output_names, outputs))
    logits = torch.from_numpy(np.asarray(by_name["logits"][0]))
    hidden = torch.from_numpy(np.asarray(by_name["last_hidden_state"][0])).float()
    probs = torch.softmax(logits, dim=-1)
    pred_ids = logits.argmax(dim=-1)
    attention_mask = encoding["attention_mask"][0].detach().cpu()
    token_strings = processor.tokenizer.convert_ids_to_tokens(encoding["input_ids"][0].tolist())

    first_token_for_word = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or int(attention_mask[token_idx].item()) == 0:
            continue
        first_token_for_word.setdefault(int(word_idx), token_idx)
    missing = [idx for idx in range(len(words)) if idx not in first_token_for_word]
    if missing:
        raise ValueError(f"Missing token alignment for {len(missing)} words; first missing={missing[:20]}")

    predictions = []
    word_hidden = []
    token_debug = []
    for word_idx, word in enumerate(words):
        token_idx = first_token_for_word[word_idx]
        pred_id = int(pred_ids[token_idx].item())
        predictions.append(
            {
                "word_idx": word_idx,
                "text": word,
                "label": id2label[pred_id],
                "confidence": float(probs[token_idx, pred_id].item()),
            }
        )
        word_hidden.append(hidden[token_idx])
    for token_idx, word_idx in enumerate(word_ids[:120]):
        pred_id = int(pred_ids[token_idx].item())
        token_debug.append(
            {
                "token_idx": token_idx,
                "token": token_strings[token_idx],
                "word_idx": word_idx,
                "word": None if word_idx is None or word_idx >= len(words) else words[word_idx],
                "pred_label": id2label[pred_id],
                "confidence": float(probs[token_idx, pred_id].item()),
            }
        )
    return {
        "encoding": encoding,
        "logits": logits,
        "last_hidden_state": hidden,
        "predictions": predictions,
        "word_hidden": torch.stack(word_hidden, dim=0),
        "word_token_indices": [first_token_for_word[idx] for idx in range(len(words))],
        "encoding_shapes": {key: list(value.shape) for key, value in encoding.items() if hasattr(value, "shape")},
        "token_debug": token_debug,
        "providers": session.get_providers(),
    }
