#!/usr/bin/env python
"""Train an item category head on top of frozen LayoutLMv3 hidden states."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.layoutlmv3_angle_inputs import build_sample_angle_features
from ml.span_relg.feature_cache import load_layoutlmv3_for_feature_cache, select_device


DEFAULT_BIO_DIRS = [
    "processed_data/custom_rotated_receipt_v2_bio",
    "processed_data/wildreceipt_custom_structure_rotated_receipt_v2_bio",
    "processed_data/cord_bio",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze LayoutLMv3, extract ITEM_NAME span hidden states, and train "
            "an attention-pooling item category classifier head."
        )
    )
    parser.add_argument("--bio_dirs", nargs="*", default=DEFAULT_BIO_DIRS)
    parser.add_argument("--taxonomy", default="schemas/item_semantic_categories.json")
    parser.add_argument(
        "--layoutlm_checkpoint",
        default="models/layoutlmv3-base",
    )
    parser.add_argument("--cord_raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--output_dir",
        default="models/item-category-layoutlmv3-base-head-valcov25-layers04812-attnstats-reg-ls",
    )
    parser.add_argument(
        "--out_dir",
        default="outputs/item_category_layoutlmv3_base_head_valcov25_layers04812_attnstats_reg_ls",
    )
    parser.add_argument("--feature_cache", default=None)
    parser.add_argument("--epochs", type=int, default=180)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=5e-2)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument(
        "--classifier_layers",
        type=int,
        default=2,
        help="Number of linear layers in the category MLP, including the output layer.",
    )
    parser.add_argument(
        "--span_self_attention_layers",
        type=int,
        default=0,
        help="Optional TransformerEncoder layers over ITEM_NAME span tokens before pooling.",
    )
    parser.add_argument("--span_attention_heads", type=int, default=4)
    parser.add_argument(
        "--hidden_layer_indices",
        default="0,4,8,12",
        help=(
            "Comma-separated LayoutLMv3 hidden-state indices to concatenate "
            "for each token. Use -1 for the last layer, 0 for embeddings, "
            "for example: 0,4,8,12."
        ),
    )
    parser.add_argument(
        "--word_token_mode",
        choices=("first", "all"),
        default="all",
        help=(
            "How to map OCR words to token hidden states. 'first' matches the "
            "older behavior; 'all' keeps every subword token for each ITEM_NAME "
            "word so lexical category cues inside long OCR tokens are preserved."
        ),
    )
    parser.add_argument(
        "--pooling_mode",
        choices=("attention", "attention_stats"),
        default="attention_stats",
        help=(
            "How to pool token hidden states inside an ITEM_NAME span. "
            "'attention' keeps the original learned attention pool. "
            "'attention_stats' concatenates attention, mean, max, and first-token "
            "pools so the head can retain stronger lexical cues from multi-token items."
        ),
    )
    parser.add_argument(
        "--class_weight_power",
        type=float,
        default=0.9,
        help=(
            "Exponent applied to inverse-frequency class weights. 1.0 keeps the "
            "original fully balanced loss; lower values temper rare-class weights."
        ),
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.02,
        help="Optional CrossEntropy label smoothing for the category head.",
    )
    parser.add_argument(
        "--exclude_other",
        action="store_true",
        default=True,
        help="Exclude OTHER category from category-head training. Enabled by default.",
    )
    parser.add_argument(
        "--include_other",
        action="store_false",
        dest="exclude_other",
        help="Include OTHER category in category-head training.",
    )
    parser.add_argument("--max_records_per_split", type=int, default=None)
    parser.add_argument("--max_examples_per_split", type=int, default=None)
    parser.add_argument(
        "--min_validation_per_label",
        type=int,
        default=25,
        help=(
            "If > 0, move held-out examples from train to validation until each "
            "label has at least this many validation examples where possible. "
            "This is useful when a source validation split is missing classes, "
            "which otherwise makes all-label macro F1 mathematically capped."
        ),
    )
    parser.add_argument("--rebuild_cache", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_taxonomy(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["categories"])


def iter_jsonl(path: Path):
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            yield json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc


def path_from_record(value: str | None) -> Path | None:
    if not value:
        return None
    raw = str(value)
    if len(raw) > 2 and raw[1] == ":":
        drive = raw[0].lower()
        rest = raw[2:].replace("\\", "/").lstrip("/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(raw)


def image_to_rgb(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return ImageOps.exif_transpose(value).convert("RGB")
    return ImageOps.exif_transpose(Image.open(value)).convert("RGB")


def load_raw_cord_dataset(raw_data_dir: str):
    from datasets import load_from_disk

    data_dir = Path(raw_data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"CORD raw dataset not found: {raw_data_dir}")
    return load_from_disk(str(data_dir))


def infer_split_from_file(path: Path) -> str:
    split = path.stem
    if split in {"valid", "val"}:
        return "validation"
    if split not in {"train", "validation", "test"}:
        return "train"
    return split


def collect_records(
    bio_dirs: list[Path],
    taxonomy: list[str],
    *,
    exclude_other: bool,
    max_records_per_split: int | None,
) -> dict[str, list[dict[str, Any]]]:
    allowed = set(taxonomy)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    seen_per_split = Counter()
    for bio_dir in bio_dirs:
        if not bio_dir.exists():
            print(f"warning: missing bio_dir skipped: {bio_dir}")
            continue
        for jsonl_path in sorted(bio_dir.glob("*.jsonl")):
            split = infer_split_from_file(jsonl_path)
            for record in iter_jsonl(jsonl_path):
                if max_records_per_split is not None and seen_per_split[split] >= max_records_per_split:
                    continue
                annotations = []
                for annotation in record.get("item_category_annotations", []):
                    category = annotation.get("category")
                    if category not in allowed:
                        continue
                    if exclude_other and category == "OTHER":
                        continue
                    indices = annotation.get("item_name_word_indices")
                    if not isinstance(indices, list) or not indices:
                        continue
                    annotations.append(
                        {
                            "item_name_text": annotation.get("item_name_text", ""),
                            "item_name_word_indices": [int(idx) for idx in indices],
                            "category": category,
                            "confidence": annotation.get("confidence"),
                            "rule": annotation.get("rule"),
                        }
                    )
                if not annotations:
                    continue
                copied = dict(record)
                copied["_category_annotations"] = annotations
                copied["_bio_dir"] = str(bio_dir)
                copied["_jsonl_path"] = str(jsonl_path)
                copied["_split"] = split
                by_split[split].append(copied)
                seen_per_split[split] += 1
    return by_split


def clamp_norm_box(box: Any) -> list[int]:
    values = [int(round(float(v))) for v in box]
    if len(values) != 4:
        raise ValueError(f"normalized box must have 4 values, got {box}")
    x0, y0, x1, y1 = values
    x0 = max(0, min(x0, 1000))
    y0 = max(0, min(y0, 1000))
    x1 = max(0, min(x1, 1000))
    y1 = max(0, min(y1, 1000))
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def normalize_pixel_box(box: Any, width: int, height: int) -> list[int]:
    values = [float(v) for v in box]
    if len(values) != 4:
        return [0, 0, 0, 0]
    x0, y0, x1, y1 = values
    return [
        max(0, min(int(1000 * x0 / max(1, width)), 1000)),
        max(0, min(int(1000 * y0 / max(1, height)), 1000)),
        max(0, min(int(1000 * x1 / max(1, width)), 1000)),
        max(0, min(int(1000 * y1 / max(1, height)), 1000)),
    ]


def parse_hidden_layer_indices(value: str) -> list[int]:
    indices = []
    for part in str(value or "-1").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            indices.append(int(part))
        except ValueError as exc:
            raise ValueError(f"Invalid hidden layer index {part!r} in {value!r}") from exc
    return indices or [-1]


def materialize_record(record: dict[str, Any], raw_cord_dataset=None) -> dict[str, Any]:
    split = record.get("_split") or record.get("split") or "train"
    source = record.get("source") or Path(record.get("_bio_dir", "")).name
    if source == "cord" or str(record.get("id", "")).startswith(("train_", "validation_", "test_")) and not record.get("image"):
        if raw_cord_dataset is None:
            raise RuntimeError("CORD raw dataset is required for CORD records.")
        index = int(record.get("index", str(record.get("id", "0")).split("_")[-1]))
        raw = raw_cord_dataset[split][index]
        image = image_to_rgb(raw["image"])
    else:
        image_path = path_from_record(record.get("image") or record.get("image_path") or record.get("source_image"))
        if image_path is None or not image_path.exists():
            raise FileNotFoundError(f"Image not found for record {record.get('id')}: {image_path}")
        image = image_to_rgb(image_path)

    width, height = image.size
    words = [str(word) for word in record.get("words", [])]
    normalized_boxes = record.get("normalized_boxes") or []
    boxes = record.get("boxes") or []
    if len(normalized_boxes) != len(words):
        if len(boxes) == len(words):
            normalized_boxes = [normalize_pixel_box(box, width, height) for box in boxes]
        else:
            raise ValueError(f"record {record.get('id')} has mismatched words/boxes")
    normalized_boxes = [clamp_norm_box(box) for box in normalized_boxes]
    boxes = boxes if len(boxes) == len(words) else [[0, 0, 0, 0] for _ in words]
    word_payloads = record.get("word_payloads") or [{} for _ in words]
    if len(word_payloads) != len(words):
        word_payloads = [{} for _ in words]
    for idx, payload in enumerate(word_payloads):
        if isinstance(payload, dict):
            payload.setdefault("text", words[idx])
            payload.setdefault("box", boxes[idx] if idx < len(boxes) else None)

    return {
        "id": record.get("id") or record.get("capture_id") or f"{split}_{record.get('index', 0)}",
        "source": source,
        "split": split,
        "image": image,
        "width": width,
        "height": height,
        "words": words,
        "boxes": boxes,
        "normalized_boxes": normalized_boxes,
        "word_payloads": word_payloads,
        "rotation_deg": record.get("rotation_deg"),
        "annotations": record["_category_annotations"],
    }


def encode_word_hidden(
    sample: dict[str, Any],
    processor,
    model,
    device: torch.device,
    *,
    max_length: int,
    uses_angle_features: bool,
    angle_config: dict[str, Any],
    word_token_mode: str,
    hidden_layer_indices: list[int],
) -> tuple[torch.Tensor, dict[int, list[int]], dict[str, Any]]:
    if uses_angle_features:
        build_sample_angle_features(
            sample,
            angle_encoding_mode=angle_config.get("angle_encoding_mode", "angle_quad"),
            angle_feature_dim=int(angle_config.get("angle_feature_dim") or 0) or None,
        )

    encoding = processor(
        sample["image"],
        sample["words"],
        boxes=sample["normalized_boxes"],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    word_ids = encoding.word_ids(batch_index=0)
    model_inputs = {
        key: value.to(device)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"}
    }

    angle_shape = None
    if uses_angle_features:
        from ml.angle_geometry import align_angle_features_to_tokens

        angle_tensor = align_angle_features_to_tokens(
            encoding,
            sample.get("angle_features") or [],
            batch_index=0,
            feature_dim=int(angle_config.get("angle_feature_dim") or 0) or None,
        )
        model_inputs["angle_features"] = angle_tensor.unsqueeze(0).to(device)
        angle_shape = list(model_inputs["angle_features"].shape)

    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
    if getattr(outputs, "hidden_states", None):
        hidden_states = list(outputs.hidden_states)
        selected_layers = []
        resolved_layer_indices = []
        for raw_idx in hidden_layer_indices:
            idx = raw_idx if raw_idx >= 0 else len(hidden_states) + raw_idx
            if idx < 0 or idx >= len(hidden_states):
                raise IndexError(
                    f"hidden layer index {raw_idx} resolved to {idx}, "
                    f"but model returned {len(hidden_states)} hidden states"
                )
            selected_layers.append(hidden_states[idx][0].detach().cpu())
            resolved_layer_indices.append(idx)
        token_hidden = torch.cat(selected_layers, dim=-1)
    else:
        resolved_layer_indices = [-1]
        token_hidden = outputs.last_hidden_state[0].detach().cpu()

    token_indices_for_word: dict[int, list[int]] = defaultdict(list)
    attention_mask = encoding["attention_mask"][0]
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None or int(attention_mask[token_idx].item()) == 0:
            continue
        word_idx = int(word_idx)
        if word_token_mode == "first" and word_idx in token_indices_for_word:
            continue
        token_indices_for_word[word_idx].append(int(token_idx))

    debug = {
        "encoding_shapes": {key: list(value.shape) for key, value in encoding.items() if hasattr(value, "shape")},
        "angle_features_shape": angle_shape,
        "num_tokenized_words": len(token_indices_for_word),
        "num_word_tokens": sum(len(indices) for indices in token_indices_for_word.values()),
        "num_words": len(sample["words"]),
        "hidden_layer_indices": hidden_layer_indices,
        "resolved_hidden_layer_indices": resolved_layer_indices,
        "token_hidden_shape": list(token_hidden.shape),
    }
    return token_hidden, dict(token_indices_for_word), debug


def extract_span_features(
    by_split: dict[str, list[dict[str, Any]]],
    processor,
    model,
    device: torch.device,
    *,
    raw_cord_dataset,
    max_length: int,
    uses_angle_features: bool,
    angle_config: dict[str, Any],
    word_token_mode: str,
    hidden_layer_indices: list[int],
    max_examples_per_split: int | None,
) -> dict[str, list[dict[str, Any]]]:
    features_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    skipped = []
    for split, records in by_split.items():
        print(f"extracting {split}: records={len(records)}")
        pbar = tqdm(records, desc=f"LayoutLMv3 hidden {split}")
        for record in pbar:
            if max_examples_per_split is not None and len(features_by_split[split]) >= max_examples_per_split:
                break
            try:
                sample = materialize_record(record, raw_cord_dataset)
                token_hidden, token_indices_for_word, debug = encode_word_hidden(
                    sample,
                    processor,
                    model,
                    device,
                    max_length=max_length,
                    uses_angle_features=uses_angle_features,
                    angle_config=angle_config,
                    word_token_mode=word_token_mode,
                    hidden_layer_indices=hidden_layer_indices,
                )
            except Exception as exc:
                skipped.append({"id": record.get("id"), "split": split, "reason": str(exc)})
                continue

            for annotation in sample["annotations"]:
                indices = [int(idx) for idx in annotation["item_name_word_indices"]]
                token_indices = [
                    token_idx
                    for idx in indices
                    for token_idx in token_indices_for_word.get(idx, [])
                ]
                if not token_indices:
                    skipped.append(
                        {
                            "id": sample["id"],
                            "split": split,
                            "reason": "span_truncated_or_missing",
                            "indices": indices,
                        }
                    )
                    continue
                hidden = token_hidden[token_indices].contiguous()
                features_by_split[split].append(
                    {
                        "hidden": hidden,
                        "label": annotation["category"],
                        "text": annotation.get("item_name_text", ""),
                        "word_indices": indices,
                        "record_id": sample["id"],
                        "source": sample["source"],
                        "split": split,
                    }
                )
                if max_examples_per_split is not None and len(features_by_split[split]) >= max_examples_per_split:
                    break
    features_by_split["_skipped"] = skipped  # type: ignore[index]
    return features_by_split


class SpanFeatureDataset(Dataset):
    def __init__(self, examples: list[dict[str, Any]], label2id: dict[str, int]):
        self.examples = examples
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        example = self.examples[idx]
        return {
            "hidden": example["hidden"].float(),
            "label_id": self.label2id[example["label"]],
            "label": example["label"],
            "text": example["text"],
            "record_id": example["record_id"],
        }


def collate_span_features(batch: list[dict[str, Any]]) -> dict[str, Any]:
    max_len = max(item["hidden"].shape[0] for item in batch)
    hidden_dim = batch[0]["hidden"].shape[-1]
    hidden = torch.zeros((len(batch), max_len, hidden_dim), dtype=torch.float32)
    mask = torch.zeros((len(batch), max_len), dtype=torch.bool)
    labels = torch.tensor([item["label_id"] for item in batch], dtype=torch.long)
    for row, item in enumerate(batch):
        length = item["hidden"].shape[0]
        hidden[row, :length] = item["hidden"]
        mask[row, :length] = True
    return {
        "hidden": hidden,
        "mask": mask,
        "labels": labels,
        "texts": [item["text"] for item in batch],
        "record_ids": [item["record_id"] for item in batch],
    }


def ensure_validation_label_coverage(
    features_by_split: dict[str, list[dict[str, Any]]],
    labels: list[str],
    *,
    min_validation_per_label: int,
    seed: int,
) -> dict[str, Any]:
    if min_validation_per_label <= 0:
        return {
            "enabled": False,
            "min_validation_per_label": min_validation_per_label,
            "moved_examples": 0,
            "before_validation_counts": dict(Counter(example["label"] for example in features_by_split["validation"])),
            "after_validation_counts": dict(Counter(example["label"] for example in features_by_split["validation"])),
            "remaining_missing_or_underfilled": [],
        }

    rng = random.Random(seed)
    before_counts = Counter(example["label"] for example in features_by_split["validation"])
    train_examples = list(features_by_split["train"])
    validation_examples = list(features_by_split["validation"])
    train_indices_by_label: dict[str, list[int]] = defaultdict(list)
    for idx, example in enumerate(train_examples):
        train_indices_by_label[example["label"]].append(idx)
    for indices in train_indices_by_label.values():
        rng.shuffle(indices)

    moved_indices = set()
    moved_examples = []
    validation_counts = Counter(before_counts)
    for label in labels:
        needed = max(0, min_validation_per_label - validation_counts[label])
        if needed <= 0:
            continue
        candidates = train_indices_by_label.get(label, [])
        for idx in candidates:
            if needed <= 0:
                break
            if idx in moved_indices:
                continue
            moved_indices.add(idx)
            example = dict(train_examples[idx])
            example["_moved_from_train_to_validation"] = True
            moved_examples.append(example)
            validation_counts[label] += 1
            needed -= 1

    if moved_indices:
        features_by_split["train"] = [
            example for idx, example in enumerate(train_examples) if idx not in moved_indices
        ]
        validation_examples.extend(moved_examples)
        rng.shuffle(validation_examples)
        features_by_split["validation"] = validation_examples

    after_counts = Counter(example["label"] for example in features_by_split["validation"])
    remaining = [
        {"label": label, "count": after_counts[label], "target": min_validation_per_label}
        for label in labels
        if after_counts[label] < min_validation_per_label
    ]
    summary = {
        "enabled": True,
        "min_validation_per_label": min_validation_per_label,
        "moved_examples": len(moved_indices),
        "before_validation_counts": dict(before_counts),
        "after_validation_counts": dict(after_counts),
        "remaining_missing_or_underfilled": remaining,
    }
    print("validation label coverage:", summary)
    return summary


class ItemCategoryAttentionHead(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_labels: int,
        mlp_hidden: int = 256,
        dropout: float = 0.2,
        classifier_layers: int = 2,
        span_self_attention_layers: int = 0,
        span_attention_heads: int = 4,
        pooling_mode: str = "attention",
    ):
        super().__init__()
        classifier_layers = max(1, int(classifier_layers))
        span_self_attention_layers = max(0, int(span_self_attention_layers))
        span_attention_heads = max(1, int(span_attention_heads))
        if pooling_mode not in {"attention", "attention_stats"}:
            raise ValueError(f"Unsupported pooling_mode: {pooling_mode}")
        if hidden_size % span_attention_heads != 0:
            print(
                f"warning: hidden_size={hidden_size} is not divisible by "
                f"span_attention_heads={span_attention_heads}; using 1 head."
            )
            span_attention_heads = 1
        self.classifier_layers = classifier_layers
        self.span_self_attention_layers = span_self_attention_layers
        self.pooling_mode = pooling_mode
        if span_self_attention_layers:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=span_attention_heads,
                dim_feedforward=max(hidden_size, mlp_hidden * 2),
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.span_encoder = nn.TransformerEncoder(
                encoder_layer,
                num_layers=span_self_attention_layers,
                norm=nn.LayerNorm(hidden_size),
            )
        else:
            self.span_encoder = None
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden),
            nn.Tanh(),
            nn.Linear(mlp_hidden, 1),
        )
        pooled_size = hidden_size * 4 if pooling_mode == "attention_stats" else hidden_size
        layers: list[nn.Module] = [nn.LayerNorm(pooled_size), nn.Dropout(dropout)]
        if classifier_layers == 1:
            layers.append(nn.Linear(pooled_size, num_labels))
        else:
            in_dim = pooled_size
            for _ in range(classifier_layers - 1):
                layers.extend(
                    [
                        nn.Linear(in_dim, mlp_hidden),
                        nn.GELU(),
                        nn.LayerNorm(mlp_hidden),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = mlp_hidden
            layers.append(nn.Linear(in_dim, num_labels))
        self.classifier = nn.Sequential(*layers)

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.span_encoder is not None:
            hidden = self.span_encoder(hidden, src_key_padding_mask=~mask)
            hidden = hidden.masked_fill(~mask.unsqueeze(-1), 0.0)
        scores = self.attention(hidden).squeeze(-1)
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        attention_pooled = torch.sum(hidden * weights.unsqueeze(-1), dim=1)
        if self.pooling_mode == "attention_stats":
            mask_float = mask.unsqueeze(-1).to(hidden.dtype)
            lengths = mask_float.sum(dim=1).clamp_min(1.0)
            mean_pooled = (hidden * mask_float).sum(dim=1) / lengths
            masked_hidden = hidden.masked_fill(~mask.unsqueeze(-1), torch.finfo(hidden.dtype).min)
            max_pooled = masked_hidden.max(dim=1).values
            first_indices = mask.to(torch.long).argmax(dim=1)
            first_pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), first_indices]
            pooled = torch.cat([attention_pooled, mean_pooled, max_pooled, first_pooled], dim=-1)
        else:
            pooled = attention_pooled
        logits = self.classifier(pooled)
        return {"logits": logits, "attention_weights": weights, "pooled": pooled}


def make_loaders(
    features_by_split: dict[str, list[dict[str, Any]]],
    label2id: dict[str, int],
    batch_size: int,
    seed: int,
) -> dict[str, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return {
        split: DataLoader(
            SpanFeatureDataset(features_by_split[split], label2id),
            batch_size=batch_size,
            shuffle=(split == "train"),
            collate_fn=collate_span_features,
            generator=generator if split == "train" else None,
        )
        for split in ("train", "validation", "test")
    }


def evaluate_head(head: nn.Module, loader: DataLoader, device: torch.device, id2label: dict[int, str], criterion) -> dict[str, Any]:
    head.eval()
    total_loss = 0.0
    total = 0
    gold = []
    pred = []
    with torch.no_grad():
        for batch in loader:
            hidden = batch["hidden"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = head(hidden, mask)
            loss = criterion(outputs["logits"], labels)
            total_loss += float(loss.detach().cpu().item()) * labels.numel()
            total += labels.numel()
            pred_ids = outputs["logits"].argmax(dim=-1).detach().cpu().tolist()
            gold_ids = labels.detach().cpu().tolist()
            gold.extend(id2label[idx] for idx in gold_ids)
            pred.extend(id2label[idx] for idx in pred_ids)
    labels_list = [id2label[idx] for idx in sorted(id2label)]
    return {
        "loss": total_loss / max(1, total),
        "accuracy": float(accuracy_score(gold, pred)) if gold else 0.0,
        "macro_f1": float(f1_score(gold, pred, labels=labels_list, average="macro", zero_division=0)) if gold else 0.0,
        "weighted_f1": float(f1_score(gold, pred, labels=labels_list, average="weighted", zero_division=0)) if gold else 0.0,
        "gold": gold,
        "pred": pred,
    }


def train_head(
    features_by_split: dict[str, list[dict[str, Any]]],
    labels: list[str],
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ItemCategoryAttentionHead, dict[str, Any], list[dict[str, Any]]]:
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    loaders = make_loaders(features_by_split, label2id, args.batch_size, args.seed)
    hidden_size = int(features_by_split["train"][0]["hidden"].shape[-1])
    head = ItemCategoryAttentionHead(
        hidden_size,
        len(labels),
        args.hidden_dim,
        args.dropout,
        args.classifier_layers,
        args.span_self_attention_layers,
        args.span_attention_heads,
        args.pooling_mode,
    ).to(device)

    train_counts = Counter(example["label"] for example in features_by_split["train"])
    weights = torch.tensor(
        [
            (len(features_by_split["train"]) / (len(labels) * max(1, train_counts[label])))
            ** float(args.class_weight_power)
            for label in labels
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=float(args.label_smoothing))
    optimizer = torch.optim.AdamW(head.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    history = []
    best = {"epoch": None, "validation_macro_f1": -1.0, "validation_loss": None}
    best_state = None
    for epoch in range(1, args.epochs + 1):
        head.train()
        total_loss = 0.0
        total = 0
        for batch in loaders["train"]:
            hidden = batch["hidden"].to(device)
            mask = batch["mask"].to(device)
            labels_tensor = batch["labels"].to(device)
            optimizer.zero_grad(set_to_none=True)
            outputs = head(hidden, mask)
            loss = criterion(outputs["logits"], labels_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * labels_tensor.numel()
            total += labels_tensor.numel()

        train_metrics = evaluate_head(head, loaders["train"], device, id2label, criterion)
        val_metrics = evaluate_head(head, loaders["validation"], device, id2label, criterion)
        row = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total),
            "train_eval_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "train_weighted_f1": train_metrics["weighted_f1"],
            "validation_loss": val_metrics["loss"],
            "validation_accuracy": val_metrics["accuracy"],
            "validation_macro_f1": val_metrics["macro_f1"],
            "validation_weighted_f1": val_metrics["weighted_f1"],
        }
        history.append(row)
        is_best = row["validation_macro_f1"] > best["validation_macro_f1"] or (
            row["validation_macro_f1"] == best["validation_macro_f1"]
            and (best["validation_loss"] is None or row["validation_loss"] < best["validation_loss"])
        )
        if is_best:
            best = {
                "epoch": epoch,
                "validation_macro_f1": row["validation_macro_f1"],
                "validation_loss": row["validation_loss"],
            }
            best_state = {key: value.detach().cpu().clone() for key, value in head.state_dict().items()}
        if epoch == 1 or epoch == args.epochs or epoch % max(1, args.epochs // 10) == 0:
            print(
                f"Epoch {epoch:03d}/{args.epochs} "
                f"train_loss={row['train_loss']:.4f} "
                f"val_loss={row['validation_loss']:.4f} "
                f"val_macro_f1={row['validation_macro_f1']:.4f}"
            )
    if best_state is not None:
        head.load_state_dict(best_state)
    test_metrics = evaluate_head(head, loaders["test"], device, id2label, criterion)
    best["test_accuracy"] = test_metrics["accuracy"]
    best["test_macro_f1"] = test_metrics["macro_f1"]
    best["test_weighted_f1"] = test_metrics["weighted_f1"]
    best["test_loss"] = test_metrics["loss"]
    best["test_gold"] = test_metrics["gold"]
    best["test_pred"] = test_metrics["pred"]
    return head, best, history


def plot_history(history: list[dict[str, Any]], out_path: Path) -> str | None:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"warning: matplotlib unavailable, plot skipped: {exc}")
        return None
    epochs = [row["epoch"] for row in history]
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(epochs, [row["train_loss"] for row in history], label="train loss", color="#c0392b")
    ax1.plot(epochs, [row["validation_loss"] for row in history], label="validation loss", color="#e67e22")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.grid(True, alpha=0.25)
    ax2 = ax1.twinx()
    ax2.plot(epochs, [row["train_macro_f1"] for row in history], label="train macro F1", color="#2980b9")
    ax2.plot(epochs, [row["validation_macro_f1"] for row in history], label="validation macro F1", color="#27ae60")
    ax2.set_ylabel("macro F1")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="center right")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return str(out_path)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    hidden_layer_indices = parse_hidden_layer_indices(args.hidden_layer_indices)

    output_dir = Path(args.output_dir)
    out_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cache = Path(args.feature_cache) if args.feature_cache else out_dir / "span_hidden_features.pt"

    taxonomy = load_taxonomy(Path(args.taxonomy))
    labels = [label for label in taxonomy if not (args.exclude_other and label == "OTHER")]
    device = select_device(args.device)
    print(f"device: {device}")
    print(f"layoutlm_checkpoint: {args.layoutlm_checkpoint}")

    if feature_cache.exists() and not args.rebuild_cache:
        print(f"Loading cached features: {feature_cache}")
        cache = torch.load(feature_cache, map_location="cpu", weights_only=False)
        features_by_split = cache["features_by_split"]
        uses_angle_features = cache.get("uses_angle_features")
        angle_config = cache.get("angle_config", {})
    else:
        by_split = collect_records(
            [Path(value) for value in args.bio_dirs],
            taxonomy,
            exclude_other=args.exclude_other,
            max_records_per_split=args.max_records_per_split,
        )
        print("records by split:", {split: len(rows) for split, rows in by_split.items()})
        if not by_split["train"] or not by_split["validation"]:
            raise SystemExit("Need at least train and validation records with item category annotations.")

        raw_cord_dataset = None
        needs_cord = any(
            row.get("source") == "cord" or str(row.get("id", "")).startswith(("train_", "validation_", "test_"))
            for rows in by_split.values()
            for row in rows
        )
        if needs_cord:
            raw_cord_dataset = load_raw_cord_dataset(args.cord_raw_data_dir)

        processor, layout_model, layout_device, uses_angle_features, angle_config = load_layoutlmv3_for_feature_cache(
            args.layoutlm_checkpoint,
            local_files_only=args.local_files_only,
            device=args.device,
        )
        features_by_split = extract_span_features(
            by_split,
            processor,
            layout_model,
            layout_device,
            raw_cord_dataset=raw_cord_dataset,
            max_length=args.max_length,
            uses_angle_features=uses_angle_features,
            angle_config=angle_config,
            word_token_mode=args.word_token_mode,
            hidden_layer_indices=hidden_layer_indices,
            max_examples_per_split=args.max_examples_per_split,
        )
        feature_cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "features_by_split": features_by_split,
                "labels": labels,
                "layoutlm_checkpoint": args.layoutlm_checkpoint,
                "uses_angle_features": uses_angle_features,
                "angle_config": angle_config,
                "word_token_mode": args.word_token_mode,
                "hidden_layer_indices": hidden_layer_indices,
            },
            feature_cache,
        )
        print(f"Saved feature cache: {feature_cache}")

    for split in ("train", "validation", "test"):
        features_by_split[split] = [
            example for example in features_by_split[split]
            if example["label"] in set(labels)
        ]
        if not features_by_split[split]:
            raise SystemExit(f"No {split} span features available.")
    validation_coverage_summary = ensure_validation_label_coverage(
        features_by_split,
        labels,
        min_validation_per_label=args.min_validation_per_label,
        seed=args.seed,
    )
    counts_by_split = {
        split: dict(Counter(example["label"] for example in features_by_split[split]).most_common())
        for split in ("train", "validation", "test")
    }
    print("span feature counts:", {split: len(features_by_split[split]) for split in ("train", "validation", "test")})

    head, best, history = train_head(features_by_split, labels, args, device)

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {str(idx): label for label, idx in label2id.items()}
    torch.save(
        {
            "state_dict": head.state_dict(),
            "hidden_size": int(features_by_split["train"][0]["hidden"].shape[-1]),
            "num_labels": len(labels),
            "labels": labels,
            "label2id": label2id,
            "id2label": id2label,
            "mlp_hidden": args.hidden_dim,
            "dropout": args.dropout,
            "classifier_layers": args.classifier_layers,
            "span_self_attention_layers": args.span_self_attention_layers,
            "span_attention_heads": args.span_attention_heads,
            "word_token_mode": args.word_token_mode,
            "hidden_layer_indices": hidden_layer_indices,
            "pooling_mode": args.pooling_mode,
            "class_weight_power": args.class_weight_power,
            "label_smoothing": args.label_smoothing,
            "min_validation_per_label": args.min_validation_per_label,
            "validation_coverage_summary": validation_coverage_summary,
            "layoutlm_checkpoint": args.layoutlm_checkpoint,
            "pooling": args.pooling_mode,
            "span_encoder": "transformer" if args.span_self_attention_layers else None,
            "layoutlm_frozen": True,
        },
        output_dir / "item_category_head.pt",
    )
    save_json(
        output_dir / "labels.json",
        {
            "label_list": labels,
            "label2id": label2id,
            "id2label": id2label,
            "taxonomy": taxonomy,
        },
    )

    plot_path = plot_history(history, out_dir / "training_curve.png")
    save_json(out_dir / "training_history.json", {"history": history})
    save_json(
        out_dir / "metrics_summary.json",
        {
            "layoutlm_checkpoint": args.layoutlm_checkpoint,
            "layoutlm_frozen": True,
            "uses_angle_features": bool(uses_angle_features),
            "angle_config": angle_config,
            "feature_cache": str(feature_cache),
            "output_dir": str(output_dir),
            "num_labels": len(labels),
            "labels": labels,
            "classifier_layers": args.classifier_layers,
            "span_self_attention_layers": args.span_self_attention_layers,
            "span_attention_heads": args.span_attention_heads,
            "word_token_mode": args.word_token_mode,
            "hidden_layer_indices": hidden_layer_indices,
            "pooling_mode": args.pooling_mode,
            "class_weight_power": args.class_weight_power,
            "label_smoothing": args.label_smoothing,
            "min_validation_per_label": args.min_validation_per_label,
            "validation_coverage_summary": validation_coverage_summary,
            "counts_by_split": counts_by_split,
            "num_train": len(features_by_split["train"]),
            "num_validation": len(features_by_split["validation"]),
            "num_test": len(features_by_split["test"]),
            "best_epoch": best["epoch"],
            "best_validation_macro_f1": best["validation_macro_f1"],
            "best_validation_loss": best["validation_loss"],
            "test_accuracy": best["test_accuracy"],
            "test_macro_f1": best["test_macro_f1"],
            "test_weighted_f1": best["test_weighted_f1"],
            "test_loss": best["test_loss"],
            "plot_path": plot_path,
            "note": "Category labels are keyword_rules_v1 metadata and should be manually audited before production use.",
        },
    )
    save_json(
        out_dir / "test_classification_report.json",
        classification_report(best["test_gold"], best["test_pred"], labels=labels, zero_division=0, output_dict=True),
    )
    (out_dir / "test_classification_report.txt").write_text(
        classification_report(best["test_gold"], best["test_pred"], labels=labels, zero_division=0),
        encoding="utf-8",
    )
    cm = confusion_matrix(best["test_gold"], best["test_pred"], labels=labels)
    save_json(out_dir / "confusion_matrix.json", {"labels": labels, "matrix": cm.tolist()})
    save_json(
        out_dir / "run_config.json",
        {
            "bio_dirs": args.bio_dirs,
            "taxonomy": args.taxonomy,
            "layoutlm_checkpoint": args.layoutlm_checkpoint,
            "cord_raw_data_dir": args.cord_raw_data_dir,
            "local_files_only": args.local_files_only,
            "output_dir": args.output_dir,
            "out_dir": args.out_dir,
            "feature_cache": str(feature_cache),
            "epochs": args.epochs,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "max_length": args.max_length,
            "classifier_layers": args.classifier_layers,
            "span_self_attention_layers": args.span_self_attention_layers,
            "span_attention_heads": args.span_attention_heads,
            "word_token_mode": args.word_token_mode,
            "hidden_layer_indices": hidden_layer_indices,
            "pooling_mode": args.pooling_mode,
            "class_weight_power": args.class_weight_power,
            "label_smoothing": args.label_smoothing,
            "exclude_other": args.exclude_other,
            "min_validation_per_label": args.min_validation_per_label,
        },
    )

    print("LayoutLMv3 item category head training passed.")
    print(json.dumps(json.loads((out_dir / "metrics_summary.json").read_text(encoding="utf-8")), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
