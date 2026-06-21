"""Input helpers for angle-aware LayoutLMv3 training and inference."""

from __future__ import annotations

from typing import Any

import torch

from ml.angle_geometry import (
    ANGLE_FEATURE_DIM,
    align_batch_angle_features_to_tokens,
    build_angle_features_for_words,
)
from ml.layoutlmv3_training import apply_word_ignores_to_encoding


def extract_word_payloads_from_label_json(payload: dict[str, Any]) -> list[dict[str, Any]]:
    words = payload.get("words")
    return list(words) if isinstance(words, list) else []


def build_sample_angle_features(sample: dict[str, Any]) -> dict[str, Any]:
    """Attach ``angle_features`` and debug stats to a sample dict."""

    if sample.get("angle_features") is not None:
        return sample
    word_payloads = sample.get("word_payloads") or []
    features, debug = build_angle_features_for_words(
        word_payloads,
        boxes=sample.get("boxes") or [],
        image_width=sample.get("width"),
        image_height=sample.get("height"),
    )
    if not features:
        features = [[0.0] * ANGLE_FEATURE_DIM for _ in sample.get("words", [])]
        debug = [
            {"word_idx": idx, "angle_deg": None, "has_angle": False, "quad_source": None, "feature_dim": ANGLE_FEATURE_DIM}
            for idx in range(len(features))
        ]
    sample["angle_features"] = features
    sample["angle_debug"] = debug
    sample["num_words_with_angle"] = sum(1 for row in debug if row.get("has_angle"))
    sample["num_words_without_angle"] = len(features) - sample["num_words_with_angle"]
    return sample


def encoding_with_angle_features(
    processor,
    samples: list[dict[str, Any]],
    max_length: int,
    *,
    include_labels: bool = True,
    first_subword_only: bool = False,
):
    """Batch encode samples and add token-aligned ``angle_features``.

    ``angle_features`` are not understood by Hugging Face processors, so this
    helper creates normal LayoutLMv3 inputs first and then appends an aligned
    tensor that angle-aware models can consume.
    """

    for sample in samples:
        build_sample_angle_features(sample)
    kwargs = {
        "images": [sample["image"] for sample in samples],
        "text": [sample["words"] for sample in samples],
        "boxes": [sample["normalized_boxes"] for sample in samples],
        "padding": "max_length",
        "truncation": True,
        "max_length": max_length,
        "return_tensors": "pt",
    }
    if include_labels:
        kwargs["word_labels"] = [sample["label_ids"] for sample in samples]
    encoding = processor(**kwargs)
    if include_labels:
        if "labels" not in encoding or encoding["labels"].shape != encoding["input_ids"].shape:
            raise RuntimeError("Processor did not produce labels shaped like input_ids.")
        ignored = [sample.get("ignore_word_indices", []) for sample in samples]
        encoding = apply_word_ignores_to_encoding(encoding, ignored)
    encoding["angle_features"] = align_batch_angle_features_to_tokens(
        encoding,
        [sample.get("angle_features", []) for sample in samples],
        first_subword_only=first_subword_only,
    )
    return encoding


def move_layoutlm_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    model_keys = {
        "input_ids",
        "attention_mask",
        "bbox",
        "pixel_values",
        "token_type_ids",
        "labels",
        "angle_features",
    }
    return {key: value.to(device) for key, value in batch.items() if key in model_keys and hasattr(value, "to")}
