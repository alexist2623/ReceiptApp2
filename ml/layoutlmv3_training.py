"""Shared LayoutLMv3 training helpers.

This module keeps the existing training scripts backward compatible while
adding support for per-word ignore labels. Hugging Face processors need a
valid word label id, so ignored words are passed as O and then masked to -100
after tokenization using ``BatchEncoding.word_ids``.
"""

from __future__ import annotations

from typing import Iterable

import torch


IGNORE_LABEL_VALUES = {None, "", "IGNORE", "IGNORED", "__IGNORE__", -100}


def is_ignore_label(label) -> bool:
    if label is None:
        return True
    if isinstance(label, int):
        return label == -100
    return str(label).strip().upper() in IGNORE_LABEL_VALUES


def labels_to_ids_with_ignore(labels: Iterable, label2id: dict[str, int]) -> tuple[list[int], list[int]]:
    """Convert word labels to ids and return word indices that should be ignored."""

    label_ids: list[int] = []
    ignored: list[int] = []
    for idx, label in enumerate(labels):
        if is_ignore_label(label):
            label_ids.append(int(label2id["O"]))
            ignored.append(idx)
            continue
        if label not in label2id:
            raise KeyError(f"Unknown label: {label}")
        label_ids.append(int(label2id[label]))
    return label_ids, ignored


def apply_word_ignores_to_encoding(encoding, ignored_word_indices_by_sample: list[Iterable[int]]):
    """Set token labels to -100 for ignored word indices."""

    if "labels" not in encoding:
        return encoding
    labels = encoding["labels"]
    if not torch.is_tensor(labels):
        labels = torch.as_tensor(labels)
    for batch_index, ignored_words in enumerate(ignored_word_indices_by_sample):
        ignored = {int(idx) for idx in ignored_words or []}
        if not ignored:
            continue
        try:
            word_ids = encoding.word_ids(batch_index=batch_index)
        except Exception as exc:  # pragma: no cover - tokenizer/version specific
            raise RuntimeError("Cannot apply ignored word labels because encoding.word_ids is unavailable.") from exc
        for token_index, word_idx in enumerate(word_ids):
            if word_idx is not None and int(word_idx) in ignored:
                labels[batch_index, token_index] = -100
    encoding["labels"] = labels
    return encoding


def encode_layoutlmv3_with_ignore(
    processor,
    samples: list[dict],
    max_length: int,
):
    """Batch encode samples and apply per-word ignore masks.

    Each sample must contain image, words, normalized_boxes, and label_ids.
    Optionally, samples may contain ignore_word_indices.
    """

    encoding = processor(
        [sample["image"] for sample in samples],
        [sample["words"] for sample in samples],
        boxes=[sample["normalized_boxes"] for sample in samples],
        word_labels=[sample["label_ids"] for sample in samples],
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    if "labels" not in encoding or encoding["labels"].shape != encoding["input_ids"].shape:
        raise RuntimeError("Processor did not produce labels shaped like input_ids.")
    ignored = [sample.get("ignore_word_indices", []) for sample in samples]
    return apply_word_ignores_to_encoding(encoding, ignored)
