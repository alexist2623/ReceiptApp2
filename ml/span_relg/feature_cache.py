import torch
from transformers import AutoModelForTokenClassification, AutoProcessor

from .geometry import box1000_to_unit, pair_geometry_dim, pair_geometry_features
from .schema import CONTEXT_FIELD, is_candidate_dep_field, is_dependent_field, is_head_field
from .span_utils import span_pool_hidden


def select_device(device="auto"):
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_layoutlmv3(checkpoint, local_files_only=False, device="auto"):
    device_obj = select_device(device)
    processor = AutoProcessor.from_pretrained(
        checkpoint,
        apply_ocr=False,
        local_files_only=local_files_only,
    )
    model = AutoModelForTokenClassification.from_pretrained(
        checkpoint,
        local_files_only=local_files_only,
    )
    model.to(device_obj)
    model.eval()
    model.requires_grad_(False)
    return processor, model, device_obj


def compute_word_hidden(image, words, normalized_boxes, processor, model, device, max_length=512):
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
    model_inputs = {
        key: value.to(device)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids"}
    }
    with torch.no_grad():
        outputs = model(**model_inputs, output_hidden_states=True, return_dict=True)
        if getattr(outputs, "hidden_states", None):
            hidden = outputs.hidden_states[-1][0].detach().cpu()
        else:
            hidden = outputs.last_hidden_state[0].detach().cpu()

    first_token_for_word = {}
    for token_idx, word_idx in enumerate(word_ids):
        if word_idx is None:
            continue
        if encoding["attention_mask"][0, token_idx].item() == 0:
            continue
        first_token_for_word.setdefault(int(word_idx), token_idx)

    missing = [idx for idx in range(len(words)) if idx not in first_token_for_word]
    if missing:
        raise ValueError(f"Missing word hidden states for {len(missing)} words; first missing={missing[:20]}")

    token_indices = [first_token_for_word[idx] for idx in range(len(words))]
    word_hidden = torch.stack([hidden[token_idx] for token_idx in token_indices], dim=0)
    return {
        "word_hidden": word_hidden,
        "word_token_indices": token_indices,
        "encoding_shapes": {key: list(value.shape) for key, value in encoding.items() if hasattr(value, "shape")},
    }


def make_nodes(sample_info, word_hidden, include_context_tokens="all", span_pooling="first"):
    nodes = []
    covered_words = set()
    for span in sample_info["spans"]:
        hidden = span_pool_hidden(word_hidden, span["word_indices"], span_pooling)
        node = {
            "node_id": len(nodes),
            "node_kind": "SPAN",
            "field": span["field"],
            "text": span["text"],
            "word_indices": list(span["word_indices"]),
            "first_word_idx": int(span["first_word_idx"]),
            "box": span["box"],
            "box_unit": box1000_to_unit(span["normalized_box"]),
            "hidden": hidden,
            "group_key": span.get("group_key"),
            "span_id": span.get("span_id"),
            "confidence": span.get("confidence", 1.0),
        }
        nodes.append(node)
        covered_words.update(span["word_indices"])

    if include_context_tokens != "none":
        for word_idx, word in enumerate(sample_info["words"]):
            if include_context_tokens == "o_only" and word_idx in covered_words:
                continue
            nodes.append(
                {
                    "node_id": len(nodes),
                    "node_kind": "TOKEN",
                    "field": CONTEXT_FIELD,
                    "text": word,
                    "word_indices": [word_idx],
                    "first_word_idx": word_idx,
                    "box": sample_info["boxes"][word_idx],
                    "box_unit": box1000_to_unit(sample_info["normalized_boxes"][word_idx]),
                    "hidden": word_hidden[word_idx],
                    "group_key": None,
                    "span_id": None,
                    "confidence": 1.0,
                }
            )
    return nodes


def make_candidate_pairs(nodes):
    pairs = []
    labels = []
    pair_fields = []
    pair_meta = []
    pair_geom = []
    head_nodes = [node for node in nodes if node["node_kind"] == "SPAN" and is_head_field(node["field"])]
    dep_nodes = [node for node in nodes if node["node_kind"] == "SPAN" and is_candidate_dep_field(node["field"])]
    for head in head_nodes:
        for dep in dep_nodes:
            if head["node_id"] == dep["node_id"]:
                continue
            y = (
                1
                if head.get("group_key") is not None
                and head.get("group_key") == dep.get("group_key")
                and is_dependent_field(dep["field"])
                else 0
            )
            pairs.append([head["node_id"], dep["node_id"]])
            labels.append(y)
            pair_fields.append(dep["field"])
            pair_geom.append(pair_geometry_features(head["box_unit"], dep["box_unit"]))
            pair_meta.append(
                {
                    "head_node_id": head["node_id"],
                    "dep_node_id": dep["node_id"],
                    "head_span_id": head.get("span_id"),
                    "dep_span_id": dep.get("span_id"),
                    "head_field": head["field"],
                    "dep_field": dep["field"],
                    "head_text": head["text"],
                    "dep_text": dep["text"],
                    "label": y,
                    "head_group_key": head.get("group_key"),
                    "dep_group_key": dep.get("group_key"),
                }
            )
    return pairs, labels, pair_fields, pair_meta, pair_geom


def build_cache_sample(
    data_id,
    split,
    index,
    sample_info,
    word_hidden,
    field2id,
    kind2id,
    include_context_tokens="all",
    span_pooling="first",
):
    nodes = make_nodes(sample_info, word_hidden, include_context_tokens, span_pooling)
    pairs, labels, pair_fields, pair_meta, pair_geom = make_candidate_pairs(nodes)
    hidden_dim = int(word_hidden.shape[-1]) if hasattr(word_hidden, "shape") and word_hidden.numel() else 0
    node_hidden = torch.stack([node["hidden"] for node in nodes], dim=0) if nodes else torch.empty((0, hidden_dim))
    return {
        "data_id": data_id,
        "split": split,
        "index": index,
        "image_size": {"width": sample_info["width"], "height": sample_info["height"]},
        "num_words": len(sample_info["words"]),
        "nodes": [
            {key: value for key, value in node.items() if key != "hidden"}
            for node in nodes
        ],
        "node_hidden": node_hidden,
        "node_field_ids": torch.tensor([field2id[node["field"]] for node in nodes], dtype=torch.long),
        "node_kind_ids": torch.tensor([kind2id[node["node_kind"]] for node in nodes], dtype=torch.long),
        "node_boxes": (
            torch.tensor([node["box_unit"] for node in nodes], dtype=torch.float32)
            if nodes
            else torch.empty((0, 4), dtype=torch.float32)
        ),
        "candidate_pairs": torch.tensor(pairs, dtype=torch.long) if pairs else torch.empty((0, 2), dtype=torch.long),
        "pair_labels": torch.tensor(labels, dtype=torch.float32) if labels else torch.empty((0,), dtype=torch.float32),
        "pair_fields": pair_fields,
        "pair_geom": (
            torch.tensor(pair_geom, dtype=torch.float32)
            if pair_geom
            else torch.empty((0, pair_geometry_dim()), dtype=torch.float32)
        ),
        "pair_meta": pair_meta,
    }
