"""Angle-aware LayoutLMv3 token classification wrapper.

This module intentionally keeps the public LayoutLMv3 inputs intact. The only
extra input is optional ``angle_features`` shaped ``[batch, seq_len, dim]``. If
that tensor is missing, the model behaves like a standard LayoutLMv3 token
classifier except for a newly initialized zero-input angle projection.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoProcessor
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.layoutlmv3.modeling_layoutlmv3 import LayoutLMv3Model, LayoutLMv3PreTrainedModel

from ml.angle_geometry import ANGLE_FEATURE_DIM, angle_feature_dim_for_mode, normalize_angle_encoding_mode


class AngleFeatureEncoder(nn.Module):
    def __init__(self, feature_dim: int, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, angle_features: torch.Tensor) -> torch.Tensor:
        return self.proj(angle_features)


class AngleAwareLayoutLMv3ForTokenClassification(LayoutLMv3PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.layoutlmv3 = LayoutLMv3Model(config)
        classifier_dropout = (
            config.classifier_dropout
            if getattr(config, "classifier_dropout", None) is not None
            else getattr(config, "hidden_dropout_prob", 0.1)
        )
        angle_encoding_mode = normalize_angle_encoding_mode(getattr(config, "angle_encoding_mode", "sincos_scalar"))
        angle_feature_dim = int(getattr(config, "angle_feature_dim", angle_feature_dim_for_mode(angle_encoding_mode)))
        angle_hidden_dim = int(
            getattr(config, "angle_hidden_size", getattr(config, "angle_hidden_dim", config.hidden_size))
        )
        angle_dropout = float(getattr(config, "angle_dropout", classifier_dropout))
        angle_fusion = str(getattr(config, "angle_fusion", "add"))
        if angle_fusion != "add":
            raise NotImplementedError("Only angle_fusion='add' is implemented for the current LayoutLMv3 wrapper.")
        self.angle_encoding_mode = angle_encoding_mode
        self.angle_feature_dim = angle_feature_dim
        self.angle_hidden_size = angle_hidden_dim
        self.angle_fusion = angle_fusion
        self.angle_encoder = nn.Sequential(
            nn.Linear(angle_feature_dim, angle_hidden_dim),
            nn.GELU(),
            nn.Dropout(angle_dropout),
            nn.Linear(angle_hidden_dim, config.hidden_size),
        )
        self.angle_layer_norm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def _zero_angle_features(self, sequence_output: torch.Tensor) -> torch.Tensor:
        return sequence_output.new_zeros((*sequence_output.shape[:2], self.angle_feature_dim))

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        bbox: torch.LongTensor | None = None,
        attention_mask: torch.FloatTensor | None = None,
        token_type_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        angle_features: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> TokenClassifierOutput:
        outputs = self.layoutlmv3(
            input_ids,
            bbox=bbox,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            pixel_values=pixel_values,
            **kwargs,
        )
        input_shape = input_ids.size() if input_ids is not None else inputs_embeds.size()[:-1]
        seq_length = input_shape[1]
        sequence_output = outputs[0][:, :seq_length]
        if angle_features is None:
            angle_features = self._zero_angle_features(sequence_output)
        else:
            angle_features = angle_features[:, :seq_length].to(sequence_output.device, dtype=sequence_output.dtype)
            if angle_features.shape[-1] != self.angle_feature_dim:
                raise ValueError(
                    f"angle_features last dim must be {self.angle_feature_dim}, got {angle_features.shape[-1]}"
                )
        sequence_output = self.angle_layer_norm(sequence_output + self.angle_encoder(angle_features))
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        hidden_states = outputs.hidden_states
        if hidden_states is not None:
            hidden_states = tuple(hidden_states) + (sequence_output,)
        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=hidden_states,
            attentions=outputs.attentions,
        )


def _config_with_angle(
    model_name_or_path: str | Path,
    *,
    num_labels: int | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    local_files_only: bool = False,
    angle_feature_dim: int | None = None,
    angle_hidden_size: int | None = None,
    angle_dropout: float | None = None,
    angle_fusion: str = "add",
    angle_encoding_mode: str | None = None,
) -> Any:
    kwargs: dict[str, Any] = {"local_files_only": local_files_only}
    if num_labels is not None:
        kwargs["num_labels"] = int(num_labels)
    if id2label is not None:
        kwargs["id2label"] = id2label
    if label2id is not None:
        kwargs["label2id"] = label2id
    config = AutoConfig.from_pretrained(model_name_or_path, **kwargs)
    mode = normalize_angle_encoding_mode(angle_encoding_mode or getattr(config, "angle_encoding_mode", "sincos_scalar"))
    config.use_angle_features = True
    config.angle_encoding_mode = mode
    config.angle_feature_dim = int(angle_feature_dim or getattr(config, "angle_feature_dim", angle_feature_dim_for_mode(mode)))
    config.angle_hidden_size = int(
        angle_hidden_size
        or getattr(config, "angle_hidden_size", getattr(config, "angle_hidden_dim", config.hidden_size))
    )
    config.angle_hidden_dim = config.angle_hidden_size
    config.angle_dropout = float(
        angle_dropout
        if angle_dropout is not None
        else getattr(config, "angle_dropout", getattr(config, "hidden_dropout_prob", 0.1))
    )
    config.angle_fusion = angle_fusion
    config.architectures = ["AngleAwareLayoutLMv3ForTokenClassification"]
    return config


def load_angle_aware_token_classifier(
    model_name_or_path: str | Path,
    *,
    num_labels: int | None = None,
    id2label: dict[int, str] | None = None,
    label2id: dict[str, int] | None = None,
    local_files_only: bool = False,
    ignore_mismatched_sizes: bool = True,
    angle_feature_dim: int | None = None,
    angle_hidden_size: int | None = None,
    angle_dropout: float | None = None,
    angle_fusion: str = "add",
    angle_encoding_mode: str | None = None,
) -> AngleAwareLayoutLMv3ForTokenClassification:
    """Load an angle-aware classifier from a normal or angle-aware checkpoint."""

    config = _config_with_angle(
        model_name_or_path,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        local_files_only=local_files_only,
        angle_feature_dim=angle_feature_dim,
        angle_hidden_size=angle_hidden_size,
        angle_dropout=angle_dropout,
        angle_fusion=angle_fusion,
        angle_encoding_mode=angle_encoding_mode,
    )
    return AngleAwareLayoutLMv3ForTokenClassification.from_pretrained(
        model_name_or_path,
        config=config,
        ignore_mismatched_sizes=ignore_mismatched_sizes,
        local_files_only=local_files_only,
    )


def save_angle_aware_model_bundle(path: str | Path, model, processor=None, labels_payload: dict[str, Any] | None = None):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path)
    if processor is not None:
        processor.save_pretrained(path)
    if labels_payload is not None:
        with (path / "labels.json").open("w", encoding="utf-8") as handle:
            json.dump(labels_payload, handle, ensure_ascii=False, indent=2)
    with (path / "angle_model_config.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "use_angle_features": True,
                "angle_feature_dim": int(getattr(model.config, "angle_feature_dim", ANGLE_FEATURE_DIM)),
                "angle_hidden_size": int(
                    getattr(model.config, "angle_hidden_size", getattr(model.config, "angle_hidden_dim", model.config.hidden_size))
                ),
                "angle_hidden_dim": int(
                    getattr(model.config, "angle_hidden_size", getattr(model.config, "angle_hidden_dim", model.config.hidden_size))
                ),
                "angle_dropout": float(getattr(model.config, "angle_dropout", 0.1)),
                "angle_encoding_mode": str(getattr(model.config, "angle_encoding_mode", "sincos_scalar")),
                "angle_fusion": str(getattr(model.config, "angle_fusion", "add")),
                "note": "Optional token-aligned angle_features are added after LayoutLMv3 backbone sequence output.",
            },
            handle,
            indent=2,
        )


def load_angle_aware_processor(model_name_or_path: str | Path, *, local_files_only: bool = False):
    return AutoProcessor.from_pretrained(model_name_or_path, apply_ocr=False, local_files_only=local_files_only)
