from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


BBox = list[int]


@dataclass(frozen=True)
class DetectionBox:
    image: str
    crop_id: str
    bbox: BBox
    detector: str
    confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CropRecord:
    image: str
    crop_id: str
    bbox: BBox
    crop_path: str
    padding: int = 0
    gt_text: str | None = None
    gt_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecognitionResult:
    model_id: str
    crop_id: str
    raw_text: str
    normalized_text: str
    confidence: float | None
    latency_ms: float | None
    error: str | None = None
    corrected_text_optional: str | None = None
    status: str = "ok"
    image: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GroundTruthToken:
    image: str
    token_id: str
    bbox: BBox
    text: str
    token_type: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelAvailability:
    model_id: str
    available: bool
    reason: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RunContext:
    model_dir: Path
    vendor_dir: Path
    device: str = "cpu"
    allow_package_models: bool = False
    extra: dict[str, Any] = field(default_factory=dict)
