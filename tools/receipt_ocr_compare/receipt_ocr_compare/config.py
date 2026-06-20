from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_MODELS = ("svtrv2_b", "paddleocr", "existing")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class CompareConfig:
    input_path: Path
    models: tuple[str, ...] = DEFAULT_MODELS
    mode: str = "recognition"
    detector: str = "auto"
    model_dir: Path = Path("tools/receipt_ocr_compare/models")
    vendor_dir: Path = Path("tools/receipt_ocr_compare/vendor")
    output_dir: Path = Path("tools/receipt_ocr_compare/outputs/run")
    crop_padding: int = 2
    device: str = "cpu"
    ground_truth_path: Path | None = None
    allow_package_models: bool = False


def parse_csv(value: str | None, default: tuple[str, ...] = DEFAULT_MODELS) -> tuple[str, ...]:
    if not value:
        return default
    return tuple(item.strip() for item in value.split(",") if item.strip())


def collect_images(input_path: Path) -> list[Path]:
    path = input_path.resolve()
    if path.is_file():
        return [path] if path.suffix.lower() in IMAGE_SUFFIXES else []
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.suffix.lower() in IMAGE_SUFFIXES)
    return []


def load_optional_yaml(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml  # type: ignore
    except Exception as exc:
        raise RuntimeError("PyYAML is required to load YAML config files") from exc
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded or {}

