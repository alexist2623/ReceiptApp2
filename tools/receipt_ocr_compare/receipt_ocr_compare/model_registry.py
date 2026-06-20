from __future__ import annotations

from .adapters import ExistingProjectAdapter, PaddleOCRRecognizerAdapter, SVTRv2BAdapter
from .adapters.base import BaseRecognizerAdapter
from .schemas import RunContext


ADAPTERS: dict[str, type[BaseRecognizerAdapter]] = {
    "svtrv2_b": SVTRv2BAdapter,
    "paddleocr": PaddleOCRRecognizerAdapter,
    "existing": ExistingProjectAdapter,
}


def available_model_ids() -> list[str]:
    return sorted(ADAPTERS)


def create_adapter(model_id: str, context: RunContext) -> BaseRecognizerAdapter:
    try:
        adapter_cls = ADAPTERS[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown model id: {model_id}. Available: {', '.join(available_model_ids())}") from exc
    return adapter_cls(context)


def create_adapters(model_ids: list[str] | tuple[str, ...], context: RunContext) -> list[BaseRecognizerAdapter]:
    return [create_adapter(model_id, context) for model_id in model_ids]

