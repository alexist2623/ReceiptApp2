from __future__ import annotations

from abc import ABC, abstractmethod
from time import perf_counter

from ..normalization import corrected_numeric_text_optional, normalize_text
from ..schemas import CropRecord, ModelAvailability, RecognitionResult, RunContext


class BaseRecognizerAdapter(ABC):
    model_id: str
    display_name: str

    def __init__(self, context: RunContext):
        self.context = context

    @abstractmethod
    def availability(self) -> ModelAvailability:
        raise NotImplementedError

    def recognize(self, crops: list[CropRecord]) -> list[RecognitionResult]:
        available = self.availability()
        if not available.available:
            return self.unavailable_results(crops, available.reason or "model unavailable")
        return self._recognize_available(crops)

    @abstractmethod
    def _recognize_available(self, crops: list[CropRecord]) -> list[RecognitionResult]:
        raise NotImplementedError

    def unavailable_results(self, crops: list[CropRecord], reason: str) -> list[RecognitionResult]:
        return [
            RecognitionResult(
                model_id=self.model_id,
                crop_id=crop.crop_id,
                raw_text="",
                normalized_text="",
                corrected_text_optional=None,
                confidence=None,
                latency_ms=None,
                error=reason,
                status="unavailable",
                image=crop.image,
            )
            for crop in crops
        ]

    def make_result(
        self,
        crop: CropRecord,
        raw_text: str,
        confidence: float | None,
        latency_ms: float,
        error: str | None = None,
    ) -> RecognitionResult:
        normalized = normalize_text(raw_text)
        return RecognitionResult(
            model_id=self.model_id,
            crop_id=crop.crop_id,
            raw_text=raw_text,
            normalized_text=normalized,
            corrected_text_optional=corrected_numeric_text_optional(raw_text),
            confidence=confidence,
            latency_ms=latency_ms,
            error=error,
            status="ok" if error is None else "error",
            image=crop.image,
        )

    @staticmethod
    def time_call(fn):
        start = perf_counter()
        value = fn()
        return value, (perf_counter() - start) * 1000.0
