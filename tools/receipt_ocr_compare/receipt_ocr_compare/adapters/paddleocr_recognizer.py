from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import BaseRecognizerAdapter
from ..schemas import CropRecord, ModelAvailability, RecognitionResult


class PaddleOCRRecognizerAdapter(BaseRecognizerAdapter):
    model_id = "paddleocr"
    display_name = "PaddleOCR recognizer"

    def __init__(self, context):
        super().__init__(context)
        self._engine: Any | None = None

    def availability(self) -> ModelAvailability:
        rec_dir = self._rec_model_dir()
        if not rec_dir.exists() or not any(rec_dir.iterdir()):
            return ModelAvailability(
                self.model_id,
                False,
                f"PaddleOCR local recognizer model directory is missing or empty: {rec_dir}",
                {"expected_path": str(rec_dir)},
            )
        try:
            import paddleocr  # noqa: F401  # type: ignore
        except Exception as exc:
            return ModelAvailability(self.model_id, False, f"paddleocr package is not importable: {exc}")
        return ModelAvailability(self.model_id, True, details={"rec_model_dir": str(rec_dir)})

    def _recognize_available(self, crops: list[CropRecord]) -> list[RecognitionResult]:
        try:
            engine = self._load_engine()
        except Exception as exc:
            return self.unavailable_results(crops, f"PaddleOCR recognizer initialization failed: {exc}")
        results: list[RecognitionResult] = []
        for crop in crops:
            try:
                raw, latency_ms = self.time_call(lambda: engine.ocr(str(crop.crop_path), det=False, rec=True, cls=False))
                text, confidence = _parse_paddle_recognition(raw)
                results.append(self.make_result(crop, text, confidence, latency_ms))
            except Exception as exc:
                results.append(self.make_result(crop, "", None, 0.0, error=f"PaddleOCR recognition failed: {exc}"))
        return results

    def _load_engine(self):
        if self._engine is not None:
            return self._engine
        from paddleocr import PaddleOCR  # type: ignore

        kwargs: dict[str, Any] = {"det": False, "rec": True, "use_angle_cls": False, "show_log": False}
        rec_dir = self._rec_model_dir()
        kwargs["rec_model_dir"] = str(rec_dir)
        self._engine = PaddleOCR(**kwargs)
        return self._engine

    def _rec_model_dir(self) -> Path:
        return self.context.model_dir / "paddleocr" / "rec"


def _parse_paddle_recognition(raw: Any) -> tuple[str, float | None]:
    if raw is None:
        return "", None
    if isinstance(raw, dict):
        payload = raw.get("res", raw)
        text = payload.get("rec_text") or payload.get("text") or ""
        score = payload.get("rec_score") or payload.get("score")
        return str(text), float(score) if score is not None else None
    if isinstance(raw, list):
        item = raw[0] if raw else None
        if isinstance(item, dict):
            return _parse_paddle_recognition(item)
        if isinstance(item, list) and item:
            candidate = item[0]
            if isinstance(candidate, tuple) and len(candidate) >= 2:
                return str(candidate[0]), float(candidate[1]) if candidate[1] is not None else None
            if isinstance(candidate, list) and candidate and isinstance(candidate[-1], (float, int)):
                return str(candidate[0]), float(candidate[-1])
        if isinstance(item, tuple) and len(item) >= 2:
            return str(item[0]), float(item[1]) if item[1] is not None else None
    return str(raw), None
