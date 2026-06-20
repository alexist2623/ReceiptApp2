from __future__ import annotations

import os
from pathlib import Path

from .base import BaseRecognizerAdapter
from ..runners.subprocess_runner import SubprocessRecognizerRunner
from ..schemas import CropRecord, ModelAvailability, RecognitionResult


class ExistingProjectAdapter(BaseRecognizerAdapter):
    model_id = "existing"
    display_name = "Existing project OCR"

    def availability(self) -> ModelAvailability:
        command = os.environ.get("RECEIPT_EXISTING_OCR_CMD") or self.context.extra.get("existing_project_command")
        if command:
            return ModelAvailability(self.model_id, True, details={"command": command})
        repo_root = Path.cwd()
        kotlin_engine = repo_root / "ReceiptApp" / "app" / "src" / "main" / "java" / "com" / "receiptapp" / "ocr"
        if kotlin_engine.exists():
            return ModelAvailability(
                self.model_id,
                False,
                "Existing Android ML Kit OCR code was found, but no Python/subprocess runner is configured",
                {"source_path": str(kotlin_engine)},
            )
        return ModelAvailability(self.model_id, False, "No existing OCR runner was found in this repository")

    def _recognize_available(self, crops: list[CropRecord]) -> list[RecognitionResult]:
        command = os.environ.get("RECEIPT_EXISTING_OCR_CMD") or self.context.extra.get("existing_project_command")
        if not command:
            return self.unavailable_results(crops, "existing project OCR runner is not configured")
        runner = SubprocessRecognizerRunner(model_id=self.model_id, command=command)
        return runner.run(crops)

