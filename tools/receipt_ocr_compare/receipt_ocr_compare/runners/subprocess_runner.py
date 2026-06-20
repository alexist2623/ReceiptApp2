from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

from ..normalization import corrected_numeric_text_optional, normalize_text
from ..schemas import CropRecord, RecognitionResult
from .common_protocol import crop_record_json


class SubprocessRecognizerRunner:
    def __init__(self, *, model_id: str, command: str):
        self.model_id = model_id
        self.command = command

    def run(self, crops: list[CropRecord]) -> list[RecognitionResult]:
        payload = "\n".join(json.dumps(crop_record_json(crop), ensure_ascii=False) for crop in crops) + "\n"
        proc = subprocess.run(
            shlex.split(self.command, posix=False),
            input=payload,
            text=True,
            capture_output=True,
            cwd=Path.cwd(),
            check=False,
        )
        if proc.returncode != 0:
            reason = (proc.stderr or proc.stdout or f"command exited {proc.returncode}").strip()
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
                    status="error",
                    image=crop.image,
                )
                for crop in crops
            ]
        rows: list[RecognitionResult] = []
        by_crop = {crop.crop_id: crop for crop in crops}
        for line in proc.stdout.splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            crop_id = str(item.get("crop_id", ""))
            raw_text = str(item.get("raw_text", ""))
            crop = by_crop.get(crop_id)
            rows.append(
                RecognitionResult(
                    model_id=str(item.get("model_id", self.model_id)),
                    crop_id=crop_id,
                    raw_text=raw_text,
                    normalized_text=str(item.get("normalized_text") or normalize_text(raw_text)),
                    corrected_text_optional=item.get("corrected_text_optional") or corrected_numeric_text_optional(raw_text),
                    confidence=_maybe_float(item.get("confidence")),
                    latency_ms=_maybe_float(item.get("latency_ms")),
                    error=item.get("error"),
                    status=str(item.get("status", "ok")),
                    image=str(item.get("image") or (crop.image if crop else "")) or None,
                )
            )
        seen = {row.crop_id for row in rows}
        for crop_id, crop in by_crop.items():
            if crop_id not in seen:
                rows.append(
                    RecognitionResult(
                        model_id=self.model_id,
                        crop_id=crop.crop_id,
                        raw_text="",
                        normalized_text="",
                        corrected_text_optional=None,
                        confidence=None,
                        latency_ms=None,
                        error="subprocess did not return a row for this crop",
                        status="error",
                        image=crop.image,
                    )
                )
        return rows


def _maybe_float(value) -> float | None:
    if value is None or value == "":
        return None
    return float(value)
