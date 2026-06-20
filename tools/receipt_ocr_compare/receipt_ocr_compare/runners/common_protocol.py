from __future__ import annotations

from dataclasses import asdict

from ..schemas import CropRecord


PROTOCOL_DESCRIPTION = """
Recognizer subprocess protocol:
- stdin: JSONL rows from CropRecord dataclasses.
- stdout: JSONL rows with model_id, crop_id, raw_text, normalized_text,
  corrected_text_optional, confidence, latency_ms, and error.
"""


def crop_record_json(crop: CropRecord) -> dict:
    return asdict(crop)

