from pathlib import Path

from receipt_ocr_compare.model_registry import available_model_ids, create_adapters
from receipt_ocr_compare.schemas import CropRecord, RunContext


def test_registry_contains_required_adapters(tmp_path: Path):
    assert {"svtrv2_b", "paddleocr", "existing"}.issubset(set(available_model_ids()))
    context = RunContext(model_dir=tmp_path / "models", vendor_dir=tmp_path / "vendor")
    adapters = create_adapters(["svtrv2_b", "paddleocr", "existing"], context)
    assert [adapter.model_id for adapter in adapters] == ["svtrv2_b", "paddleocr", "existing"]


def test_unavailable_model_returns_per_crop_error(tmp_path: Path):
    context = RunContext(model_dir=tmp_path / "models", vendor_dir=tmp_path / "vendor")
    adapter = create_adapters(["paddleocr"], context)[0]
    crop = CropRecord(image="x.png", crop_id="token_001", bbox=[0, 0, 10, 10], crop_path=str(tmp_path / "crop.png"))
    results = adapter.recognize([crop])
    assert results[0].status == "unavailable"
    assert results[0].error

