from pathlib import Path

from PIL import Image, ImageDraw

from receipt_ocr_compare.overlay import render_model_overlay
from receipt_ocr_compare.schemas import DetectionBox, GroundTruthToken, RecognitionResult


def test_overlay_image_generation(tmp_path: Path):
    image_path = tmp_path / "receipt.png"
    image = Image.new("RGB", (180, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.text((10, 10), "12.30", fill="black")
    image.save(image_path)

    box = DetectionBox(image=image_path.name, crop_id="token_001", bbox=[8, 8, 80, 30], detector="test")
    result = RecognitionResult(
        model_id="paddleocr",
        crop_id="token_001",
        raw_text="1230",
        normalized_text="1230",
        corrected_text_optional="1230",
        confidence=0.8,
        latency_ms=1.0,
    )
    gt = GroundTruthToken(image=image_path.name, token_id="token_001", bbox=[8, 8, 80, 30], text="12.30", token_type="number")
    out = render_model_overlay(image_path, [box], [result], output_path=tmp_path / "overlay.png", ground_truth=[gt])
    assert out.exists()
    assert Image.open(out).size == (180, 80)

