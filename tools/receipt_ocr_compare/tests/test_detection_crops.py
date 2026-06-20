from pathlib import Path

from PIL import Image

from receipt_ocr_compare.crops import create_crops
from receipt_ocr_compare.detection import detect_boxes
from receipt_ocr_compare.schemas import GroundTruthToken


def test_ground_truth_detector_maps_bbox_to_crop_id(tmp_path: Path):
    image_path = tmp_path / "receipt.png"
    Image.new("RGB", (100, 60), "white").save(image_path)
    gt = [GroundTruthToken(image=image_path.name, token_id="token_001", bbox=[10, 10, 50, 30], text="12.30", token_type="number")]
    boxes, status = detect_boxes(image_path, detector="ground_truth", model_dir=tmp_path, vendor_dir=tmp_path, ground_truth=gt)
    assert status["detector"] == "ground_truth"
    assert boxes[0].crop_id == "token_001"
    crops = create_crops(image_path, boxes, output_dir=tmp_path / "crops", padding=2, ground_truth=gt)
    assert crops[0].crop_id == "token_001"
    assert Path(crops[0].crop_path).exists()

