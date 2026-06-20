from __future__ import annotations

from pathlib import Path

from PIL import Image

from .schemas import CropRecord, DetectionBox, GroundTruthToken


def create_crops(
    image_path: Path,
    boxes: list[DetectionBox],
    *,
    output_dir: Path,
    padding: int = 0,
    ground_truth: list[GroundTruthToken] | None = None,
) -> list[CropRecord]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    output_dir.mkdir(parents=True, exist_ok=True)
    gt_by_id = {token.token_id: token for token in ground_truth or []}
    crops: list[CropRecord] = []
    for box in boxes:
        x1, y1, x2, y2 = box.bbox
        padded = [
            max(0, x1 - padding),
            max(0, y1 - padding),
            min(width, x2 + padding),
            min(height, y2 + padding),
        ]
        crop = image.crop(tuple(padded))
        crop_path = output_dir / f"{image_path.stem}_{box.crop_id}.png"
        crop.save(crop_path)
        gt = gt_by_id.get(box.crop_id)
        crops.append(
            CropRecord(
                image=image_path.name,
                crop_id=box.crop_id,
                bbox=box.bbox,
                crop_path=str(crop_path),
                padding=padding,
                gt_text=gt.text if gt else None,
                gt_type=gt.token_type if gt else None,
            )
        )
    return crops

