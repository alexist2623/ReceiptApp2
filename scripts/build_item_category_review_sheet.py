#!/usr/bin/env python
"""Build contact sheets for manual review of item semantic category labels."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


def parse_args():
    parser = argparse.ArgumentParser(description="Create image contact sheets for item category manual review.")
    parser.add_argument("--custom_root", default=str(Path.home() / "OneDrive" / "APK_Receipt2"))
    parser.add_argument(
        "--wildreceipt_root",
        default=str(Path("..") / "receipt_training_data3" / "wildreceipt" / "wildreceipt_custom_structure"),
    )
    parser.add_argument("--out_dir", default="outputs/item_category_manual_review")
    parser.add_argument("--max_custom", type=int, default=200)
    parser.add_argument("--max_wild", type=int, default=200)
    parser.add_argument("--wild_mode", choices=("frequent", "other", "all"), default="frequent")
    parser.add_argument("--exclude_dir_name", action="append", default=["Temp"])
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def image_for_label(path: Path, payload: dict[str, Any]) -> Path | None:
    capture_id = payload.get("capture_id") or path.name.replace("_labeled_v2_1.json", "")
    candidates = [
        path.with_name(f"{capture_id}.jpg"),
        path.with_name(f"{capture_id}.jpeg"),
        path.with_name(f"{capture_id}.png"),
    ]
    image_path = payload.get("image_path")
    if image_path:
        raw = Path(str(image_path))
        candidates.append(raw if raw.is_absolute() else path.parent / raw)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    jpgs = list(path.parent.glob("*.jpg")) + list(path.parent.glob("*.jpeg")) + list(path.parent.glob("*.png"))
    return jpgs[0] if jpgs else None


def safe_indices(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []
    out = []
    for item in value:
        try:
            out.append(int(item))
        except (TypeError, ValueError):
            continue
    return out


def union_box(words: list[dict[str, Any]], indices: list[int]) -> list[int] | None:
    boxes = []
    for idx in indices:
        if 0 <= idx < len(words):
            box = words[idx].get("box")
            if isinstance(box, list) and len(box) == 4:
                boxes.append([int(round(float(v))) for v in box])
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def collect_annotations(root: Path, exclude_names: set[str]) -> list[dict[str, Any]]:
    rows = []
    for label_path in sorted(root.rglob("*_labeled_v2_1.json")):
        if any(part in exclude_names for part in label_path.parts):
            continue
        payload = load_json(label_path)
        image_path = image_for_label(label_path, payload)
        words = payload.get("words") if isinstance(payload.get("words"), list) else []
        for ann_idx, annotation in enumerate(payload.get("item_category_annotations", [])):
            indices = safe_indices(annotation.get("item_name_word_indices"))
            box = union_box(words, indices)
            if not image_path or not box:
                continue
            text = annotation.get("item_name_text") or " ".join(
                str(words[idx].get("text", "")) for idx in indices if 0 <= idx < len(words)
            )
            rows.append(
                {
                    "label_path": str(label_path),
                    "image_path": str(image_path),
                    "annotation_index": ann_idx,
                    "text": text,
                    "category": annotation.get("category", "OTHER"),
                    "rule": annotation.get("rule"),
                    "confidence": annotation.get("confidence"),
                    "word_indices": indices,
                    "box": box,
                    "capture_id": payload.get("capture_id") or label_path.parent.name.replace("_receipt_ocr", ""),
                }
            )
    return rows


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "").strip()).upper()


def select_wild(rows: list[dict[str, Any]], mode: str, limit: int) -> list[dict[str, Any]]:
    if mode == "other":
        rows = [row for row in rows if row.get("category") == "OTHER"]
    if mode in {"frequent", "other"}:
        counts = Counter(normalize_text(row["text"]) for row in rows)
        seen = set()
        selected = []
        for row in sorted(rows, key=lambda r: (-counts[normalize_text(r["text"])], normalize_text(r["text"]))):
            key = normalize_text(row["text"])
            if key in seen:
                continue
            seen.add(key)
            selected.append(row)
            if len(selected) >= limit:
                break
        return selected
    return rows[:limit]


def crop_with_context(image: Image.Image, box: list[int]) -> Image.Image:
    width, height = image.size
    x0, y0, x1, y1 = box
    pad_x = max(40, int((x1 - x0) * 1.4))
    pad_y = max(35, int((y1 - y0) * 2.2))
    left = max(0, x0 - pad_x)
    top = max(0, y0 - pad_y)
    right = min(width, x1 + pad_x)
    bottom = min(height, y1 + pad_y)
    crop = image.crop((left, top, right, bottom)).convert("RGB")
    draw = ImageDraw.Draw(crop)
    draw.rectangle([x0 - left, y0 - top, x1 - left, y1 - top], outline=(255, 0, 0), width=3)
    return crop


def make_sheet(rows: list[dict[str, Any]], out_path: Path, manifest_path: Path) -> None:
    font = ImageFont.load_default()
    cell_w, cell_h = 420, 260
    cols = 2
    rows_per_sheet = (len(rows) + cols - 1) // cols
    sheet = Image.new("RGB", (cell_w * cols, cell_h * rows_per_sheet), "white")
    draw = ImageDraw.Draw(sheet)
    manifest = []
    for idx, row in enumerate(rows):
        col = idx % cols
        row_i = idx // cols
        x = col * cell_w
        y = row_i * cell_h
        try:
            image = ImageOps.exif_transpose(Image.open(row["image_path"])).convert("RGB")
            crop = crop_with_context(image, row["box"])
            crop.thumbnail((cell_w - 20, 170))
            sheet.paste(crop, (x + 10, y + 10))
        except Exception as exc:
            draw.text((x + 10, y + 10), f"image error: {exc}", fill=(160, 0, 0), font=font)
        label = f"{idx:03d} [{row['category']}] {row['text']}"
        rule = f"rule={row.get('rule')} file={Path(row['label_path']).parent.name}"
        draw.text((x + 10, y + 185), label[:70], fill=(0, 0, 0), font=font)
        draw.text((x + 10, y + 205), rule[:80], fill=(70, 70, 70), font=font)
        draw.text((x + 10, y + 225), f"indices={row['word_indices']} box={row['box']}", fill=(70, 70, 70), font=font)
        manifest.append({"review_index": idx, **row})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    exclude_names = set(args.exclude_dir_name or [])
    custom_rows = collect_annotations(Path(args.custom_root), exclude_names)[: args.max_custom]
    wild_rows = select_wild(collect_annotations(Path(args.wildreceipt_root), exclude_names), args.wild_mode, args.max_wild)
    make_sheet(custom_rows, out_dir / "custom_review_sheet.png", out_dir / "custom_review_manifest.json")
    make_sheet(wild_rows, out_dir / f"wild_{args.wild_mode}_review_sheet.png", out_dir / f"wild_{args.wild_mode}_review_manifest.json")
    print(f"custom rows: {len(custom_rows)}")
    print(f"wild rows: {len(wild_rows)}")
    print(f"out_dir: {out_dir}")


if __name__ == "__main__":
    main()
