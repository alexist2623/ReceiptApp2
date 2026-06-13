import argparse
import csv
import html
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

try:
    import torch
except Exception:  # pragma: no cover - environment diagnostics only
    torch = None


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
IMAGE_PRIORITY = {".jpg": 0, ".jpeg": 1, ".png": 2, ".webp": 3, ".bmp": 4}
JSON_EXTENSIONS = (".json",)
OCR_SUFFIXES = (
    "_ocr",
    "-ocr",
    ".ocr",
    "_ocr_result",
    "-ocr-result",
    "_ocr_results",
    "-ocr-results",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Validate an unzipped Android receipt OCR export and optionally run the ML pipeline."
    )
    parser.add_argument("--input_dir", required=True, help="Directory created by unzipping the Android export ZIP.")
    parser.add_argument("--layoutlm_checkpoint", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--relg_checkpoint", default="models/span-relg-context/best")
    parser.add_argument("--labels", default="processed_data/cord_bio/labels.json")
    parser.add_argument("--out_dir", default="outputs/exported_receipt_zip_test")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument(
        "--skip_model_inference",
        action="store_true",
        help="Only validate schema and OCR box overlays; skip LayoutLMv3 and span rel-g inference.",
    )
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def path_from_arg(value):
    path_text = str(value)
    if sys.platform.startswith("linux") and len(path_text) >= 3 and path_text[1:3] in (":\\", ":/"):
        drive = path_text[0].lower()
        rest = path_text[3:].replace("\\", "/")
        return Path(f"/mnt/{drive}/{rest}")
    return Path(path_text)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalized_capture_id(path):
    stem = Path(path).stem
    lowered = stem.lower()
    changed = True
    while changed:
        changed = False
        for suffix in OCR_SUFFIXES:
            if lowered.endswith(suffix):
                stem = stem[: -len(suffix)]
                lowered = lowered[: -len(suffix)]
                changed = True
    return stem


def safe_id(value):
    keep = []
    for char in str(value):
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    text = "".join(keep).strip("._-")
    return text or "receipt"


def list_files(directory, extensions, recursive=False):
    directory = Path(directory)
    if not directory.exists():
        return []
    pattern = "**/*" if recursive else "*"
    return sorted(
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in extensions
    )


def choose_image(paths):
    return sorted(paths, key=lambda p: (IMAGE_PRIORITY.get(p.suffix.lower(), 99), str(p)))[0]


def pair_from_dirs(image_dir, json_dir, structure):
    images = list_files(image_dir, IMAGE_EXTENSIONS)
    jsons = list_files(json_dir, JSON_EXTENSIONS)
    images_by_id = {}
    jsons_by_id = {}
    for image_path in images:
        images_by_id.setdefault(normalized_capture_id(image_path), []).append(image_path)
    for json_path in jsons:
        jsons_by_id.setdefault(normalized_capture_id(json_path), []).append(json_path)

    pairs = []
    matched_images = set()
    matched_jsons = set()
    for capture_id in sorted(set(images_by_id) & set(jsons_by_id)):
        image_path = choose_image(images_by_id[capture_id])
        json_path = sorted(jsons_by_id[capture_id])[0]
        warnings = []
        if len(images_by_id[capture_id]) > 1:
            warnings.append(f"Multiple image candidates found; using {image_path}")
        if len(jsons_by_id[capture_id]) > 1:
            warnings.append(f"Multiple OCR JSON candidates found; using {json_path}")
        matched_images.add(image_path.resolve())
        matched_jsons.add(json_path.resolve())
        pairs.append(
            {
                "capture_id": safe_id(capture_id),
                "image_path": str(image_path),
                "ocr_json_path": str(json_path),
                "structure": structure,
                "warnings": warnings,
            }
        )

    unmatched_images = [str(path) for path in images if path.resolve() not in matched_images]
    unmatched_jsons = [str(path) for path in jsons if path.resolve() not in matched_jsons]
    return pairs, unmatched_images, unmatched_jsons


def find_exported_pairs(input_dir):
    input_dir = Path(input_dir)
    all_pairs = []
    all_unmatched_images = []
    all_unmatched_jsons = []

    root_images = input_dir / "images"
    root_jsons = input_dir / "ocr_json"
    if root_images.exists() and root_jsons.exists():
        pairs, unmatched_images, unmatched_jsons = pair_from_dirs(root_images, root_jsons, "images_ocr_json")
        all_pairs.extend(pairs)
        all_unmatched_images.extend(unmatched_images)
        all_unmatched_jsons.extend(unmatched_jsons)

    for child in sorted(path for path in input_dir.iterdir() if path.is_dir()):
        image_dir = child / "images"
        json_dir = child / "ocr_json"
        if image_dir.exists() and json_dir.exists():
            pairs, unmatched_images, unmatched_jsons = pair_from_dirs(image_dir, json_dir, "nested")
            all_pairs.extend(pairs)
            all_unmatched_images.extend(unmatched_images)
            all_unmatched_jsons.extend(unmatched_jsons)

    pairs, unmatched_images, unmatched_jsons = pair_from_dirs(input_dir, input_dir, "flat")
    all_pairs.extend(pairs)
    all_unmatched_images.extend(unmatched_images)
    all_unmatched_jsons.extend(unmatched_jsons)

    deduped = []
    seen = set()
    for pair in all_pairs:
        key = (Path(pair["image_path"]).resolve(), Path(pair["ocr_json_path"]).resolve())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(pair)
    return {
        "pairs": deduped,
        "unmatched_images": sorted(set(all_unmatched_images)),
        "unmatched_jsons": sorted(set(all_unmatched_jsons)),
    }


def load_image_size(image_path):
    image = Image.open(image_path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB"), image.size


def json_image_size(payload):
    width = payload.get("image_width")
    height = payload.get("image_height")
    image_info = payload.get("image") if isinstance(payload.get("image"), dict) else {}
    if width is None:
        width = image_info.get("width")
    if height is None:
        height = image_info.get("height")
    return width, height


def coerce_box(raw_box):
    if not isinstance(raw_box, list) or len(raw_box) != 4:
        return None
    try:
        return [int(round(float(value))) for value in raw_box]
    except (TypeError, ValueError):
        return None


def box_outside_image(box, width, height):
    left, top, right, bottom = box
    return left < 0 or top < 0 or right > width or bottom > height


def clamp_box(box, width, height):
    left, top, right, bottom = box
    left = max(0, min(left, width - 1))
    right = max(0, min(right, width - 1))
    top = max(0, min(top, height - 1))
    bottom = max(0, min(bottom, height - 1))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def validate_ocr_json(json_path, image_path):
    warnings = []
    errors = []
    try:
        image, (image_width, image_height) = load_image_size(image_path)
    except Exception as exc:
        return {
            "valid": False,
            "payload": None,
            "image": None,
            "image_width": None,
            "image_height": None,
            "json_image_width": None,
            "json_image_height": None,
            "word_count": 0,
            "blank_text_count": 0,
            "invalid_box_count": 0,
            "out_of_image_box_count": 0,
            "warnings": warnings,
            "errors": [f"Could not open image: {exc}"],
        }
    try:
        with Path(json_path).open("r", encoding="utf-8-sig") as handle:
            payload = json.load(handle)
    except Exception as exc:
        return {
            "valid": False,
            "payload": None,
            "image": image,
            "image_width": image_width,
            "image_height": image_height,
            "json_image_width": None,
            "json_image_height": None,
            "word_count": 0,
            "blank_text_count": 0,
            "invalid_box_count": 0,
            "out_of_image_box_count": 0,
            "warnings": warnings,
            "errors": [f"Could not parse OCR JSON: {exc}"],
        }

    if not isinstance(payload, dict):
        errors.append("OCR JSON top-level value must be an object.")
        payload = {}
    json_width, json_height = json_image_size(payload)
    if json_width is None or json_height is None:
        warnings.append("image_width/image_height missing; using actual image size.")
        json_width, json_height = image_width, image_height
    elif int(json_width) != image_width or int(json_height) != image_height:
        warnings.append(
            f"JSON image size {json_width}x{json_height} differs from actual image size {image_width}x{image_height}."
        )

    words = payload.get("words")
    if not isinstance(words, list):
        errors.append("top-level words must exist and be a list.")
        words = []

    blank_text_count = 0
    invalid_box_count = 0
    out_of_image_box_count = 0
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            invalid_box_count += 1
            warnings.append(f"words[{index}] is not an object.")
            continue
        text = word.get("text")
        if text is None or not str(text).strip():
            blank_text_count += 1
        box = coerce_box(word.get("box"))
        if box is None:
            invalid_box_count += 1
            continue
        left, top, right, bottom = box
        if left > right or top > bottom or right <= left or bottom <= top:
            invalid_box_count += 1
            continue
        if box_outside_image(box, image_width, image_height):
            out_of_image_box_count += 1

    if not words:
        errors.append("words is empty; model inference cannot run.")
    return {
        "valid": not errors,
        "payload": payload,
        "image": image,
        "top_level_keys": sorted(payload.keys()),
        "image_width": image_width,
        "image_height": image_height,
        "json_image_width": int(json_width) if json_width is not None else None,
        "json_image_height": int(json_height) if json_height is not None else None,
        "word_count": len(words),
        "blank_text_count": blank_text_count,
        "invalid_box_count": invalid_box_count,
        "out_of_image_box_count": out_of_image_box_count,
        "warnings": warnings,
        "errors": errors,
    }


def draw_ocr_box_overlay(image_path, payload, out_path, validation):
    image, (width, height) = load_image_size(image_path)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    words = payload.get("words", []) if isinstance(payload, dict) else []
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            continue
        raw_box = word.get("box")
        box = coerce_box(raw_box)
        color = (34, 197, 94)
        if box is None or len(box) != 4:
            continue
        left, top, right, bottom = box
        invalid = left > right or top > bottom or right <= left or bottom <= top
        outside = not invalid and box_outside_image(box, width, height)
        if invalid:
            color = (239, 68, 68)
            clamped = clamp_box([min(left, right), min(top, bottom), max(left, right), max(top, bottom)], width, height)
        elif outside:
            color = (245, 158, 11)
            clamped = clamp_box(box, width, height)
        else:
            clamped = box
        if clamped is None:
            continue
        draw.rectangle(clamped, outline=color, width=2)
        if index < 200:
            label = str(word.get("text", "")).strip()[:24]
            if label:
                draw.text((clamped[0], max(0, clamped[1] - 12)), label, fill=color, font=font)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path)
    return str(out_path)


def copy_to_staging(pair, out_dir):
    capture_id = pair["capture_id"]
    image_path = Path(pair["image_path"])
    json_path = Path(pair["ocr_json_path"])
    staged_root = Path(out_dir) / "staged_user_receipts"
    staged_image_dir = staged_root / "images"
    staged_json_dir = staged_root / "ocr_json"
    staged_image_dir.mkdir(parents=True, exist_ok=True)
    staged_json_dir.mkdir(parents=True, exist_ok=True)
    staged_image = staged_image_dir / f"{capture_id}{image_path.suffix.lower()}"
    staged_json = staged_json_dir / f"{capture_id}.json"
    shutil.copy2(image_path, staged_image)
    shutil.copy2(json_path, staged_json)
    return staged_image, staged_json


def run_subprocess(command, cwd):
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
    }


def choose_batch_script(root_dir):
    candidates = [
        root_dir / "scripts" / "batch_infer_user_receipts.py",
        root_dir / "scripts" / "batch_infer_user_ocr_json.py",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def run_layoutlmv3_batch(args, staged_root, out_dir, root_dir):
    batch_script = choose_batch_script(root_dir)
    if batch_script is None:
        return {
            "ok": False,
            "error": "No batch inference script found: expected scripts/batch_infer_user_receipts.py or scripts/batch_infer_user_ocr_json.py",
        }
    command = [
        sys.executable,
        str(batch_script),
        "--input_dir",
        str(staged_root),
        "--checkpoint",
        str(path_from_arg(args.layoutlm_checkpoint)),
        "--labels",
        str(path_from_arg(args.labels)),
        "--out_dir",
        str(Path(out_dir) / "layoutlmv3_predictions"),
        "--device",
        args.device,
        "--max_length",
        str(args.max_length),
    ]
    if args.local_files_only:
        command.append("--local_files_only")
    if args.debug:
        command.append("--debug")
    result = run_subprocess(command, root_dir)
    return {"ok": result["returncode"] == 0, "script": str(batch_script), **result}


def run_relg_for_prediction(args, capture_id, prediction_json, out_dir, root_dir):
    relg_script = root_dir / "scripts" / "infer_user_span_relg.py"
    if not relg_script.exists():
        return {"ok": False, "error": "scripts/infer_user_span_relg.py not found."}
    out_json = Path(out_dir) / "grouped" / f"{capture_id}_grouped_relg.json"
    out_overlay = Path(out_dir) / "grouped_overlays" / f"{capture_id}_span_relg_overlay.png"
    command = [
        sys.executable,
        str(relg_script),
        "--prediction_json",
        str(prediction_json),
        "--layoutlm_checkpoint",
        str(path_from_arg(args.layoutlm_checkpoint)),
        "--relg_checkpoint",
        str(path_from_arg(args.relg_checkpoint)),
        "--out_json",
        str(out_json),
        "--out_overlay",
        str(out_overlay),
        "--device",
        args.device,
        "--max_length",
        str(args.max_length),
    ]
    if args.local_files_only:
        command.append("--local_files_only")
    if args.debug:
        command.append("--debug")
    result = run_subprocess(command, root_dir)
    return {"ok": result["returncode"] == 0, "out_json": str(out_json), "out_overlay": str(out_overlay), **result}


def validate_grouped_json(path):
    info = {
        "item_count": None,
        "price_missing_item_count": None,
        "menu_name_missing_item_count": None,
        "rel_g_prob_min": None,
        "rel_g_prob_max": None,
        "rel_g_prob_avg": None,
        "warnings": [],
    }
    if not path or not Path(path).exists():
        info["warnings"].append("grouped rel-g JSON not found.")
        return info
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("items")
    if not isinstance(items, list):
        info["warnings"].append("grouped JSON does not contain an items array.")
        return info
    probs = []
    price_missing = 0
    menu_missing = 0
    for item in items:
        if not isinstance(item, dict):
            continue
        if not (item.get("price") or item.get("menu_price")):
            price_missing += 1
        if not (item.get("item_name") or item.get("menu_name") or item.get("name")):
            menu_missing += 1
        if item.get("rel_g_prob") is not None:
            try:
                probs.append(float(item["rel_g_prob"]))
            except (TypeError, ValueError):
                pass
    info["item_count"] = len(items)
    info["price_missing_item_count"] = price_missing
    info["menu_name_missing_item_count"] = menu_missing
    if probs:
        info["rel_g_prob_min"] = min(probs)
        info["rel_g_prob_max"] = max(probs)
        info["rel_g_prob_avg"] = sum(probs) / len(probs)
    return info


def field_text(value):
    if isinstance(value, dict):
        return value.get("text") or ""
    return ""


def field_prob(value):
    if isinstance(value, dict) and value.get("rel_prob") is not None:
        try:
            return float(value["rel_prob"])
        except (TypeError, ValueError):
            return None
    return None


def grouped_mapping_rows(row):
    grouped_path = row.get("grouped_relg_json")
    if not grouped_path or not Path(grouped_path).exists():
        return []
    with Path(grouped_path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    items = payload.get("items")
    if not isinstance(items, list):
        return []
    output = []
    for item in items:
        if not isinstance(item, dict):
            continue
        price = item.get("price") or item.get("menu_price")
        output.append(
            {
                "capture_id": row.get("capture_id", ""),
                "item_index": item.get("item_index"),
                "item_name": field_text(item.get("item_name") or item.get("menu_name") or item.get("name")),
                "menu_name": field_text(item.get("menu_name") or item.get("item_name") or item.get("name")),
                "price": field_text(price),
                "price_rel_prob": field_prob(price),
                "quantity": field_text(item.get("quantity") or item.get("count")),
                "unit_price": field_text(item.get("unit_price")),
                "warnings": "; ".join(item.get("warnings") or []),
            }
        )
    return output


def write_mapping_outputs(out_dir, rows):
    mapping_rows = []
    for row in rows:
        mapping_rows.extend(grouped_mapping_rows(row))
    csv_path = Path(out_dir) / "menu_price_mapping.csv"
    md_path = Path(out_dir) / "menu_price_mapping.md"
    fieldnames = [
        "capture_id",
        "item_index",
        "item_name",
        "menu_name",
        "price",
        "price_rel_prob",
        "quantity",
        "unit_price",
        "warnings",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in mapping_rows:
            writer.writerow(item)
    lines = [
        "# Item Price Mapping",
        "",
        "| capture_id | item | item_name | price | price_rel_prob | quantity | unit_price | warnings |",
        "|---|---:|---|---|---:|---|---|---|",
    ]
    for item in mapping_rows:
        lines.append(
            "| {capture_id} | {item_index} | {item_name} | {price} | {price_rel_prob} | {quantity} | {unit_price} | {warnings} |".format(
                capture_id=str(item.get("capture_id", "")).replace("|", "\\|"),
                item_index=item.get("item_index", ""),
                item_name=str(item.get("item_name") or item.get("menu_name", "")).replace("|", "\\|"),
                price=str(item.get("price", "")).replace("|", "\\|"),
                price_rel_prob="" if item.get("price_rel_prob") is None else f"{item['price_rel_prob']:.4f}",
                quantity=str(item.get("quantity", "")).replace("|", "\\|"),
                unit_price=str(item.get("unit_price", "")).replace("|", "\\|"),
                warnings=str(item.get("warnings", "")).replace("|", "\\|"),
            )
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(csv_path), str(md_path), mapping_rows


def write_csv_summary(path, rows):
    fieldnames = [
        "capture_id",
        "ocr_valid",
        "image_width",
        "image_height",
        "json_image_width",
        "json_image_height",
        "word_count",
        "invalid_box_count",
        "out_of_image_box_count",
        "ocr_overlay_path",
        "layoutlmv3_prediction_json",
        "layoutlmv3_overlay_path",
        "grouped_relg_json",
        "item_count",
        "warning_count",
        "error_count",
    ]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def rel_path(path, base):
    if not path:
        return ""
    try:
        return Path(path).resolve().relative_to(Path(base).resolve()).as_posix()
    except Exception:
        return str(path)


def write_html_gallery(path, rows, out_dir):
    cards = []
    for row in rows:
        ocr_overlay = html.escape(rel_path(row.get("ocr_overlay_path"), out_dir))
        pred_overlay = html.escape(rel_path(row.get("layoutlmv3_overlay_path"), out_dir))
        relg_overlay = html.escape(rel_path(row.get("grouped_relg_overlay_path"), out_dir))
        grouped = html.escape(rel_path(row.get("grouped_relg_json"), out_dir))
        ocr_json = html.escape(rel_path(row.get("staged_ocr_json_path"), out_dir))
        warnings = "<br>".join(html.escape(item) for item in row.get("warnings", [])[:5])
        errors = "<br>".join(html.escape(item) for item in row.get("errors", [])[:5])
        mapping_rows = grouped_mapping_rows(row)
        if mapping_rows:
            mapping_table = """
  <table>
    <thead><tr><th>#</th><th>Item name</th><th>Price</th><th>Rel prob</th><th>Quantity</th><th>Unit price</th><th>Warnings</th></tr></thead>
    <tbody>
"""
            for item in mapping_rows:
                prob = "" if item.get("price_rel_prob") is None else f"{item['price_rel_prob']:.3f}"
                mapping_table += (
                    "      <tr>"
                    f"<td>{html.escape(str(item.get('item_index', '')))}</td>"
                    f"<td>{html.escape(str(item.get('item_name') or item.get('menu_name', '')))}</td>"
                    f"<td>{html.escape(str(item.get('price', '')))}</td>"
                    f"<td>{html.escape(prob)}</td>"
                    f"<td>{html.escape(str(item.get('quantity', '')))}</td>"
                    f"<td>{html.escape(str(item.get('unit_price', '')))}</td>"
                    f"<td>{html.escape(str(item.get('warnings', '')))}</td>"
                    "</tr>\n"
                )
            mapping_table += "    </tbody>\n  </table>"
        else:
            mapping_table = "<p>No grouped item-price mapping available.</p>"
        cards.append(
            f"""
<section class="card">
  <h2>{html.escape(row['capture_id'])}</h2>
  <p>words={row.get('word_count')} invalid_boxes={row.get('invalid_box_count')} items={row.get('item_count')}</p>
  <p><a href="{ocr_json}">OCR JSON</a> | <a href="{grouped}">Grouped JSON</a></p>
  <h3>Item -> Price Mapping</h3>
  {mapping_table}
  <div class="images">
    <figure><figcaption>OCR box overlay</figcaption><img src="{ocr_overlay}" alt="OCR overlay"></figure>
    <figure><figcaption>LayoutLMv3 overlay</figcaption><img src="{pred_overlay}" alt="LayoutLMv3 overlay"></figure>
    <figure><figcaption>Rel-G mapping overlay</figcaption><img src="{relg_overlay}" alt="Rel-G mapping overlay"></figure>
  </div>
  <p class="warn">{warnings}</p>
  <p class="err">{errors}</p>
</section>
"""
        )
    content = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Exported Receipt ZIP Test</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 24px; background: #f6f7f9; color: #172033; }}
    .card {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; margin: 0 0 24px; padding: 16px; }}
    .images {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d8dee9; }}
    table {{ width: 100%; border-collapse: collapse; margin: 12px 0 18px; font-size: 14px; }}
    th, td {{ border: 1px solid #d8dee9; padding: 6px 8px; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f7; }}
    figcaption {{ font-weight: 700; margin-bottom: 8px; }}
    .warn {{ color: #a16207; }}
    .err {{ color: #b91c1c; }}
  </style>
</head>
<body>
  <h1>Exported Receipt ZIP Test</h1>
  {''.join(cards)}
</body>
</html>
"""
    Path(path).write_text(content, encoding="utf-8")


def print_environment():
    print(f"WSL/conda Python path: {sys.executable}")
    if torch is None:
        print("torch: not importable")
        print("cuda availability: unknown")
        return
    print(f"torch version: {torch.__version__}")
    cuda = torch.cuda.is_available()
    print(f"cuda availability: {cuda}")
    if cuda:
        print(f"cuda device: {torch.cuda.get_device_name(0)}")


def main():
    args = parse_args()
    root_dir = Path(__file__).resolve().parents[1]
    input_dir = path_from_arg(args.input_dir)
    out_dir = path_from_arg(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_environment()
    print(f"input_dir: {input_dir}")
    print(f"out_dir: {out_dir}")
    if not input_dir.exists():
        print(f"ERROR: input_dir not found: {input_dir}", file=sys.stderr)
        raise SystemExit(1)

    discovery = find_exported_pairs(input_dir)
    pairs = discovery["pairs"]
    if args.max_samples is not None:
        pairs = pairs[: args.max_samples]
    print(f"discovered pair count: {len(pairs)}")
    print(f"unmatched images count: {len(discovery['unmatched_images'])}")
    print(f"unmatched json count: {len(discovery['unmatched_jsons'])}")

    run_config = {
        "input_dir": str(input_dir),
        "layoutlm_checkpoint": args.layoutlm_checkpoint,
        "relg_checkpoint": args.relg_checkpoint,
        "labels": args.labels,
        "out_dir": str(out_dir),
        "device": args.device,
        "local_files_only": args.local_files_only,
        "max_length": args.max_length,
        "max_samples": args.max_samples,
        "skip_model_inference": args.skip_model_inference,
        "unmatched_images": discovery["unmatched_images"],
        "unmatched_jsons": discovery["unmatched_jsons"],
    }
    save_json(out_dir / "run_config.json", run_config)

    rows = []
    staged_pairs = []
    valid_pairs = []
    for pair in pairs:
        image_path = Path(pair["image_path"])
        json_path = Path(pair["ocr_json_path"])
        validation = validate_ocr_json(json_path, image_path)
        warnings = list(pair.get("warnings", [])) + validation["warnings"]
        errors = list(validation["errors"])
        ocr_overlay_path = ""
        if validation.get("payload"):
            ocr_overlay_path = draw_ocr_box_overlay(
                image_path,
                validation["payload"],
                out_dir / "ocr_overlay" / f"{pair['capture_id']}_ocr_overlay.png",
                validation,
            )
        staged_image = ""
        staged_json = ""
        if validation["valid"]:
            staged_image, staged_json = copy_to_staging(pair, out_dir)
            valid_pairs.append(pair)
            staged_pairs.append(
                {
                    "capture_id": pair["capture_id"],
                    "image_path": str(image_path),
                    "ocr_json_path": str(json_path),
                    "staged_image_path": str(staged_image),
                    "staged_ocr_json_path": str(staged_json),
                    "structure": pair["structure"],
                }
            )
        row = {
            "capture_id": pair["capture_id"],
            "image_path": str(image_path),
            "ocr_json_path": str(json_path),
            "staged_image_path": str(staged_image),
            "staged_ocr_json_path": str(staged_json),
            "ocr_valid": validation["valid"],
            "image_width": validation["image_width"],
            "image_height": validation["image_height"],
            "json_image_width": validation["json_image_width"],
            "json_image_height": validation["json_image_height"],
            "word_count": validation["word_count"],
            "blank_text_count": validation["blank_text_count"],
            "invalid_box_count": validation["invalid_box_count"],
            "out_of_image_box_count": validation["out_of_image_box_count"],
            "ocr_overlay_path": ocr_overlay_path,
            "layoutlmv3_prediction_json": "",
            "layoutlmv3_overlay_path": "",
            "grouped_relg_json": "",
            "grouped_relg_overlay_path": "",
            "item_count": None,
            "warnings": warnings,
            "errors": errors,
        }
        rows.append(row)
        print(
            f"{pair['capture_id']}: image={image_path} json={json_path} "
            f"size={validation['image_width']}x{validation['image_height']} "
            f"json_size={validation['json_image_width']}x{validation['json_image_height']} "
            f"words={validation['word_count']} invalid_boxes={validation['invalid_box_count']} "
            f"overlay={ocr_overlay_path}"
        )

    save_json(out_dir / "staged_pairs.json", staged_pairs)

    layout_result = {"ok": False, "skipped": True}
    if args.skip_model_inference:
        print("skip_model_inference enabled; not running LayoutLMv3 or span rel-g.")
    elif not valid_pairs:
        print("No valid OCR pairs; not running LayoutLMv3 or span rel-g.")
    else:
        for required_path, label in [
            (path_from_arg(args.layoutlm_checkpoint), "LayoutLMv3 checkpoint"),
            (path_from_arg(args.labels), "labels.json"),
        ]:
            if not required_path.exists():
                message = f"{label} not found: {required_path}"
                print(f"ERROR: {message}", file=sys.stderr)
                for row in rows:
                    row["errors"].append(message)
                break
        else:
            staged_root = out_dir / "staged_user_receipts"
            layout_result = run_layoutlmv3_batch(args, staged_root, out_dir, root_dir)
            save_json(out_dir / "layoutlmv3_subprocess.json", layout_result)
            print(f"LayoutLMv3 prediction status: {'ok' if layout_result['ok'] else 'failed'}")
            if not layout_result["ok"]:
                print(layout_result.get("stderr") or layout_result.get("error"), file=sys.stderr)

    prediction_dir = out_dir / "layoutlmv3_predictions" / "predictions"
    overlay_dir = out_dir / "layoutlmv3_predictions" / "overlays"
    if layout_result.get("ok"):
        relg_checkpoint = path_from_arg(args.relg_checkpoint)
        relg_available = relg_checkpoint.exists()
        if not relg_available:
            message = f"span rel-g checkpoint not found: {relg_checkpoint}"
            print(f"ERROR: {message}", file=sys.stderr)
        for row in rows:
            capture_id = row["capture_id"]
            prediction_json = prediction_dir / f"{capture_id}_prediction.json"
            prediction_overlay = overlay_dir / f"{capture_id}_overlay.png"
            if prediction_json.exists():
                row["layoutlmv3_prediction_json"] = str(prediction_json)
            if prediction_overlay.exists():
                row["layoutlmv3_overlay_path"] = str(prediction_overlay)
            if prediction_json.exists() and relg_available:
                relg_result = run_relg_for_prediction(args, capture_id, prediction_json, out_dir, root_dir)
                save_json(out_dir / "grouped" / f"{capture_id}_relg_subprocess.json", relg_result)
                if relg_result["ok"] and Path(relg_result["out_json"]).exists():
                    grouped_info = validate_grouped_json(relg_result["out_json"])
                    row["grouped_relg_json"] = relg_result["out_json"]
                    row["grouped_relg_overlay_path"] = relg_result.get("out_overlay", "")
                    row["item_count"] = grouped_info["item_count"]
                    row["price_missing_item_count"] = grouped_info["price_missing_item_count"]
                    row["menu_name_missing_item_count"] = grouped_info["menu_name_missing_item_count"]
                    row["rel_g_prob_min"] = grouped_info["rel_g_prob_min"]
                    row["rel_g_prob_max"] = grouped_info["rel_g_prob_max"]
                    row["rel_g_prob_avg"] = grouped_info["rel_g_prob_avg"]
                    row["warnings"].extend(grouped_info["warnings"])
                else:
                    row["errors"].append(relg_result.get("stderr") or relg_result.get("error") or "rel-g failed")
            elif prediction_json.exists() and not relg_available:
                row["errors"].append(f"span rel-g checkpoint not found: {relg_checkpoint}")

    summary = {
        "input_dir": str(input_dir),
        "out_dir": str(out_dir),
        "discovered_pair_count": len(pairs),
        "unmatched_image_count": len(discovery["unmatched_images"]),
        "unmatched_json_count": len(discovery["unmatched_jsons"]),
        "schema_valid_count": sum(1 for row in rows if row["ocr_valid"]),
        "layoutlmv3_ran": bool(layout_result.get("ok")),
        "rows": rows,
    }
    for row in rows:
        row["warning_count"] = len(row.get("warnings", []))
        row["error_count"] = len(row.get("errors", []))
    save_json(out_dir / "test_summary.json", summary)
    write_csv_summary(out_dir / "test_summary.csv", rows)
    mapping_csv, mapping_md, mapping_rows = write_mapping_outputs(out_dir, rows)
    write_html_gallery(out_dir / "index.html", rows, out_dir)

    print(f"final summary path: {out_dir / 'test_summary.json'}")
    print(f"summary CSV path: {out_dir / 'test_summary.csv'}")
    print(f"menu-price mapping CSV path: {mapping_csv}")
    print(f"menu-price mapping Markdown path: {mapping_md}")
    print(f"menu-price mapping rows: {len(mapping_rows)}")
    print(f"HTML gallery path: {out_dir / 'index.html'}")
    print("Exported receipt ZIP folder test complete.")


if __name__ == "__main__":
    main()
