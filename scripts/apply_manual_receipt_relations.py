import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply manually inspected receipt relation decisions to labeled receipt JSON files."
    )
    parser.add_argument("--input_dir", required=True, help="Directory containing *_receipt_ocr folders.")
    parser.add_argument(
        "--decisions_jsonl",
        required=True,
        help="JSONL with manual relation decisions. This script does not infer relations.",
    )
    parser.add_argument("--backup_dir", default=None, help="Optional backup dir before writing labels.")
    parser.add_argument("--overlay_dir", default=None, help="Optional dir for BIO and relation overlays.")
    parser.add_argument(
        "--overlay_in_sample_dir",
        action="store_true",
        help="Write <capture_id>_overlay.jpg next to the image, combining BIO and relation overlays.",
    )
    parser.add_argument("--coordinate_mode", choices=["strict", "auto-scale"], default="strict")
    parser.add_argument("--overwrite_backup", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_decisions(path):
    decisions = {}
    with Path(path).open("r", encoding="utf-8-sig") as handle:
        for line_no, line in enumerate(handle, start=1):
            raw = line.strip()
            if not raw or raw.startswith("#"):
                continue
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                fail(f"{path}:{line_no}: invalid JSON: {exc}")
            capture_id = payload.get("capture_id") or payload.get("id")
            if not capture_id:
                fail(f"{path}:{line_no}: decision missing capture_id")
            decisions[str(capture_id)] = payload
    return decisions


def folder_paths(input_dir, capture_id):
    folder = Path(input_dir) / f"{capture_id}_receipt_ocr"
    image_path = folder / f"{capture_id}.jpg"
    init_path = folder / f"{capture_id}_init_labeled.json"
    v2_path = folder / f"{capture_id}_labeled_v2_1.json"
    return folder, image_path, init_path, v2_path


def parse_box(box):
    if not isinstance(box, list) or len(box) != 4:
        return None
    try:
        x0, y0, x1, y1 = [int(round(float(value))) for value in box]
    except (TypeError, ValueError):
        return None
    if x1 <= x0 or y1 <= y0:
        return None
    return [x0, y0, x1, y1]


def union_boxes(boxes):
    boxes = [box for box in boxes if box is not None]
    if not boxes:
        return None
    return [
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    ]


def normalize_indices(value):
    if value is None:
        return []
    if isinstance(value, int):
        return [value]
    if isinstance(value, str):
        try:
            return [int(value)]
        except ValueError:
            return []
    if isinstance(value, list):
        result = []
        for item in value:
            try:
                result.append(int(item))
            except (TypeError, ValueError):
                pass
        return result
    return []


def text_for_indices(words, indices):
    return " ".join(str(words[idx].get("text", "")) for idx in indices if 0 <= idx < len(words)).strip()


def field_for_indices(words, indices, fallback):
    for idx in indices:
        if 0 <= idx < len(words):
            field = words[idx].get("field")
            if field:
                return field
            label = str(words[idx].get("label", "O"))
            if label.startswith(("B-", "I-")):
                return label[2:]
    return fallback


def box_for_indices(words, indices):
    return union_boxes(parse_box(words[idx].get("box")) for idx in indices if 0 <= idx < len(words))


def ensure_indices_valid(words, indices, capture_id, role):
    if not indices:
        fail(f"{capture_id}: {role} word indices are empty")
    bad = [idx for idx in indices if idx < 0 or idx >= len(words)]
    if bad:
        fail(f"{capture_id}: {role} word indices out of range: {bad}, word_count={len(words)}")
    box = box_for_indices(words, indices)
    if box is None:
        fail(f"{capture_id}: {role} word indices have no valid box: {indices}")


def clear_relation_ids(words):
    for word in words:
        if not isinstance(word, dict):
            continue
        word["relation_ids_as_head"] = []
        word["relation_ids_as_tail"] = []
        word["rel_g_edge_ids_as_head"] = []
        word["rel_g_edge_ids_as_tail"] = []


def add_relation_to_words(words, relation):
    rid = relation["relation_id"]
    for idx in relation["head_word_indices"]:
        words[idx].setdefault("relation_ids_as_head", []).append(rid)
    for idx in relation["tail_word_indices"]:
        words[idx].setdefault("relation_ids_as_tail", []).append(rid)


def add_edge_to_words(words, edge):
    rid = edge["relation_id"]
    for idx in edge["head_word_indices"]:
        words[idx].setdefault("rel_g_edge_ids_as_head", []).append(rid)
    for idx in edge["dep_word_indices"]:
        words[idx].setdefault("rel_g_edge_ids_as_tail", []).append(rid)


def relation_from_decision(words, item, relation_id, default_type="item_attribute"):
    head_indices = normalize_indices(item.get("head_word_indices"))
    tail_indices = normalize_indices(item.get("tail_word_indices") or item.get("dep_word_indices"))
    capture_id = item.get("_capture_id", "<unknown>")
    ensure_indices_valid(words, head_indices, capture_id, "head")
    ensure_indices_valid(words, tail_indices, capture_id, "tail")
    head_field = item.get("head_field") or field_for_indices(words, head_indices, "ITEM_NAME")
    tail_field = item.get("tail_field") or item.get("dep_field") or field_for_indices(words, tail_indices, "ITEM_PRICE")
    relation = {
        "relation_id": relation_id,
        "relation_type": item.get("relation_type") or default_type,
        "head_field": head_field,
        "head_word_indices": head_indices,
        "head_text": item.get("head_text") or text_for_indices(words, head_indices),
        "head_box": box_for_indices(words, head_indices),
        "tail_field": tail_field,
        "tail_word_indices": tail_indices,
        "tail_text": item.get("tail_text") or item.get("dep_text") or text_for_indices(words, tail_indices),
        "tail_box": box_for_indices(words, tail_indices),
    }
    if item.get("group_id") is not None:
        relation["group_id"] = item["group_id"]
    if item.get("head_span_id") is not None:
        relation["head_span_id"] = item["head_span_id"]
    if item.get("tail_span_id") is not None:
        relation["tail_span_id"] = item["tail_span_id"]
    return relation


def relg_edge_from_relation(relation, edge_id):
    return {
        "relation_id": edge_id,
        "relation_type": "rel_g_positive",
        "group_id": relation.get("group_id"),
        "head_field": relation["head_field"],
        "head_word_indices": relation["head_word_indices"],
        "head_text": relation["head_text"],
        "head_span_id": relation.get("head_span_id"),
        "dep_field": relation["tail_field"],
        "dep_word_indices": relation["tail_word_indices"],
        "dep_text": relation["tail_text"],
        "dep_span_id": relation.get("tail_span_id"),
    }


def apply_decision(payload, decision):
    words = payload.get("words")
    if not isinstance(words, list):
        fail(f"{decision.get('capture_id')}: label JSON missing words list")
    clear_relation_ids(words)
    capture_id = str(decision.get("capture_id"))
    raw_items = []
    for key in ("item_relations", "summary_relations", "payment_relations", "relations"):
        for item in decision.get(key, []) or []:
            item = dict(item)
            item["_source_key"] = key
            item["_capture_id"] = capture_id
            raw_items.append(item)

    relations = []
    item_relations = []
    summary_relations = []
    payment_relations = []
    rel_g_edges = []
    for idx, item in enumerate(raw_items):
        source_key = item.get("_source_key")
        default_type = {
            "item_relations": "item_attribute",
            "summary_relations": "summary_amount",
            "payment_relations": "payment_attribute",
        }.get(source_key, item.get("relation_type") or "item_attribute")
        relation = relation_from_decision(words, item, f"r_{idx:06d}", default_type=default_type)
        relations.append(relation)
        if source_key == "summary_relations" or relation["relation_type"] in {"summary_amount", "tax_amount"}:
            summary_relations.append(relation)
        elif source_key == "payment_relations" or relation["relation_type"].startswith("payment"):
            payment_relations.append(relation)
        else:
            item_relations.append(relation)
        add_relation_to_words(words, relation)
        if relation["relation_type"] == "item_attribute" and relation["head_field"] == "ITEM_NAME":
            edge = relg_edge_from_relation(relation, f"g_{len(rel_g_edges):06d}")
            rel_g_edges.append(edge)
            add_edge_to_words(words, edge)

    payload["relations"] = relations
    payload["item_relations"] = item_relations
    payload["summary_relations"] = summary_relations
    payload["payment_relations"] = payment_relations
    payload["rel_g_edges"] = rel_g_edges
    payload["manual_relation_labeling"] = {
        "source": "human_visual_inspection_by_codex",
        "note": "Relations were manually specified from image/label review; no trained model inference was used.",
        "decision_notes": decision.get("notes", []),
    }
    return {
        "relation_count": len(relations),
        "item_relation_count": len(item_relations),
        "summary_relation_count": len(summary_relations),
        "payment_relation_count": len(payment_relations),
        "rel_g_edge_count": len(rel_g_edges),
    }


def backup_file(path, input_dir, backup_dir, overwrite):
    if not path.exists() or backup_dir is None:
        return
    relative = path.relative_to(input_dir)
    target = backup_dir / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and not overwrite:
        return
    shutil.copy2(path, target)


def run_overlay(script, args):
    cmd = [sys.executable, str(script)] + args
    result = subprocess.run(cmd, text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"overlay command failed: {' '.join(cmd)}\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def fit_width(image, width):
    if image.width <= width:
        return image
    ratio = width / image.width
    return image.resize((width, max(1, int(round(image.height * ratio)))))


def combine_overlay_images(bio_path, rel_path, out_path, max_panel_width=1300):
    left = fit_width(Image.open(bio_path).convert("RGB"), max_panel_width)
    right = fit_width(Image.open(rel_path).convert("RGB"), max_panel_width)
    width = left.width + right.width
    height = max(left.height, right.height)
    canvas = Image.new("RGB", (width, height), "white")
    canvas.paste(left, (0, 0))
    canvas.paste(right, (left.width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path, quality=90)


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    decisions = read_decisions(args.decisions_jsonl)
    if not input_dir.exists():
        fail(f"input_dir not found: {input_dir}")
    backup_dir = Path(args.backup_dir) if args.backup_dir else None
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else None
    root = Path(__file__).resolve().parents[1]
    bio_overlay_script = root / "scripts" / "overlay_labeled_receipt_json.py"
    rel_overlay_script = root / "scripts" / "overlay_labeled_relations.py"

    summary = {
        "input_dir": str(input_dir),
        "decisions_jsonl": str(args.decisions_jsonl),
        "backup_dir": str(backup_dir) if backup_dir else None,
        "overlay_dir": str(overlay_dir) if overlay_dir else None,
        "processed": [],
        "errors": [],
    }
    for capture_id, decision in decisions.items():
        try:
            folder, image_path, init_path, v2_path = folder_paths(input_dir, capture_id)
            if not folder.exists():
                fail(f"{capture_id}: folder not found: {folder}")
            if not image_path.exists():
                fail(f"{capture_id}: image not found: {image_path}")
            label_path = v2_path if v2_path.exists() else init_path
            if not label_path.exists():
                fail(f"{capture_id}: no label JSON found")
            payload = load_json(label_path)
            metrics = apply_decision(payload, decision)
            if not args.dry_run:
                if backup_dir:
                    backup_file(init_path, input_dir, backup_dir, args.overwrite_backup)
                    backup_file(v2_path, input_dir, backup_dir, args.overwrite_backup)
                if init_path.exists():
                    save_json(init_path, payload)
                if v2_path.exists():
                    save_json(v2_path, payload)
                if overlay_dir or args.overlay_in_sample_dir:
                    cleanup_temp_overlay_dir = False
                    if overlay_dir:
                        sample_overlay_dir = overlay_dir / capture_id
                    else:
                        sample_overlay_dir = folder / ".overlay_tmp"
                        cleanup_temp_overlay_dir = True
                    sample_overlay_dir.mkdir(parents=True, exist_ok=True)
                    bio_out = sample_overlay_dir / f"{capture_id}_bio_overlay.png"
                    bio_summary = sample_overlay_dir / f"{capture_id}_bio_summary.json"
                    rel_out = sample_overlay_dir / f"{capture_id}_relations_overlay.png"
                    rel_summary = sample_overlay_dir / f"{capture_id}_relations_summary.json"
                    run_overlay(
                        bio_overlay_script,
                        [
                            "--image",
                            str(image_path),
                            "--label_json",
                            str(label_path),
                            "--out",
                            str(bio_out),
                            "--summary_out",
                            str(bio_summary),
                            "--coordinate_mode",
                            args.coordinate_mode,
                            "--show_text",
                        ],
                    )
                    run_overlay(
                        rel_overlay_script,
                        [
                            "--image",
                            str(image_path),
                            "--label_json",
                            str(label_path),
                            "--out",
                            str(rel_out),
                            "--summary_out",
                            str(rel_summary),
                            "--coordinate_mode",
                            args.coordinate_mode,
                            "--relation_source",
                            "all",
                        ],
                    )
                    metrics["bio_overlay"] = str(bio_out)
                    metrics["relations_overlay"] = str(rel_out)
                    if args.overlay_in_sample_dir:
                        combined_out = folder / f"{capture_id}_overlay.jpg"
                        combine_overlay_images(bio_out, rel_out, combined_out)
                        metrics["combined_overlay"] = str(combined_out)
                    if cleanup_temp_overlay_dir:
                        shutil.rmtree(sample_overlay_dir, ignore_errors=True)
            metrics["capture_id"] = capture_id
            summary["processed"].append(metrics)
            print(f"applied {capture_id}: {metrics}")
        except Exception as exc:
            summary["errors"].append({"capture_id": capture_id, "error": repr(exc)})
            print(f"ERROR {capture_id}: {exc}", file=sys.stderr)

    if overlay_dir and not args.dry_run:
        save_json(overlay_dir / "manual_relation_apply_summary.json", summary)
    print(f"processed={len(summary['processed'])} errors={len(summary['errors'])}")
    if summary["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
