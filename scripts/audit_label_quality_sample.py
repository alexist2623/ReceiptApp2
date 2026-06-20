import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="Sample labeled receipts and build review overlays/contact sheets.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--out_dir", default="outputs/label_quality_audit_100")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--exclude_dir_name", default="Temp")
    parser.add_argument("--include_augmented", action="store_true", default=True)
    parser.add_argument("--tile_width", type=int, default=720)
    parser.add_argument("--tile_height", type=int, default=960)
    return parser.parse_args()


def collect_labels(input_dir, exclude_dir_name):
    root = Path(input_dir)
    labels = []
    for path in root.rglob("*_labeled_v2_1.json"):
        parts = set(path.parts)
        if exclude_dir_name and exclude_dir_name in parts:
            continue
        capture_id = path.name[: -len("_labeled_v2_1.json")]
        image = path.parent / f"{capture_id}.jpg"
        if not image.exists():
            candidates = list(path.parent.glob("*.jpg")) + list(path.parent.glob("*.jpeg")) + list(path.parent.glob("*.png"))
            image = candidates[0] if candidates else None
        if image is None or not image.exists():
            continue
        base_id = capture_id.split("_aug_")[0]
        labels.append(
            {
                "id": capture_id,
                "base_id": base_id,
                "is_augmented": "_aug_" in capture_id,
                "image": str(image),
                "label_json": str(path),
            }
        )
    labels.sort(key=lambda item: (item["base_id"], item["is_augmented"], item["id"]))
    return labels


def balanced_sample(labels, num_samples):
    originals = [item for item in labels if not item["is_augmented"]]
    augmented_by_base = {}
    for item in labels:
        if item["is_augmented"]:
            augmented_by_base.setdefault(item["base_id"], []).append(item)

    sample = []
    seen = set()
    for item in originals:
        sample.append(item)
        seen.add(item["id"])
        if len(sample) >= num_samples:
            return sample[:num_samples]

    bases = sorted(augmented_by_base)
    offset = 0
    while len(sample) < num_samples:
        added = False
        for base in bases:
            items = augmented_by_base.get(base, [])
            if offset < len(items) and items[offset]["id"] not in seen:
                sample.append(items[offset])
                seen.add(items[offset]["id"])
                added = True
                if len(sample) >= num_samples:
                    return sample[:num_samples]
        if not added:
            break
        offset += 1

    for item in labels:
        if item["id"] not in seen:
            sample.append(item)
            seen.add(item["id"])
            if len(sample) >= num_samples:
                break
    return sample[:num_samples]


def run_overlay(sample, out_dir):
    bio_dir = out_dir / "bio_overlays"
    relation_dir = out_dir / "relation_overlays"
    summary_dir = out_dir / "summaries"
    for directory in (bio_dir, relation_dir, summary_dir):
        directory.mkdir(parents=True, exist_ok=True)

    failures = []
    for index, item in enumerate(sample):
        prefix = f"{index:03d}_{item['id']}"
        commands = [
            [
                "python",
                "scripts/overlay_labeled_receipt_json.py",
                "--image",
                item["image"],
                "--label_json",
                item["label_json"],
                "--out",
                str(bio_dir / f"{prefix}_bio.png"),
                "--summary_out",
                str(summary_dir / f"{prefix}_bio_summary.json"),
                "--coordinate_mode",
                "strict",
                "--show_text",
                "--draw_legend",
            ],
            [
                "python",
                "scripts/overlay_labeled_relations.py",
                "--image",
                item["image"],
                "--label_json",
                item["label_json"],
                "--relation_source",
                "all",
                "--coordinate_mode",
                "strict",
                "--out",
                str(relation_dir / f"{prefix}_rel.png"),
                "--summary_out",
                str(summary_dir / f"{prefix}_rel_summary.json"),
            ],
        ]
        for command in commands:
            result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            if result.returncode:
                failures.append(
                    {
                        "id": item["id"],
                        "command": command,
                        "stdout_tail": result.stdout[-1200:],
                        "stderr_tail": result.stderr[-1200:],
                    }
                )
    return failures


def make_contact_sheets(src_dir, out_dir, prefix, tile_width, tile_height, cols=5, rows=2):
    font = ImageFont.load_default()
    files = sorted(src_dir.glob("*.png"))
    per_sheet = cols * rows
    sheets = []
    for sheet_index in range((len(files) + per_sheet - 1) // per_sheet):
        subset = files[sheet_index * per_sheet : (sheet_index + 1) * per_sheet]
        sheet = Image.new("RGB", (cols * tile_width, rows * tile_height), (245, 245, 245))
        draw = ImageDraw.Draw(sheet)
        for tile_index, path in enumerate(subset):
            image = Image.open(path).convert("RGB")
            image.thumbnail((tile_width, tile_height - 28), Image.LANCZOS)
            x0 = (tile_index % cols) * tile_width
            y0 = (tile_index // cols) * tile_height
            sheet.paste(image, (x0 + (tile_width - image.width) // 2, y0 + 26))
            draw.rectangle([x0, y0, x0 + tile_width - 1, y0 + tile_height - 1], outline=(80, 80, 80), width=2)
            draw.text((x0 + 4, y0 + 5), path.name[:100], fill=(0, 0, 0), font=font)
        out_path = out_dir / f"{prefix}_sheet_{sheet_index + 1:02d}.jpg"
        sheet.save(out_path, quality=92)
        sheets.append(str(out_path))
    return sheets


def summarize(sample, out_dir):
    label_counts = Counter()
    relation_counts = Counter()
    issues = []
    skipped_relations = 0
    summary_dir = out_dir / "summaries"
    for index, item in enumerate(sample):
        with Path(item["label_json"]).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        words = payload.get("words") or []
        relations = payload.get("relations") or []
        if not relations:
            for key in ("item_relations", "summary_relations", "payment_relations", "rel_g_edges"):
                relations.extend(payload.get(key) or [])

        item_name_words = 0
        item_price_words = 0
        for word in words:
            label = str(word.get("label") or word.get("bio_label") or "O")
            label_counts[label] += 1
            if "ITEM_NAME" in label:
                item_name_words += 1
            if "ITEM_PRICE" in label:
                item_price_words += 1
        for relation in relations:
            relation_type = relation.get("relation_type") or relation.get("type") or ("rel_g_edge" if "dep_span_id" in relation else "unknown")
            relation_counts[relation_type] += 1

        if item_price_words and not item_name_words:
            issues.append({"id": item["id"], "type": "has_item_price_no_item_name", "item_price_words": item_price_words})
        if item_name_words and not item_price_words:
            issues.append({"id": item["id"], "type": "has_item_name_no_item_price", "item_name_words": item_name_words})
        if not relations:
            issues.append({"id": item["id"], "type": "no_relations"})

        relation_summary = summary_dir / f"{index:03d}_{item['id']}_rel_summary.json"
        if relation_summary.exists():
            try:
                skipped_relations += int(json.loads(relation_summary.read_text(encoding="utf-8")).get("skipped_relation_count") or 0)
            except Exception:
                pass

    return {
        "num_sampled": len(sample),
        "original_count_in_sample": sum(not item["is_augmented"] for item in sample),
        "augmented_count_in_sample": sum(item["is_augmented"] for item in sample),
        "label_counts_top": label_counts.most_common(50),
        "relation_counts": relation_counts.most_common(),
        "skipped_relation_count_from_overlay": skipped_relations,
        "num_programmatic_issues": len(issues),
        "programmatic_issues": issues[:300],
    }


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = collect_labels(args.input_dir, args.exclude_dir_name)
    sample = balanced_sample(labels, args.num_samples)
    (out_dir / "sample_manifest.json").write_text(
        json.dumps({"num_available": len(labels), "num_sampled": len(sample), "samples": sample}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    failures = run_overlay(sample, out_dir)
    (out_dir / "overlay_failures.json").write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")
    bio_sheets = make_contact_sheets(out_dir / "bio_overlays", out_dir, "bio", args.tile_width, args.tile_height)
    relation_sheets = make_contact_sheets(out_dir / "relation_overlays", out_dir, "relations", args.tile_width, args.tile_height)
    summary = summarize(sample, out_dir)
    summary.update({"num_available": len(labels), "overlay_failure_count": len(failures), "bio_sheets": bio_sheets, "relation_sheets": relation_sheets})
    (out_dir / "audit_programmatic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
