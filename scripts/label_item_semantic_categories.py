#!/usr/bin/env python
"""Attach semantic item categories to receipt training data.

This script does not rewrite BIO labels. It adds category metadata that can be
used later by an item-category classifier.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


TAXONOMY = [
    "FOOD",
    "DRINK",
    "GROCERY",
    "SNACK",
    "ALCOHOL",
    "HOUSEHOLD",
    "PERSONAL_CARE",
    "HEALTH",
    "CLOTHING",
    "ELECTRONICS",
    "TRANSPORT",
    "SERVICE",
    "ENTERTAINMENT",
    "TAX_FEE",
    "OTHER",
]

MANUAL_OVERRIDES: dict[str, str] = {}


# Ordered from narrow/high-confidence classes to broader fallback classes.
CATEGORY_RULES: list[tuple[str, float, list[str]]] = [
    (
        "TAX_FEE",
        0.95,
        [
            "TAX",
            "FEE",
            "DEPOSIT",
            "CONTAINER DEPOSIT",
            "ENVIRO",
            "ENVIRONMENT",
            "SURCHARGE",
            "SERVICE CHARGE",
            "CUP FEE",
            "BAG FEE",
            "BOTTLE DEPOSIT",
        ],
    ),
    (
        "TRANSPORT",
        0.95,
        [
            "TRANSIT",
            "BUS",
            "TRAIN",
            "SUBWAY",
            "SKYTRAIN",
            "ZONE MONTHLY PASS",
            "MONTHLY PASS",
            "PARKING",
            "TAXI",
            "UBER",
            "LYFT",
            "FARE",
        ],
    ),
    (
        "ALCOHOL",
        0.95,
        [
            "BEER",
            "WINE",
            "SAKE",
            "SOJU",
            "VODKA",
            "WHISKY",
            "WHISKEY",
            "TEQUILA",
            "RUM",
            "GIN",
            "LAGER",
            "ALE",
            "IPA",
            "SAPPORO",
            "CORONA",
            "STELLA",
            "MOJITO",
            "MIMOSA",
            "BLOODY MARY",
            "TALL CAN",
            "CIDER",
        ],
    ),
    (
        "PERSONAL_CARE",
        0.9,
        [
            "SHAMPOO",
            "CONDITIONER",
            "SOAP",
            "BODY WASH",
            "LOTION",
            "SKINCARE",
            "MAKEUP",
            "MAK",
            "TOOTHPASTE",
            "TOOTHBRUSH",
            "ORAL B",
            "CREST",
            "DOVE",
            "DEODORANT",
            "RAZOR",
            "COSMETIC",
        ],
    ),
    (
        "HEALTH",
        0.9,
        [
            "VITAMIN",
            "MEDICINE",
            "PHARMACY",
            "ADVIL",
            "TYLENOL",
            "ASPIRIN",
            "SUPPLEMENT",
            "BANDAGE",
            "MASK",
            "FIRST AID",
        ],
    ),
    (
        "HOUSEHOLD",
        0.9,
        [
            "NAPKIN",
            "TISSUE",
            "PAPER TOWEL",
            "BOUNTY",
            "PUREX",
            "CASCADE",
            "LYSOL",
            "WIPE",
            "DETERGENT",
            "CLEANER",
            "LAUNDRY",
            "DISH SOAP",
            "GARBAGE BAG",
            "FOIL",
            "BATTERY",
        ],
    ),
    (
        "ELECTRONICS",
        0.9,
        [
            "PHONE",
            "CHARGER",
            "CABLE",
            "USB",
            "HDMI",
            "EARPHONE",
            "HEADPHONE",
            "LAPTOP",
            "KEYBOARD",
            "MOUSE",
            "SCREEN",
            "CAMERA",
            "SD CARD",
            "TECH",
        ],
    ),
    (
        "CLOTHING",
        0.9,
        [
            "SHIRT",
            "TSHIRT",
            "T-SHIRT",
            "PANTS",
            "JEANS",
            "DRESS",
            "SHOE",
            "SOCK",
            "JACKET",
            "COAT",
            "HAT",
            "BAG",
            "HANDBAG",
            "SHOULDER BAG",
            "SHOULDERBAG",
            "BELT",
        ],
    ),
    (
        "ENTERTAINMENT",
        0.9,
        [
            "MOVIE",
            "CINEMA",
            "TICKET",
            "GAME",
            "TOY",
            "BOOK",
            "MUSIC",
            "CONCERT",
            "MUSEUM",
            "ARCADE",
        ],
    ),
    (
        "DRINK",
        0.88,
        [
            "COFFEE",
            "LATTE",
            "MOCHA",
            "AMERICANO",
            "CAPPUCCINO",
            "ESPRESSO",
            "TEA",
            "COKE",
            "COCA COLA",
            "PEPSI",
            "SODA",
            "WATER",
            "JUICE",
            "LEMONADE",
            "MILK",
            "SMOOTHIE",
            "DRINK",
            "BEVERAGE",
            "FLAT WHITE",
            "MACCHIATO",
            "MATCHA",
        ],
    ),
    (
        "SNACK",
        0.86,
        [
            "CHIP",
            "CHIPS",
            "CHOCOLATE",
            "CANDY",
            "COOKIE",
            "COOKIES",
            "WAFER",
            "WAFFLE",
            "TIRAMISU",
            "CHEESECAKE",
            "CAKE",
            "PIE",
            "DESSERT",
            "ICE CREAM",
            "NACHO",
            "POPCORN",
            "DONUT",
        ],
    ),
    (
        "GROCERY",
        0.84,
        [
            "BANANA",
            "BANANAS",
            "PINEAPPLE",
            "APPLE",
            "ORANGE",
            "DATE",
            "DATES",
            "ALMOND",
            "PEANUT",
            "YOGURT",
            "YOG",
            "YGRT",
            "EGG",
            "EGGS",
            "BREAD",
            "CEREAL",
            "FLOUR",
            "SUGAR",
            "RICE BAG",
            "GARLIC",
            "ONION",
            "PRODUCE",
            "VEGETABLE",
            "FRUIT",
            "GROCERY",
        ],
    ),
    (
        "FOOD",
        0.82,
        [
            "CHICKEN",
            "PORK",
            "BEEF",
            "STEAK",
            "BURGER",
            "FRIES",
            "PIZZA",
            "RICE",
            "NOODLE",
            "SOUP",
            "SALAD",
            "SANDWICH",
            "CROISSANT",
            "BUN",
            "SAUSAGE",
            "FISH",
            "SEAFOOD",
            "SUSHI",
            "TACO",
            "GUAC",
            "NACHOS",
            "NAAN",
            "CURRY",
            "PLATTER",
            "BACON",
            "DUCK",
            "MEAL",
            "FOOD",
        ],
    ),
    (
        "SERVICE",
        0.8,
        [
            "SERVICE",
            "DELIVERY",
            "INSTALLATION",
            "REPAIR",
            "MEMBERSHIP",
            "SUBSCRIPTION",
            "PREPAY",
        ],
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attach semantic item categories to labeled receipt JSON and BIO JSONL data."
    )
    parser.add_argument(
        "--custom_root",
        default=str(Path.home() / "OneDrive" / "APK_Receipt2"),
        help="Root containing custom *_receipt_ocr folders.",
    )
    parser.add_argument(
        "--wildreceipt_root",
        default=str(Path("..") / "receipt_training_data3" / "wildreceipt" / "wildreceipt_custom_structure"),
        help="Root containing converted WildReceipt *_receipt_ocr folders.",
    )
    parser.add_argument(
        "--bio_dirs",
        nargs="*",
        default=[
            "processed_data/custom_rotated_receipt_v2_bio",
            "processed_data/wildreceipt_rotated_receipt_v2_bio",
            "processed_data/cord_bio",
        ],
        help="BIO JSONL directories to annotate in-place.",
    )
    parser.add_argument(
        "--summary_out",
        default="outputs/item_category_labeling/item_category_labeling_summary.json",
    )
    parser.add_argument(
        "--manual_overrides",
        default="schemas/item_semantic_category_manual_overrides.json",
        help="Optional manually reviewed normalized item text -> category overrides.",
    )
    parser.add_argument(
        "--backup_dir",
        default=None,
        help="Backup directory. Defaults to outputs/item_category_labeling_backups/<timestamp>.",
    )
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_backup", action="store_true")
    parser.add_argument(
        "--exclude_dir_name",
        action="append",
        default=["Temp"],
        help="Directory name to skip; can be passed more than once.",
    )
    return parser.parse_args()


def normalize_text(text: str) -> str:
    value = str(text or "")
    value = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", value)
    value = value.replace("&", " AND ")
    value = re.sub(r"[^0-9A-Za-z]+", " ", value)
    value = re.sub(r"\s+", " ", value).strip().upper()
    return value


def load_manual_overrides(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    overrides = payload.get("overrides", payload)
    if not isinstance(overrides, dict):
        return {}
    return {normalize_text(key): str(value) for key, value in overrides.items() if str(value) in TAXONOMY}


def categorize_item_text(text: str) -> dict[str, Any]:
    normalized = normalize_text(text)
    if not normalized:
        return {
            "category": "OTHER",
            "confidence": 0.0,
            "rule": "empty_text",
            "normalized_text": normalized,
        }
    override = MANUAL_OVERRIDES.get(normalized)
    if override:
        return {
            "category": override,
            "confidence": 1.0,
            "rule": "manual_override",
            "normalized_text": normalized,
        }

    padded = f" {normalized} "
    compact_normalized = normalized.replace(" ", "")
    for category, confidence, keywords in CATEGORY_RULES:
        for keyword in keywords:
            key = normalize_text(keyword)
            if not key:
                continue
            # Prefer phrase/token boundaries. Short substrings such as FEE,
            # ALE, HAT, TECH, TRAIN, and LATTE create noisy labels when matched
            # inside COFFEE, TAMALES, HATER, WHITECHEESE, PULLBACKTRAINSET, or
            # PLATTER. For OCR-concatenated item names, allow compact substring
            # matching only for longer keywords.
            whole_phrase_match = f" {key} " in padded
            compact_key = key.replace(" ", "")
            compact_substring_match = len(compact_key) >= 6 and compact_key in compact_normalized
            if whole_phrase_match or compact_substring_match:
                return {
                    "category": category,
                    "confidence": confidence,
                    "rule": keyword,
                    "normalized_text": normalized,
                }

    return {
        "category": "OTHER",
        "confidence": 0.35,
        "rule": "fallback_other",
        "normalized_text": normalized,
    }


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def copy_backup(path: Path, backup_root: Path, source_root: Path | None = None) -> None:
    try:
        rel = path.relative_to(source_root) if source_root else path
    except ValueError:
        rel = Path(path.drive.replace(":", "")) / Path(*path.parts[1:])
    out = backup_root / rel
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, out)


def label_from_word(word: dict[str, Any]) -> str:
    label = word.get("label") or word.get("field") or ""
    if label.startswith("B-") or label.startswith("I-"):
        return label[2:]
    return label


def text_for_indices(words: list[dict[str, Any]], indices: list[int]) -> str:
    return " ".join(str(words[idx].get("text", "")).strip() for idx in indices if 0 <= idx < len(words)).strip()


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


def unique_annotation_key(indices: list[int]) -> tuple[int, ...]:
    return tuple(sorted(set(indices)))


def filter_item_name_indices(words: list[Any], indices: list[int]) -> list[int]:
    return [
        idx
        for idx in indices
        if 0 <= idx < len(words)
        and isinstance(words[idx], dict)
        and label_from_word(words[idx]) == "ITEM_NAME"
    ]


def clear_labeled_json_category_metadata(obj: dict[str, Any]) -> None:
    word_keys = (
        "semantic_item_category",
        "semantic_item_category_confidence",
        "semantic_item_category_rule",
    )
    words = obj.get("words") if isinstance(obj.get("words"), list) else []
    for word in words:
        if isinstance(word, dict):
            for key in word_keys:
                word.pop(key, None)

    for group in obj.get("item_groups") or []:
        if isinstance(group, dict):
            for key in word_keys:
                group.pop(key, None)

    for relation_key in ("relations", "item_relations", "rel_g_edges"):
        for relation in obj.get(relation_key) or []:
            if isinstance(relation, dict):
                for key in (
                    "head_semantic_item_category",
                    "head_semantic_item_category_confidence",
                    "head_semantic_item_category_rule",
                ):
                    relation.pop(key, None)

    for span in obj.get("spans") or []:
        if isinstance(span, dict):
            for key in word_keys:
                span.pop(key, None)


def collect_item_annotations_from_labeled(obj: dict[str, Any]) -> list[dict[str, Any]]:
    words = obj.get("words") if isinstance(obj.get("words"), list) else []
    annotations: dict[tuple[int, ...], dict[str, Any]] = {}

    item_groups = obj.get("item_groups")
    if isinstance(item_groups, list):
        for group in item_groups:
            indices = safe_indices(group.get("head_word_indices") or group.get("item_name_word_indices"))
            if not indices:
                fields = group.get("fields") if isinstance(group.get("fields"), dict) else {}
                indices = safe_indices(fields.get("ITEM_NAME"))
            indices = filter_item_name_indices(words, indices)
            if not indices:
                continue
            text = group.get("head_text") or group.get("expected_item_name") or text_for_indices(words, indices)
            key = unique_annotation_key(indices)
            annotations[key] = {
                "source": "item_groups",
                "group_id": group.get("group_id"),
                "item_name_text": text,
                "item_name_word_indices": indices,
            }

    relation_lists = []
    for key in ("item_relations", "relations", "rel_g_edges"):
        value = obj.get(key)
        if isinstance(value, list):
            relation_lists.extend((key, relation) for relation in value)

    for source, relation in relation_lists:
        head_field = relation.get("head_field")
        if head_field != "ITEM_NAME":
            continue
        indices = safe_indices(relation.get("head_word_indices"))
        indices = filter_item_name_indices(words, indices)
        if not indices:
            continue
        text = relation.get("head_text") or text_for_indices(words, indices)
        key = unique_annotation_key(indices)
        annotations.setdefault(
            key,
            {
                "source": source,
                "group_id": relation.get("group_id"),
                "item_name_text": text,
                "item_name_word_indices": indices,
            },
        )

    if not annotations:
        current: list[int] = []
        for idx, word in enumerate(words):
            field = label_from_word(word)
            label = word.get("label", "")
            if field == "ITEM_NAME":
                if label.startswith("B-") and current:
                    text = text_for_indices(words, current)
                    annotations[unique_annotation_key(current)] = {
                        "source": "words",
                        "group_id": None,
                        "item_name_text": text,
                        "item_name_word_indices": current,
                    }
                    current = []
                current.append(idx)
            elif current:
                text = text_for_indices(words, current)
                annotations[unique_annotation_key(current)] = {
                    "source": "words",
                    "group_id": None,
                    "item_name_text": text,
                    "item_name_word_indices": current,
                }
                current = []
        if current:
            text = text_for_indices(words, current)
            annotations[unique_annotation_key(current)] = {
                "source": "words",
                "group_id": None,
                "item_name_text": text,
                "item_name_word_indices": current,
            }

    return list(annotations.values())


def attach_labeled_json_categories(obj: dict[str, Any]) -> dict[str, Any]:
    words = obj.get("words") if isinstance(obj.get("words"), list) else []
    clear_labeled_json_category_metadata(obj)
    annotations = collect_item_annotations_from_labeled(obj)
    category_counts = Counter()

    for annotation in annotations:
        result = categorize_item_text(annotation.get("item_name_text", ""))
        annotation.update(
            {
                "category": result["category"],
                "confidence": result["confidence"],
                "rule": result["rule"],
                "method": "keyword_rules_v1",
            }
        )
        category_counts[result["category"]] += 1

        for idx in annotation.get("item_name_word_indices", []):
            if 0 <= idx < len(words) and isinstance(words[idx], dict):
                words[idx]["semantic_item_category"] = result["category"]
                words[idx]["semantic_item_category_confidence"] = result["confidence"]
                words[idx]["semantic_item_category_rule"] = result["rule"]

    by_key = {unique_annotation_key(a["item_name_word_indices"]): a for a in annotations}
    item_groups = obj.get("item_groups")
    if isinstance(item_groups, list):
        for group in item_groups:
            indices = safe_indices(group.get("head_word_indices") or group.get("item_name_word_indices"))
            if not indices:
                fields = group.get("fields") if isinstance(group.get("fields"), dict) else {}
                indices = safe_indices(fields.get("ITEM_NAME"))
            annotation = by_key.get(unique_annotation_key(indices))
            if annotation:
                group["semantic_item_category"] = annotation["category"]
                group["semantic_item_category_confidence"] = annotation["confidence"]
                group["semantic_item_category_rule"] = annotation["rule"]

    for key in ("relations", "item_relations", "rel_g_edges"):
        value = obj.get(key)
        if not isinstance(value, list):
            continue
        for relation in value:
            if relation.get("head_field") != "ITEM_NAME":
                continue
            annotation = by_key.get(unique_annotation_key(safe_indices(relation.get("head_word_indices"))))
            if annotation:
                relation["head_semantic_item_category"] = annotation["category"]
                relation["head_semantic_item_category_confidence"] = annotation["confidence"]

    spans = obj.get("spans")
    if isinstance(spans, list):
        for span in spans:
            if span.get("field") != "ITEM_NAME":
                continue
            annotation = by_key.get(unique_annotation_key(safe_indices(span.get("word_indices"))))
            if annotation:
                span["semantic_item_category"] = annotation["category"]
                span["semantic_item_category_confidence"] = annotation["confidence"]

    obj["item_semantic_category_taxonomy"] = TAXONOMY
    obj["item_semantic_category_labeling"] = {
        "schema_version": "item_semantic_categories_v1",
        "method": "keyword_rules_v1",
        "script": "scripts/label_item_semantic_categories.py",
    }
    obj["item_category_annotations"] = annotations
    obj["item_semantic_category_counts"] = dict(sorted(category_counts.items()))
    return obj


def field_from_bio_label(label: str) -> tuple[str, str]:
    if not isinstance(label, str):
        return "", ""
    if label.startswith("B-") or label.startswith("I-"):
        return label[:1], label[2:]
    return "", label


def collect_item_name_spans(labels: list[str]) -> list[list[int]]:
    spans: list[list[int]] = []
    current: list[int] = []
    for idx, label in enumerate(labels):
        prefix, field = field_from_bio_label(label)
        is_item = field in {"ITEM_NAME", "MENU_NM"}
        if is_item:
            if prefix == "B" and current:
                spans.append(current)
                current = []
            current.append(idx)
        elif current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return spans


def attach_bio_record_categories(record: dict[str, Any]) -> dict[str, Any]:
    words = record.get("words") if isinstance(record.get("words"), list) else []
    labels = record.get("labels") if isinstance(record.get("labels"), list) else []
    word_payloads = record.get("word_payloads") if isinstance(record.get("word_payloads"), list) else []

    semantic_categories = [None] * len(words)
    annotations = []
    for span_id, indices in enumerate(collect_item_name_spans(labels)):
        text = " ".join(str(words[idx]) for idx in indices if 0 <= idx < len(words)).strip()
        result = categorize_item_text(text)
        annotation = {
            "span_id": f"item_category_{span_id:04d}",
            "item_name_text": text,
            "item_name_word_indices": indices,
            "category": result["category"],
            "confidence": result["confidence"],
            "rule": result["rule"],
            "method": "keyword_rules_v1",
        }
        annotations.append(annotation)
        for idx in indices:
            if 0 <= idx < len(semantic_categories):
                semantic_categories[idx] = result["category"]
            if 0 <= idx < len(word_payloads) and isinstance(word_payloads[idx], dict):
                word_payloads[idx]["semantic_item_category"] = result["category"]
                word_payloads[idx]["semantic_item_category_confidence"] = result["confidence"]
                word_payloads[idx]["semantic_item_category_rule"] = result["rule"]

    record["item_semantic_category_taxonomy"] = TAXONOMY
    record["semantic_item_categories"] = semantic_categories
    record["item_category_annotations"] = annotations
    record["item_semantic_category_labeling"] = {
        "schema_version": "item_semantic_categories_v1",
        "method": "keyword_rules_v1",
        "script": "scripts/label_item_semantic_categories.py",
    }
    return record


def should_skip(path: Path, exclude_names: set[str]) -> bool:
    return any(part in exclude_names for part in path.parts)


def process_labeled_root(
    root: Path,
    backup_root: Path,
    dry_run: bool,
    no_backup: bool,
    exclude_names: set[str],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "root": str(root),
        "exists": root.exists(),
        "files_seen": 0,
        "files_written": 0,
        "item_annotations": 0,
        "category_counts": Counter(),
        "skipped_files": [],
    }
    if not root.exists():
        return summary

    for path in sorted(root.rglob("*_labeled_v2_1.json")):
        if should_skip(path, exclude_names):
            summary["skipped_files"].append({"path": str(path), "reason": "excluded_dir"})
            continue
        summary["files_seen"] += 1
        obj = read_json(path)
        updated = attach_labeled_json_categories(obj)
        annotations = updated.get("item_category_annotations", [])
        summary["item_annotations"] += len(annotations)
        summary["category_counts"].update(a.get("category", "OTHER") for a in annotations)
        if not dry_run:
            if not no_backup:
                copy_backup(path, backup_root / root.name, root)
            write_json(path, updated)
            summary["files_written"] += 1

    summary["category_counts"] = dict(summary["category_counts"].most_common())
    return summary


def process_bio_dir(
    bio_dir: Path,
    backup_root: Path,
    dry_run: bool,
    no_backup: bool,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "bio_dir": str(bio_dir),
        "exists": bio_dir.exists(),
        "files_seen": 0,
        "files_written": 0,
        "records": 0,
        "item_annotations": 0,
        "category_counts": Counter(),
    }
    if not bio_dir.exists():
        return summary

    for path in sorted(bio_dir.glob("*.jsonl")):
        if path.name.startswith("."):
            continue
        summary["files_seen"] += 1
        out_lines = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            updated = attach_bio_record_categories(record)
            annotations = updated.get("item_category_annotations", [])
            summary["records"] += 1
            summary["item_annotations"] += len(annotations)
            summary["category_counts"].update(a.get("category", "OTHER") for a in annotations)
            out_lines.append(json.dumps(updated, ensure_ascii=False))

        if not dry_run:
            if not no_backup:
                copy_backup(path, backup_root / "processed_data", Path("processed_data"))
            path.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
            summary["files_written"] += 1

    summary["category_counts"] = dict(summary["category_counts"].most_common())
    return summary


def main() -> None:
    args = parse_args()
    global MANUAL_OVERRIDES
    MANUAL_OVERRIDES = load_manual_overrides(Path(args.manual_overrides) if args.manual_overrides else None)
    if MANUAL_OVERRIDES:
        print(f"manual overrides loaded: {len(MANUAL_OVERRIDES)}")
    backup_root = Path(args.backup_dir) if args.backup_dir else Path("outputs") / "item_category_labeling_backups" / datetime.now().strftime("%Y%m%d_%H%M%S")
    exclude_names = set(args.exclude_dir_name or [])

    roots = [Path(args.custom_root), Path(args.wildreceipt_root)]
    bio_dirs = [Path(value) for value in args.bio_dirs]

    summary: dict[str, Any] = {
        "taxonomy": TAXONOMY,
        "method": "keyword_rules_v1",
        "manual_overrides": args.manual_overrides,
        "manual_override_count": len(MANUAL_OVERRIDES),
        "dry_run": args.dry_run,
        "backup_dir": None if args.no_backup else str(backup_root),
        "labeled_roots": [],
        "bio_dirs": [],
        "total_category_counts": Counter(),
    }

    for root in roots:
        root_summary = process_labeled_root(root, backup_root, args.dry_run, args.no_backup, exclude_names)
        summary["labeled_roots"].append(root_summary)
        summary["total_category_counts"].update(root_summary.get("category_counts", {}))

    for bio_dir in bio_dirs:
        bio_summary = process_bio_dir(bio_dir, backup_root, args.dry_run, args.no_backup)
        summary["bio_dirs"].append(bio_summary)
        summary["total_category_counts"].update(bio_summary.get("category_counts", {}))

    summary["total_category_counts"] = dict(summary["total_category_counts"].most_common())
    summary["notes"] = [
        "BIO labels were not changed.",
        "Semantic categories were attached to item/category metadata for classifier training.",
        "OTHER indicates no keyword rule matched and should be reviewed before using as high-quality ground truth.",
    ]

    summary_out = Path(args.summary_out)
    if not args.dry_run:
        write_json(summary_out, summary)
    else:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_out: {summary_out}")
    print("total_category_counts:")
    print(json.dumps(summary["total_category_counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
