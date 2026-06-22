import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.span_relg.cord_spans import make_gold_spans_from_cord
from ml.angle_geometry import ANGLE_FEATURE_DIM, angle_feature_dim_for_mode, build_angle_features_for_words
from ml.span_relg.feature_cache import build_cache_sample, compute_word_hidden, load_layoutlmv3_for_feature_cache
from ml.span_relg.schema import ALL_FIELDS


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build cached span-level rel-g training data from CORD-v2 and a frozen LayoutLMv3 checkpoint."
    )
    parser.add_argument("--raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--checkpoint", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--out_dir", default="processed_data/span_relg")
    parser.add_argument("--split", default=None, help="Optional single split to process.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional per-split sample limit.")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--include_context_tokens", default="all", choices=("all", "o_only", "none"))
    parser.add_argument("--span_pooling", default="first", choices=("first", "mean"))
    parser.add_argument("--group_key_strategy", default="group", choices=("group", "group_sub", "group_row"))
    parser.add_argument("--use_angle_features", default="auto", choices=("auto", "true", "false"))
    parser.add_argument("--disable_angle_features", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def fail(message):
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def save_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def prepare_out_dir(out_dir, overwrite):
    out_dir = Path(out_dir)
    if out_dir.exists():
        if not overwrite:
            fail(f"{out_dir} already exists. Use --overwrite to rebuild it.")
        print(f"Removing existing output directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def split_lengths(dataset):
    return {split: len(dataset[split]) for split in dataset.keys()}


def build_one_split(
    dataset,
    split,
    out_dir,
    processor,
    layout_model,
    device,
    args,
    field2id,
    kind2id,
    angle_config,
):
    split_dir = out_dir / split
    split_dir.mkdir(parents=True, exist_ok=True)
    total = len(dataset[split])
    limit = min(total, args.max_samples) if args.max_samples is not None else total
    records = []
    skipped = []
    counters = Counter()
    field_counts = Counter()
    pair_field_counts = Counter()
    iterator = range(limit)
    for index in tqdm(iterator, desc=f"cache {split}", unit="sample"):
        sample_id = f"{split}_{index:06d}"
        try:
            sample_info = make_gold_spans_from_cord(
                dataset[split][index],
                group_key_strategy=args.group_key_strategy,
            )
            if bool(getattr(layout_model, "uses_angle_features", False)):
                mode = angle_config.get("angle_encoding_mode") or "sincos_scalar"
                dim = int(angle_config.get("angle_feature_dim") or angle_feature_dim_for_mode(mode))
                existing = sample_info.get("angle_features") or []
                existing_dim = len(existing[0]) if existing else 0
                if existing_dim != dim:
                    angle_result = build_angle_features_for_words(
                        sample_info.get("word_payloads") or [],
                        boxes=sample_info.get("boxes") or [],
                        image_width=sample_info.get("width"),
                        image_height=sample_info.get("height"),
                        mode=mode,
                    )
                    sample_info["angle_features"] = angle_result["angle_features"]
                    sample_info["angle_debug"] = angle_result["word_angles"]
            word_features = compute_word_hidden(
                sample_info["image"],
                sample_info["words"],
                sample_info["normalized_boxes"],
                processor,
                layout_model,
                device,
                max_length=args.max_length,
                word_angle_features=(
                    sample_info.get("angle_features")
                    if getattr(layout_model, "uses_angle_features", False) or args.use_angle_features == "true"
                    else None
                ),
                uses_angle_features=bool(getattr(layout_model, "uses_angle_features", False)),
                angle_feature_dim=int(angle_config.get("angle_feature_dim") or 0),
            )
            cache = build_cache_sample(
                sample_id,
                split,
                index,
                sample_info,
                word_features["word_hidden"],
                field2id,
                kind2id,
                include_context_tokens=args.include_context_tokens,
                span_pooling=args.span_pooling,
            )
            cache["word_token_indices"] = word_features["word_token_indices"]
            cache["encoding_shapes"] = word_features["encoding_shapes"]
            cache["uses_angle_features"] = word_features.get("uses_angle_features", False)
            cache["angle_features_shape"] = word_features.get("angle_features_shape")
            if cache["candidate_pairs"].numel() == 0:
                raise ValueError("No candidate rel-g pairs found.")
            sample_path = split_dir / f"{sample_id}.pt"
            torch.save(cache, sample_path)
            records.append({"id": sample_id, "split": split, "index": index, "path": str(sample_path)})
            counters["input_samples"] += 1
            counters["written_samples"] += 1
            counters["nodes"] += len(cache["nodes"])
            counters["span_nodes"] += sum(1 for node in cache["nodes"] if node.get("node_kind") == "SPAN")
            counters["context_nodes"] += sum(1 for node in cache["nodes"] if node.get("node_kind") == "TOKEN")
            counters["candidate_pairs"] += int(cache["pair_labels"].numel())
            counters["positive_pairs"] += int(cache["pair_labels"].sum().item())
            counters["negative_pairs"] += int(cache["pair_labels"].numel() - cache["pair_labels"].sum().item())
            for node in cache["nodes"]:
                if node.get("node_kind") == "SPAN":
                    field_counts[node.get("field")] += 1
            pair_field_counts.update(cache.get("pair_fields", []))
            if args.debug and len(records) <= 2:
                print(f"\nDEBUG {sample_id}")
                print(f"  words={cache['num_words']} nodes={len(cache['nodes'])} pairs={len(cache['pair_meta'])}")
                for meta in cache["pair_meta"][:20]:
                    print(
                        "  pair "
                        f"{meta['head_text']!r} -> {meta['dep_text']!r} "
                        f"{meta['dep_field']} label={meta['label']}"
                    )
        except Exception as exc:
            skipped.append({"id": sample_id, "split": split, "index": index, "error": repr(exc)})
            counters["input_samples"] += 1
            counters["skipped_samples"] += 1
            print(f"WARNING: skipped {sample_id}: {exc}")
    return {
        "records": records,
        "skipped": skipped,
        "summary": dict(counters),
        "field_counts": dict(field_counts),
        "pair_field_counts": dict(pair_field_counts),
    }


def main():
    args = parse_args()
    raw_data_dir = Path(args.raw_data_dir)
    checkpoint = Path(args.checkpoint)
    if not raw_data_dir.exists():
        fail(f"CORD-v2 dataset not found at {raw_data_dir}. Run download step first.")
    if not checkpoint.exists() and args.local_files_only:
        fail(f"LayoutLMv3 checkpoint not found: {checkpoint}")

    out_dir = prepare_out_dir(args.out_dir, args.overwrite)
    print(f"raw_data_dir: {raw_data_dir}")
    print(f"checkpoint: {args.checkpoint}")
    print(f"out_dir: {out_dir}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")

    dataset = load_from_disk(str(raw_data_dir))
    available = list(dataset.keys())
    print(f"available splits: {split_lengths(dataset)}")
    if args.split is not None and args.split not in dataset:
        fail(f"Split {args.split!r} not found. Available splits: {available}")
    splits = [args.split] if args.split else available

    print("Loading frozen LayoutLMv3 feature extractor...")
    use_angle_features = "false" if args.disable_angle_features else args.use_angle_features
    if use_angle_features == "true":
        use_angle_features_for_model = True
    elif use_angle_features == "false":
        use_angle_features_for_model = False
    else:
        use_angle_features_for_model = "auto"
    processor, layout_model, device, layout_uses_angle, angle_config = load_layoutlmv3_for_feature_cache(
        args.checkpoint,
        args.local_files_only,
        args.device,
    )
    if use_angle_features_for_model is False:
        layout_model.uses_angle_features = False
        layout_uses_angle = False
        angle_config = {"use_angle_features": False, "angle_feature_dim": 0, "angle_encoding_mode": "none"}
    elif use_angle_features_for_model is True and not layout_uses_angle:
        print("WARNING: --use_angle_features true was requested, but checkpoint is not angle-aware.")
    print(f"selected device: {device}")
    print(f"use_angle_features: {use_angle_features}")
    print(f"layout_model_uses_angle_features: {getattr(layout_model, 'uses_angle_features', False)}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    field2id = {field: idx for idx, field in enumerate(ALL_FIELDS)}
    kind2id = {"SPAN": 0, "TOKEN": 1}
    manifest = {"splits": {}, "records": []}
    summary = {
        "raw_data_dir": str(raw_data_dir),
        "checkpoint": args.checkpoint,
        "out_dir": str(out_dir),
        "max_samples": args.max_samples,
        "max_length": args.max_length,
        "include_context_tokens": args.include_context_tokens,
        "span_pooling": args.span_pooling,
        "group_key_strategy": args.group_key_strategy,
        "use_angle_features": use_angle_features,
        "angle_feature_dim": int(angle_config.get("angle_feature_dim") or ANGLE_FEATURE_DIM),
        "angle_encoding_mode": angle_config.get("angle_encoding_mode", "sincos_scalar"),
        "angle_config": angle_config,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "splits": {},
        "skipped_samples": [],
        "field_counts": {},
        "pair_field_counts": {},
    }
    field_counts = Counter()
    pair_field_counts = Counter()
    hidden_dim = None

    for split in splits:
        result = build_one_split(dataset, split, out_dir, processor, layout_model, device, args, field2id, kind2id, angle_config)
        manifest["splits"][split] = result["records"]
        manifest["records"].extend(result["records"])
        summary["splits"][split] = result["summary"]
        summary["skipped_samples"].extend(result["skipped"])
        field_counts.update(result["field_counts"])
        pair_field_counts.update(result["pair_field_counts"])
        if hidden_dim is None and result["records"]:
            first = torch.load(result["records"][0]["path"], map_location="cpu", weights_only=False)
            hidden_dim = int(first["node_hidden"].shape[-1])

    schema = {
        "field_list": ALL_FIELDS,
        "field2id": field2id,
        "kind2id": kind2id,
        "hidden_dim": hidden_dim,
        "candidate_head_fields": ["ITEM_NAME"],
        "candidate_dep_fields": [
            "ITEM_PRICE",
            "ITEM_QTY",
            "ITEM_UNIT_PRICE",
            "ITEM_CODE",
            "ITEM_SKU",
            "ITEM_DISCOUNT",
            "ITEM_OPTION",
            "ITEM_TAX_FLAG",
            "ITEM_ETC",
        ],
        "notes": [
            "LayoutLMv3 is used only as a frozen feature extractor.",
            "Pair labels are binary same-group rel-g labels, not group_id classes.",
            "Old CORD MENU_* labels are canonicalized to receipt schema v2 ITEM_* fields.",
            "No rel-s or token serialization is implemented here.",
            "Angle-aware LayoutLMv3 checkpoints are supported; CORD-only cache uses zero angle features unless angle_features are provided by the sample builder.",
        ],
        "angle_feature_dim": int(angle_config.get("angle_feature_dim") or ANGLE_FEATURE_DIM),
        "angle_encoding_mode": angle_config.get("angle_encoding_mode", "sincos_scalar"),
        "angle_config": angle_config,
        "uses_angle_features": bool(getattr(layout_model, "uses_angle_features", False)),
    }
    summary["field_counts"] = dict(field_counts)
    summary["pair_field_counts"] = dict(pair_field_counts)
    summary["num_cached_samples"] = len(manifest["records"])
    summary["hidden_dim"] = hidden_dim
    save_json(out_dir / "schema.json", schema)
    save_json(out_dir / "manifest.json", manifest)
    save_json(out_dir / "summary.json", summary)

    print("\nCache build summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:5000])
    print(f"schema path: {out_dir / 'schema.json'}")
    print(f"manifest path: {out_dir / 'manifest.json'}")
    print(f"summary path: {out_dir / 'summary.json'}")
    print("Span rel-g cache build passed.")


if __name__ == "__main__":
    main()
