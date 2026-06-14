import argparse
import json
import shutil
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge two span rel-g cache manifests without modifying the source caches."
    )
    parser.add_argument("--cord_cache_dir", required=True)
    parser.add_argument("--user_cache_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--user_repeat", type=int, default=1)
    parser.add_argument("--splits", default="train,validation")
    parser.add_argument("--overwrite", action="store_true")
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


def prepare_out_dir(path, overwrite):
    out_dir = Path(path)
    if out_dir.exists():
        if not overwrite:
            fail(f"{out_dir} already exists. Use --overwrite to replace only the merged manifest output.")
        print(f"Removing existing merged cache directory: {out_dir}")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def load_cache_meta(cache_dir):
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.json"
    schema_path = cache_dir / "schema.json"
    summary_path = cache_dir / "summary.json"
    if not manifest_path.exists():
        fail(f"manifest not found: {manifest_path}")
    if not schema_path.exists():
        fail(f"schema not found: {schema_path}")
    return {
        "dir": cache_dir,
        "manifest_path": manifest_path,
        "schema_path": schema_path,
        "summary_path": summary_path if summary_path.exists() else None,
        "manifest": load_json(manifest_path),
        "schema": load_json(schema_path),
        "summary": load_json(summary_path) if summary_path.exists() else {},
    }


def assert_schema_compatible(left, right):
    keys = ["field_list", "field2id", "kind2id", "hidden_dim"]
    mismatches = []
    for key in keys:
        if left["schema"].get(key) != right["schema"].get(key):
            mismatches.append(key)
    if mismatches:
        fail(
            "span rel-g cache schemas are incompatible: "
            + ", ".join(mismatches)
            + ". Rebuild both caches with the same schema/checkpoint before merging."
        )


def normalize_records(cache_meta, split):
    records = cache_meta["manifest"].get("splits", {}).get(split, [])
    if not isinstance(records, list):
        fail(f"{cache_meta['manifest_path']} split {split!r} is not a list.")
    out = []
    for idx, record in enumerate(records):
        if not isinstance(record, dict) or not record.get("path"):
            fail(f"Invalid record in {cache_meta['manifest_path']} split {split!r} at index {idx}.")
        path = Path(record["path"])
        if not path.is_absolute():
            path = ROOT_DIR / path
        if not path.exists():
            fail(f"cache sample path not found: {path}")
        clone = deepcopy(record)
        clone["path"] = str(path)
        out.append(clone)
    return out


def add_records(manifest, split, records, source_name, repeat=1):
    repeat = max(1, int(repeat))
    for repeat_idx in range(repeat):
        for record in records:
            clone = deepcopy(record)
            base_id = clone.get("id") or Path(clone["path"]).stem
            if repeat > 1:
                clone["id"] = f"{source_name}_r{repeat_idx:02d}_{base_id}"
            else:
                clone["id"] = f"{source_name}_{base_id}"
            clone["source_cache"] = source_name
            clone["source_id"] = base_id
            clone["split"] = split
            manifest["splits"].setdefault(split, []).append(clone)
            manifest["records"].append(clone)


def summarize_records(manifest):
    counts = Counter()
    by_source = Counter()
    by_split = {}
    for split, records in manifest.get("splits", {}).items():
        by_split[split] = len(records)
        counts["records"] += len(records)
        for record in records:
            by_source[record.get("source_cache", "unknown")] += 1
    return {"total_records": counts["records"], "split_records": by_split, "source_records": dict(by_source)}


def main():
    args = parse_args()
    if args.user_repeat < 1:
        fail("--user_repeat must be >= 1")
    splits = [part.strip() for part in args.splits.split(",") if part.strip()]
    if not splits:
        fail("--splits must contain at least one split name")

    cord = load_cache_meta(args.cord_cache_dir)
    user = load_cache_meta(args.user_cache_dir)
    assert_schema_compatible(cord, user)
    out_dir = prepare_out_dir(args.out_dir, args.overwrite)

    merged_manifest = {"splits": {}, "records": []}
    source_counts = {}
    for split in splits:
        cord_records = normalize_records(cord, split)
        user_records = normalize_records(user, split)
        source_counts[split] = {"cord": len(cord_records), "user": len(user_records), "user_repeat": args.user_repeat}
        add_records(merged_manifest, split, cord_records, "cord", repeat=1)
        add_records(merged_manifest, split, user_records, "user", repeat=args.user_repeat)

    schema = deepcopy(cord["schema"])
    schema.setdefault("notes", [])
    schema["notes"] = list(schema["notes"]) + [
        "Merged cache manifest from CORD span rel-g cache and user hand-labeled span rel-g cache.",
        "Source cache files are referenced by absolute path and are not copied or modified.",
    ]
    summary = {
        "cord_cache_dir": str(Path(args.cord_cache_dir)),
        "user_cache_dir": str(Path(args.user_cache_dir)),
        "out_dir": str(out_dir),
        "user_repeat": args.user_repeat,
        "splits": splits,
        "source_counts": source_counts,
        **summarize_records(merged_manifest),
    }
    save_json(out_dir / "schema.json", schema)
    save_json(out_dir / "manifest.json", merged_manifest)
    save_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"schema path: {out_dir / 'schema.json'}")
    print(f"manifest path: {out_dir / 'manifest.json'}")
    print(f"summary path: {out_dir / 'summary.json'}")
    print("Span rel-g cache merge passed.")


if __name__ == "__main__":
    main()
