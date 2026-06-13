import json
from pathlib import Path

import torch


def load_json(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _as_existing_path(dataset_dir, value):
    if value is None:
        return None
    path = Path(value)
    if not path.is_absolute():
        path = Path(dataset_dir) / path
        if not path.exists():
            path = Path(value)
    return path if path.exists() else None


def _paths_from_manifest_entry(dataset_dir, entry, split):
    if isinstance(entry, str):
        path = _as_existing_path(dataset_dir, entry)
        return [path] if path else []
    if isinstance(entry, dict):
        for key in ("path", "file", "cache", "cache_path"):
            path = _as_existing_path(dataset_dir, entry.get(key))
            if path:
                return [path]
        if entry.get("split") == split:
            path = _as_existing_path(dataset_dir, entry.get("path"))
            return [path] if path else []
    if isinstance(entry, list):
        paths = []
        for item in entry:
            paths.extend(_paths_from_manifest_entry(dataset_dir, item, split))
        return paths
    return []


def resolve_split_cache(dataset_dir: str, split: str) -> Path:
    dataset_dir = Path(dataset_dir)
    direct = dataset_dir / f"{split}.pt"
    if direct.exists():
        return direct

    manifest_path = dataset_dir / "manifest.json"
    if manifest_path.exists():
        manifest = load_json(manifest_path)
        splits = manifest.get("splits")
        if isinstance(splits, dict) and split in splits:
            paths = _paths_from_manifest_entry(dataset_dir, splits[split], split)
            if len(paths) == 1:
                return paths[0]
            if len(paths) > 1:
                return manifest_path
        files = manifest.get("files")
        if isinstance(files, dict) and split in files:
            paths = _paths_from_manifest_entry(dataset_dir, files[split], split)
            if len(paths) == 1:
                return paths[0]
        samples = manifest.get("samples")
        if isinstance(samples, list):
            paths = [
                _as_existing_path(dataset_dir, item.get("path"))
                for item in samples
                if isinstance(item, dict) and item.get("split") == split
            ]
            paths = [path for path in paths if path]
            if len(paths) == 1:
                return paths[0]
            if len(paths) > 1:
                return manifest_path

    candidates = sorted(dataset_dir.rglob(f"*{split}*.pt"))
    if not candidates:
        candidates = sorted((dataset_dir / split).glob("*.pt")) if (dataset_dir / split).exists() else []
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        if manifest_path.exists():
            return manifest_path
        preview = "\n".join(str(path) for path in candidates[:30])
        raise FileNotFoundError(f"Multiple cache candidates for split {split!r}:\n{preview}")
    raise FileNotFoundError(f"No cache file found for split {split!r} under {dataset_dir}")


def records_from_manifest(dataset_dir: str, split: str):
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = load_json(manifest_path)
    records = []
    split_entry = manifest.get("splits", {}).get(split) if isinstance(manifest.get("splits"), dict) else None
    if isinstance(split_entry, list):
        for item in split_entry:
            if not isinstance(item, dict):
                continue
            path = _as_existing_path(dataset_dir, item.get("path"))
            if path:
                record = dict(item)
                record["path"] = str(path)
                records.append(record)
    elif split_entry is not None:
        for path in _paths_from_manifest_entry(dataset_dir, split_entry, split):
            records.append({"id": path.stem, "split": split, "path": str(path)})

    files = manifest.get("files")
    if not records and isinstance(files, dict) and split in files:
        for path in _paths_from_manifest_entry(dataset_dir, files[split], split):
            records.append({"id": path.stem, "split": split, "path": str(path)})

    samples = manifest.get("samples")
    if not records and isinstance(samples, list):
        for item in samples:
            if isinstance(item, dict) and item.get("split") == split:
                path = _as_existing_path(dataset_dir, item.get("path"))
                if path:
                    record = dict(item)
                    record["path"] = str(path)
                    records.append(record)
    return records


def load_split_cache(dataset_dir: str, split: str):
    source_path = resolve_split_cache(dataset_dir, split)
    records = records_from_manifest(dataset_dir, split)
    if records:
        return {"source_path": source_path, "records": records, "object": None}
    obj = torch.load(source_path, map_location="cpu", weights_only=False)
    if isinstance(obj, list):
        records = [{"id": item.get("data_id", f"{split}_{idx:06d}") if isinstance(item, dict) else f"{split}_{idx:06d}", "split": split, "sample": item} for idx, item in enumerate(obj)]
    elif isinstance(obj, dict) and "samples" in obj and isinstance(obj["samples"], list):
        records = [{"id": item.get("data_id", f"{split}_{idx:06d}") if isinstance(item, dict) else f"{split}_{idx:06d}", "split": split, "sample": item} for idx, item in enumerate(obj["samples"])]
    elif isinstance(obj, dict) and "node_hidden" in obj:
        records = [{"id": obj.get("data_id", source_path.stem), "split": split, "sample": obj}]
    else:
        raise TypeError(f"Unsupported cache object type from {source_path}: {type(obj)}")
    return {"source_path": source_path, "records": records, "object": obj}


def summarize_cache_object(cache_info):
    records = cache_info.get("records") or []
    first = None
    if records:
        first_record = records[0]
        first = first_record.get("sample")
        if first is None and first_record.get("path"):
            first = torch.load(first_record["path"], map_location="cpu", weights_only=False)
    return {
        "source_path": str(cache_info.get("source_path")),
        "record_count": len(records),
        "first_record_keys": list(records[0].keys()) if records else [],
        "first_sample_type": str(type(first)) if first is not None else None,
        "first_sample_keys": list(first.keys()) if isinstance(first, dict) else None,
    }


def resolve_model_config(checkpoint_dir: str) -> Path:
    checkpoint_dir = Path(checkpoint_dir)
    for name in ("config.json", "model_config.json"):
        path = checkpoint_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No config.json or model_config.json found under {checkpoint_dir}")


def resolve_field_vocab(dataset_dir, checkpoint_dir):
    checked = []
    for path in (
        Path(dataset_dir) / "field_vocab.json",
        Path(checkpoint_dir) / "field_vocab.json",
        Path(dataset_dir) / "schema.json",
        Path(checkpoint_dir) / "schema.json",
        Path(dataset_dir) / "manifest.json",
        resolve_model_config(checkpoint_dir),
    ):
        checked.append(str(path))
        if not path.exists() or path.suffix != ".json":
            continue
        obj = load_json(path)
        for key in ("field_vocab", "field2id", "id2field", "field_label2id", "node_field_vocab"):
            if key in obj:
                vocab = obj[key]
                if key == "id2field" and isinstance(vocab, dict):
                    vocab = {value: int(idx) for idx, value in vocab.items()}
                return {"source": str(path), "key": key, "vocab": vocab, "checked": checked}
        if "field_list" in obj:
            return {
                "source": str(path),
                "key": "field_list",
                "vocab": {field: idx for idx, field in enumerate(obj["field_list"])},
                "checked": checked,
            }
    raise FileNotFoundError(f"Could not resolve field vocab. Checked: {checked}")
