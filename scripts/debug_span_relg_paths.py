import argparse
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.span_relg.io_utils import (
    load_json,
    load_split_cache,
    resolve_field_vocab,
    resolve_model_config,
    summarize_cache_object,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug span rel-g cache/checkpoint path resolution.")
    parser.add_argument("--dataset_dir", default="processed_data/span_relg")
    parser.add_argument("--checkpoint", default="models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best")
    parser.add_argument("--split", default="test")
    return parser.parse_args()


def load_first_sample(cache_info):
    records = cache_info.get("records") or []
    if not records:
        return None
    record = records[0]
    if "sample" in record:
        return record["sample"]
    if record.get("path"):
        return torch.load(record["path"], map_location="cpu", weights_only=False)
    return None


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    checkpoint = Path(args.checkpoint)

    cache_info = load_split_cache(dataset_dir, args.split)
    cache_summary = summarize_cache_object(cache_info)
    first_sample = load_first_sample(cache_info)

    config_path = resolve_model_config(checkpoint)
    config = load_json(config_path)
    field_vocab = resolve_field_vocab(dataset_dir, checkpoint)
    model_path = checkpoint / "model.pt"

    print(f"resolved split cache path: {cache_info['source_path']}")
    print(f"split cache object type: {type(cache_info.get('object'))}")
    print(f"sample count estimate: {cache_summary['record_count']}")
    print(f"first record keys: {cache_summary['first_record_keys']}")
    print(f"first sample type: {cache_summary['first_sample_type']}")
    print(f"first sample keys: {cache_summary['first_sample_keys']}")
    if isinstance(first_sample, dict):
        print(f"first sample data_id: {first_sample.get('data_id')}")
        print(f"first sample split/index: {first_sample.get('split')}/{first_sample.get('index')}")
        print(f"first sample node count: {len(first_sample.get('nodes', []))}")
        pairs = first_sample.get("candidate_pairs")
        print(f"first sample candidate pair shape: {tuple(pairs.shape) if hasattr(pairs, 'shape') else None}")
    print(f"resolved config path: {config_path}")
    print(f"config keys: {list(config.keys())}")
    print(f"resolved field vocab source: {field_vocab['source']}::{field_vocab['key']}")
    print(f"field vocab size: {len(field_vocab['vocab'])}")
    print(f"model.pt exists: {model_path.exists()}")


if __name__ == "__main__":
    main()
