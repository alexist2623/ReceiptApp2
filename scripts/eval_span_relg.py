import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.span_relg.decode import decode_edges_to_items
from ml.span_relg.io_utils import (
    load_json,
    load_split_cache,
    resolve_field_vocab,
    resolve_model_config,
    summarize_cache_object,
)
from ml.span_relg.metrics import aggregate_metrics
from ml.span_relg.model import SpanRelGModel


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained span-level rel-g checkpoint.")
    parser.add_argument("--dataset_dir", default="processed_data/span_relg")
    parser.add_argument("--checkpoint", default="models/span-relg-context/best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out_dir", default="outputs/span_relg_eval")
    parser.add_argument("--sweep_thresholds", action="store_true")
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


def select_device(device):
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            fail("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CacheDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        record = self.records[idx]
        if "sample" in record:
            return record["sample"]
        return torch.load(record["path"], map_location="cpu", weights_only=False)


def collate(samples):
    batch_size = len(samples)
    max_nodes = max(sample["node_hidden"].shape[0] for sample in samples)
    hidden_dim = samples[0]["node_hidden"].shape[-1]
    node_hidden = torch.zeros(batch_size, max_nodes, hidden_dim, dtype=torch.float32)
    node_field_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_kind_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_boxes = torch.zeros(batch_size, max_nodes, 4, dtype=torch.float32)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    candidate_rows = []
    labels = []
    pair_counts = []
    for batch_idx, sample in enumerate(samples):
        n_nodes = sample["node_hidden"].shape[0]
        node_hidden[batch_idx, :n_nodes] = sample["node_hidden"].float()
        node_field_ids[batch_idx, :n_nodes] = sample["node_field_ids"].long()
        node_kind_ids[batch_idx, :n_nodes] = sample["node_kind_ids"].long()
        node_boxes[batch_idx, :n_nodes] = sample["node_boxes"].float()
        node_mask[batch_idx, :n_nodes] = True
        pairs = sample["candidate_pairs"].long()
        pair_counts.append(pairs.shape[0])
        if pairs.numel():
            candidate_rows.append(torch.cat([torch.full((pairs.shape[0], 1), batch_idx, dtype=torch.long), pairs], dim=1))
            labels.append(sample["pair_labels"].float())
    return {
        "samples": samples,
        "node_hidden": node_hidden,
        "node_field_ids": node_field_ids,
        "node_kind_ids": node_kind_ids,
        "node_boxes": node_boxes,
        "node_mask": node_mask,
        "candidate_pairs": torch.cat(candidate_rows, dim=0) if candidate_rows else torch.empty((0, 3), dtype=torch.long),
        "pair_labels": torch.cat(labels, dim=0) if labels else torch.empty((0,), dtype=torch.float32),
        "pair_counts": pair_counts,
    }


def move_batch(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key in {"node_hidden", "node_field_ids", "node_kind_ids", "node_boxes", "node_mask", "candidate_pairs", "pair_labels"}
    }


def load_model(checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        fail(f"Span rel-g checkpoint not found: {checkpoint}")
    config_path = resolve_model_config(checkpoint)
    config = load_json(config_path)
    model = SpanRelGModel(**config)
    state_path = checkpoint / "model.pt"
    if not state_path.exists():
        fail(f"Span rel-g checkpoint missing. Run train_span_relg.py first. Missing: {state_path}")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config, config_path


def split_probs(probs, counts):
    values = probs.detach().cpu().tolist()
    output = []
    offset = 0
    for count in counts:
        output.append(values[offset : offset + count])
        offset += count
    return output


def run_inference(model, records, batch_size, device, split):
    loader = DataLoader(CacheDataset(records), batch_size=batch_size, shuffle=False, collate_fn=collate)
    all_samples = []
    probs_by_sample = []
    total_loss = 0.0
    total_pairs = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"eval {split}", unit="batch"):
            model_batch = move_batch(batch, device)
            output = model(**model_batch)
            pair_count = int(model_batch["pair_labels"].numel())
            total_loss += float(output["loss"].item()) * pair_count
            total_pairs += pair_count
            all_samples.extend(batch["samples"])
            probs_by_sample.extend(split_probs(output["probs"], batch["pair_counts"]))
    return all_samples, probs_by_sample, total_loss / total_pairs if total_pairs else None, total_pairs


def edge_payload(sample, probs, threshold):
    rows = []
    for idx, meta in enumerate(sample.get("pair_meta", [])):
        prob = float(probs[idx])
        gold = int(sample["pair_labels"][idx].item())
        pred = int(prob >= threshold)
        rows.append(
            {
                "head_span_id": meta.get("head_span_id"),
                "dep_span_id": meta.get("dep_span_id"),
                "head_field": meta.get("head_field"),
                "dep_field": meta.get("dep_field"),
                "head_text": meta.get("head_text"),
                "dep_text": meta.get("dep_text"),
                "gold": gold,
                "prob": prob,
                "pred": pred,
                "correct": bool(gold == pred),
                "head_group_key": meta.get("head_group_key"),
                "dep_group_key": meta.get("dep_group_key"),
            }
        )
    return rows


def gold_probs(sample):
    return [float(v.item()) for v in sample["pair_labels"]]


def simple_item(item):
    def text(value):
        if isinstance(value, dict):
            return value.get("text")
        return value

    return {
        "menu_name": text(item.get("menu_name")),
        "price": text(item.get("price")),
        "count": text(item.get("count")),
        "unit_price": text(item.get("unit_price")),
        "rel_g_edges": item.get("rel_g_edges", []),
        "warnings": item.get("warnings", []),
    }


def decoded_item_errors(samples, probs_by_sample, threshold):
    no_price = []
    multiple_price = []
    collision_errors = []
    for sample, probs in zip(samples, probs_by_sample):
        decoded = decode_edges_to_items(sample, probs, threshold=threshold)
        for item in decoded.get("items", []):
            menu_text = item.get("menu_name", {}).get("text") if isinstance(item.get("menu_name"), dict) else item.get("menu_name")
            if item.get("price") is None:
                no_price.append({"data_id": sample.get("data_id"), "index": sample.get("index"), "menu_name": menu_text, "item": simple_item(item)})
            price_edges = [edge for edge in item.get("rel_g_edges", []) if edge.get("dep_field") == "MENU_PRICE"]
            if len(price_edges) > 1:
                multiple_price.append({"data_id": sample.get("data_id"), "index": sample.get("index"), "menu_name": menu_text, "price_edges": price_edges})
        selected_by_dep = {}
        for idx, meta in enumerate(sample.get("pair_meta", [])):
            if float(probs[idx]) < threshold:
                continue
            selected_by_dep.setdefault(meta.get("dep_span_id"), []).append({"data_id": sample.get("data_id"), **meta, "prob": float(probs[idx])})
        for dep_span_id, edges in selected_by_dep.items():
            if len(edges) > 1:
                collision_errors.append({"data_id": sample.get("data_id"), "dep_span_id": dep_span_id, "edges": edges})
    return no_price, multiple_price, collision_errors


def flattened_metrics(metrics, samples, probs_by_sample, threshold, eval_loss, split, checkpoint, dataset_dir, device):
    labels = []
    for sample in samples:
        labels.extend(int(v.item()) for v in sample["pair_labels"])
    no_price, multiple_price, collision_errors = decoded_item_errors(samples, probs_by_sample, threshold)
    edge = metrics.get("edge", {})
    item_price = metrics.get("item_price_pair", metrics.get("menu_price_pair", {}))
    menu_price = metrics.get("menu_price_pair", item_price)
    field_edges = metrics.get("field_edges", {})
    return {
        "split": split,
        "threshold": threshold,
        "num_samples": len(samples),
        "num_candidate_pairs": len(labels),
        "num_positive_pairs": sum(labels),
        "num_negative_pairs": len(labels) - sum(labels),
        "eval_loss": eval_loss,
        "edge_precision": edge.get("precision", 0.0),
        "edge_recall": edge.get("recall", 0.0),
        "edge_f1": edge.get("f1", 0.0),
        "item_price_pair_precision": item_price.get("precision", 0.0),
        "item_price_pair_recall": item_price.get("recall", 0.0),
        "item_price_pair_f1": item_price.get("f1", 0.0),
        "menu_price_pair_precision": menu_price.get("precision", 0.0),
        "menu_price_pair_recall": menu_price.get("recall", 0.0),
        "menu_price_pair_f1": menu_price.get("f1", 0.0),
        "item_qty_pair_f1": field_edges.get("ITEM_QTY", field_edges.get("MENU_CNT", {})).get("f1", 0.0),
        "item_unit_price_pair_f1": field_edges.get("ITEM_UNIT_PRICE", field_edges.get("MENU_UNITPRICE", {})).get("f1", 0.0),
        "menu_cnt_pair_f1": field_edges.get("ITEM_QTY", field_edges.get("MENU_CNT", {})).get("f1", 0.0),
        "menu_unitprice_pair_f1": field_edges.get("ITEM_UNIT_PRICE", field_edges.get("MENU_UNITPRICE", {})).get("f1", 0.0),
        "hard_negative_false_positive_count": metrics.get("hard_negative_false_positive_count", 0),
        "store_false_positive_count": metrics.get("store_false_positive_count", 0),
        "total_subtotal_false_positive_count": metrics.get("total_subtotal_false_positive_count", 0),
        "dependent_collision_count": metrics.get("dependent_collision_count", 0),
        "no_price_item_count": len(no_price),
        "multiple_price_item_count": len(multiple_price),
        "device": str(device),
        "checkpoint": str(checkpoint),
        "dataset_dir": str(dataset_dir),
        "nested_metrics": metrics,
    }


def write_outputs(out_dir, samples, probs_by_sample, threshold, metrics, summary):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "metrics_summary.json", summary)

    hard_examples = metrics.get("hard_negative_false_positive_examples", [])
    no_price, multiple_price, collision_errors = decoded_item_errors(samples, probs_by_sample, threshold)
    save_json(out_dir / "hard_negative_errors.json", hard_examples)
    save_json(out_dir / "collision_errors.json", collision_errors)
    save_json(out_dir / "no_price_items.json", no_price)
    save_json(
        out_dir / "debug_examples.json",
        {
            "hard_negative_examples": hard_examples[:5],
            "collision_examples": collision_errors[:5],
            "no_price_examples": no_price[:5],
            "multiple_price_examples": multiple_price[:5],
        },
    )

    with (out_dir / "edge_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for sample, probs in zip(samples, probs_by_sample):
            handle.write(
                json.dumps(
                    {
                        "data_id": sample.get("data_id"),
                        "split": sample.get("split"),
                        "index": sample.get("index"),
                        "candidate_edges": edge_payload(sample, probs, threshold),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with (out_dir / "item_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for sample, probs in zip(samples, probs_by_sample):
            decoded = decode_edges_to_items(sample, probs, threshold=threshold)
            gold_decoded = decode_edges_to_items(sample, gold_probs(sample), threshold=0.5)
            handle.write(
                json.dumps(
                    {
                        "data_id": sample.get("data_id"),
                        "split": sample.get("split"),
                        "index": sample.get("index"),
                        "items": [simple_item(item) for item in decoded.get("items", [])],
                        "gold_items": [simple_item(item) for item in gold_decoded.get("items", [])],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def threshold_sweep(samples, probs_by_sample):
    rows = []
    best = None
    for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        metrics = aggregate_metrics(samples, probs_by_sample, threshold)
        edge = metrics["edge"]
        menu_price = metrics["menu_price_pair"]
        row = {
            "threshold": threshold,
            "edge_precision": edge["precision"],
            "edge_recall": edge["recall"],
            "edge_f1": edge["f1"],
            "menu_price_pair_precision": menu_price["precision"],
            "menu_price_pair_recall": menu_price["recall"],
            "menu_price_pair_f1": menu_price["f1"],
            "hard_negative_false_positive_count": metrics["hard_negative_false_positive_count"],
            "dependent_collision_count": metrics["dependent_collision_count"],
        }
        rows.append(row)
        if (
            best is None
            or row["menu_price_pair_f1"] > best["menu_price_pair_f1"]
            or (
                row["menu_price_pair_f1"] == best["menu_price_pair_f1"]
                and row["menu_price_pair_precision"] > best["menu_price_pair_precision"]
            )
            or (
                row["menu_price_pair_f1"] == best["menu_price_pair_f1"]
                and row["menu_price_pair_precision"] == best["menu_price_pair_precision"]
                and row["hard_negative_false_positive_count"] < best["hard_negative_false_positive_count"]
            )
        ):
            best = row
    return {"thresholds": rows, "best_threshold": best}


def _load_sample_from_record(record):
    if "sample" in record:
        return record["sample"]
    return torch.load(record["path"], map_location="cpu", weights_only=False)


def validate_cache_model_compatibility(records, config):
    num_fields = int(config.get("num_fields", 0))
    num_kinds = int(config.get("num_kinds", 0))
    hidden_dim = int(config.get("hidden_dim", 0))
    max_field_id = -1
    max_kind_id = -1
    hidden_dims = set()
    examples = []
    for record in records:
        sample = _load_sample_from_record(record)
        if sample["node_field_ids"].numel():
            sample_max_field = int(sample["node_field_ids"].max().item())
            max_field_id = max(max_field_id, sample_max_field)
        if sample["node_kind_ids"].numel():
            sample_max_kind = int(sample["node_kind_ids"].max().item())
            max_kind_id = max(max_kind_id, sample_max_kind)
        hidden_dims.add(int(sample["node_hidden"].shape[-1]))
        if len(examples) < 3:
            examples.append({"data_id": sample.get("data_id"), "max_field_id": max_field_id, "max_kind_id": max_kind_id})
    issues = []
    if num_fields and max_field_id >= num_fields:
        issues.append(f"cache max node_field_id {max_field_id} >= checkpoint num_fields {num_fields}")
    if num_kinds and max_kind_id >= num_kinds:
        issues.append(f"cache max node_kind_id {max_kind_id} >= checkpoint num_kinds {num_kinds}")
    if hidden_dim and hidden_dims and hidden_dims != {hidden_dim}:
        issues.append(f"cache hidden dims {sorted(hidden_dims)} != checkpoint hidden_dim {hidden_dim}")
    if issues:
        fail(
            "Span rel-g cache/checkpoint mismatch. "
            + "; ".join(issues)
            + ". Rebuild cache with the checkpoint field vocab or retrain rel-g on this cache before evaluation. "
            + f"examples={examples}"
        )


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    checkpoint = Path(args.checkpoint)
    cache_info = load_split_cache(dataset_dir, args.split)
    config_path = resolve_model_config(checkpoint)
    field_vocab = resolve_field_vocab(dataset_dir, checkpoint)
    device = select_device(args.device)
    model, config, model_config_path = load_model(checkpoint, device)

    print(f"resolved split cache path: {cache_info['source_path']}")
    print(f"resolved config path: {model_config_path}")
    print(f"field vocab source: {field_vocab['source']}::{field_vocab['key']}")
    print(f"cache summary: {json.dumps(summarize_cache_object(cache_info), ensure_ascii=False, indent=2)}")
    validate_cache_model_compatibility(cache_info["records"], config)

    samples, probs_by_sample, eval_loss, total_pairs = run_inference(model, cache_info["records"], args.batch_size, device, args.split)
    metrics = aggregate_metrics(samples, probs_by_sample, args.threshold)
    summary = flattened_metrics(metrics, samples, probs_by_sample, args.threshold, eval_loss, args.split, checkpoint, dataset_dir, device)
    out_dir = Path(args.out_dir)
    write_outputs(out_dir, samples, probs_by_sample, args.threshold, metrics, summary)
    save_json(
        out_dir / "run_config.json",
        {
            "dataset_dir": str(dataset_dir),
            "checkpoint": str(checkpoint),
            "split": args.split,
            "threshold": args.threshold,
            "device": args.device,
            "resolved_split_cache_path": str(cache_info["source_path"]),
            "resolved_config_path": str(config_path),
            "field_vocab_source": field_vocab["source"],
            "field_vocab_key": field_vocab["key"],
        },
    )

    if args.sweep_thresholds:
        sweep = threshold_sweep(samples, probs_by_sample)
        save_json(out_dir / "threshold_sweep.json", sweep)
        print(f"best threshold: {sweep['best_threshold']}")

    if args.debug and samples:
        first = samples[0]
        print(f"first sample: {first['data_id']} pairs={len(first['pair_meta'])}")
        for meta, prob in list(zip(first["pair_meta"], probs_by_sample[0]))[:30]:
            print(f"  {meta['head_text']!r} -> {meta['dep_text']!r} {meta['dep_field']} y={meta['label']} p={prob:.3f}")

    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"selected device: {device}")
    print(f"metrics_summary path: {out_dir / 'metrics_summary.json'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:5000])
    print("Span rel-g evaluation passed.")


if __name__ == "__main__":
    main()
