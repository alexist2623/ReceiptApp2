import argparse
import json
import sys
from pathlib import Path

import torch
from datasets import load_from_disk

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.span_relg.cord_spans import ensure_pil_rgb
from ml.span_relg.decode import decode_edges_to_items
from ml.span_relg.io_utils import load_json, load_split_cache, resolve_field_vocab, resolve_model_config
from ml.span_relg.model import SpanRelGModel
from ml.span_relg.visualization import draw_span_relg_overlay


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize span-level rel-g predictions on a CORD image.")
    parser.add_argument("--raw_data_dir", default="../receipt_training_data2")
    parser.add_argument("--dataset_dir", default="processed_data/span_relg")
    parser.add_argument("--checkpoint", default="models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best")
    parser.add_argument("--split", default="test")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.84)
    parser.add_argument("--out_dir", default="outputs/span_relg_overlay")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--show_gt", action="store_true")
    parser.add_argument("--show_pred", action="store_true")
    parser.add_argument("--errors_only", action="store_true")
    parser.add_argument("--filename_suffix", default=None)
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


def load_model(checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        fail(f"Span rel-g checkpoint not found: {checkpoint}")
    config_path = resolve_model_config(checkpoint)
    config = load_json(config_path)
    model = SpanRelGModel(**config)
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, config_path


def run_one(model, sample, device):
    node_hidden = sample["node_hidden"].unsqueeze(0).float().to(device)
    node_field_ids = sample["node_field_ids"].unsqueeze(0).long().to(device)
    node_kind_ids = sample["node_kind_ids"].unsqueeze(0).long().to(device)
    node_boxes = sample["node_boxes"].unsqueeze(0).float().to(device)
    node_mask = torch.ones(1, sample["node_hidden"].shape[0], dtype=torch.bool, device=device)
    if sample["candidate_pairs"].numel():
        batch_col = torch.zeros(sample["candidate_pairs"].shape[0], 1, dtype=torch.long)
        candidate_pairs = torch.cat([batch_col, sample["candidate_pairs"].long()], dim=1).to(device)
    else:
        candidate_pairs = torch.empty((0, 3), dtype=torch.long, device=device)
    with torch.no_grad():
        output = model(
            node_hidden=node_hidden,
            node_field_ids=node_field_ids,
            node_kind_ids=node_kind_ids,
            node_boxes=node_boxes,
            node_mask=node_mask,
            candidate_pairs=candidate_pairs,
        )
    return output["probs"].detach().cpu().tolist()


def load_sample_from_record(record):
    if "sample" in record:
        return record["sample"]
    return torch.load(record["path"], map_location="cpu", weights_only=False)


def find_record(records, index):
    for record in records:
        if int(record.get("index", -1)) == index:
            return record
    if 0 <= index < len(records):
        return records[index]
    fail(f"Cached sample with index {index} not found.")


def classify_edges(sample, probs, threshold):
    pred_edges = []
    gold_edges = []
    correct_edges = []
    wrong_edges = []
    missed_edges = []
    hard_negative_errors = []
    for idx, meta in enumerate(sample.get("pair_meta", [])):
        prob = float(probs[idx])
        gold = int(sample["pair_labels"][idx].item()) == 1
        pred = prob >= threshold
        base = dict(meta)
        base.update(
            {
                "pair_index": idx,
                "head_node_id": meta["head_node_id"],
                "dep_node_id": meta["dep_node_id"],
                "prob": prob,
                "gold": int(gold),
                "pred": int(pred),
                "correct": bool(gold and pred),
            }
        )
        if gold:
            gold_edges.append(dict(base, status="gold"))
        if pred:
            status = "correct" if gold else "wrong"
            edge = dict(base, status=status)
            if not gold and (str(meta.get("dep_field", "")).startswith("TOTAL_") or str(meta.get("dep_field", "")).startswith("SUBTOTAL_")):
                edge["hard_negative"] = True
                hard_negative_errors.append(edge)
            pred_edges.append(edge)
            if gold:
                correct_edges.append(edge)
            else:
                wrong_edges.append(edge)
        elif gold:
            missed_edges.append(dict(base, status="missed"))
    return {
        "pred_edges": pred_edges,
        "gold_edges": gold_edges,
        "correct_edges": correct_edges,
        "wrong_edges": wrong_edges,
        "missed_edges": missed_edges,
        "hard_negative_errors": hard_negative_errors,
    }


def edges_for_drawing(edge_sets, show_gt, show_pred, errors_only):
    if errors_only:
        return edge_sets["wrong_edges"] + edge_sets["missed_edges"] + edge_sets["hard_negative_errors"]
    edges = []
    if show_pred or not show_gt:
        edges.extend(edge_sets["pred_edges"])
    if show_gt:
        selected = {(edge["head_span_id"], edge["dep_span_id"]) for edge in edges}
        for edge in edge_sets["missed_edges"]:
            key = (edge["head_span_id"], edge["dep_span_id"])
            if key not in selected:
                edges.append(edge)
    return edges


def main():
    args = parse_args()
    raw_data_dir = Path(args.raw_data_dir)
    dataset_dir = Path(args.dataset_dir)
    checkpoint = Path(args.checkpoint)
    if not raw_data_dir.exists():
        fail(f"CORD-v2 dataset not found at {raw_data_dir}.")
    cache_info = load_split_cache(dataset_dir, args.split)
    field_vocab = resolve_field_vocab(dataset_dir, checkpoint)
    device = select_device(args.device)
    model, config_path = load_model(checkpoint, device)
    record = find_record(cache_info["records"], args.index)
    sample = load_sample_from_record(record)

    raw_index = int(sample.get("index", args.index))
    index_mismatch = raw_index != args.index
    dataset = load_from_disk(str(raw_data_dir))
    if args.split not in dataset:
        fail(f"Raw dataset split {args.split!r} not found.")
    image = ensure_pil_rgb(dataset[args.split][raw_index]["image"])
    probs = run_one(model, sample, device)
    decoded = decode_edges_to_items(sample, probs, threshold=args.threshold)
    edge_sets = classify_edges(sample, probs, args.threshold)
    draw_edges = edges_for_drawing(edge_sets, args.show_gt, args.show_pred, args.errors_only)

    out_dir = Path(args.out_dir)
    suffix = f"_{args.filename_suffix}" if args.filename_suffix else ""
    png_path = out_dir / f"{args.split}_{args.index:06d}{suffix}_relg.png"
    json_path = out_dir / f"{args.split}_{args.index:06d}{suffix}_relg_debug.json"
    draw_span_relg_overlay(image, sample, draw_edges, png_path, title=f"{sample['data_id']} span rel-g")
    save_json(
        json_path,
        {
            "split": args.split,
            "index": args.index,
            "raw_index_used": raw_index,
            "index_mismatch": index_mismatch,
            "threshold": args.threshold,
            "image_width": image.size[0],
            "image_height": image.size[1],
            "resolved_split_cache_path": str(cache_info["source_path"]),
            "resolved_config_path": str(config_path),
            "field_vocab_source": field_vocab["source"],
            "spans": [node for node in sample.get("nodes", []) if node.get("node_kind") == "SPAN"],
            **edge_sets,
            "decoded_items": decoded.get("items", []),
        },
    )
    if args.debug:
        print(f"resolved split cache path: {cache_info['source_path']}")
        print(f"resolved config path: {config_path}")
        print(f"field vocab source: {field_vocab['source']}::{field_vocab['key']}")
        print(f"raw index used: {raw_index} mismatch={index_mismatch}")
        for edge in draw_edges[:30]:
            print(f"{edge['status']} {edge['head_text']!r} -> {edge['dep_text']!r} {edge['dep_field']} p={edge['prob']:.3f}")
    print(f"overlay path: {png_path}")
    print(f"json path: {json_path}")
    print("Span rel-g visualization passed.")


if __name__ == "__main__":
    main()
