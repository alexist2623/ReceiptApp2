import argparse
import json
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.span_relg.io_utils import load_split_cache, resolve_field_vocab, resolve_model_config, summarize_cache_object
from ml.span_relg.metrics import aggregate_metrics
from scripts.eval_span_relg import (
    flattened_metrics,
    load_model,
    run_inference,
    save_json,
    threshold_sweep,
    write_outputs,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate rel-g checkpoint with ONNX-derived LayoutLMv3 hidden cache.")
    parser.add_argument("--dataset_dir", default="processed_data/span_relg_cord_onnx_int8_dynamic")
    parser.add_argument("--relg_checkpoint", default="models/span-relg-f1search-2layer-itempricew2-resume-lr5e5-50ep/best")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--threshold", type=float, default=0.84)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--out_dir", default="outputs/quantization/cord_relg_hidden_impact")
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def select_device(device):
    if device == "cpu":
        return torch.device("cpu")
    if device == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    checkpoint = Path(args.relg_checkpoint)
    cache_info = load_split_cache(dataset_dir, args.split)
    config_path = resolve_model_config(checkpoint)
    field_vocab = resolve_field_vocab(dataset_dir, checkpoint)
    device = select_device(args.device)
    model, _, model_config_path = load_model(checkpoint, device)
    samples, probs_by_sample, eval_loss, _ = run_inference(model, cache_info["records"], args.batch_size, device, args.split)
    metrics = aggregate_metrics(samples, probs_by_sample, args.threshold)
    summary = flattened_metrics(metrics, samples, probs_by_sample, args.threshold, eval_loss, args.split, checkpoint, dataset_dir, device)
    out_dir = Path(args.out_dir)
    write_outputs(out_dir, samples, probs_by_sample, args.threshold, metrics, summary)
    save_json(out_dir / f"relg_metrics_onnx_int8_{args.split}.json", summary)
    sweep = threshold_sweep(samples, probs_by_sample)
    save_json(out_dir / f"threshold_sweep_onnx_int8_{args.split}.json", sweep)
    with (out_dir / f"decoded_preview_onnx_int8_{args.split}.jsonl").open("w", encoding="utf-8") as handle:
        for sample, probs in list(zip(samples, probs_by_sample))[:20]:
            handle.write(json.dumps({"data_id": sample.get("data_id"), "num_pairs": len(sample.get("pair_meta", []))}, ensure_ascii=False) + "\n")
    save_json(
        out_dir / "run_config.json",
        {
            "dataset_dir": str(dataset_dir),
            "relg_checkpoint": str(checkpoint),
            "split": args.split,
            "threshold": args.threshold,
            "device": args.device,
            "resolved_split_cache_path": str(cache_info["source_path"]),
            "resolved_config_path": str(config_path),
            "resolved_model_config_path": str(model_config_path),
            "field_vocab_source": field_vocab["source"],
            "field_vocab_key": field_vocab["key"],
            "cache_summary": summarize_cache_object(cache_info),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2)[:5000])
    print(f"metrics path: {out_dir / f'relg_metrics_onnx_int8_{args.split}.json'}")
    print(f"sweep path: {out_dir / f'threshold_sweep_onnx_int8_{args.split}.json'}")
    print("Rel-g hidden impact evaluation passed.")


if __name__ == "__main__":
    main()
