import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.span_relg.model import SpanRelGModel


class SpanRelGOnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, node_hidden, node_field_ids, node_kind_ids, node_boxes, node_mask, candidate_pairs):
        outputs = self.model(
            node_hidden=node_hidden,
            node_field_ids=node_field_ids,
            node_kind_ids=node_kind_ids,
            node_boxes=node_boxes,
            node_mask=node_mask,
            candidate_pairs=candidate_pairs,
        )
        return outputs["logits"], outputs["probs"]


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Export the trained span-level rel-g parser to ONNX.")
    parser.add_argument("--checkpoint", default="models/span-relg-item-policy-mixed/best")
    parser.add_argument("--out_dir", default="models/layoutlmv3-item-policy-onnx/int8_dynamic")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--sample_nodes", type=int, default=64)
    parser.add_argument("--sample_pairs", type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint = Path(args.checkpoint)
    out_dir = Path(args.out_dir)
    config_path = checkpoint / "model_config.json"
    model_path = checkpoint / "model.pt"
    schema_path = checkpoint / "schema.json"
    if not config_path.exists():
        raise FileNotFoundError(f"model_config.json not found: {config_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"model.pt not found: {model_path}")
    if not schema_path.exists():
        raise FileNotFoundError(f"schema.json not found: {schema_path}")

    config = load_json(config_path)
    model = SpanRelGModel(**config)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.eval()
    model.requires_grad_(False)

    n_nodes = args.sample_nodes
    n_pairs = args.sample_pairs
    hidden_dim = int(config["hidden_dim"])
    node_hidden = torch.randn(1, n_nodes, hidden_dim, dtype=torch.float32)
    node_field_ids = torch.zeros(1, n_nodes, dtype=torch.long)
    node_kind_ids = torch.zeros(1, n_nodes, dtype=torch.long)
    node_boxes = torch.rand(1, n_nodes, 4, dtype=torch.float32)
    node_boxes[..., 2:] = torch.maximum(node_boxes[..., :2] + 0.01, node_boxes[..., 2:])
    node_boxes = node_boxes.clamp(0.0, 1.0)
    node_mask = torch.ones(1, n_nodes, dtype=torch.float32)
    head = torch.randint(0, n_nodes, (n_pairs, 1), dtype=torch.long)
    dep = torch.randint(0, n_nodes, (n_pairs, 1), dtype=torch.long)
    batch = torch.zeros(n_pairs, 1, dtype=torch.long)
    candidate_pairs = torch.cat([batch, head, dep], dim=1)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_model = out_dir / "span_relg.onnx"
    out_external_data = out_dir / "span_relg.onnx.data"
    if out_external_data.exists():
        out_external_data.unlink()
    wrapper = SpanRelGOnnxWrapper(model)
    wrapper.eval()
    torch.onnx.export(
        wrapper,
        (node_hidden, node_field_ids, node_kind_ids, node_boxes, node_mask, candidate_pairs),
        str(out_model),
        input_names=[
            "node_hidden",
            "node_field_ids",
            "node_kind_ids",
            "node_boxes",
            "node_mask",
            "candidate_pairs",
        ],
        output_names=["logits", "probs"],
        dynamic_axes={
            "node_hidden": {1: "num_nodes"},
            "node_field_ids": {1: "num_nodes"},
            "node_kind_ids": {1: "num_nodes"},
            "node_boxes": {1: "num_nodes"},
            "node_mask": {1: "num_nodes"},
            "candidate_pairs": {0: "num_pairs"},
            "logits": {0: "num_pairs"},
            "probs": {0: "num_pairs"},
        },
        opset_version=args.opset,
        do_constant_folding=True,
        external_data=False,
    )
    shutil.copy2(schema_path, out_dir / "span_relg_schema.json")
    shutil.copy2(config_path, out_dir / "span_relg_model_config.json")
    manifest = {
        "source_checkpoint": str(checkpoint),
        "onnx_model": str(out_model),
        "schema": str(out_dir / "span_relg_schema.json"),
        "model_config": str(out_dir / "span_relg_model_config.json"),
        "opset": args.opset,
        "inputs": {
            "node_hidden": ["batch", "num_nodes", hidden_dim],
            "node_field_ids": ["batch", "num_nodes"],
            "node_kind_ids": ["batch", "num_nodes"],
            "node_boxes": ["batch", "num_nodes", 4],
            "node_mask": ["batch", "num_nodes"],
            "candidate_pairs": ["num_pairs", 3],
        },
        "outputs": {"logits": ["num_pairs"], "probs": ["num_pairs"]},
    }
    with (out_dir / "span_relg_export_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print(f"Exported span rel-g ONNX: {out_model}")
    print(f"Copied schema: {out_dir / 'span_relg_schema.json'}")
    print(f"Copied config: {out_dir / 'span_relg_model_config.json'}")


if __name__ == "__main__":
    main()
