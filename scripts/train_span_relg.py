import argparse
import json
import math
import shutil
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
from ml.span_relg.metrics import aggregate_metrics
from ml.span_relg.model import SpanRelGModel


def parse_args():
    parser = argparse.ArgumentParser(description="Train the span-level rel-g parser on cached CORD span graphs.")
    parser.add_argument("--dataset_dir", default="processed_data/span_relg")
    parser.add_argument("--output_dir", default="models/span-relg-context")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--d_model", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--best_metric",
        default="menu_price_pair_f1",
        choices=("item_price_pair_f1", "menu_price_pair_f1", "edge_f1", "eval_loss"),
        help="Metric used to save best/. eval_loss is minimized; F1 metrics are maximized.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        default=None,
        help="Optional span rel-g checkpoint directory containing model.pt and model_config.json.",
    )
    parser.add_argument(
        "--plot_path",
        default=None,
        help="Optional path for the train/validation curve PNG. Defaults to output_dir/training_curve.png.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite_output_dir", action="store_true")
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


class SpanRelGCacheDataset(Dataset):
    def __init__(self, records):
        self.records = list(records)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        return torch.load(self.records[idx]["path"], map_location="cpu", weights_only=False)


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_manifest(dataset_dir):
    dataset_dir = Path(dataset_dir)
    manifest_path = dataset_dir / "manifest.json"
    schema_path = dataset_dir / "schema.json"
    if not manifest_path.exists():
        fail(f"{manifest_path} not found. Run build_span_relg_dataset.py first.")
    if not schema_path.exists():
        fail(f"{schema_path} not found. Run build_span_relg_dataset.py first.")
    return load_json(manifest_path), load_json(schema_path)


def collate_graphs(samples):
    batch_size = len(samples)
    max_nodes = max(int(sample["node_hidden"].shape[0]) for sample in samples)
    hidden_dim = int(samples[0]["node_hidden"].shape[-1])
    node_hidden = torch.zeros(batch_size, max_nodes, hidden_dim, dtype=torch.float32)
    node_field_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_kind_ids = torch.zeros(batch_size, max_nodes, dtype=torch.long)
    node_boxes = torch.zeros(batch_size, max_nodes, 4, dtype=torch.float32)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    pair_rows = []
    pair_labels = []
    pair_counts = []
    for batch_idx, sample in enumerate(samples):
        n_nodes = int(sample["node_hidden"].shape[0])
        node_hidden[batch_idx, :n_nodes] = sample["node_hidden"].float()
        node_field_ids[batch_idx, :n_nodes] = sample["node_field_ids"].long()
        node_kind_ids[batch_idx, :n_nodes] = sample["node_kind_ids"].long()
        node_boxes[batch_idx, :n_nodes] = sample["node_boxes"].float()
        node_mask[batch_idx, :n_nodes] = True
        pairs = sample["candidate_pairs"].long()
        labels = sample["pair_labels"].float()
        pair_counts.append(int(pairs.shape[0]))
        if pairs.numel():
            batch_col = torch.full((pairs.shape[0], 1), batch_idx, dtype=torch.long)
            pair_rows.append(torch.cat([batch_col, pairs], dim=1))
            pair_labels.append(labels)
    if pair_rows:
        candidate_pairs = torch.cat(pair_rows, dim=0)
        labels = torch.cat(pair_labels, dim=0)
    else:
        candidate_pairs = torch.empty((0, 3), dtype=torch.long)
        labels = torch.empty((0,), dtype=torch.float32)
    return {
        "samples": samples,
        "node_hidden": node_hidden,
        "node_field_ids": node_field_ids,
        "node_kind_ids": node_kind_ids,
        "node_boxes": node_boxes,
        "node_mask": node_mask,
        "candidate_pairs": candidate_pairs,
        "pair_labels": labels,
        "pair_counts": pair_counts,
    }


def move_batch(batch, device):
    return {
        key: value.to(device)
        for key, value in batch.items()
        if key
        in {
            "node_hidden",
            "node_field_ids",
            "node_kind_ids",
            "node_boxes",
            "node_mask",
            "candidate_pairs",
            "pair_labels",
        }
    }


def split_probs_by_sample(probs, pair_counts):
    probs = probs.detach().cpu().tolist()
    out = []
    offset = 0
    for count in pair_counts:
        out.append(probs[offset : offset + count])
        offset += count
    return out


def evaluate(model, loader, device, threshold):
    model.eval()
    all_samples = []
    probs_by_sample = []
    total_loss = 0.0
    total_pairs = 0
    with torch.no_grad():
        for batch in loader:
            if batch["candidate_pairs"].numel() == 0:
                continue
            model_batch = move_batch(batch, device)
            output = model(**model_batch)
            pair_count = int(model_batch["pair_labels"].numel())
            total_loss += float(output["loss"].item()) * pair_count
            total_pairs += pair_count
            all_samples.extend(batch["samples"])
            probs_by_sample.extend(split_probs_by_sample(output["probs"], batch["pair_counts"]))
    metrics = aggregate_metrics(all_samples, probs_by_sample, threshold) if all_samples else {}
    metrics["eval_loss"] = total_loss / total_pairs if total_pairs else None
    metrics["num_samples"] = len(all_samples)
    metrics["num_pairs"] = total_pairs
    model.train()
    return metrics, all_samples, probs_by_sample


def prepare_output_dir(output_dir, overwrite):
    output_dir = Path(output_dir)
    if output_dir.exists():
        if not overwrite:
            fail(f"{output_dir} already exists. Use --overwrite_output_dir to replace it.")
        print(f"Removing existing output directory: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def save_checkpoint(path, model, schema, config, metrics):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path / "model.pt")
    save_json(path / "schema.json", schema)
    save_json(path / "model_config.json", config)
    save_json(path / "metrics.json", metrics)


def best_initial_value(metric_name):
    return math.inf if metric_name == "eval_loss" else -math.inf


def is_better_metric(value, best_value, metric_name):
    if value is None:
        return False
    if metric_name == "eval_loss":
        return float(value) < float(best_value)
    return float(value) > float(best_value)


def plot_training_history(path, history, best_epoch=None, best_metric="menu_price_pair_f1"):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"WARNING: could not import matplotlib; skipped plot: {exc}")
        return None

    if not history:
        return None
    epochs = [row["epoch"] for row in history]
    train_loss = [row.get("train_loss") for row in history]
    eval_loss = [row.get("eval_loss") for row in history]
    f1 = [row.get("menu_price_pair_f1") for row in history]

    fig, loss_ax = plt.subplots(figsize=(11, 6))
    loss_ax.plot(epochs, train_loss, label="train loss", color="#2f6fed", linewidth=2)
    loss_ax.plot(epochs, eval_loss, label="val loss", color="#e07a2f", linewidth=2)
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylabel("Loss")
    loss_ax.grid(True, alpha=0.25)

    f1_ax = loss_ax.twinx()
    f1_ax.plot(epochs, f1, label="validation MENU_PRICE pair F1", color="#24945a", linewidth=2)
    f1_ax.set_ylabel("F1")
    f1_ax.set_ylim(0.0, 1.05)

    if best_epoch is not None:
        loss_ax.axvline(best_epoch, color="#555555", linestyle="--", linewidth=1.5, alpha=0.8)
        loss_ax.text(
            best_epoch,
            max(v for v in eval_loss + train_loss if v is not None),
            f" best {best_metric}: {best_epoch}",
            rotation=90,
            va="top",
            ha="right",
            fontsize=9,
            color="#555555",
        )

    lines, labels = loss_ax.get_legend_handles_labels()
    f1_lines, f1_labels = f1_ax.get_legend_handles_labels()
    loss_ax.legend(lines + f1_lines, labels + f1_labels, loc="upper right")
    fig.tight_layout()

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def label_distribution(records):
    counts = Counter()
    for record in records:
        sample = torch.load(record["path"], map_location="cpu", weights_only=False)
        counts.update(int(v.item()) for v in sample["pair_labels"])
    return dict(counts)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    dataset_dir = Path(args.dataset_dir)
    manifest, schema = load_manifest(dataset_dir)
    output_dir = prepare_output_dir(args.output_dir, args.overwrite_output_dir)
    device = select_device(args.device)
    print(f"dataset_dir: {dataset_dir}")
    print(f"output_dir: {output_dir}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    print(f"selected device: {device}")
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)}")

    split_records = manifest.get("splits", {})
    train_records = split_records.get("train", [])
    val_records = split_records.get("validation") or split_records.get("test") or []
    if not train_records:
        fail("No train samples in manifest.")
    if not val_records:
        print("WARNING: no validation/test split in manifest; train metrics will be used as fallback.")
        val_records = train_records

    print(f"train samples: {len(train_records)}")
    print(f"validation samples: {len(val_records)}")
    if args.debug:
        print(f"train pair label distribution: {label_distribution(train_records)}")
        print(f"validation pair label distribution: {label_distribution(val_records)}")

    first = torch.load(train_records[0]["path"], map_location="cpu", weights_only=False)
    hidden_dim = int(schema.get("hidden_dim") or first["node_hidden"].shape[-1])
    num_fields = len(schema["field_list"])
    num_kinds = len(schema["kind2id"])
    model_config = {
        "hidden_dim": hidden_dim,
        "num_fields": num_fields,
        "num_kinds": num_kinds,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "dropout": args.dropout,
    }
    if args.resume_from_checkpoint:
        resume_dir = Path(args.resume_from_checkpoint)
        if not (resume_dir / "model.pt").exists():
            fail(f"{resume_dir / 'model.pt'} not found.")
        if not (resume_dir / "model_config.json").exists():
            fail(f"{resume_dir / 'model_config.json'} not found.")
        model_config = load_json(resume_dir / "model_config.json")
        print(f"resume_from_checkpoint: {resume_dir}")
    model = SpanRelGModel(**model_config).to(device)
    if args.resume_from_checkpoint:
        state = torch.load(Path(args.resume_from_checkpoint) / "model.pt", map_location="cpu", weights_only=False)
        model.load_state_dict(state)
        print("Loaded model weights from resume checkpoint.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    train_loader = DataLoader(
        SpanRelGCacheDataset(train_records),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_graphs,
    )
    val_loader = DataLoader(
        SpanRelGCacheDataset(val_records),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_graphs,
    )

    total_pos = 0
    total_neg = 0
    for record in train_records:
        sample = torch.load(record["path"], map_location="cpu", weights_only=False)
        labels = sample["pair_labels"]
        total_pos += int(labels.sum().item())
        total_neg += int(labels.numel() - labels.sum().item())
    pos_weight = torch.tensor([total_neg / max(total_pos, 1)], dtype=torch.float32, device=device)
    print(f"train positive pairs: {total_pos}")
    print(f"train negative pairs: {total_neg}")
    print(f"pos_weight: {pos_weight.item():.4f}")

    history = []
    best_score = best_initial_value(args.best_metric)
    best_epoch = None
    initial_loss = None
    final_loss = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_pairs = 0
        for batch in tqdm(train_loader, desc=f"epoch {epoch:03d}/{args.epochs:03d}", unit="batch"):
            if batch["candidate_pairs"].numel() == 0:
                continue
            model_batch = move_batch(batch, device)
            output = model(**model_batch, pos_weight=pos_weight)
            loss = output["loss"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            pair_count = int(model_batch["pair_labels"].numel())
            running_loss += float(loss.item()) * pair_count
            running_pairs += pair_count
            if args.debug and epoch == 1 and running_pairs == pair_count:
                print("first batch shapes:")
                for key, value in model_batch.items():
                    print(f"  {key}: {tuple(value.shape)}")
                print(f"first batch loss: {loss.item():.6f}")
        train_loss = running_loss / running_pairs if running_pairs else None
        if initial_loss is None:
            initial_loss = train_loss
        final_loss = train_loss
        eval_metrics, eval_samples, probs_by_sample = evaluate(model, val_loader, device, args.threshold)
        edge_f1 = eval_metrics.get("edge", {}).get("f1", 0.0)
        item_price_metric = eval_metrics.get("item_price_pair", eval_metrics.get("menu_price_pair", {}))
        menu_price_metric = eval_metrics.get("menu_price_pair", item_price_metric)
        item_price_f1 = item_price_metric.get("f1", 0.0)
        menu_price_f1 = menu_price_metric.get("f1", item_price_f1)
        metric_values = {
            "edge_f1": edge_f1,
            "item_price_pair_f1": item_price_f1,
            "menu_price_pair_f1": menu_price_f1,
            "eval_loss": eval_metrics.get("eval_loss"),
        }
        score = metric_values[args.best_metric]
        best_so_far = is_better_metric(score, best_score, args.best_metric)
        if best_so_far:
            best_score = score
            best_epoch = epoch
            save_checkpoint(output_dir / "best", model, schema, model_config, eval_metrics)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "eval_loss": eval_metrics.get("eval_loss"),
            "edge_f1": edge_f1,
            "item_price_pair_f1": item_price_f1,
            "item_price_pair_precision": item_price_metric.get("precision", 0.0),
            "item_price_pair_recall": item_price_metric.get("recall", 0.0),
            "menu_price_pair_f1": menu_price_f1,
            "menu_price_pair_precision": menu_price_metric.get("precision", 0.0),
            "menu_price_pair_recall": menu_price_metric.get("recall", 0.0),
            "best_metric": args.best_metric,
            "best_metric_value": score,
            "best_so_far": best_so_far,
        }
        history.append(row)
        print(
            f"Epoch {epoch:03d}/{args.epochs:03d} | "
            f"train_loss={train_loss:.6f} | "
            f"val_loss={eval_metrics.get('eval_loss'):.6f} | "
            f"edge_f1={edge_f1:.4f} | "
            f"menu_price_pair_f1={menu_price_f1:.4f} | "
            f"best_epoch={best_epoch}"
        )

    final_metrics, eval_samples, probs_by_sample = evaluate(model, val_loader, device, args.threshold)
    save_checkpoint(output_dir / "last", model, schema, model_config, final_metrics)
    best_metrics = load_json(output_dir / "best" / "metrics.json") if (output_dir / "best" / "metrics.json").exists() else {}
    config = {
        "dataset_dir": str(dataset_dir),
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "threshold": args.threshold,
        "best_metric": args.best_metric,
        "resume_from_checkpoint": args.resume_from_checkpoint,
        "seed": args.seed,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "hidden_dim": hidden_dim,
        "num_fields": num_fields,
        "num_kinds": num_kinds,
    }
    save_json(output_dir / "training_config.json", config)
    save_json(output_dir / "training_history.json", {"history": history})
    save_json(
        output_dir / "best_metrics.json",
        {"best_epoch": best_epoch, "best_metric": args.best_metric, "best_metric_value": best_score, "metrics": best_metrics},
    )
    save_json(output_dir / "final_metrics.json", final_metrics)

    plot_path = Path(args.plot_path) if args.plot_path else output_dir / "training_curve.png"
    saved_plot_path = plot_training_history(plot_path, history, best_epoch=best_epoch, best_metric=args.best_metric)

    preview_path = output_dir / "validation_decoded_preview.jsonl"
    with preview_path.open("w", encoding="utf-8") as handle:
        for sample, probs in list(zip(eval_samples, probs_by_sample))[:20]:
            decoded = decode_edges_to_items(sample, probs, threshold=args.threshold)
            handle.write(json.dumps({"id": sample["data_id"], **decoded}, ensure_ascii=False) + "\n")

    print(f"initial_loss: {initial_loss}")
    print(f"final_loss: {final_loss}")
    print(f"best_epoch: {best_epoch}")
    print(f"best checkpoint: {output_dir / 'best'}")
    print(f"last checkpoint: {output_dir / 'last'}")
    print(f"training_history path: {output_dir / 'training_history.json'}")
    if saved_plot_path:
        print(f"training_curve path: {saved_plot_path}")
    print("Span rel-g debug/full training passed.")


if __name__ == "__main__":
    main()
