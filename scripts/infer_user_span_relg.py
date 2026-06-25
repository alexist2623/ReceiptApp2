import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.span_relg.decode import decode_edges_to_items
from ml.span_relg.feature_cache import build_cache_sample, compute_word_hidden, load_layoutlmv3
from ml.span_relg.model import SpanRelGModel
from ml.span_relg.schema import ALL_FIELDS
from ml.span_relg.span_utils import bio_predictions_to_spans
from ml.span_relg.visualization import draw_span_relg_overlay, draw_user_item_mapping_overlay
from ml.receipt_schema import field_for_vocab


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run span-level rel-g grouping on a user OCR prediction JSON produced by infer_user_ocr_json.py."
    )
    parser.add_argument("--prediction_json", required=True)
    parser.add_argument("--layoutlm_checkpoint", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--relg_checkpoint", default="models/span-relg-context/best")
    parser.add_argument("--out_json", default=None)
    parser.add_argument("--out_overlay", default=None)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--include_context_tokens", default="all", choices=("all", "o_only", "none"))
    parser.add_argument("--span_pooling", default="first", choices=("first", "mean"))
    parser.add_argument("--debug", action="store_true")
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


def load_image(path):
    image = Image.open(path)
    image = ImageOps.exif_transpose(image)
    return image.convert("RGB")


def load_rel_model(checkpoint, device):
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        fail(f"Span rel-g checkpoint not found: {checkpoint}. Train span rel-g first.")
    config = load_json(checkpoint / "model_config.json")
    model = SpanRelGModel(**config)
    state = torch.load(checkpoint / "model.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    schema_path = checkpoint / "schema.json"
    schema = load_json(schema_path) if schema_path.exists() else None
    return model, config, schema


def sample_from_prediction_json(payload):
    image_path = payload.get("image_path")
    if not image_path:
        fail("prediction_json does not contain image_path; cannot recompute LayoutLMv3 features.")
    image = load_image(image_path)
    width, height = image.size
    predictions = payload.get("predictions")
    if not isinstance(predictions, list) or not predictions:
        fail("prediction_json does not contain non-empty predictions list.")
    words = []
    boxes = []
    normalized_boxes = []
    clean_predictions = []
    for item in predictions:
        text = str(item.get("text", "")).strip()
        box = item.get("box")
        norm_box = item.get("normalized_box")
        label = item.get("label") or item.get("pred_label") or "O"
        if not text or not isinstance(box, list) or not isinstance(norm_box, list):
            continue
        word_idx = len(words)
        words.append(text)
        boxes.append([int(v) for v in box])
        normalized_boxes.append([int(v) for v in norm_box])
        clean_predictions.append(
            {
                "word_idx": word_idx,
                "text": text,
                "box": boxes[-1],
                "normalized_box": normalized_boxes[-1],
                "label": label,
                "confidence": float(item.get("confidence", 1.0)),
                "source": item.get("source"),
                "line_index": item.get("line_index"),
                "block_index": item.get("block_index"),
            }
        )
    if not words:
        fail("No valid word predictions found in prediction_json.")
    spans = bio_predictions_to_spans(clean_predictions, width, height)
    return {
        "image": image,
        "width": width,
        "height": height,
        "words": words,
        "boxes": boxes,
        "normalized_boxes": normalized_boxes,
        "predictions": clean_predictions,
        "spans": spans,
        "image_path": image_path,
    }


def run_rel_model(model, cache, device):
    node_hidden = cache["node_hidden"].unsqueeze(0).float().to(device)
    node_field_ids = cache["node_field_ids"].unsqueeze(0).long().to(device)
    node_kind_ids = cache["node_kind_ids"].unsqueeze(0).long().to(device)
    node_boxes = cache["node_boxes"].unsqueeze(0).float().to(device)
    node_mask = torch.ones(1, cache["node_hidden"].shape[0], dtype=torch.bool, device=device)
    pairs = cache["candidate_pairs"].long()
    if pairs.numel():
        candidate_pairs = torch.cat([torch.zeros(pairs.shape[0], 1, dtype=torch.long), pairs], dim=1).to(device)
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


def main():
    args = parse_args()
    prediction_path = Path(args.prediction_json)
    if not prediction_path.exists():
        fail(f"prediction_json not found: {prediction_path}")
    if args.local_files_only and not Path(args.layoutlm_checkpoint).exists():
        fail(f"LayoutLMv3 checkpoint not found: {args.layoutlm_checkpoint}")
    payload = load_json(prediction_path)
    sample_info = sample_from_prediction_json(payload)
    print(f"prediction_json: {prediction_path}")
    print(f"image_path: {sample_info['image_path']}")
    print(f"image_size: {sample_info['width']}x{sample_info['height']}")
    print(f"word_count: {len(sample_info['words'])}")
    print(f"span_count: {len(sample_info['spans'])}")
    print(f"span label distribution: {dict(Counter(span['field'] for span in sample_info['spans']))}")

    processor, layout_model, device = load_layoutlmv3(args.layoutlm_checkpoint, args.local_files_only, args.device)
    rel_model, rel_config, rel_schema = load_rel_model(args.relg_checkpoint, device)
    schema = rel_schema or {"field2id": {field: idx for idx, field in enumerate(ALL_FIELDS)}, "kind2id": {"SPAN": 0, "TOKEN": 1}}
    field2id = schema.get("field2id") or {field: idx for idx, field in enumerate(schema["field_list"])}
    kind2id = schema.get("kind2id") or {"SPAN": 0, "TOKEN": 1}
    unsupported_spans = [span for span in sample_info["spans"] if field_for_vocab(span.get("field"), field2id) is None]
    if unsupported_spans:
        unsupported_counts = Counter(span.get("field", "UNKNOWN") for span in unsupported_spans)
        print(
            "warning: dropping predicted spans not present in span rel-g schema: "
            f"{dict(unsupported_counts)}"
        )
        sample_info["spans"] = [span for span in sample_info["spans"] if field_for_vocab(span.get("field"), field2id) is not None]

    word_features = compute_word_hidden(
        sample_info["image"],
        sample_info["words"],
        sample_info["normalized_boxes"],
        processor,
        layout_model,
        device,
        max_length=args.max_length,
    )
    cache = build_cache_sample(
        prediction_path.stem,
        "user",
        0,
        sample_info,
        word_features["word_hidden"],
        field2id,
        kind2id,
        include_context_tokens=args.include_context_tokens,
        span_pooling=args.span_pooling,
    )
    probs = run_rel_model(rel_model, cache, device)
    decoded = decode_edges_to_items(cache, probs, threshold=args.threshold)
    edges = []
    for idx, (meta, prob) in enumerate(zip(cache["pair_meta"], probs)):
        edge = dict(meta)
        edge.update(
            {
                "pair_index": idx,
                "head_node_id": meta["head_node_id"],
                "dep_node_id": meta["dep_node_id"],
                "prob": float(prob),
                "selected": float(prob) >= args.threshold,
            }
        )
        edges.append(edge)

    default_dir = Path("outputs/user_ocr_inference")
    out_json = Path(args.out_json) if args.out_json else default_dir / f"{prediction_path.stem}_span_relg.json"
    out_overlay = Path(args.out_overlay) if args.out_overlay else default_dir / f"{prediction_path.stem}_span_relg_overlay.png"
    save_json(
        out_json,
        {
            "prediction_json": str(prediction_path),
            "layoutlm_checkpoint": args.layoutlm_checkpoint,
            "relg_checkpoint": args.relg_checkpoint,
            "threshold": args.threshold,
            "num_words": len(sample_info["words"]),
            "num_spans": len(sample_info["spans"]),
            "dropped_unsupported_spans": unsupported_spans,
            "num_candidate_edges": len(edges),
            "selected_edges": [edge for edge in edges if edge["selected"]],
            "all_candidate_edges": edges,
            **decoded,
        },
    )
    selected_edges = [edge for edge in edges if edge["selected"]]
    edge_overlay = out_overlay.with_name(f"{out_overlay.stem}_edges{out_overlay.suffix}")
    draw_user_item_mapping_overlay(sample_info["image"], decoded.get("items", []), out_overlay, title=f"{prediction_path.stem} item mapping")
    draw_span_relg_overlay(sample_info["image"], cache, selected_edges, edge_overlay, title=f"{prediction_path.stem} rel-g edges")
    if args.debug:
        for span in sample_info["spans"][:30]:
            print(f"span {span['span_id']}: {span['field']} {span['text']!r}")
        for edge in edges[:50]:
            print(f"edge {edge['head_text']!r} -> {edge['dep_text']!r} {edge['dep_field']} p={edge['prob']:.3f}")
    print(f"relg JSON path: {out_json}")
    print(f"relg overlay path: {out_overlay}")
    print(f"relg edge debug overlay path: {edge_overlay}")
    print("User span rel-g inference passed.")


if __name__ == "__main__":
    main()
