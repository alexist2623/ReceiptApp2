from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.angle_geometry import build_angle_features_for_words, angle_feature_dim_for_mode, clamp_box, normalize_box_1000
from ml.receipt_schema import canonicalize_label
from ml.span_relg.decode import decode_edges_to_items
from ml.span_relg.feature_cache import build_cache_sample
from ml.span_relg.io_utils import resolve_field_vocab
from ml.span_relg.span_utils import bio_predictions_to_spans
from ml.span_relg.visualization import draw_span_relg_overlay, draw_user_item_mapping_overlay
from scripts.eval_predicted_span_relg_e2e import (
    attach_boxes_to_predictions,
    edge_payload,
    filter_candidate_spans,
    load_layoutlmv3,
    load_rel_model,
    resolve_kind_vocab,
    run_layout_prediction,
    run_rel_model,
    select_device,
)
from scripts.infer_user_ocr_json import draw_label, draw_rectangle, label_to_color
from scripts.overlay_paddleocr_vs_existing_ocr import run_paddle_ocr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run PaddleOCR output through the current LayoutLMv3 + span rel-g pipeline and draw overlays."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--ocr_json",
        default=None,
        help="Use an existing OCR JSON with words[].text/box/quad instead of running PaddleOCR in this script.",
    )
    parser.add_argument("--out_dir", default="outputs/paddleocr_current_model_pipeline")
    parser.add_argument("--layoutlm_checkpoint", default="models/layoutlmv3-angle-mixed-public-user-35ep/best")
    parser.add_argument(
        "--relg_checkpoint",
        default="models/span-relg-cord-wild-custom-angle-mixed-public-user-35ep-100ep/best",
    )
    parser.add_argument(
        "--span_relg_dataset_dir",
        default="processed_data/span_relg_cord_wild_custom_angle_mixed_public_user_35ep",
    )
    parser.add_argument("--paddle_model_dir", default="tools/receipt_ocr_compare/models")
    parser.add_argument("--threshold", type=float, default=0.84)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument(
        "--return_word_box",
        action="store_true",
        help="Use PaddleOCR recognizer with return_word_box=True and feed word-level boxes into LayoutLMv3/rel-g.",
    )
    parser.add_argument("--max_text_len", type=int, default=24)
    parser.add_argument("--mobile_preview_width", type=int, default=1400)
    return parser.parse_args()


def fail(message: str) -> None:
    print(f"ERROR: {message}", file=sys.stderr)
    raise SystemExit(1)


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_image(path: Path) -> Image.Image:
    return ImageOps.exif_transpose(Image.open(path)).convert("RGB")


def flatten_paddleocr3_word_boxes(res: dict, width: int, height: int) -> tuple[list[dict], int]:
    text_words = res.get("text_word") or []
    text_word_boxes = res.get("text_word_boxes") or []
    rec_texts = res.get("rec_texts") or []
    rec_scores = res.get("rec_scores") or []
    words: list[dict] = []
    skipped = 0

    for line_idx, (line_words, line_boxes) in enumerate(zip(text_words, text_word_boxes)):
        if not isinstance(line_words, list) or not isinstance(line_boxes, list):
            skipped += 1
            continue
        for line_word_idx, (text, raw_box) in enumerate(zip(line_words, line_boxes)):
            text = str(text or "").strip()
            if not text or not isinstance(raw_box, list) or len(raw_box) != 4:
                skipped += 1
                continue
            try:
                box = [int(round(float(v))) for v in raw_box]
            except (TypeError, ValueError):
                skipped += 1
                continue
            box = clamp_box(box, width, height)
            if box is None:
                skipped += 1
                continue
            words.append(
                {
                    "text": text,
                    "box": box,
                    "confidence": rec_scores[line_idx] if line_idx < len(rec_scores) else None,
                    "source": "paddleocr3_return_word_box",
                    "line_idx": line_idx,
                    "line_word_idx": line_word_idx,
                    "word_idx": len(words),
                    "line_text": rec_texts[line_idx] if line_idx < len(rec_texts) else None,
                }
            )
    return words, skipped


def run_paddleocr3_return_word_box(image_path: Path, width: int, height: int) -> tuple[dict, dict]:
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang="en",
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="PP-OCRv5_mobile_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
        return_word_box=True,
    )
    results = list(ocr.predict(str(image_path), return_word_box=True))
    if not results:
        fail("PaddleOCR returned no results.")
    raw_json = results[0].json
    res = raw_json.get("res") if isinstance(raw_json, dict) else None
    if not isinstance(res, dict):
        fail("PaddleOCR result JSON does not contain res dict.")
    if res.get("return_word_box") is not True:
        fail("PaddleOCR result did not confirm return_word_box=True.")
    words, skipped = flatten_paddleocr3_word_boxes(res, width, height)
    if not words:
        fail("PaddleOCR returned no valid text_word_boxes.")
    payload = {
        "image_width": width,
        "image_height": height,
        "ocr_engine": "paddleocr3",
        "coordinate_space": "image_pixels",
        "return_word_box_requested": True,
        "return_word_box_native_available": True,
        "word_box_source": "PaddleOCR.result.res.text_word_boxes",
        "raw_box_count": len(res.get("dt_polys") or []),
        "line_count": len(res.get("rec_texts") or []),
        "word_count": len(words),
        "skipped_word_box_count": skipped,
        "words": words,
    }
    return payload, raw_json


def paddle_words_to_ocr_json(
    image: Image.Image,
    image_path: Path,
    paddle_model_dir: Path,
    out_path: Path,
    *,
    return_word_box: bool = False,
) -> dict:
    if return_word_box:
        payload, raw_json = run_paddleocr3_return_word_box(image_path, image.width, image.height)
        save_json(out_path, payload)
        save_json(out_path.with_name(f"{out_path.stem}_raw_paddleocr3_result.json"), raw_json)
        return payload

    paddle_boxes = run_paddle_ocr(image_path=image_path, model_dir=paddle_model_dir, recognize=True)
    words = []
    for item in paddle_boxes:
        text = str(item.get("text") or "").strip()
        box = item.get("box")
        if not text or not isinstance(box, list) or len(box) != 4:
            continue
        words.append(
            {
                "text": text,
                "box": [int(v) for v in box],
                "quad": item.get("quad"),
                "confidence": item.get("confidence"),
                "source": "paddleocr",
            }
        )
    payload = {
        "image_width": image.width,
        "image_height": image.height,
        "ocr_engine": "paddleocr",
        "coordinate_space": "image_pixels",
        "return_word_box_requested": False,
        "return_word_box_native_available": False,
        "raw_box_count": len(paddle_boxes),
        "words": words,
    }
    save_json(out_path, payload)
    return payload


def make_mobile_preview(path: Path, max_width: int) -> Path:
    image = Image.open(path).convert("RGB")
    if max_width > 0 and image.width > max_width:
        new_height = int(image.height * max_width / image.width)
        resampling = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize((max_width, new_height), resampling)
    preview = path.with_name(f"{path.stem}_mobile_preview{path.suffix}")
    image.save(preview)
    return preview


def draw_prediction_label_overlay(
    image: Image.Image,
    predictions: list[dict],
    out_path: Path,
    max_text_len: int,
) -> Path:
    output = image.copy().convert("RGB")
    draw = ImageDraw.Draw(output)
    font = ImageFont.load_default()
    for item in predictions:
        label = item.get("label") or "O"
        color = label_to_color(label)
        draw_rectangle(draw, item["box"], color, width=3 if label != "O" else 1)
        text = f"{label} {float(item.get('confidence', 0.0)):.2f}\n{str(item.get('text', ''))[:max_text_len]}"
        draw_label(draw, item["box"], text, color, font, output.size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(out_path)
    return out_path


def prepare_words(ocr_payload: dict, width: int, height: int) -> tuple[list[str], list[list[int]], list[list[int]], list[dict]]:
    words: list[str] = []
    boxes: list[list[int]] = []
    normalized_boxes: list[list[int]] = []
    word_payloads: list[dict] = []
    for item in ocr_payload.get("words", []):
        text = str(item.get("text") or "").strip()
        box = clamp_box(item.get("box"), width, height)
        if not text or box is None:
            continue
        words.append(text)
        boxes.append(box)
        normalized_boxes.append(normalize_box_1000(box, width, height))
        word_payloads.append(
            {
                "text": text,
                "box": box,
                "quad": item.get("quad"),
                "confidence": item.get("confidence"),
                "source": item.get("source"),
            }
        )
    if not words:
        fail("No valid PaddleOCR words after filtering.")
    return words, boxes, normalized_boxes, word_payloads


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    for path, label in [
        (image_path, "image"),
        (Path(args.layoutlm_checkpoint), "layoutlm_checkpoint"),
        (Path(args.relg_checkpoint), "relg_checkpoint"),
        (Path(args.span_relg_dataset_dir), "span_relg_dataset_dir"),
        (Path(args.paddle_model_dir), "paddle_model_dir"),
    ]:
        if not path.exists():
            fail(f"{label} not found: {path}")

    stem = image_path.stem
    out_dir = Path(args.out_dir) / stem
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"cuda_device: {torch.cuda.get_device_name(0)}")
    device = select_device(args.device)
    print(f"selected_device: {device}")

    image = load_image(image_path)
    width, height = image.size
    if args.ocr_json:
        source_ocr_json = Path(args.ocr_json)
        if not source_ocr_json.exists():
            fail(f"ocr_json not found: {source_ocr_json}")
        mode_suffix = "external_ocr"
        ocr_json_path = out_dir / f"{stem}_{source_ocr_json.stem}.json"
        ocr_payload = json.loads(source_ocr_json.read_text(encoding="utf-8"))
        save_json(ocr_json_path, ocr_payload)
    else:
        mode_suffix = "paddle_return_word_box" if args.return_word_box else "paddle"
        ocr_json_path = out_dir / f"{stem}_{mode_suffix}_ocr_full.json"
        ocr_payload = paddle_words_to_ocr_json(
            image,
            image_path,
            Path(args.paddle_model_dir),
            ocr_json_path,
            return_word_box=args.return_word_box,
        )
    words, boxes, normalized_boxes, word_payloads = prepare_words(ocr_payload, width, height)
    print(f"paddle_raw_box_count: {ocr_payload.get('raw_box_count')}")
    print(f"return_word_box_requested: {ocr_payload.get('return_word_box_requested')}")
    print(f"return_word_box_native_available: {ocr_payload.get('return_word_box_native_available')}")
    print(f"paddle_valid_word_count: {len(words)}")

    field_vocab = resolve_field_vocab(args.span_relg_dataset_dir, args.relg_checkpoint)
    field2id = {str(key): int(value) for key, value in field_vocab["vocab"].items()}
    kind2id, kind_source = resolve_kind_vocab(args.span_relg_dataset_dir, args.relg_checkpoint)
    processor, layout_model, _label2id, id2label, label_source = load_layoutlmv3(
        args.layoutlm_checkpoint,
        args.local_files_only,
        device,
        use_angle_features="auto",
    )
    rel_model, rel_config, rel_config_path = load_rel_model(args.relg_checkpoint, device)
    include_context_tokens = rel_config.get("include_context_tokens", "all")
    span_pooling = rel_config.get("span_pooling", "first")

    angle_features = None
    angle_debug = []
    if getattr(layout_model, "uses_angle_features", False):
        angle_config = getattr(layout_model, "angle_feature_config", {})
        angle_mode = angle_config.get("angle_encoding_mode") or "sincos_scalar"
        angle_result = build_angle_features_for_words(
            word_payloads,
            boxes=boxes,
            image_width=width,
            image_height=height,
            mode=angle_mode,
        )
        angle_features = angle_result["angle_features"]
        angle_debug = angle_result["word_angles"]
        expected_dim = int(angle_config.get("angle_feature_dim") or angle_feature_dim_for_mode(angle_mode))
        if angle_features and len(angle_features[0]) != expected_dim:
            fail(f"angle feature dim mismatch: {len(angle_features[0])} != {expected_dim}")

    layout = run_layout_prediction(
        image,
        words,
        normalized_boxes,
        processor,
        layout_model,
        device,
        id2label,
        args.max_length,
        angle_features=angle_features,
    )
    predictions = attach_boxes_to_predictions(layout["predictions"], boxes, normalized_boxes)
    for item in predictions:
        item["raw_label"] = item.get("label")
        item["canonical_label"] = canonicalize_label(item.get("label"))
        item["label"] = item["canonical_label"]

    spans = bio_predictions_to_spans(predictions, width, height)
    filtered_spans, dropped_spans = filter_candidate_spans(spans, field2id)
    sample_info = {
        "image": image,
        "width": width,
        "height": height,
        "words": words,
        "boxes": boxes,
        "normalized_boxes": normalized_boxes,
        "angle_features": angle_features,
        "predictions": predictions,
        "spans": filtered_spans,
    }
    cache = build_cache_sample(
        f"{stem}_paddleocr",
        "paddleocr",
        0,
        sample_info,
        layout["word_hidden"],
        field2id,
        kind2id,
        include_context_tokens=include_context_tokens,
        span_pooling=span_pooling,
    )
    probs = run_rel_model(rel_model, cache, device)
    edges = edge_payload(cache, probs, args.threshold)
    selected_edges = [edge for edge in edges if edge.get("pred") == 1]
    decoded = decode_edges_to_items(cache, probs, threshold=args.threshold)

    label_counts = Counter(item["label"] for item in predictions)
    span_counts = Counter(span["field"] for span in filtered_spans)

    prediction_json = out_dir / f"{stem}_{mode_suffix}_layoutlmv3_prediction.json"
    grouped_json = out_dir / f"{stem}_{mode_suffix}_relg_grouped.json"
    debug_json = out_dir / f"{stem}_{mode_suffix}_pipeline_debug.json"
    label_overlay = out_dir / f"{stem}_{mode_suffix}_label_overlay.png"
    mapping_overlay = out_dir / f"{stem}_{mode_suffix}_mapping_overlay.png"
    graph_overlay = out_dir / f"{stem}_{mode_suffix}_relg_graph_overlay.png"

    save_json(
        prediction_json,
        {
            "image_path": str(image_path),
            "ocr_json_path": str(ocr_json_path),
            "layoutlm_checkpoint": args.layoutlm_checkpoint,
            "return_word_box_requested": args.return_word_box,
            "return_word_box_native_available": ocr_payload.get("return_word_box_native_available"),
            "num_words": len(words),
            "predictions": predictions,
            "label_counts": dict(label_counts),
            "encoding_shapes": layout.get("encoding_shapes"),
            "angle_features_shape": layout.get("angle_features_shape"),
        },
    )
    save_json(
        grouped_json,
        {
            "image_path": str(image_path),
            "ocr_json_path": str(ocr_json_path),
            "layoutlm_checkpoint": args.layoutlm_checkpoint,
            "relg_checkpoint": args.relg_checkpoint,
            "return_word_box_requested": args.return_word_box,
            "return_word_box_native_available": ocr_payload.get("return_word_box_native_available"),
            "threshold": args.threshold,
            "num_words": len(words),
            "num_spans": len(filtered_spans),
            "num_candidate_edges": len(edges),
            "num_selected_edges": len(selected_edges),
            "selected_edges": selected_edges,
            "all_candidate_edges": edges,
            "decoded": decoded,
        },
    )
    save_json(
        debug_json,
        {
            "label_source": label_source,
            "field_vocab_source": field_vocab.get("source"),
            "field_vocab_key": field_vocab.get("key"),
            "kind_source": kind_source,
            "relg_config_path": str(rel_config_path),
            "label_counts": dict(label_counts),
            "span_counts": dict(span_counts),
            "dropped_spans": dropped_spans,
            "angle_debug_preview": angle_debug[:30],
            "token_debug_preview": layout.get("token_debug", [])[:80],
        },
    )
    draw_prediction_label_overlay(image, predictions, label_overlay, args.max_text_len)
    draw_user_item_mapping_overlay(image, decoded.get("items", []), mapping_overlay, title=f"{stem} PaddleOCR Item Mapping")
    draw_span_relg_overlay(image, cache, selected_edges, graph_overlay, title=f"{stem} PaddleOCR Rel-G Graph")

    previews = [make_mobile_preview(path, args.mobile_preview_width) for path in (label_overlay, mapping_overlay, graph_overlay)]

    print(f"paddle_ocr_json: {ocr_json_path}")
    print(f"prediction_json: {prediction_json}")
    print(f"grouped_json: {grouped_json}")
    print(f"debug_json: {debug_json}")
    print(f"label_overlay: {label_overlay}")
    print(f"mapping_overlay: {mapping_overlay}")
    print(f"graph_overlay: {graph_overlay}")
    for preview in previews:
        print(f"preview: {preview}")
    print(f"label_counts: {dict(label_counts.most_common())}")
    print(f"span_counts: {dict(span_counts.most_common())}")
    print(f"candidate_edges: {len(edges)}")
    print(f"selected_edges: {len(selected_edges)}")
    print(f"decoded_items: {len(decoded.get('items', []))}")
    print("PaddleOCR current model pipeline passed.")


if __name__ == "__main__":
    main()
