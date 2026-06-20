from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from receipt_ocr_compare.config import CompareConfig, collect_images, parse_csv  # noqa: E402
from receipt_ocr_compare.crops import create_crops  # noqa: E402
from receipt_ocr_compare.detection import detect_boxes, gt_tokens_for_image, load_ground_truth_jsonl  # noqa: E402
from receipt_ocr_compare.manifest import create_run_manifest, repo_root, sha256_file, write_json  # noqa: E402
from receipt_ocr_compare.metrics import confusion_matrix_rows, per_token_comparison, summarize_model_metrics  # noqa: E402
from receipt_ocr_compare.model_registry import available_model_ids, create_adapters  # noqa: E402
from receipt_ocr_compare.overlay import render_comparison_overlay, render_model_overlay  # noqa: E402
from receipt_ocr_compare.schemas import RecognitionResult, RunContext  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "compare":
        return compare_command(args)
    parser.print_help()
    return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Receipt OCR model token overlay comparison")
    subparsers = parser.add_subparsers(dest="command")
    compare = subparsers.add_parser("compare", help="Run OCR token comparison")
    compare.add_argument("--input", required=True, help="Image file or directory")
    compare.add_argument("--models", default="svtrv2_b,paddleocr,existing", help="Comma-separated model ids")
    compare.add_argument("--mode", choices=["recognition", "end-to-end"], default="recognition")
    compare.add_argument("--detector", choices=["auto", "existing", "paddleocr", "simple", "ground_truth"], default="auto")
    compare.add_argument("--model-dir", default="tools/receipt_ocr_compare/models")
    compare.add_argument("--vendor-dir", default="tools/receipt_ocr_compare/vendor")
    compare.add_argument("--output", required=True)
    compare.add_argument("--ground-truth", default=None)
    compare.add_argument("--crop-padding", type=int, default=2)
    compare.add_argument("--device", choices=["cpu", "gpu"], default="cpu")
    compare.add_argument("--numeric-only-overlay", action="store_true")
    compare.add_argument("--mismatches-only-overlay", action="store_true")
    compare.add_argument("--hide-confidence", action="store_true")
    compare.add_argument("--hide-boxes", action="store_true")
    return parser


def compare_command(args: argparse.Namespace) -> int:
    config = CompareConfig(
        input_path=Path(args.input),
        models=parse_csv(args.models),
        mode=args.mode,
        detector=args.detector,
        model_dir=Path(args.model_dir),
        vendor_dir=Path(args.vendor_dir),
        output_dir=Path(args.output),
        crop_padding=args.crop_padding,
        device=args.device,
        ground_truth_path=Path(args.ground_truth) if args.ground_truth else None,
        allow_package_models=False,
    )
    if config.mode == "end-to-end":
        print("End-to-end mode is available only for adapters with configured full-pipeline runners; metrics are kept separate.")
    return run_compare(
        config,
        numeric_only_overlay=args.numeric_only_overlay,
        mismatches_only_overlay=args.mismatches_only_overlay,
        show_confidence=not args.hide_confidence,
        show_boxes=not args.hide_boxes,
    )


def run_compare(
    config: CompareConfig,
    *,
    numeric_only_overlay: bool = False,
    mismatches_only_overlay: bool = False,
    show_confidence: bool = True,
    show_boxes: bool = True,
) -> int:
    images = collect_images(config.input_path)
    if not images:
        raise SystemExit(f"No supported image files found at {config.input_path}")

    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir = output_dir / "overlays"
    crops_dir = output_dir / "crops"
    gt_by_image = load_ground_truth_jsonl(config.ground_truth_path)

    context = RunContext(
        model_dir=config.model_dir,
        vendor_dir=config.vendor_dir,
        device=config.device,
        allow_package_models=config.allow_package_models,
    )
    adapters = create_adapters(config.models, context)

    all_boxes = []
    all_crops = []
    all_results: list[RecognitionResult] = []
    gt_flat = []
    detector_records: list[dict[str, Any]] = []
    model_records = []

    for adapter in adapters:
        availability = adapter.availability()
        model_records.append(
            {
                **availability.to_dict(),
                "framework_version": _framework_version(adapter.model_id),
                "checkpoint_path": _checkpoint_path(config.model_dir, adapter.model_id),
                "checkpoint_checksum": _checkpoint_checksum(config.model_dir, adapter.model_id),
                "source_repository": _source_repository(adapter.model_id),
                "license": _license(adapter.model_id),
            }
        )

    for image_path in images:
        gt_tokens = gt_tokens_for_image(gt_by_image, image_path)
        gt_flat.extend(gt_tokens)
        try:
            boxes, detector_status = detect_boxes(
                image_path,
                detector=config.detector,
                model_dir=config.model_dir,
                vendor_dir=config.vendor_dir,
                ground_truth=gt_tokens,
            )
        except Exception as exc:
            boxes = []
            detector_status = {"detector": config.detector, "available": False, "error": str(exc)}
        detector_records.append({"image": image_path.name, **detector_status, "box_count": len(boxes)})
        all_boxes.extend(boxes)
        crops = create_crops(
            image_path,
            boxes,
            output_dir=crops_dir,
            padding=config.crop_padding,
            ground_truth=gt_tokens,
        )
        all_crops.extend(crops)

        image_results_by_model: dict[str, list[RecognitionResult]] = {}
        for adapter in adapters:
            results = adapter.recognize(crops)
            image_results_by_model[adapter.model_id] = results
            all_results.extend(results)
            overlay_path = overlays_dir / f"{image_path.stem}_{adapter.model_id}.png"
            render_model_overlay(
                image_path,
                boxes,
                results,
                output_path=overlay_path,
                ground_truth=gt_tokens,
                numeric_only=numeric_only_overlay,
                mismatches_only=mismatches_only_overlay,
                show_confidence=show_confidence,
                show_boxes=show_boxes,
            )
        comparison_inputs = [overlays_dir / f"{image_path.stem}_{adapter.model_id}.png" for adapter in adapters]
        if comparison_inputs:
            render_comparison_overlay(
                comparison_inputs,
                [adapter.model_id for adapter in adapters],
                overlays_dir / f"{image_path.stem}_comparison.png",
            )

    _write_jsonl(output_dir / "predictions.jsonl", [result.to_dict() for result in all_results])
    _write_jsonl(output_dir / "detector_boxes.jsonl", [box.to_dict() for box in all_boxes])
    _write_jsonl(output_dir / "crops_manifest.jsonl", [crop.to_dict() for crop in all_crops])

    if gt_flat:
        per_token_rows = per_token_comparison(gt_flat, all_results)
        summary_rows = summarize_model_metrics(per_token_rows)
        confusion_rows = confusion_matrix_rows(per_token_rows)
        _write_csv(output_dir / "per_token_results.csv", per_token_rows)
        _write_csv(output_dir / "model_summary.csv", summary_rows)
        _write_csv(output_dir / "confusion_matrix.csv", confusion_rows)
    else:
        _write_csv(output_dir / "per_token_results.csv", [{"note": "Visual comparison only: no ground truth provided."}])
        _write_csv(output_dir / "model_summary.csv", [{"note": "Visual comparison only: no ground truth provided."}])
        _write_csv(output_dir / "confusion_matrix.csv", [{"note": "Visual comparison only: no ground truth provided."}])

    manifest = create_run_manifest(
        repo=repo_root(Path.cwd()),
        input_images=images,
        model_records=model_records,
        device=config.device,
        preprocessing={"crop_padding": config.crop_padding},
        detector={"selected": config.detector, "records": detector_records},
        normalization={"primary_metric_text": "raw_text", "ocr_confusion_corrections_in_primary_metric": False},
    )
    write_json(output_dir / "run_manifest.json", manifest)
    print(f"Wrote comparison outputs to {output_dir}")
    if not gt_flat:
        print("Visual comparison only: no ground truth provided.")
    return 0


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()}) or ["note"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fieldnames})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


def _framework_version(model_id: str) -> str | None:
    package = "paddleocr" if model_id == "paddleocr" else None
    if package is None:
        return None
    try:
        module = __import__(package)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


def _checkpoint_path(model_dir: Path, model_id: str) -> str | None:
    root = model_dir / model_id
    if model_id == "paddleocr":
        root = model_dir / "paddleocr"
    for suffix in ("*.pth", "*.pt", "*.ckpt", "*.safetensors", "*.onnx", "*.pdparams", "*.pdmodel", "*.bin"):
        matches = sorted(root.rglob(suffix))
        if matches:
            return str(matches[0])
    return None


def _checkpoint_checksum(model_dir: Path, model_id: str) -> str | None:
    path = _checkpoint_path(model_dir, model_id)
    return sha256_file(Path(path)) if path else None


def _source_repository(model_id: str) -> str | None:
    return {
        "svtrv2_b": "https://github.com/Topdu/OpenOCR",
        "paddleocr": "https://github.com/PaddlePaddle/PaddleOCR",
        "existing": "ReceiptApp/app/src/main/java/com/receiptapp/ocr",
    }.get(model_id)


def _license(model_id: str) -> str | None:
    return {
        "svtrv2_b": "Apache-2.0",
        "paddleocr": "Apache-2.0",
        "existing": "project license",
    }.get(model_id)


if __name__ == "__main__":
    raise SystemExit(main())
