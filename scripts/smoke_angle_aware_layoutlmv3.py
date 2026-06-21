import argparse
import json
import sys
from pathlib import Path

import torch
from transformers import AutoProcessor

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from ml.angle_geometry import ANGLE_FEATURE_DIM, angle_deg_to_feature
from ml.layoutlmv3_angle_inputs import encoding_with_angle_features
from ml.layoutlmv3_angle_model import (
    load_angle_aware_token_classifier,
    save_angle_aware_model_bundle,
)
from scripts.smoke_finetune_user_labels_v2 import (
    fail,
    load_label_schema,
    load_labeled_sample,
    save_json,
    select_device,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test angle-aware LayoutLMv3 token classification.")
    parser.add_argument("--image", required=True)
    parser.add_argument("--label_json", required=True)
    parser.add_argument("--label_schema", default="schemas/receipt_labels_v2.json")
    parser.add_argument("--model_name_or_path", default="models/layoutlmv3-cord-full/best")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--out_dir", default="outputs/smoke_angle_aware_layoutlmv3")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--synthetic_angle_deg", type=float, default=None)
    parser.add_argument("--debug", action="store_true")
    return parser.parse_args()


def inject_synthetic_angles(sample, angle_deg):
    if angle_deg is None:
        return
    sample["angle_features"] = [
        angle_deg_to_feature(angle_deg, box=box, image_width=sample["width"], image_height=sample["height"])
        for box in sample["boxes"]
    ]
    sample["angle_debug"] = [
        {
            "word_idx": idx,
            "angle_deg": float(angle_deg),
            "has_angle": True,
            "quad_source": "synthetic",
            "feature_dim": ANGLE_FEATURE_DIM,
        }
        for idx in range(len(sample["words"]))
    ]
    sample["num_words_with_angle"] = len(sample["words"])
    sample["num_words_without_angle"] = 0


def main():
    args = parse_args()
    image_path = Path(args.image)
    label_json_path = Path(args.label_json)
    schema_path = Path(args.label_schema)
    if not image_path.exists():
        fail(f"image not found: {image_path}")
    if not label_json_path.exists():
        fail(f"label_json not found: {label_json_path}")
    if not schema_path.exists():
        fail(f"label_schema not found: {schema_path}")
    if args.local_files_only and not Path(args.model_name_or_path).exists():
        fail(f"model checkpoint not found: {args.model_name_or_path}")

    print(f"python: {sys.executable}")
    print(f"torch: {torch.__version__}")
    print(f"cuda_available: {torch.cuda.is_available()}")
    device = select_device(args.device)
    print(f"selected_device: {device}")
    if torch.cuda.is_available():
        print(f"cuda_device_name: {torch.cuda.get_device_name(0)}")

    labels_payload, label_list, label2id, id2label = load_label_schema(schema_path)
    sample = load_labeled_sample(image_path, label_json_path, label2id)
    inject_synthetic_angles(sample, args.synthetic_angle_deg)
    print(f"image_size: {sample['width']}x{sample['height']}")
    print(f"num_words: {len(sample['words'])}")
    print(f"num_labels: {len(label_list)}")
    print(f"angle_feature_dim: {ANGLE_FEATURE_DIM}")
    print(f"num_words_with_angle: {sample['num_words_with_angle']}")
    print(f"num_words_without_angle: {sample['num_words_without_angle']}")

    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path,
        apply_ocr=False,
        local_files_only=args.local_files_only,
    )
    model = load_angle_aware_token_classifier(
        args.model_name_or_path,
        num_labels=len(label_list),
        id2label={idx: label for idx, label in id2label.items()},
        label2id=label2id,
        local_files_only=args.local_files_only,
        ignore_mismatched_sizes=True,
    )
    model.to(device)
    model.train()
    encoding = encoding_with_angle_features(processor, [sample], args.max_length, include_labels=True)
    shapes = {key: list(value.shape) for key, value in encoding.items() if hasattr(value, "shape")}
    print(f"encoding_shapes: {shapes}")
    non_ignored = int((encoding["labels"] != -100).sum().item())
    print(f"non_ignored_token_labels: {non_ignored}")
    if non_ignored == 0:
        fail("No non-ignored token labels were produced.")

    batch = {
        key: value.to(device)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask", "bbox", "pixel_values", "token_type_ids", "labels", "angle_features"}
    }
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    losses = []
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch)
        loss = outputs.loss
        if loss is None or not torch.isfinite(loss):
            fail(f"Invalid loss at step {step}: {loss}")
        loss.backward()
        optimizer.step()
        value = float(loss.detach().cpu().item())
        losses.append(value)
        print(f"step {step}/{args.steps} loss={value:.6f}")

    out_dir = Path(args.out_dir)
    checkpoint_dir = out_dir / "angle_smoke_checkpoint"
    save_angle_aware_model_bundle(checkpoint_dir, model, processor, labels_payload)
    print(f"saved_angle_checkpoint: {checkpoint_dir}")

    reloaded = load_angle_aware_token_classifier(
        checkpoint_dir,
        num_labels=len(label_list),
        id2label={idx: label for idx, label in id2label.items()},
        label2id=label2id,
        local_files_only=True,
        ignore_mismatched_sizes=False,
    )
    reloaded.to(device)
    reloaded.eval()
    with torch.no_grad():
        reload_outputs = reloaded(**batch)
    reload_shape = list(reload_outputs.logits.shape)
    print(f"reloaded_logits_shape: {reload_shape}")

    report = {
        "image": str(image_path),
        "label_json": str(label_json_path),
        "model_name_or_path": args.model_name_or_path,
        "angle_feature_dim": ANGLE_FEATURE_DIM,
        "num_words": len(sample["words"]),
        "num_words_with_angle": sample["num_words_with_angle"],
        "num_words_without_angle": sample["num_words_without_angle"],
        "encoding_shapes": shapes,
        "train_losses": losses,
        "saved_angle_checkpoint": str(checkpoint_dir),
        "reloaded_logits_shape": reload_shape,
        "angle_debug_preview": sample["angle_debug"][:50],
    }
    report_path = out_dir / f"{label_json_path.stem}_angle_smoke_report.json"
    save_json(report_path, report)
    if args.debug:
        print(json.dumps(report, ensure_ascii=False, indent=2)[:4000])
    print(f"angle smoke report path: {report_path}")
    print("Angle-aware LayoutLMv3 smoke run passed.")


if __name__ == "__main__":
    main()
