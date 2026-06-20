from __future__ import annotations

import io
import csv
import sys
from datetime import datetime
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import streamlit as st  # noqa: E402
from PIL import Image  # noqa: E402

from cli import run_compare  # noqa: E402
from receipt_ocr_compare.config import CompareConfig  # noqa: E402
from receipt_ocr_compare.model_registry import available_model_ids  # noqa: E402


ROOT = THIS_DIR.parents[1]
DEFAULT_OUTPUT_ROOT = THIS_DIR / "outputs"


def main() -> None:
    st.set_page_config(page_title="Receipt OCR Compare", layout="wide")
    st.title("Receipt OCR Token Overlay Comparison")

    with st.sidebar:
        mode = st.radio("Mode", ["recognition", "end-to-end"], index=0)
        detector = st.selectbox("Detector", ["auto", "existing", "paddleocr", "simple", "ground_truth"], index=0)
        selected_models = st.multiselect("Models", available_model_ids(), default=["svtrv2_b", "paddleocr", "existing"])
        device = st.radio("Device", ["cpu", "gpu"], index=0, horizontal=True)
        crop_padding = st.slider("Crop padding", 0, 24, 2)
        view_mode = st.radio("Token view", ["all tokens", "numeric tokens"], index=0)
        show_confidence = st.checkbox("Show confidence", value=True)
        show_boxes = st.checkbox("Show bounding boxes", value=True)
        mismatches_only = st.checkbox("Mismatch only", value=False)

    upload_tab, batch_tab = st.tabs(["Upload", "Batch directory"])
    uploaded = None
    directory_input = ""
    with upload_tab:
        uploaded = st.file_uploader("Receipt image", type=["png", "jpg", "jpeg", "bmp", "tif", "tiff", "webp"])
    with batch_tab:
        directory_input = st.text_input("Image directory", value="")

    gt_upload = st.file_uploader("Ground truth JSONL", type=["jsonl"])

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Original image", use_container_width=True)

    if st.button("Run comparison", type="primary", disabled=not selected_models or (uploaded is None and not directory_input)):
        run_id = datetime.utcnow().strftime("ui_%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        input_path: Path
        gt_path: Path | None = None

        if uploaded is not None:
            input_path = output_dir / "input" / uploaded.name
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_bytes(uploaded.getvalue())
        else:
            input_path = Path(directory_input).expanduser().resolve()

        if gt_upload is not None:
            gt_path = output_dir / "ground_truth.jsonl"
            gt_path.write_bytes(gt_upload.getvalue())

        config = CompareConfig(
            input_path=input_path,
            models=tuple(selected_models),
            mode=mode,
            detector=detector,
            model_dir=THIS_DIR / "models",
            vendor_dir=THIS_DIR / "vendor",
            output_dir=output_dir,
            crop_padding=crop_padding,
            device=device,
            ground_truth_path=gt_path,
            allow_package_models=False,
        )
        try:
            run_compare(
                config,
                numeric_only_overlay=view_mode == "numeric tokens",
                mismatches_only_overlay=mismatches_only,
                show_confidence=show_confidence,
                show_boxes=show_boxes,
            )
        except Exception as exc:
            st.error(str(exc))
            return
        render_results(output_dir, selected_models, show_confidence, show_boxes)


def render_results(output_dir: Path, model_ids: list[str], show_confidence: bool, show_boxes: bool) -> None:
    st.success(f"Outputs: {output_dir}")
    summary_path = output_dir / "model_summary.csv"
    per_token_path = output_dir / "per_token_results.csv"
    predictions_path = output_dir / "predictions.jsonl"
    if summary_path.exists():
        st.subheader("Summary")
        st.dataframe(_read_csv(summary_path), use_container_width=True)
        st.download_button("Download summary CSV", summary_path.read_bytes(), file_name="model_summary.csv", mime="text/csv")
    if per_token_path.exists():
        st.download_button("Download token CSV", per_token_path.read_bytes(), file_name="per_token_results.csv", mime="text/csv")
    if predictions_path.exists():
        st.download_button(
            "Download predictions JSONL",
            predictions_path.read_bytes(),
            file_name="predictions.jsonl",
            mime="application/jsonl",
        )
    overlays = sorted((output_dir / "overlays").glob("*.png"))
    if overlays:
        st.subheader("Overlays")
        cols = st.columns(min(3, len(overlays)))
        for idx, overlay in enumerate(overlays):
            with cols[idx % len(cols)]:
                st.image(Image.open(overlay), caption=overlay.name, use_container_width=True)
                st.download_button(
                    "Download PNG",
                    overlay.read_bytes(),
                    file_name=overlay.name,
                    mime="image/png",
                    key=str(overlay),
                )
    st.caption(f"confidence={show_confidence}, boxes={show_boxes}")


def _read_csv(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(path.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()
