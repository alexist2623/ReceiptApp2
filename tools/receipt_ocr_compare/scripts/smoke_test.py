from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = SCRIPT_DIR.parents[1]


def main() -> int:
    sample_image = SCRIPT_DIR / "sample_data" / "synthetic_receipt.png"
    sample_gt = SCRIPT_DIR / "sample_data" / "synthetic_receipt_gt.jsonl"
    output_dir = SCRIPT_DIR / "outputs" / "smoke_test"
    if not sample_image.exists() or not sample_gt.exists():
        print("Synthetic sample data is missing; run tests or regenerate sample_data first.", file=sys.stderr)
        return 1
    cmd = [
        sys.executable,
        "-m",
        "tools.receipt_ocr_compare.cli",
        "compare",
        "--input",
        str(sample_image),
        "--models",
        "svtrv2_b,paddleocr,existing",
        "--mode",
        "recognition",
        "--detector",
        "ground_truth",
        "--ground-truth",
        str(sample_gt),
        "--model-dir",
        str(SCRIPT_DIR / "models"),
        "--vendor-dir",
        str(SCRIPT_DIR / "vendor"),
        "--output",
        str(output_dir),
    ]
    result = subprocess.run(cmd, cwd=REPO_ROOT, text=True, capture_output=True, check=False)
    print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)
    if result.returncode != 0:
        return result.returncode
    required = [
        output_dir / "predictions.jsonl",
        output_dir / "per_token_results.csv",
        output_dir / "model_summary.csv",
        output_dir / "run_manifest.json",
        output_dir / "overlays" / "synthetic_receipt_comparison.png",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        print(f"Smoke test output missing: {missing}", file=sys.stderr)
        return 1
    print("Smoke test passed. Real model integration is skipped when checkpoints are absent.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

