from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from receipt_ocr_compare.manifest import scan_gitmodules, scan_nested_git, sha256_file  # noqa: E402
from receipt_ocr_compare.model_registry import create_adapters  # noqa: E402
from receipt_ocr_compare.schemas import RunContext  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify model/source availability without downloading")
    parser.add_argument("--model-dir", default="tools/receipt_ocr_compare/models")
    parser.add_argument("--vendor-dir", default="tools/receipt_ocr_compare/vendor")
    parser.add_argument("--models", default="svtrv2_b,paddleocr,existing")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    vendor_dir = Path(args.vendor_dir)
    failures = []
    nested = scan_nested_git(vendor_dir)
    gitmodules = scan_gitmodules(vendor_dir)
    print(f"nested_git={nested}")
    print(f"gitmodules={gitmodules}")
    if nested:
        failures.append(f"nested .git directories found: {nested}")
    if gitmodules:
        failures.append(f".gitmodules files found: {gitmodules}")

    context = RunContext(model_dir=model_dir, vendor_dir=vendor_dir)
    for adapter in create_adapters([item.strip() for item in args.models.split(",") if item.strip()], context):
        status = adapter.availability()
        print(json.dumps(status.to_dict(), ensure_ascii=False))
        if args.strict and not status.available:
            failures.append(f"{adapter.model_id}: {status.reason}")

    manifest_path = model_dir / "model_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for record in manifest.get("models", []):
            path = Path(record["local_checkpoint_path"])
            if not path.exists():
                failures.append(f"manifest model file missing: {path}")
                continue
            actual = sha256_file(path)
            if actual != record.get("checksum"):
                failures.append(f"checksum mismatch for {path}")
    else:
        print(f"model manifest not found: {manifest_path}")

    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

