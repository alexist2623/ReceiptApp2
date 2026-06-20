from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from receipt_ocr_compare.manifest import sha256_file, utc_now_iso, write_json  # noqa: E402


PADDLE_HF_REPOS = {
    "det": "PaddlePaddle/PP-OCRv4_mobile_det",
    "rec": "PaddlePaddle/PP-OCRv4_mobile_rec",
}
PADDLE_ALLOWED_SUFFIXES = {".pdmodel", ".pdiparams", ".yml", ".yaml", ".txt", ".json"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download OCR model checkpoints into the repository")
    parser.add_argument("--models", required=True, help="Comma-separated: svtrv2_b,paddleocr")
    parser.add_argument("--model-dir", default="tools/receipt_ocr_compare/models")
    parser.add_argument("--svtrv2-url", default=None, help="Direct URL for an SVTRv2-B checkpoint archive or file")
    parser.add_argument("--svtrv2-license", default="Apache-2.0")
    args = parser.parse_args(argv)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = model_dir / "model_manifest.json"
    existing_manifest = load_manifest(manifest_path)
    records: list[dict[str, Any]] = []
    failures: list[str] = []

    for model_id in [item.strip() for item in args.models.split(",") if item.strip()]:
        if model_id == "paddleocr":
            records.extend(download_paddleocr(model_dir, existing_manifest))
        elif model_id == "svtrv2_b":
            if not args.svtrv2_url:
                failures.append(
                    "svtrv2_b requires --svtrv2-url because OpenOCR publishes SVTRv2-B checkpoints via external model links."
                )
                continue
            records.append(
                download_single_file(
                    model_name="svtrv2_b",
                    url=args.svtrv2_url,
                    destination_dir=model_dir / "svtrv2_b",
                    source_repository="https://github.com/Topdu/OpenOCR",
                    source_revision="user-supplied",
                    license_name=args.svtrv2_license,
                    framework_version=framework_version("torch"),
                    existing_manifest=existing_manifest,
                )
            )
        else:
            failures.append(f"unknown model: {model_id}")

    if records:
        write_json(manifest_path, {"models": merge_records(existing_manifest.get("models", []), records)})
        print(f"Wrote {manifest_path}")
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    return 0


def download_paddleocr(model_dir: Path, existing_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for component, repo in PADDLE_HF_REPOS.items():
        api_payload = hf_model_info(repo)
        sha = api_payload.get("sha") or "main"
        siblings = api_payload.get("siblings", [])
        files = [
            item["rfilename"]
            for item in siblings
            if Path(item.get("rfilename", "")).suffix.lower() in PADDLE_ALLOWED_SUFFIXES
            and not item.get("rfilename", "").startswith(".")
        ]
        if not files:
            raise SystemExit(f"No PaddleOCR model files discovered in Hugging Face repo {repo}")
        for filename in files:
            url = f"https://huggingface.co/{repo}/resolve/{urllib.parse.quote(str(sha))}/{urllib.parse.quote(filename)}"
            records.append(
                download_single_file(
                    model_name="paddleocr",
                    url=url,
                    destination_dir=model_dir / "paddleocr" / component,
                    source_repository=f"https://huggingface.co/{repo}",
                    source_revision=str(sha),
                    license_name=str(api_payload.get("cardData", {}).get("license") or "apache-2.0"),
                    framework_version=framework_version("paddleocr"),
                    existing_manifest=existing_manifest,
                    checkpoint_filename=filename,
                )
            )
    return records


def download_single_file(
    *,
    model_name: str,
    url: str,
    destination_dir: Path,
    source_repository: str,
    source_revision: str,
    license_name: str,
    framework_version: str | None,
    existing_manifest: dict[str, Any],
    checkpoint_filename: str | None = None,
) -> dict[str, Any]:
    destination_dir.mkdir(parents=True, exist_ok=True)
    filename = checkpoint_filename or Path(urllib.parse.urlparse(url).path).name or f"{model_name}.ckpt"
    destination = destination_dir / filename
    destination.parent.mkdir(parents=True, exist_ok=True)
    previous_checksum = manifest_checksum(existing_manifest, destination)
    if destination.exists():
        checksum = sha256_file(destination)
        if previous_checksum and checksum != previous_checksum:
            raise SystemExit(f"Checksum mismatch for existing model file {destination}; refusing to overwrite")
        return model_record(
            model_name,
            url,
            source_repository,
            source_revision,
            filename,
            destination,
            checksum,
            license_name,
            framework_version,
            reused=True,
        )
    partial = destination.with_suffix(destination.suffix + ".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(destination)
    checksum = sha256_file(destination)
    return model_record(
        model_name,
        url,
        source_repository,
        source_revision,
        filename,
        destination,
        checksum,
        license_name,
        framework_version,
        reused=False,
    )


def model_record(
    model_name: str,
    source_url: str,
    source_repository: str,
    source_revision: str,
    checkpoint_filename: str,
    local_checkpoint_path: Path,
    checksum: str,
    license_name: str,
    framework_version: str | None,
    reused: bool,
) -> dict[str, Any]:
    return {
        "model_name": model_name,
        "source_url": source_url,
        "source_repository": source_repository,
        "source_revision": source_revision,
        "checkpoint_filename": checkpoint_filename,
        "local_checkpoint_path": str(local_checkpoint_path),
        "checksum": checksum,
        "license": license_name,
        "download_time": utc_now_iso(),
        "framework_version": framework_version,
        "reused_existing_file": reused,
    }


def hf_model_info(repo: str) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{repo}"
    with urllib.request.urlopen(url) as response:
        return json.loads(response.read().decode("utf-8"))


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"models": []}
    return json.loads(path.read_text(encoding="utf-8"))


def manifest_checksum(manifest: dict[str, Any], destination: Path) -> str | None:
    destination_str = str(destination)
    for record in manifest.get("models", []):
        if record.get("local_checkpoint_path") == destination_str:
            return record.get("checksum")
    return None


def merge_records(old_records: list[dict[str, Any]], new_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged = {(record["model_name"], record["local_checkpoint_path"]): record for record in old_records}
    for record in new_records:
        merged[(record["model_name"], record["local_checkpoint_path"])] = record
    return sorted(merged.values(), key=lambda row: (row["model_name"], row["local_checkpoint_path"]))


def framework_version(package: str) -> str | None:
    try:
        module = __import__(package)
        return str(getattr(module, "__version__", "unknown"))
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
