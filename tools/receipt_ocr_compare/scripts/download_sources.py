from __future__ import annotations

import argparse
import json
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from receipt_ocr_compare.manifest import scan_gitmodules, scan_nested_git, utc_now_iso, write_json  # noqa: E402


SOURCES = {
    "openocr": {
        "repo": "Topdu/OpenOCR",
        "revision": "main",
        "target": "OpenOCR",
        "license": "Apache-2.0",
    },
    "paddleocr": {
        "repo": "PaddlePaddle/PaddleOCR",
        "revision": "main",
        "target": "PaddleOCR",
        "license": "Apache-2.0",
    },
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Download source archives without nested Git metadata")
    parser.add_argument("--sources", required=True, help="Comma-separated: openocr,paddleocr")
    parser.add_argument("--vendor-dir", default="tools/receipt_ocr_compare/vendor")
    parser.add_argument("--force", action="store_true", help="Replace an existing target directory")
    args = parser.parse_args(argv)

    vendor_dir = Path(args.vendor_dir)
    vendor_dir.mkdir(parents=True, exist_ok=True)
    downloads_dir = vendor_dir.parent / "downloads"
    downloads_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for source_id in [item.strip() for item in args.sources.split(",") if item.strip()]:
        if source_id not in SOURCES:
            raise SystemExit(f"Unknown source: {source_id}. Available: {', '.join(sorted(SOURCES))}")
        spec = SOURCES[source_id]
        sha = resolve_github_revision(spec["repo"], spec["revision"])
        zip_url = f"https://github.com/{spec['repo']}/archive/{sha}.zip"
        archive_path = downloads_dir / f"{source_id}-{sha}.zip"
        target = vendor_dir / spec["target"]
        if target.exists() and _contains_only_placeholders(target):
            shutil.rmtree(target)
        if target.exists() and any(target.iterdir()) and not args.force:
            raise SystemExit(f"Target already exists and is not empty: {target}. Use --force to replace it.")
        if target.exists() and args.force:
            shutil.rmtree(target)
        if target.exists():
            target.rmdir()
        download(zip_url, archive_path)
        extracted = extract_single_root(archive_path, vendor_dir)
        extracted.rename(target)
        remove_disallowed_metadata(target)
        nested = scan_nested_git(vendor_dir)
        gitmodules = scan_gitmodules(vendor_dir)
        if nested or gitmodules:
            raise SystemExit(f"Disallowed Git metadata found after extraction: nested={nested}, gitmodules={gitmodules}")
        records.append(
            {
                "source": source_id,
                "repository": f"https://github.com/{spec['repo']}",
                "revision": sha,
                "archive_url": zip_url,
                "local_path": str(target),
                "license": spec["license"],
                "download_time": utc_now_iso(),
            }
        )
    write_json(vendor_dir / "source_manifest.json", {"sources": records})
    print(f"Wrote {vendor_dir / 'source_manifest.json'}")
    return 0


def resolve_github_revision(repo: str, revision: str) -> str:
    url = f"https://api.github.com/repos/{repo}/commits/{revision}"
    with urllib.request.urlopen(url) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload["sha"]


def download(url: str, path: Path) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with urllib.request.urlopen(url) as response, partial.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    partial.replace(path)


def extract_single_root(archive_path: Path, vendor_dir: Path) -> Path:
    before = {path.resolve() for path in vendor_dir.iterdir()}
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(vendor_dir)
    after = {path.resolve() for path in vendor_dir.iterdir()}
    created = sorted(after - before)
    if len(created) != 1 or not created[0].is_dir():
        raise RuntimeError(f"Expected one extracted root from {archive_path}, got {created}")
    return created[0]


def remove_disallowed_metadata(root: Path) -> None:
    for path in root.rglob(".git"):
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    for path in root.rglob(".github"):
        if path.is_dir():
            shutil.rmtree(path)
    for path in root.rglob(".gitmodules"):
        path.unlink()


def _contains_only_placeholders(path: Path) -> bool:
    if not path.exists():
        return False
    items = list(path.rglob("*"))
    files = [item for item in items if item.is_file()]
    dirs = [item for item in items if item.is_dir()]
    return not dirs and all(item.name == ".gitkeep" for item in files)


if __name__ == "__main__":
    raise SystemExit(main())
