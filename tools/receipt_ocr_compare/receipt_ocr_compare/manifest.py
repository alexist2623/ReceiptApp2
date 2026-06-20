from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(args: list[str], cwd: Path) -> str | None:
    try:
        return subprocess.check_output(["git", *args], cwd=cwd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def repo_root(start: Path) -> Path:
    found = git_value(["rev-parse", "--show-toplevel"], start)
    return Path(found).resolve() if found else start.resolve()


def scan_nested_git(root: Path) -> list[str]:
    root = root.resolve()
    found: list[str] = []
    for dirpath, dirnames, _ in os.walk(root):
        path = Path(dirpath).resolve()
        if path == root:
            if ".git" in dirnames:
                dirnames.remove(".git")
            continue
        if ".git" in dirnames:
            found.append(str(path / ".git"))
            dirnames.remove(".git")
    return sorted(found)


def scan_gitmodules(root: Path) -> list[str]:
    return sorted(str(path) for path in root.rglob(".gitmodules"))


def create_run_manifest(
    *,
    repo: Path,
    input_images: list[Path],
    model_records: list[dict[str, Any]],
    device: str,
    preprocessing: dict[str, Any],
    detector: dict[str, Any],
    normalization: dict[str, Any],
) -> dict[str, Any]:
    image_records = []
    for image in input_images:
        image_records.append(
            {
                "path": str(image),
                "checksum": sha256_file(image) if image.exists() else None,
            }
        )
    return {
        "git_branch": git_value(["branch", "--show-current"], repo),
        "git_commit": git_value(["rev-parse", "HEAD"], repo),
        "run_time": utc_now_iso(),
        "input_images": image_records,
        "models": model_records,
        "python_version": platform.python_version(),
        "device": device,
        "cuda_availability": _cuda_availability(),
        "preprocessing_settings": preprocessing,
        "detector_settings": detector,
        "normalization_settings": normalization,
        "nested_git_directories": scan_nested_git(repo),
        "gitmodules": scan_gitmodules(repo),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _cuda_availability() -> dict[str, Any]:
    try:
        import torch  # type: ignore

        return {"torch_available": True, "cuda_available": bool(torch.cuda.is_available())}
    except Exception as exc:
        return {"torch_available": False, "cuda_available": False, "error": str(exc)}

