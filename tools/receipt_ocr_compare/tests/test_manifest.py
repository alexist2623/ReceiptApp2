from pathlib import Path

from receipt_ocr_compare.manifest import create_run_manifest, scan_gitmodules, scan_nested_git, sha256_file


def test_manifest_sha256_and_nested_git_scan(tmp_path: Path):
    payload = tmp_path / "file.txt"
    payload.write_text("hello", encoding="utf-8")
    assert sha256_file(payload) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"

    nested = tmp_path / "vendor" / "OpenOCR" / ".git"
    nested.mkdir(parents=True)
    assert scan_nested_git(tmp_path) == [str(nested.resolve())]


def test_gitmodules_scan(tmp_path: Path):
    gitmodules = tmp_path / ".gitmodules"
    gitmodules.write_text("[submodule]\n", encoding="utf-8")
    assert scan_gitmodules(tmp_path) == [str(gitmodules)]


def test_run_manifest_generation(tmp_path: Path):
    image = tmp_path / "receipt.png"
    image.write_bytes(b"not-a-real-image-but-hashable")
    manifest = create_run_manifest(
        repo=tmp_path,
        input_images=[image],
        model_records=[{"model_name": "paddleocr"}],
        device="cpu",
        preprocessing={"crop_padding": 2},
        detector={"selected": "ground_truth"},
        normalization={"primary_metric_text": "raw_text"},
    )
    assert manifest["input_images"][0]["checksum"] == sha256_file(image)
    assert manifest["normalization_settings"]["primary_metric_text"] == "raw_text"
