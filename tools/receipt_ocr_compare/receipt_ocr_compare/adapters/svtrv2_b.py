from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

from .base import BaseRecognizerAdapter
from ..runners.subprocess_runner import SubprocessRecognizerRunner
from ..schemas import CropRecord, ModelAvailability, RecognitionResult


class SVTRv2BAdapter(BaseRecognizerAdapter):
    model_id = "svtrv2_b"
    display_name = "SVTRv2-B / OpenOCR"

    def availability(self) -> ModelAvailability:
        vendor_root = self.context.vendor_dir / "OpenOCR"
        checkpoint = self._checkpoint_path()
        if not vendor_root.exists() or not any(vendor_root.iterdir()):
            return ModelAvailability(
                self.model_id,
                False,
                f"OpenOCR source is missing. Run scripts/download_sources.py with --sources openocr. Expected: {vendor_root}",
            )
        if (vendor_root / ".git").exists():
            return ModelAvailability(self.model_id, False, f"nested .git directory is not allowed: {vendor_root / '.git'}")
        if checkpoint is None:
            return ModelAvailability(
                self.model_id,
                False,
                f"SVTRv2-B checkpoint is missing under {self.context.model_dir / 'svtrv2_b'}",
            )
        command = self.context.extra.get("svtrv2_b_command")
        if not command:
            infer_script = vendor_root / "tools" / "infer_rec.py"
            if not infer_script.exists():
                return ModelAvailability(
                    self.model_id,
                    False,
                    "OpenOCR source is present, but no configured JSONL runner or tools/infer_rec.py was found",
                    {"checkpoint": str(checkpoint)},
                )
            config = self._config_path(vendor_root)
            if config is None:
                return ModelAvailability(
                    self.model_id,
                    False,
                    "OpenOCR source is present, but no SVTRv2 inference config was found",
                    {"checkpoint": str(checkpoint), "vendor_root": str(vendor_root)},
                )
        return ModelAvailability(self.model_id, True, details={"checkpoint": str(checkpoint), "vendor_root": str(vendor_root)})

    def _recognize_available(self, crops: list[CropRecord]) -> list[RecognitionResult]:
        command = self.context.extra.get("svtrv2_b_command")
        if not command:
            return self._recognize_with_openocr(crops)
        runner = SubprocessRecognizerRunner(model_id=self.model_id, command=command)
        return runner.run(crops)

    def _recognize_with_openocr(self, crops: list[CropRecord]) -> list[RecognitionResult]:
        vendor_root = self.context.vendor_dir / "OpenOCR"
        infer_script = vendor_root / "tools" / "infer_rec.py"
        config = self._config_path(vendor_root)
        checkpoint = self._checkpoint_path()
        if config is None or checkpoint is None:
            return self.unavailable_results(crops, "OpenOCR SVTRv2-B config or checkpoint is missing")
        results: list[RecognitionResult] = []
        env = os.environ.copy()
        env["PYTHONPATH"] = str(vendor_root) + os.pathsep + env.get("PYTHONPATH", "")
        for crop in crops:
            cmd = [
                sys.executable,
                str(infer_script),
                "--c",
                str(config),
                "--o",
                f"Global.infer_img={crop.crop_path}",
                f"Global.pretrained_model={checkpoint}",
            ]
            try:
                proc, latency_ms = self.time_call(
                    lambda: subprocess.run(cmd, cwd=vendor_root, env=env, text=True, capture_output=True, check=False)
                )
                if proc.returncode != 0:
                    results.append(
                        self.make_result(crop, "", None, latency_ms, error=(proc.stderr or proc.stdout).strip())
                    )
                    continue
                text, confidence = _parse_openocr_stdout(proc.stdout, crop.crop_path)
                results.append(self.make_result(crop, text, confidence, latency_ms))
            except Exception as exc:
                results.append(self.make_result(crop, "", None, 0.0, error=f"OpenOCR inference failed: {exc}"))
        return results

    def _checkpoint_path(self) -> Path | None:
        model_root = self.context.model_dir / "svtrv2_b"
        for suffix in ("*.pth", "*.pt", "*.ckpt", "*.safetensors", "*.onnx", "*.pdparams"):
            matches = sorted(model_root.glob(suffix))
            if matches:
                return matches[0]
        return None

    def _config_path(self, vendor_root: Path) -> Path | None:
        candidates = [
            vendor_root / "configs" / "rec" / "svtrv2" / "svtrv2_smtr_gtc_rctc_infer.yml",
            vendor_root / "configs" / "rec" / "svtrv2" / "svtrv2_rctc_infer.yml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        matches = sorted((vendor_root / "configs" / "rec" / "svtrv2").glob("*infer*.yml"))
        return matches[0] if matches else None


def _parse_openocr_stdout(stdout: str, crop_path: str) -> tuple[str, float | None]:
    basename = Path(crop_path).name
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    for line in reversed(lines):
        if basename in line:
            parsed = _parse_text_score(line)
            if parsed[0]:
                return parsed
    for line in reversed(lines):
        parsed = _parse_text_score(line)
        if parsed[0]:
            return parsed
    return "", None


def _parse_text_score(line: str) -> tuple[str, float | None]:
    score_match = re.search(r"(?:score|confidence)[:=]\s*([0-9.]+)", line, re.IGNORECASE)
    confidence = float(score_match.group(1)) if score_match else None
    text_match = re.search(r"(?:text|rec_text|pred|result)[:=]\s*['\"]?([^,'\"\]\)]+)", line, re.IGNORECASE)
    if text_match:
        return text_match.group(1).strip(), confidence
    if "\t" in line:
        parts = [part.strip() for part in line.split("\t") if part.strip()]
        if len(parts) >= 2:
            return parts[-1], confidence
    return "", confidence
