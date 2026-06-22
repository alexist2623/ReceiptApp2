import json
import subprocess
import sys
import tempfile
from pathlib import Path

from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="wildreceipt_rotation_smoke_") as tmp:
        tmp_path = Path(tmp)
        wild = tmp_path / "wildreceipt"
        images = wild / "image_files"
        images.mkdir(parents=True)
        image_path = images / "sample.jpg"
        Image.new("RGB", (160, 100), "white").save(image_path)
        record = {
            "file_name": "image_files/sample.jpg",
            "receipt_id": "sample",
            "annotations": [
                {"text": "MILK", "box": [20, 20, 70, 40], "label": "PROD_ITEM_VALUE"},
                {"text": "$2.99", "box": [100, 20, 145, 40], "label": "PROD_PRICE_VALUE"},
                {"text": "TOTAL", "box": [20, 70, 75, 88], "label": "TOTAL_KEY"},
                {"text": "$2.99", "box": [100, 70, 145, 88], "label": "TOTAL_VALUE"},
            ],
        }
        (wild / "train.txt").write_text(json.dumps(record) + "\n", encoding="utf-8")
        (wild / "test.txt").write_text(json.dumps(record) + "\n", encoding="utf-8")
        (wild / "class_list.txt").write_text("", encoding="utf-8")
        out_dir = tmp_path / "out"
        cmd = [
            sys.executable,
            str(root / "scripts" / "convert_wildreceipt_rotated_to_receipt_v2_bio.py"),
            "--wildreceipt_root",
            str(wild),
            "--out_dir",
            str(out_dir),
            "--rotation_degrees",
            "0,10",
        ]
        subprocess.run(cmd, check=True, cwd=root)
        rows = []
        for split in ("train", "validation", "test"):
            path = out_dir / f"{split}.jsonl"
            if path.exists():
                rows.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        if not rows:
            raise RuntimeError("converter produced no rows")
        rotated = [row for row in rows if abs(float(row.get("rotation_deg", 0.0))) > 0.0]
        if not rotated:
            raise RuntimeError("converter produced no rotated rows")
        payload = rotated[0]["word_payloads"][0]
        if "quad" not in payload or "angle_deg" not in payload:
            raise RuntimeError(f"word payload missing quad/angle: {payload}")
        print(json.dumps({"rows": len(rows), "rotated_rows": len(rotated), "sample_angle": payload["angle_deg"]}, indent=2))
        print("WildReceipt rotation augmentation smoke passed.")


if __name__ == "__main__":
    main()
