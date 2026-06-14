# Android Export Coordinates

## Invariant

Every exported Android receipt must satisfy:

```text
actual JPG size == OCR JSON image_width/image_height == OCR JSON image.width/height
```

`words[].box` is always in actual saved JPG pixels.

## What To Use For Training

Use the exported ZIP contents as a pair:

```text
<capture_id>.jpg
<capture_id>_ocr.json
<capture_id>_export_validation.json
```

Do not use a JPG copied from a gallery, chat app, cloud preview, or image-only
share. Those copies may be resized or recompressed without updating the OCR
JSON.

## Quick PC Check

```bash
python - <<'PY'
from PIL import Image
import json
from pathlib import Path

folder = Path("path/to/unzipped_export")
for jpg in folder.rglob("*.jpg"):
    stem = jpg.stem
    candidates = list(jpg.parent.glob(f"{stem}*_ocr.json")) + list(jpg.parent.glob(f"{stem}.json"))
    if not candidates:
        continue
    ocr = json.load(open(candidates[0], encoding="utf-8"))
    print(
        jpg.name,
        Image.open(jpg).size,
        ocr.get("image_width"),
        ocr.get("image_height"),
        ocr.get("image", {}).get("width"),
        ocr.get("image", {}).get("height"),
    )
PY
```

All three sizes must match.

## Install / Manual Verification

1. Remove any old APK:

   ```bash
   adb uninstall com.receiptapp
   ```

2. Install the current debug APK:

   ```bash
   cd ReceiptApp
   ./gradlew :app:installDebug
   ```

3. Capture a receipt.
4. On the review screen, confirm:
   - actual JPG size equals OCR JSON top-level size
   - actual JPG size equals OCR JSON `image.width/height`
   - validation status is `OK`
   - app version/build information is visible
5. Share ZIP.
6. Unzip on PC and run the quick check above or:

   ```bash
   python scripts/validate_receipt_export_coordinates.py \
     --input_dir path/to/unzipped_export \
     --strict \
     --out_json outputs/coordinate_validation_summary.json
   ```

If a mismatch is reported, the export is invalid for training.
