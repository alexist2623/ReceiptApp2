# Receipt OCR JSON Schema

Schema version: `receipt_ocr_v1`

The app writes OCR JSON next to the canonical image:

```text
files/receipts/<capture_id>/<capture_id>_ocr.json
```

The top-level `words` array is required for the Python pipeline. Each word must include:

- `text`
- `box`

`box` is `[left, top, right, bottom]` in integer pixel coordinates.
Those coordinates must be in the saved JPG pixel space, not the original camera
sensor size and not a resized preview size.

## Coordinate Space

All boxes are relative to the saved canonical image:

```text
image.coordinateSpace = "saved_canonical_image_pixels"
```

The image used for OCR and the image sent to the computer are the same upright canonical bitmap. EXIF/camera rotation is applied before OCR and before saving.

The coordinate invariant is:

```text
saved JPG size == OCR input bitmap size == OCR JSON image_width/image_height == words[].box coordinate space
```

`image_width` / `image_height` must exactly match the actual saved JPG size.
`image.width` / `image.height` must also match that same size. A mismatched
image/JSON pair must not be used for training.

## Required Top-Level Fields

- `schemaVersion`
- `captureId`
- `createdAtUtc`
- `device`
- `app`
- `image`
- `ocr`
- `blocks`
- `lines`
- `words`
- `image_width`
- `image_height`

`image_width` and `image_height` are duplicated at the top level for Python compatibility.

## Example

```json
{
  "schemaVersion": "receipt_ocr_v1",
  "captureId": "20260613T143012Z_ab12cd34",
  "createdAtUtc": "2026-06-13T14:30:12Z",
  "image_width": 1080,
  "image_height": 1920,
  "image": {
    "fileName": "20260613T143012Z_ab12cd34.jpg",
    "width": 1080,
    "height": 1920,
    "mimeType": "image/jpeg",
    "coordinateSpace": "saved_canonical_image_pixels",
    "exifOrientationApplied": true,
    "rotationDegreesApplied": 90
  },
  "ocr": {
    "engine": "mlkit_text_recognition_v2",
    "script": "latin",
    "source": "on_device_ocr",
    "confidenceAvailable": false
  },
  "words": [
    {
      "wordId": "w_000001",
      "blockId": "b_000001",
      "lineId": "l_000001",
      "wordIndexInLine": 0,
      "globalWordIndex": 0,
      "text": "americano",
      "box": [120, 340, 310, 385]
    }
  ],
  "blocks": [],
  "lines": []
}
```

## Notes

- Do not split line text manually to create words.
- Use ML Kit text elements as word-like units.
- Do not merge OCR tokens. If OCR emits `$` and `16.99` as separate elements,
  keep both elements as separate `words`.
- Price splitting is handled later by BIO labels and span normalization. For
  example, label `$` as `B-ITEM_PRICE` and `16.99` as `I-ITEM_PRICE`; the Python
  span layer can normalize the recovered span to `$16.99`.
- Clamp boxes to the canonical image bounds.
- Skip empty text and boxes that cannot be recovered from bounding boxes or corner points.
- Run `scripts/validate_receipt_export_coordinates.py --strict` before using
  exported Android data for labeling or fine-tuning.
