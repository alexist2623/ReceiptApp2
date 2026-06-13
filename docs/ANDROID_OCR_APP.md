# Android OCR App

## Current Role

The Android app currently handles:

1. receipt photo capture,
2. gallery image import,
3. canonical upright image saving,
4. ML Kit OCR,
5. OCR JSON generation,
6. OCR overlay review,
7. ZIP sharing,
8. development HTTP upload.

The app does not yet run:

- LayoutLMv3 inference,
- ITEM_NAME/ITEM_PRICE or legacy MENU_NM/MENU_PRICE prediction logic beyond OCR,
- span-level rel-g grouping,
- group_id prediction,
- heuristic item-price grouping.

The app preserves ML Kit element-level raw OCR tokens. It does not merge tokens
such as `$` and `16.99`, and it does not split line text manually. Receipt schema
v2 labels and BIO span normalization are applied later in the computer pipeline.

## Why Inference Is Abstracted

Inference is separated behind `ReceiptInferenceEngine` so the app can use server-side inference now and later replace it with on-device LayoutLMv3/span-relg inference.

Current implementations:

- `ServerReceiptInferenceEngine`
- `OnDeviceReceiptInferenceEngine` stub
- `MockReceiptInferenceEngine`

## Usage

1. Open the app.
2. Grant camera permission.
3. Capture a receipt or import one from the gallery.
4. Wait for OCR.
5. Review OCR boxes on the canonical image.
6. Share the ZIP or upload to a development server.
7. Run the Python pipeline on the computer.

## Server URL

Set the server base URL in Settings.

Example:

```text
http://192.168.0.10:8000
```

The upload endpoint is:

```text
POST /api/receipt/ocr
```

Multipart fields:

- `capture_id`
- `image`
- `ocr_json`

If upload fails, use ZIP sharing.

## OCR Script

The app supports an OCR script setting:

- `LATIN`
- `KOREAN`

The app currently uses bundled ML Kit recognizers. Bundled recognizers increase app size. Unbundled recognizers may reduce app size but can require model download on first use.

## Coordinate Warning

The JSON boxes are in saved canonical image pixels:

```text
image.coordinateSpace = "saved_canonical_image_pixels"
```

The OCR input bitmap and exported image are the same upright bitmap. This prevents EXIF rotation from invalidating OCR coordinates.

## JSON Schema

See [RECEIPT_OCR_JSON_SCHEMA.md](RECEIPT_OCR_JSON_SCHEMA.md).

## Future On-Device Plan

See [ANDROID_ON_DEVICE_MODEL_PLAN.md](ANDROID_ON_DEVICE_MODEL_PLAN.md).
