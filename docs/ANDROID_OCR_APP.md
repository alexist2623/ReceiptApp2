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

The app now has an on-device development path for:

- ML Kit OCR,
- LayoutLMv3 INT8 ONNX field prediction,
- span-level rel-g ONNX item/summary grouping,
- dashboard and receipt-history UI backed by saved inference results.

It does not predict `group_id` directly, and it does not use a y-coordinate or
line-distance heuristic to connect items to prices. If the rel-g ONNX artifacts
are missing, on-device inference fails with a model-file error.

The app preserves ML Kit element-level raw OCR tokens. It does not merge tokens
such as `$` and `16.99`, and it does not split line text manually. Receipt schema
v2 labels and BIO span normalization are applied later in the computer pipeline.

## Why Inference Is Abstracted

Inference is separated behind `ReceiptInferenceEngine` so the app can use the
on-device LayoutLMv3/span-relg path or a server-side implementation later.

Current implementations:

- `ServerReceiptInferenceEngine`
- `OnDeviceReceiptInferenceEngine`
- `MockReceiptInferenceEngine`

## Usage

1. Open the app.
2. Grant camera permission.
3. Capture a receipt or import one from the gallery.
4. Wait for OCR.
5. Review OCR boxes on the canonical image.
6. Share the ZIP or upload to a development server.
7. If model artifacts are present, run on-device INT8 LayoutLMv3 + rel-g from the app.
8. If model artifacts are not present, share the ZIP and run the Python pipeline on the computer.

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
