# Android On-Device Model Plan

The Android app currently performs only:

1. receipt image capture or gallery import,
2. canonical upright image saving,
3. on-device ML Kit OCR,
4. OCR JSON generation,
5. ZIP export or development HTTP upload.

It does not run LayoutLMv3 or the span-level rel-g parser on device yet.

## Future Assets

Future on-device inference would require these assets:

- LayoutLMv3 token classification ONNX model
- tokenizer vocab, merges, and tokenizer config
- image processor config
- label map
- span-relg field schema
- span-relg model ONNX, or a lightweight Kotlin/ONNX implementation

Do not commit these model files until the on-device packaging plan is explicit.

## ONNX Runtime Mobile Considerations

- model size and APK/AAB size
- memory pressure during LayoutLMv3 image and token processing
- latency on CPU
- quantization strategy
- CPU, XNNPACK, or NNAPI execution provider behavior
- tokenizer and image preprocessing parity with the Python pipeline
- bbox normalization to LayoutLMv3 0-1000 coordinates

## App Interface

The app already separates inference behind `ReceiptInferenceEngine`.

- `ServerReceiptInferenceEngine` uploads canonical image + OCR JSON to a computer/server.
- `OnDeviceReceiptInferenceEngine` is a stub for future ONNX Runtime Mobile work.
- `MockReceiptInferenceEngine` is available for UI testing.

This lets the app switch from server inference to on-device inference without changing capture/OCR/export code.
