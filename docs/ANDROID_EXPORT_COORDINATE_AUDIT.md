# Android Export Coordinate Audit

## Repo State

| item | value |
|---|---|
| remote | `origin https://github.com/alexist2623/ReceiptApp2.git` |
| local branch | `main` |
| local HEAD before changes | `77f2facb8ae3f98d5054adade09ae6cfd93bb07f` |
| GitHub main HEAD checked | `77f2facb8ae3f98d5054adade09ae6cfd93bb07f` |
| Android root files | `settings.gradle.kts`, `build.gradle.kts`, `app/build.gradle.kts`, `app/src/main/AndroidManifest.xml` all present |

Local HEAD and GitHub main matched before this audit patch. The working tree
already contained coordinate-validation edits from the previous pass.

## Resize / Downscale Search Summary

| file | line/keyword | meaning | suspicious? | action |
|---|---|---|---|---|
| `capture/CanonicalImageWriter.kt` | `BitmapFactory.decodeFile`, `rotateIfNeeded`, `bitmap.compress` | Decodes source, applies EXIF rotation, saves JPEG. No resize call was found. | No resize, but saved-file size was not independently authoritative before the validation patch. | Re-read saved JPEG bounds after compress and use that size in `ImageInfoDto`. |
| `capture/ImageRotationUtils.kt` | `Bitmap.createBitmap` | Rotates bitmap through matrix. | Not a downscale by itself. | Leave as-is. |
| `ocr/MlKitOcrEngine.kt` | `BitmapFactory.decodeFile`, `InputImage.fromBitmap(bitmap, 0)` | OCR runs on the saved canonical JPEG decoded from disk. | Size mismatch should be impossible if metadata tracks saved JPEG. | Validate decoded bitmap, saved file, and imageInfo all match. |
| `ocr/OcrJsonMapper.kt` | `image_width`, `image_height`, `clampBox` | JSON size is copied from `ImageInfoDto`; boxes are clamped to that size. | Depends on `ImageInfoDto` correctness. | Add debug metadata and keep coordinate-space invariant. |
| `receipt/ReceiptRepository.kt` | `saveOcrPayload` | Previously wrote JSON directly. | Yes, stale/mismatched payload could be saved. | Block save through `ReceiptExportValidator`. |
| `export/ZipExportService.kt` | `export`, `zipFiles` | Previously wrote image/JSON directly. | Yes, mismatched pair could be shared. | Block export and add validation summary JSON. |
| `export/ShareReceiptIntentFactory.kt` | `FileProvider` | Shares the ZIP file created by `ZipExportService`. | Not a resize path. | Leave as-is. |
| `ui/OcrOverlayComposable.kt` | `scale` | Scales boxes only for on-screen display. | Not export data mutation. | Leave as visual transform only. |
| `AndroidManifest.xml` | `FileProvider` | Grants URI access to exported ZIP. | Not a resize path. | Leave as-is. |

No `createScaledBitmap`, `inSampleSize`, `decodeSampledBitmap`, or hard-coded
`1536/2048/3000/4000` resize path was found in Android source.

## Conclusion

The main code path should not generate `image=1536x2048` and
`OCR JSON=3000x4000` from the same fresh capture. If that mismatch appears, the
likely causes are a stale APK, stale capture JSON, a manually downscaled image
paired with old JSON, or an image shared outside the app ZIP.

The fix makes the app fail closed: save, upload, and ZIP export now validate the
actual saved JPEG size against OCR JSON dimensions before data leaves the app.
