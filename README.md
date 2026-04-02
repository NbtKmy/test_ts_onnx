# YOLO26m Object Detection in the Browser

A demo app that runs object detection entirely in your browser using [ONNX Web Runtime](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) and [Ultralytics YOLO26m](https://docs.ultralytics.com/models/yolo26/).

**Live demo:** https://nbtkmy.github.io/test_ts_onnx/

## Features

- **Local file upload** — your image is never sent to any server; all inference runs on your device
- **IIIF Manifest support** — load images from IIIF v2 / v3 manifests (auto-detected)
- **CPU / WebGPU toggle** — switch execution backend at runtime; WebGPU is ~18x faster after warmup
- **Adjustable threshold** — tune the detection score threshold with a slider

## Performance (YOLO26m, 640×640 input)

| Backend | Inference time |
|---|---|
| CPU (single-threaded) | ~3000 ms |
| CPU (multi-threaded via SharedArrayBuffer) | ~850 ms |
| WebGPU (first run) | ~500 ms |
| WebGPU (subsequent runs) | **~70 ms** |

## Tech Stack

- [Vite](https://vite.dev/) + TypeScript
- [ONNX Web Runtime](https://github.com/microsoft/onnxruntime) (`onnxruntime-web`)
- [coi-serviceworker](https://github.com/gzuidhof/coi-serviceworker) — enables `SharedArrayBuffer` on GitHub Pages for multi-threaded WASM

## Development

```bash
npm install
npm run dev
```

## Build & Deploy

```bash
npm run build
```

Deployed automatically to GitHub Pages via GitHub Actions on push to `main`.
