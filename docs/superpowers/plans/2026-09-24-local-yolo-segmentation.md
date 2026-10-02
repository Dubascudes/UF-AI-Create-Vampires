# Local YOLO Person Segmentation Implementation Plan

> **For agentic workers:** Execute inline in this session. Tests precede implementation; do not change the active mirror until local inference works.

**Goal:** Compare local YOLO26n-seg person masks with the current Selfie Segmenter and use YOLO as an optional input/output mask source in the hybrid mirror.

**Architecture:** A local FastAPI process on port 8001 loads one pretrained YOLO26n-seg model on the RTX 4070 Ti SUPER. It accepts small JPEG frames and returns a union of class-0 person masks as raw bytes with dimensions and inference time in headers. Vite proxies `/seg` requests to it. The browser can select YOLO or MediaPipe; the rest of the learned-room and JoyAI pipeline is unchanged.

**Tech Stack:** Python 3, PyTorch CUDA, Ultralytics 8.4.161, FastAPI, Vite, TypeScript, Vitest, pytest.

**Spec:** User request in this conversation to try YOLO locally because Selfie Segmenter misses hands and full bodies.

## Global Constraints

- Keep MediaPipe selectable for A/B comparison.
- Do not silently show raw people if local segmentation fails.
- Keep the learned room and generated-output checks in the active pipeline.
- Report measured local inference time separately from full output FPS and delay.
- Keep downloaded model and environment out of source control.

### Task 1: Local model and service

**Files:** Create `server/person_segment.py`, `tests/test_person_segment.py`, `scripts/setup-yolo.py`; modify `.gitignore`.

**Interfaces:** `POST /seg/mask` consumes `image/jpeg`, returns `application/octet-stream` with one byte per source pixel, `X-Mask-Width`, `X-Mask-Height`, `X-Inference-Ms`; `GET /seg/health` returns model and device.

- [x] Write a failing Python test for union of multiple person masks and no detection.
- [x] Implement mask conversion and run the test.
- [x] Build a local Python environment using existing system CUDA PyTorch; install pinned Ultralytics.
- [x] Download official `yolo26n-seg.pt` into `models/` and validate one local CUDA inference.
- [x] Start service and verify health and binary mask response on a test image.

### Task 2: Browser backend selection

**Files:** Create `src/person-segmentation.ts`, `src/person-segmentation.test.ts`; modify `src/hybrid-main.ts`, `vite.config.ts`, `package.json`, `scripts/dev-all.mjs`.

**Interfaces:** `fetchYoloPersonMask(canvas)` returns `{ values: Float32Array, width, height, inferenceMs }`; MediaPipe continues to return its existing confidence mask.

- [x] Write a failing Vitest test for mask decoding and malformed responses.
- [x] Implement browser mask client and run test.
- [x] Add backend selector and route both camera and generated frames through selected backend.
- [x] Preserve fail-closed behavior if YOLO service becomes unavailable.
- [x] Run the full tests and build.

### Task 3: Local comparison and documentation

**Files:** Modify `README.md`; create result artifact under `runs/` if suitable local test frames exist.

- [x] Measure local service inference on a reproducible test image and record model, device, resolution, warmup, and timing.
- [x] Explain how to start the service and compare masks/FPS using the webcam; avoid claiming visual improvement until webcam observations support it.
- [x] Verify the browser page loads and endpoint responds.
