"""Local CUDA person-mask service for the hybrid mirror."""

from __future__ import annotations

from io import BytesIO
import os
import struct
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import Response
from PIL import Image


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "yolo26n-seg.pt"
app = FastAPI(title="Absent local person segmenter")
_model = None
_lock = Lock()


def union_person_masks(masks: np.ndarray | None, width: int, height: int) -> bytes:
    """Return one 8-bit mask in source-image order from class-filtered instance masks."""
    if width < 1 or height < 1:
        raise ValueError("Invalid source dimensions")
    if masks is None or len(masks) == 0:
        return bytes(width * height)
    if masks.ndim != 3:
        raise ValueError("Expected instance masks with shape N,H,W")
    merged = (np.max(masks, axis=0) >= 0.5).astype(np.uint8) * 255
    if merged.shape != (height, width):
        merged = np.asarray(
            Image.fromarray(merged).resize((width, height), Image.Resampling.NEAREST)
        )
    return np.ascontiguousarray(merged).tobytes()


def decode_rgba_request(message: bytes) -> tuple[int, np.ndarray]:
    if len(message) < 8:
        raise ValueError("Frame header is missing")
    request_id, width, height = struct.unpack_from("<IHH", message)
    if not 1 <= width <= 1280 or not 1 <= height <= 720 or len(message) != 8 + width * height * 4:
        raise ValueError("RGBA frame dimensions do not match payload")
    rgba = np.frombuffer(message, dtype=np.uint8, offset=8).reshape(height, width, 4)
    return request_id, np.ascontiguousarray(rgba[:, :, :3])


def encode_mask_response(request_id: int, width: int, height: int, inference_ms: float, mask: bytes) -> bytes:
    if len(mask) != width * height:
        raise ValueError("Person mask dimensions do not match payload")
    return struct.pack("<IHHf", request_id, width, height, inference_ms) + mask


def model():
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"YOLO weights missing: {MODEL_PATH}. Run npm run segment:setup.")
        os.environ.setdefault("YOLO_CONFIG_DIR", str(MODEL_PATH.parent.parent / ".yolo-config"))
        import torch
        from ultralytics import YOLO

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable to the local YOLO process")
        loaded = YOLO(str(MODEL_PATH))
        loaded.predict(np.zeros((384, 640, 3), dtype=np.uint8), device=0, imgsz=640,
                       classes=[0], retina_masks=True, verbose=False)
        _model = loaded
    return _model


@app.get("/seg/health")
def health() -> dict[str, str]:
    try:
        model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"model": MODEL_PATH.name, "device": "cuda:0"}


@app.post("/seg/mask")
async def person_mask(request: Request) -> Response:
    if request.headers.get("content-type", "").split(";")[0] != "image/jpeg":
        raise HTTPException(status_code=415, detail="Send an image/jpeg frame")
    body = await request.body()
    if len(body) > 2_000_000:
        raise HTTPException(status_code=413, detail="Frame is too large")
    try:
        image = Image.open(BytesIO(body)).convert("RGB")
        if not 1 <= image.width <= 1280 or not 1 <= image.height <= 720:
            raise ValueError("Frame dimensions exceed 1280 x 720")
    except (OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="Invalid JPEG frame") from exc
    try:
        with _lock:
            start = perf_counter()
            prediction = model().predict(np.asarray(image), device=0, imgsz=640,
                                         classes=[0], retina_masks=True, verbose=False)[0]
            elapsed = (perf_counter() - start) * 1000
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    masks = prediction.masks.data.cpu().numpy() if prediction.masks is not None else None
    payload = union_person_masks(masks, image.width, image.height)
    return Response(payload, media_type="application/octet-stream", headers={
        "X-Mask-Width": str(image.width), "X-Mask-Height": str(image.height),
        "X-Inference-Ms": f"{elapsed:.1f}",
    })


@app.websocket("/seg/stream")
async def person_mask_stream(socket: WebSocket) -> None:
    await socket.accept()
    try:
        while True:
            message = await socket.receive_bytes()
            try:
                request_id, image = decode_rgba_request(message)
            except ValueError:
                await socket.close(code=1003)
                return
            with _lock:
                start = perf_counter()
                prediction = model().predict(image, device=0, imgsz=640,
                                             classes=[0], retina_masks=True, verbose=False)[0]
                elapsed = (perf_counter() - start) * 1000
            masks = prediction.masks.data.cpu().numpy() if prediction.masks is not None else None
            height, width = image.shape[:2]
            await socket.send_bytes(encode_mask_response(
                request_id, width, height, elapsed, union_person_masks(masks, width, height)
            ))
    except WebSocketDisconnect:
        return
