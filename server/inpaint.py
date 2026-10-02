"""Local GPU-backed LaMa inpainting endpoint for the fixed-camera mirror."""

from __future__ import annotations

import base64
import binascii
import struct
from io import BytesIO
from pathlib import Path
from threading import Lock
from time import perf_counter

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from PIL import Image, ImageFilter
from pydantic import BaseModel


MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "lama_fp32.onnx"
MODEL_SIZE = (512, 512)
app = FastAPI(title="Absent local inpainter")
_session: ort.InferenceSession | None = None
_lock = Lock()


class InpaintRequest(BaseModel):
    image: str
    mask: str


def session() -> ort.InferenceSession:
    global _session
    if _session is None:
        if not MODEL_PATH.is_file():
            raise RuntimeError(f"LaMa model missing: {MODEL_PATH}. Run npm run inpaint:setup.")
        if "CUDAExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError("ONNX Runtime CUDA provider is unavailable")
        options = ort.SessionOptions()
        options.log_severity_level = 3
        _session = ort.InferenceSession(
            str(MODEL_PATH), sess_options=options, providers=["CUDAExecutionProvider"]
        )
        _session.disable_fallback()
    return _session


def decode_png(data_url: str, mode: str) -> Image.Image:
    try:
        encoded = data_url.split(",", 1)[1] if data_url.startswith("data:") else data_url
        return Image.open(BytesIO(base64.b64decode(encoded, validate=True))).convert(mode)
    except (ValueError, OSError, IndexError, binascii.Error) as exc:
        raise HTTPException(status_code=400, detail="Invalid image or mask data") from exc


def inpaint(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, float]:
    if image.size != mask.size or not 1 <= image.width <= 1024 or not 1 <= image.height <= 1024:
        raise HTTPException(status_code=400, detail="Image and mask dimensions must match and be at most 1024")
    start = perf_counter()
    mask = mask.filter(ImageFilter.MaxFilter(9))
    source = np.asarray(image.resize(MODEL_SIZE, Image.Resampling.BILINEAR), dtype=np.float32)
    masked = np.asarray(mask.resize(MODEL_SIZE, Image.Resampling.NEAREST), dtype=np.uint8)
    inputs = {
        "image": np.ascontiguousarray(source.transpose(2, 0, 1)[None] / 255.0),
        "mask": np.ascontiguousarray((masked[None, None] > 127).astype(np.float32)),
    }
    with _lock:
        generated = session().run(None, inputs)[0]
    pixels = np.clip(generated[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    result = Image.fromarray(pixels, "RGB").resize(image.size, Image.Resampling.BILINEAR)
    # Keep actual camera evidence intact; only the requested hole is generated.
    result.paste(image, mask=Image.eval(mask, lambda value: 255 - value))
    return result, (perf_counter() - start) * 1000


def inpaint_letterboxed(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, float]:
    """Inpaints an image of any aspect ratio: scales its long side to the model size and pads
    the rest (edge-extended, unmasked) instead of stretching it to a square."""
    start = perf_counter()
    width, height = image.size
    scale = MODEL_SIZE[0] / max(width, height)
    small_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    small = np.asarray(image.resize(small_size, Image.Resampling.LANCZOS), dtype=np.float32)
    small_mask = mask.resize(small_size, Image.Resampling.BILINEAR).point(lambda value: 255 if value > 127 else 0)
    small_mask = np.asarray(small_mask.filter(ImageFilter.MaxFilter(5)), dtype=np.uint8)
    pad = ((0, MODEL_SIZE[1] - small_size[1]), (0, MODEL_SIZE[0] - small_size[0]))
    inputs = {
        "image": np.ascontiguousarray(np.pad(small, pad + ((0, 0),), mode="edge").transpose(2, 0, 1)[None] / 255.0),
        "mask": np.ascontiguousarray((np.pad(small_mask, pad) > 127).astype(np.float32)[None, None]),
    }
    with _lock:
        generated = session().run(None, inputs)[0]
    pixels = np.clip(generated[0].transpose(1, 2, 0), 0, 255).astype(np.uint8)[: small_size[1], : small_size[0]]
    result = Image.fromarray(pixels, "RGB").resize(image.size, Image.Resampling.BICUBIC)
    # Keep actual camera evidence intact; only the requested hole is generated.
    hole = Image.fromarray(small_mask, "L").resize(image.size, Image.Resampling.BILINEAR)
    result.paste(image, mask=Image.eval(hole, lambda value: 255 - value))
    return result, (perf_counter() - start) * 1000


def encode(image: Image.Image, format: str, **options) -> bytes:
    encoded = BytesIO()
    image.save(encoded, format=format, **options)
    return encoded.getvalue()


@app.get("/api/health")
def health() -> dict[str, str]:
    try:
        loaded = session()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"model": "LaMa fp32", "provider": loaded.get_providers()[0]}


@app.post("/api/inpaint")
def inpaint_route(request: InpaintRequest) -> Response:
    image = decode_png(request.image, "RGB")
    mask = decode_png(request.mask, "L")
    try:
        output, elapsed = inpaint(image, mask)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(encode(output, "PNG"), media_type="image/png", headers={"X-Inference-Ms": f"{elapsed:.1f}"})


@app.post("/api/inpaint-frame")
async def inpaint_frame_route(request: Request) -> Response:
    """Binary variant for the phone: u32 JPEG length, JPEG, u16 mask width, u16 mask height,
    mask bytes (0–255, rows top-down). The mask is scaled to the image, which may be any
    aspect ratio up to 2048 px per side. Returns a JPEG of the same size."""
    body = await request.body()
    try:
        (jpeg_length,) = struct.unpack_from(">I", body, 0)
        image = Image.open(BytesIO(body[4:4 + jpeg_length])).convert("RGB")
        mask_width, mask_height = struct.unpack_from(">HH", body, 4 + jpeg_length)
        mask_bytes = body[8 + jpeg_length:]
        if len(mask_bytes) != mask_width * mask_height:
            raise ValueError("mask size mismatch")
        mask = Image.frombytes("L", (mask_width, mask_height), mask_bytes)
        if not (1 <= image.width <= 2048 and 1 <= image.height <= 2048):
            raise ValueError("image too large")
    except (struct.error, ValueError, OSError) as exc:
        raise HTTPException(status_code=400, detail="Invalid frame") from exc
    try:
        output, elapsed = inpaint_letterboxed(image, mask)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return Response(encode(output, "JPEG", quality=90), media_type="image/jpeg", headers={"X-Inference-Ms": f"{elapsed:.1f}"})
