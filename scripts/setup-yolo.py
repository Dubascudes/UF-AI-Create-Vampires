"""Download and warm the local YOLO26n-seg checkpoint on CUDA."""

import os
from pathlib import Path

import numpy as np
import torch


root = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str(root / ".yolo-config"))
from ultralytics import YOLO  # noqa: E402

weights = root / "models" / "yolo26n-seg.pt"
weights.parent.mkdir(exist_ok=True)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; check the NVIDIA driver and PyTorch build")
model = YOLO(str(weights))
result = model.predict(np.zeros((384, 640, 3), dtype=np.uint8), device=0, imgsz=640,
                       classes=[0], retina_masks=True, verbose=False)[0]
print(f"YOLO ready: {weights} ({weights.stat().st_size} bytes), CUDA {torch.cuda.get_device_name(0)}, "
      f"warmup inference {result.speed['inference']:.1f} ms")
