"""Prepare a short, matched video and white-person-mask video for Wan VACE."""

from pathlib import Path
import argparse

import cv2
import numpy as np
from ultralytics import YOLO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--fps", type=int, default=16)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(args.source))
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or source_fps <= 0:
        raise RuntimeError(f"Cannot read source video: {args.source}")
    model = YOLO("models/yolo26n-seg.pt")
    size = (832, 480)
    codec = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(str(args.output_dir / "source.mp4"), codec, args.fps, size)
    masks = cv2.VideoWriter(str(args.output_dir / "mask.mp4"), codec, args.fps, size, isColor=True)
    if not video.isOpened() or not masks.isOpened():
        raise RuntimeError("Cannot open output video writers")
    coverage = []
    try:
        for index in range(args.frames):
            capture.set(cv2.CAP_PROP_POS_FRAMES, round(index * source_fps / args.fps))
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Source ended before requested frame {index}")
            frame = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            result = model.predict(frame, classes=[0], conf=0.25, imgsz=640, verbose=False)[0]
            person = np.zeros((size[1], size[0]), dtype=np.uint8)
            if result.masks is not None:
                for segment in result.masks.data.cpu().numpy():
                    person = np.maximum(person, cv2.resize((segment > 0.5).astype(np.uint8) * 255, size))
            person = cv2.dilate(person, np.ones((9, 9), dtype=np.uint8))
            coverage.append(float(np.count_nonzero(person) / person.size))
            video.write(frame)
            masks.write(cv2.cvtColor(person, cv2.COLOR_GRAY2BGR))
    finally:
        capture.release()
        video.release()
        masks.release()
    print(f"frames={args.frames} fps={args.fps} coverage_min={min(coverage):.3f} "
          f"coverage_max={max(coverage):.3f} coverage_mean={np.mean(coverage):.3f}")


if __name__ == "__main__":
    main()
