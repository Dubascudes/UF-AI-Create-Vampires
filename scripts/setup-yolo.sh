#!/usr/bin/env bash
set -euo pipefail
python3 -m venv --system-site-packages yolo-env
yolo-env/bin/python -m pip install --no-deps \
  ultralytics==8.4.161 torchvision==0.21.0 ultralytics-thop==2.1.6 \
  polars==1.44.2 polars-runtime-32==1.44.2 \
  nvidia-ml-py==13.610.43 ultralytics-platform==0.1.62
mkdir -p .yolo-config/Ultralytics
yolo-env/bin/python scripts/setup-yolo.py
