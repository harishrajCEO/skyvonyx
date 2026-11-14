#!/usr/bin/env bash
# One-shot environment setup for a Colab / fresh Linux GPU box.
# Equivalent to notebook Cell 2. Usage:  bash scripts/bootstrap_colab.sh
set -euo pipefail
cd "$(dirname "$0")/.."

bash scripts/clone_repos.sh

# SAM 2's CUDA extension is optional for image prediction and frequently fails to
# compile on Colab's toolchain; skipping it is the upstream-documented fallback.
export SAM2_BUILD_CUDA=0

pip install -q -r requirements-colab.txt
pip install -q -e vendor/sahi
pip install -q -e vendor/sam2
pip install -q -e vendor/segment-anything
# Optional, only when an RF-DETR fine-tune exists:
# pip install -q -e vendor/rf-detr

python - <<'EOF'
import torch, sahi, ultralytics, rasterio
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
print("sahi", getattr(sahi, "__version__", "?"), "| ultralytics", ultralytics.__version__, "| rasterio", rasterio.__version__)
import sam2; print("sam2 importable")
EOF
python tests/test_pipeline.py | tail -3
