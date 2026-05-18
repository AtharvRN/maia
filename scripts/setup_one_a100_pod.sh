#!/usr/bin/env bash
set -euo pipefail

# Minimal setup for the real-neuron MAIA demo on a Kubernetes pod with:
# - 1x A100 80 GB
# - about 32 GB system RAM
#
# This intentionally skips GroundingDINO/SAM and the synthetic-neuron stack.

ENV_NAME="${ENV_NAME:-maia-one-a100}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required for this setup script." >&2
  exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda create -y -n "${ENV_NAME}" "python=${PYTHON_VERSION}" pip
fi

conda activate "${ENV_NAME}"
python -m pip install -U pip wheel setuptools

python -m pip install \
  torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
  --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements-one-a100.txt

python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
