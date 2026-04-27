#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)
cd "$ROOT"
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt
python -m pip install -e .
python - <<PY
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available())
PY
