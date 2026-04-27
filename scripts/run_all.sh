#!/usr/bin/env bash
set -euo pipefail
ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/.. && pwd)
cd "$ROOT"
python scripts/train.py --config configs/dqn_cartpole.yaml
python scripts/train.py --config configs/double_dqn_cartpole.yaml
python scripts/plot_results.py --log-dir logs --output-dir results/figures
