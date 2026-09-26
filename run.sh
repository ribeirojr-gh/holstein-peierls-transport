#!/usr/bin/env bash
set -euo pipefail

# S1 local fallback: spin-adapted stationary RPROP bridge.
# GitHub Actions is the primary execution environment. This runner preserves
# a self-contained local reproduction path for the full 2x2x3 comparison.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_s1_local_validation.py

echo
printf 'S1 completed. Results are under: s1-local-validation/\n'
