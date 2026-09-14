#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1g controlled single-relocation wake validation.
# This stage runs new deterministic 40x40 dynamics. No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1g_local_validation.py

echo
printf 'IP1g completed. Results are under: ip1g-local-validation/\n'
