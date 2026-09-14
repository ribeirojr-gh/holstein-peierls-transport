#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1f carrier-centered phonon-wake validation.
# This stage runs new 40x40 dynamics. No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1f_local_validation.py

echo
printf 'IP1f completed. Results are under: ip1f-local-validation/\n'
