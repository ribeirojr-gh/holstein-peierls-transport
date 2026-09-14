#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1j first-natural-hop field-driven wake validation.
# This stage reruns only the isotropic and anisotropic 40x40, 10 mV/A controls.
# No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1j_local_validation.py

echo
printf 'IP1j completed. Results are under: ip1j-local-validation/\n'
