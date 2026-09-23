#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP2a deterministic fixed-energy low-q
# Peierls ensemble construction. This stage validates the 32-member paired
# preparation family and the three preregistered candidate perturbation
# energies only. It does not propagate native/reversed outcomes or select the
# production energy.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip2a_local_validation.py

echo
printf 'IP2a completed. Results are under: ip2a-local-validation/\n'
