#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1n gauge-continuous field-release validation.
# The isotropic 40x40 system is driven until the first natural x hop is accepted,
# then cloned into field-on and zero-power held-phase continuations.
# No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1n_local_validation.py

echo
printf 'IP1n completed. Results are under: ip1n-local-validation/\n'
