#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1o frozen-electronic-surface validation.
# The isotropic 40x40 system is driven until the first natural x hop is accepted,
# then cloned into fully coupled released and frozen-electronic-surface branches.
# No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1o_local_validation.py

echo
printf 'IP1o completed. Results are under: ip1o-local-validation/\n'
