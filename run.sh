#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1s energy-preserving Peierls direction-reversal control.
# The isotropic 40x40 system is driven until the first natural x hop is accepted,
# then two zero-power fully coupled branches are compared: native released and a
# counterfactual with all non-special vx traveling directions reversed at fixed
# coordinates and fixed total energy. No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1s_local_validation.py

echo
printf 'IP1s completed. Results are under: ip1s-local-validation/\n'
