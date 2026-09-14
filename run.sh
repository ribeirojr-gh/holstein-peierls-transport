#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1q mode-resolved frozen-memory validation.
# The isotropic 40x40 system is driven until the first natural x hop is accepted,
# then a 10 ps frozen-electronic-surface continuation is sampled for q-omega and
# polarization-resolved normal-mode analysis. No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1q_local_validation.py

echo
printf 'IP1q completed. Results are under: ip1q-local-validation/\n'
