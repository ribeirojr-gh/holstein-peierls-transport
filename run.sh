#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1r traveling-wave attribution.
# The isotropic 40x40 system is driven until the first natural x hop is accepted,
# then a 5 ps frozen-electronic-surface trajectory is decomposed exactly into
# retrograde, co-moving, special and cross/interference x-current components.
# No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1r_local_validation.py

echo
printf 'IP1r completed. Results are under: ip1r-local-validation/\n'
