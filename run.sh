#!/usr/bin/env bash
set -euo pipefail

# IP2b local fallback: complete 32-member deterministic paired counterfactual
# production ensemble. GitHub Actions is the primary execution environment.
# This runner preserves a self-contained local reproduction path.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip2b_local_validation.py

echo
printf 'IP2b completed. Results are under: ip2b-local-validation/\n'
