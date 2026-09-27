#!/usr/bin/env bash
set -euo pipefail

# S2 local fallback: unified static-sector regression.
# GitHub Actions is the primary execution environment.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy==2.5.3' 'scipy==1.18.1' 'pytest==9.1.1'
python -m pip install -e . --no-deps

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_s2_local_validation.py

echo
printf 'S2 completed. Results are under: s2-local-validation/\n'
