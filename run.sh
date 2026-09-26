#!/usr/bin/env bash
set -euo pipefail

# S1R local fallback: deterministic spin-adapted singlet root-manifold audit.
# GitHub Actions is the primary execution environment. The exact numerical
# environment is pinned to match the preregistered S1R production workflow.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install 'numpy==2.5.3' 'scipy==1.18.1' 'pytest==9.1.1'
python -m pip install -e . --no-deps

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

python scripts/run_s1r_local_validation.py

echo
printf 'S1R completed. Results are under: s1r-local-validation/\n'
