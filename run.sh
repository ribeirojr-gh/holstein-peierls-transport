#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1k natural-hop wake replication/background audit.
# This stage reruns only the isotropic and anisotropic 40x40, 10 mV/A controls.
# No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1k_local_validation.py

echo
printf 'IP1k completed. Results are under: ip1k-local-validation/\n'
