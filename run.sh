#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1l inter-hop wake-memory validation.
# This stage reruns only the anisotropic 40x40, 10 mV/A control and saves the
# complete sampled intermolecular x-current trajectory for posthoc analysis.
# No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1l_local_validation.py

echo
printf 'IP1l completed. Results are under: ip1l-local-validation/\n'
