#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1u post-return Peierls escape-stability control.
# The isotropic 40x40 system is driven to the first natural hop, continued under
# zero-power held phase through a common +x recrossing and -x return. Only after the
# return is accepted does it branch into native and energy-preserving
# vx-direction-reversed fully coupled trajectories. No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1u_local_validation.py

echo
printf 'IP1u completed. Results are under: ip1u-local-validation/\n'
