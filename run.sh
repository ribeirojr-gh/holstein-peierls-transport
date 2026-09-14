#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1t post-recrossing Peierls direction control.
# The isotropic 40x40 system is driven to the first natural hop, continued under
# zero-power held phase until the first common +x recrossing is accepted, and
# only then branched into native and energy-preserving vx-direction-reversed
# fully coupled trajectories. No prior artifact ZIP is required.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1t_local_validation.py

echo
printf 'IP1t completed. Results are under: ip1t-local-validation/\n'
