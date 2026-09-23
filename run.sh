#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP2a-2 pre-intervention event-generation
# calibration. This stage evaluates only field-free preparation stability,
# numerical quality, and first persistent-event yield for the fixed eight-member
# pilot subset at the three preregistered candidate energies. It does not create
# native/reversed post-event branches.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip2a2_local_validation.py

echo
printf 'IP2a-2 completed. Results are under: ip2a2-local-validation/\n'
