#!/usr/bin/env bash
set -euo pipefail

# IP2a-4: independently preregistered complete pre-intervention calibration
# at refined dt=0.10 fs. All 24 pilot trials, three original perturbation
# energies and the original 2e-6 eV energy/work bound remain unchanged.
# This stage has no native/reversed post-event branches. Original IP2a-2
# selection remains formally failed. IP2b is blocked unless IP2a-4 passes.
#
# Invoke with "bash run.sh" so this tracked file's executable bit is unchanged.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip2a4_local_validation.py

echo
printf 'IP2a-4 completed. Results are under: ip2a4-local-validation/\n'
