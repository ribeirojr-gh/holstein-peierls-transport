#!/usr/bin/env bash
set -euo pipefail

# IP2a-3: independent numerical time-step diagnostic for pre-event
# electric-field energy/work balance. Fixed unperturbed and member-0 references,
# dt=0.2/0.1/0.05 fs, no native/reversed post-event branches. IP2a-2 remains
# formally unqualified regardless of the refinement result.
#
# Invoke as "bash run.sh" to avoid changing this tracked file's executable bit.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip2a3_local_validation.py

echo
printf 'IP2a-3 completed. Results are under: ip2a3-local-validation/\n'
