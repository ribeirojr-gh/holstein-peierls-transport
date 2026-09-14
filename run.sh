#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for IP1h fixed-boundary flux reanalysis.
# No dynamics are rerun. By default the newest complete IP1g artifact under
# ip1g-local-validation/ is selected automatically.

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

if [[ $# -gt 0 ]]; then
  python scripts/run_ip1h_local_validation.py "$1"
else
  python scripts/run_ip1h_local_validation.py
fi

echo
printf 'IP1h completed. Results are under: ip1h-local-validation/\n'
