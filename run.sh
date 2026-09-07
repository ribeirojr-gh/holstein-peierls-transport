#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for the IP1e no-dynamics size-aware recheck.
# Usage:
#   bash run.sh [path/to/ip1e-local-validation/<timestamp>]
#
# If no path is supplied, the newest directory under ip1e-local-validation/
# is selected automatically. If only ip1e-local-validation.zip is present,
# it is extracted first.

if [[ ! -d ip1e-local-validation && -f ip1e-local-validation.zip ]]; then
  unzip -q ip1e-local-validation.zip
fi

ARTIFACT_DIR="${1:-}"
if [[ -z "${ARTIFACT_DIR}" ]]; then
  if [[ ! -d ip1e-local-validation ]]; then
    echo "ERROR: ip1e-local-validation/ not found and no artifact directory was supplied." >&2
    exit 2
  fi
  ARTIFACT_DIR="$(find ip1e-local-validation -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1)"
fi

if [[ -z "${ARTIFACT_DIR}" || ! -d "${ARTIFACT_DIR}" ]]; then
  echo "ERROR: artifact directory not found: ${ARTIFACT_DIR}" >&2
  exit 2
fi

for required in baseline20-ip1d.json large40-ip1d.json weak40-ip1d.json; do
  if [[ ! -f "${ARTIFACT_DIR}/${required}" ]]; then
    echo "ERROR: missing ${ARTIFACT_DIR}/${required}" >&2
    exit 2
  fi
done

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

python scripts/run_ip1e_recheck_local_validation.py "${ARTIFACT_DIR}"

echo
printf 'Recheck completed. Results are under: ip1e-recheck-local-validation/\n'
