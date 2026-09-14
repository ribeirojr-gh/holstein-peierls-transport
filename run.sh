#!/usr/bin/env bash
set -euo pipefail

# Self-contained local runner for the IP1e no-dynamics size-aware recheck.
# Usage:
#   bash run.sh
#   bash run.sh path/to/ip1e-local-validation/<timestamp>
#   bash run.sh path/to/ip1e-local-validation.zip
#
# With no argument, the runner uses the newest directory under
# ip1e-local-validation/. If only ./ip1e-local-validation.zip is present,
# it extracts that archive first.

INPUT="${1:-}"

# If the user supplied the original validation ZIP explicitly, extract it into
# the current repository root and continue with automatic artifact discovery.
if [[ -n "${INPUT}" && -f "${INPUT}" && "${INPUT}" == *.zip ]]; then
  echo "Extracting artifact ZIP: ${INPUT}"
  unzip -q "${INPUT}" -d .
  INPUT=""
fi

# No explicit input: extract a ZIP in the repository root when present.
if [[ -z "${INPUT}" && ! -d ip1e-local-validation && -f ip1e-local-validation.zip ]]; then
  echo "Extracting ./ip1e-local-validation.zip"
  unzip -q ip1e-local-validation.zip -d .
fi

ARTIFACT_DIR="${INPUT}"
if [[ -z "${ARTIFACT_DIR}" ]]; then
  if [[ ! -d ip1e-local-validation ]]; then
    cat >&2 <<'EOF'
ERROR: no IP1e validation artifact was found.

Expected either:
  1. ./ip1e-local-validation/<timestamp>/
  2. ./ip1e-local-validation.zip
  3. an explicit directory or ZIP argument, for example:
       bash run.sh ~/Downloads/ip1e-local-validation.zip
       bash run.sh /path/to/ip1e-local-validation/20260907T183017Z
EOF
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

echo "Using IP1e artifact directory: ${ARTIFACT_DIR}"

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
