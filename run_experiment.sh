#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/grokking_mod_prime_113.yaml}"
SUBMIT_PYTHON="${SUBMIT_PYTHON:-python3}"

QUEUE_NAME="$(awk '
  $1 == "clearml:" { in_clearml=1; next }
  in_clearml && $1 ~ /^[^[:space:]].*:$/ { in_clearml=0 }
  in_clearml && $1 == "queue:" { print $2; exit }
' "${CONFIG_PATH}")"

if [[ -n "${QUEUE_NAME}" && "${QUEUE_NAME}" != "null" ]]; then
  echo "Submitting task to ClearML queue '${QUEUE_NAME}' without local training dependency sync"
  "${SUBMIT_PYTHON}" train_grokking.py --config "${CONFIG_PATH}"
else
  echo "Running locally via uv"
  uv run train_grokking.py --config "${CONFIG_PATH}"
fi
