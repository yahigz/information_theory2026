#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="${1:-configs/grokking_mod_prime_113.yaml}"

uv run train_grokking.py --config "${CONFIG_PATH}"
