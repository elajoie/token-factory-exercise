#!/usr/bin/env bash
# BFCL baseline of the untouched base model (the "before" number).
# Note: base Qwen3 thinks (<think>...</think>) before answering -> slower; BFCL strips the reasoning.
set -euo pipefail; cd "$(dirname "$0")/.."
./eval/run_bfcl.sh base_qwen3-8b_bf16 models/Qwen3-8B
