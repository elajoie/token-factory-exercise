#!/usr/bin/env bash
# Evaluate one model variant on the BFCL "python" test collection against a LIVE vLLM
# server started with the production config (serve/serve.sh) - i.e. we test what we ship.
#
#   ./eval/run_bfcl.sh <variant_name> <model_dir> [extra env for serve.sh]
#   ./eval/run_bfcl.sh base_bf16    models/Qwen3-8B
#   ./eval/run_bfcl.sh lora16_bf16  models/lora_r16-merged
#   ./eval/run_bfcl.sh lora16_fp8   models/lora_r16-fp8
#   KV_FP8=1 ./eval/run_bfcl.sh lora16_fp8_kvfp8 models/lora_r16-fp8
#
# BFCL registry entry "Qwen/Qwen3-8B" = QwenHandler in *prompting* mode; --local-model-path
# swaps in our weights while keeping BFCL's exact prompt construction and AST checker.
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANT=${1:?variant name}; MODEL_DIR=$(realpath "${2:?model dir}")
CATEGORY=${CATEGORY:-python}
PORT=${PORT:-8000}
RUN_ROOT="$PWD/eval/bfcl_runs/$VARIANT"
mkdir -p "$RUN_ROOT" logs

MODEL_PATH="$MODEL_DIR" PORT=$PORT ./serve/serve.sh > "logs/serve_${VARIANT}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true' EXIT
PORT=$PORT ./serve/wait_ready.sh

source .venv-eval/bin/activate
export BFCL_PROJECT_ROOT="$RUN_ROOT" LOCAL_SERVER_ENDPOINT=localhost LOCAL_SERVER_PORT=$PORT

start=$(date +%s)
bfcl generate --model Qwen/Qwen3-8B --test-category "$CATEGORY" \
  --skip-server-setup --local-model-path "$MODEL_DIR" --allow-overwrite \
  2>&1 | tee "logs/bfcl_generate_${VARIANT}.log"
bfcl evaluate --model Qwen/Qwen3-8B --test-category "$CATEGORY" \
  2>&1 | tee "logs/bfcl_evaluate_${VARIANT}.log"
echo "{\"variant\": \"$VARIANT\", \"model_dir\": \"$MODEL_DIR\", \"wall_s\": $(( $(date +%s) - start ))}" \
  > "$RUN_ROOT/run_meta.json"

python eval/collect_bfcl.py --runs-dir eval/bfcl_runs --out results/bfcl_summary.md
