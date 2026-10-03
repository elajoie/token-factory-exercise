#!/usr/bin/env bash
# Production-style vLLM server (OpenAI-compatible API, same interface as Nebius Token Factory).
#
#   MODEL_PATH=models/lora_r16-fp8 ./serve/serve.sh            # foreground
#   MODEL_PATH=models/lora_r16-fp8 KV_FP8=1 ./serve/serve.sh   # + FP8 KV cache
#
# Two served names: a friendly one for clients, and the path itself because BFCL
# (--skip-server-setup) sends `model=<local path>`.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv-serve/bin/activate

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
SERVED_NAME=${SERVED_NAME:-toolace-qwen3-8b}
PORT=${PORT:-8000}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-64}          # > 32 target concurrency, leaves headroom
GPU_UTIL=${GPU_UTIL:-0.90}
EXTRA_ARGS=${EXTRA_ARGS:-}

ARGS=(
  "$MODEL_PATH"
  --served-model-name "$SERVED_NAME" "$MODEL_PATH"
  --host 0.0.0.0 --port "$PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --max-num-seqs "$MAX_NUM_SEQS"
  --gpu-memory-utilization "$GPU_UTIL"
  --enable-prefix-caching                 # tool schemas repeat across requests -> big TTFT win
  --enable-chunked-prefill                # long prompts don't stall running decodes
  --chat-template serve/chat_template.jinja
  --enable-auto-tool-choice --tool-call-parser pythonic
  --generation-config vllm                # ignore HF sampling defaults; clients control sampling
  --seed 0
)
[[ "${KV_FP8:-0}" == "1" ]] && ARGS+=(--kv-cache-dtype fp8)
[[ "${NO_PREFIX_CACHE:-0}" == "1" ]] && ARGS=("${ARGS[@]/--enable-prefix-caching/--no-enable-prefix-caching}")

echo "vllm serve ${ARGS[*]} $EXTRA_ARGS"
exec vllm serve "${ARGS[@]}" $EXTRA_ARGS
