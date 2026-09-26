#!/usr/bin/env bash
# Latency / throughput sweep over client concurrency against the production server.
#
#   ./bench/run_bench.sh <variant_name> <model_dir>
#   ./bench/run_bench.sh lora16_bf16 models/lora_r16-merged
#   ./bench/run_bench.sh lora16_fp8  models/lora_r16-fp8
#   NO_PREFIX_CACHE=1 ./bench/run_bench.sh lora16_fp8_nocache models/lora_r16-fp8
#
# Two workloads per concurrency level:
#   natural : real BFCL prompts, model stops by itself (short function calls) -> realistic E2E
#   fixed128: same prompts, exactly 128 output tokens (--ignore-eos) -> stable TPOT/throughput
set -euo pipefail
cd "$(dirname "$0")/.."
VARIANT=${1:?variant}; MODEL_DIR=$(realpath "${2:?model dir}")
PORT=${PORT:-8000}
CONCURRENCY=${CONCURRENCY:-"1 8 16 24 32 48"}
NUM_PROMPTS_PER_CLIENT=${NUM_PROMPTS_PER_CLIENT:-20}
OUT="results/bench/$VARIANT"; mkdir -p "$OUT" logs

MODEL_PATH="$MODEL_DIR" PORT=$PORT ./serve/serve.sh > "logs/serve_bench_${VARIANT}.log" 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null; wait $SERVER_PID 2>/dev/null || true' EXIT
PORT=$PORT ./serve/wait_ready.sh
source .venv-serve/bin/activate

common=(--backend vllm --host localhost --port "$PORT" --endpoint /v1/completions
        --model "$MODEL_DIR" --served-model-name toolace-qwen3-8b
        --dataset-name custom --dataset-path bench/prompts_bfcl.jsonl --skip-chat-template
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,95,99
        --temperature 0 --seed 0 --save-result --result-dir "$OUT")

# warm-up (fills CUDA graphs, prefix cache realistic state is then built by the runs)
vllm bench serve "${common[@]}" --num-prompts 64 --max-concurrency 16 --custom-output-len 64 \
  --result-filename warmup.json > /dev/null

for C in $CONCURRENCY; do
  N=$(( C * NUM_PROMPTS_PER_CLIENT )); [[ $N -lt 100 ]] && N=100
  echo "=== $VARIANT | concurrency=$C | natural ==="
  vllm bench serve "${common[@]}" --num-prompts $N --max-concurrency $C --request-rate inf \
    --custom-output-len 256 --result-filename "natural_c${C}.json" | tee -a "logs/bench_${VARIANT}.log"
  echo "=== $VARIANT | concurrency=$C | fixed128 ==="
  vllm bench serve "${common[@]}" --num-prompts $N --max-concurrency $C --request-rate inf \
    --custom-output-len 128 --ignore-eos --result-filename "fixed128_c${C}.json" | tee -a "logs/bench_${VARIANT}.log"
done
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv > "$OUT/gpu_mem.csv"
python bench/summarize_bench.py --results-dir results/bench --out results/bench_summary.md
