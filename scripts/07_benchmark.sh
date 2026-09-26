#!/usr/bin/env bash
# Latency/throughput sweep (1..48 concurrent) for the candidate deployments.
#   ./scripts/07_benchmark.sh lora_r16
set -euo pipefail; cd "$(dirname "$0")/.."
NAME=${1:?short name of the winner}
./bench/run_bench.sh ${NAME}_bf16          models/${NAME}-merged
./bench/run_bench.sh ${NAME}_fp8           models/${NAME}-fp8
KV_FP8=1 ./bench/run_bench.sh ${NAME}_fp8_kvfp8 models/${NAME}-fp8
NO_PREFIX_CACHE=1 ./bench/run_bench.sh ${NAME}_fp8_nocache models/${NAME}-fp8
./bench/run_bench.sh ${NAME}_w4a16         models/${NAME}-w4a16
