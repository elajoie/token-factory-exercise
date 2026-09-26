#!/usr/bin/env bash
# Quantise the WINNER (highest BFCL python_overall) and re-check accuracy.
#   ./scripts/06_quantize_and_eval.sh models/lora_r16-merged lora_r16
set -euo pipefail; cd "$(dirname "$0")/.."
SRC=${1:?merged bf16 model dir}; NAME=${2:?short name}
[[ -d models/${NAME}-fp8   ]] || .venv-quant/bin/python quantize/quantize.py --model $SRC --scheme fp8   --out models/${NAME}-fp8
[[ -d models/${NAME}-w4a16 ]] || .venv-quant/bin/python quantize/quantize.py --model $SRC --scheme w4a16 --out models/${NAME}-w4a16
./eval/run_bfcl.sh ${NAME}_fp8          models/${NAME}-fp8
KV_FP8=1 ./eval/run_bfcl.sh ${NAME}_fp8_kvfp8 models/${NAME}-fp8
./eval/run_bfcl.sh ${NAME}_w4a16        models/${NAME}-w4a16
du -sh $SRC models/${NAME}-fp8 models/${NAME}-w4a16 | tee results/model_sizes.txt
