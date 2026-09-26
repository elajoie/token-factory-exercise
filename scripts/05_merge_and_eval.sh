#!/usr/bin/env bash
# Merge each adapter into bf16 weights and run BFCL on every variant.
#   ./scripts/05_merge_and_eval.sh                 # all runs
#   ./scripts/05_merge_and_eval.sh lora_r16        # one run
set -euo pipefail; cd "$(dirname "$0")/.."
RUNS=${@:-lora_r16 lora_r64 qlora_r16 dora_r16 full}
for r in $RUNS; do
  [[ -d runs/$r/final ]] || { echo "skip $r (not trained)"; continue; }
  if [[ $r == full ]]; then
    MODEL_DIR=models/full-merged           # symlink so later steps treat all variants alike
    mkdir -p models && ln -sfn "$PWD/runs/full/final" $MODEL_DIR
  else
    MODEL_DIR=models/${r}-merged
    [[ -d $MODEL_DIR ]] || .venv-train/bin/python train/merge_lora.py --base models/Qwen3-8B --adapter runs/$r/final --out $MODEL_DIR
  fi
  ./eval/run_bfcl.sh ${r}_bf16 $MODEL_DIR
done
