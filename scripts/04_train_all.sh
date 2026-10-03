#!/usr/bin/env bash
# Fine-tuning experiment grid. Run inside tmux: each run takes hours.
# Order = most informative first, so you can stop early if time runs out.
set -euo pipefail; cd "$(dirname "$0")/.."
source .venv-train/bin/activate
mkdir -p logs
B=models/Qwen3-8B
run() { local name=$1; shift; echo ">>> $name"; python train/train_sft.py --model $B --output-dir runs/$name "$@" 2>&1 | tee logs/train_$name.log; }

# 0) 5-minute smoke test: catches OOM / format bugs before burning hours
python train/train_sft.py --model $B --method lora --r 16 --max-train-samples 200 --epochs 1 \
  --eval-steps 10 --output-dir runs/_smoke --report-to none 2>&1 | tee logs/train_smoke.log
rm -rf runs/_smoke

run lora_r16  --method lora  --r 16
run lora_r64  --method lora  --r 64  --lr 1e-4
run qlora_r16 --method qlora --r 16
run dora_r16  --method dora  --r 16
run full      --method full  --lr 1e-5 --batch-size 1 --grad-accum 8 --liger
.venv-train/bin/python eval/collect_training.py
