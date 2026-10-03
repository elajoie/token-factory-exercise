#!/usr/bin/env bash
# One-time VM bootstrap (Nebius gpu-h100-sxm, 1 GPU / 16 vCPU / 200 GiB, Ubuntu image).
# Creates four isolated uv virtualenvs because the tools pin conflicting versions:
#   .venv-train  TRL/PEFT training      .venv-quant  llm-compressor
#   .venv-serve  vLLM + load testing    .venv-eval   BFCL (pins numpy 1.26)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "== system checks =="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
df -h "$PWD" | tail -1
FREE_GB=$(df --output=avail -BG "$PWD" | tail -1 | tr -dc '0-9')
if (( FREE_GB < 150 )); then
  echo "WARNING: only ${FREE_GB} GB free. Plan ~150 GB (4 venvs ~25 GB, base model 16 GB,"
  echo "         each merged bf16 model 16 GB, each FP8 model 9 GB, checkpoints). See RUNBOOK 'Disk'."
fi

echo "== OS packages =="
sudo apt-get update -y
sudo apt-get install -y git tmux htop jq curl build-essential python3-dev
command -v nvtop >/dev/null || sudo apt-get install -y nvtop || true

echo "== uv =="
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

export UV_LINK_MODE=hardlink          # share one package cache across the 4 venvs (saves disk)
for v in train quant serve; do
  uv venv ".venv-$v" --python 3.12
  uv pip install --python ".venv-$v/bin/python" -r "env/requirements-$v.txt"
done
uv venv .venv-eval --python 3.12
uv pip install --python .venv-eval/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-eval/bin/python -r env/requirements-eval.txt

echo "== freeze exact versions for reproducibility =="
mkdir -p env/lock
for v in train quant serve eval; do uv pip freeze --python ".venv-$v/bin/python" > "env/lock/$v.txt"; done

echo "== sanity =="
.venv-train/bin/python -c "import torch, trl, peft, transformers; print('train ok', torch.__version__, torch.cuda.is_available(), trl.__version__, peft.__version__, transformers.__version__)"
.venv-quant/bin/python -c "import llmcompressor; print('quant ok', llmcompressor.__version__)"
.venv-serve/bin/python -c "import vllm; print('serve ok', vllm.__version__)"
.venv-eval/bin/bfcl --help >/dev/null && echo "eval ok"
echo "setup complete"
