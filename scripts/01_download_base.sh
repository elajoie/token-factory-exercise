#!/usr/bin/env bash
# Download the base model once to a local dir (pinned revision recorded for reproducibility).
set -euo pipefail; cd "$(dirname "$0")/.."
source .venv-train/bin/activate
BASE_REV=${BASE_REV:-main}
hf download Qwen/Qwen3-8B --revision "$BASE_REV" --local-dir models/Qwen3-8B
python - <<'PY'
from huggingface_hub import HfApi
api = HfApi()
print("Qwen/Qwen3-8B sha:", api.model_info("Qwen/Qwen3-8B").sha)
print("Team-ACE/ToolACE sha:", api.dataset_info("Team-ACE/ToolACE").sha)
PY
