#!/usr/bin/env bash
# Data prep + contamination check + prompt-format parity test + production chat template.
set -euo pipefail; cd "$(dirname "$0")/.."
.venv-train/bin/python data/prepare_toolace.py --out-dir data/processed --val-ratio 0.05 --seed 42
.venv-eval/bin/python eval/contamination_check.py --data-dir data/processed
.venv-eval/bin/python serve/build_chat_template.py --out serve/chat_template.jinja
.venv-eval/bin/python tests/test_prompt_parity.py
.venv-eval/bin/python bench/make_bench_prompts.py --out bench/prompts_bfcl.jsonl
