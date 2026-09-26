#!/usr/bin/env bash
# Regenerate every results table.
set -euo pipefail; cd "$(dirname "$0")/.."
.venv-train/bin/python eval/collect_training.py
.venv-eval/bin/python eval/collect_bfcl.py --runs-dir eval/bfcl_runs --out results/bfcl_summary.md
.venv-serve/bin/python bench/summarize_bench.py --results-dir results/bench --out results/bench_summary.md
ls -la results
