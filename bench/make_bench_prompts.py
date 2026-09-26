#!/usr/bin/env python
"""Build a realistic load-test prompt set: BFCL python single-turn questions rendered
EXACTLY as BFCL renders them for Qwen3 (system prompt with JSON tool schemas + ChatML).

Output: bench/prompts_bfcl.jsonl  ({"prompt": "<full raw prompt>"} per line) for
`vllm bench serve --dataset-name custom --skip-chat-template`.

Run in the eval venv:
    python bench/make_bench_prompts.py --out bench/prompts_bfcl.jsonl
"""
import argparse
import copy
import json
import random
import statistics
from pathlib import Path

import bfcl_eval
from bfcl_eval.model_handler.local_inference.qwen import QwenHandler

CATS = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
        "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
        "live_irrelevance", "live_relevance"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="bench/prompts_bfcl.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    h = QwenHandler("Qwen/Qwen3-8B", 0.001, "Qwen/Qwen3-8B", False)
    data_dir = Path(bfcl_eval.__file__).parent / "data"
    prompts = []
    for c in CATS:
        for line in open(data_dir / f"BFCL_v4_{c}.json"):
            e = json.loads(line)
            if len(e["question"]) != 1:
                continue
            e = copy.deepcopy(e)
            inf = h._pre_query_processing_prompting(e)
            msgs = e["question"][0]
            prompts.append(h._format_prompt(msgs, inf["function"]))

    random.Random(a.seed).shuffle(prompts)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")
    chars = [len(p) for p in prompts]
    print(f"wrote {len(prompts)} prompts -> {a.out}; chars median={statistics.median(chars):.0f} "
          f"p95={sorted(chars)[int(0.95 * len(chars))]} (~4 chars/token)")


if __name__ == "__main__":
    main()
