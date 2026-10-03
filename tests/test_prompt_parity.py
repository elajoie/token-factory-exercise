#!/usr/bin/env python
"""Guard against train/eval/serve prompt drift (run in the eval venv).

1. training formatter  == BFCL QwenHandler prompt   (for every python BFCL entry)
2. production chat template (tools=...) == BFCL prompt (what live traffic sees)
3. assistant tool-call history renders back to pythonic form

    python tests/test_prompt_parity.py
"""
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import bfcl_eval  # noqa: E402
from bfcl_eval.model_handler.local_inference.qwen import QwenHandler  # noqa: E402
from transformers.utils.chat_template_utils import _compile_jinja_template  # noqa: E402

from common.prompt_format import render_prompt  # noqa: E402

CATS = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
        "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
        "live_irrelevance", "live_relevance"]

h = QwenHandler("Qwen/Qwen3-8B", 0.001, "Qwen/Qwen3-8B", False)
tpl = _compile_jinja_template((ROOT / "serve/chat_template.jinja").read_text())
data_dir = Path(bfcl_eval.__file__).parent / "data"

n = fail_train = fail_tpl = 0
for c in CATS:
    for line in open(data_dir / f"BFCL_v4_{c}.json"):
        e = json.loads(line)
        if len(e["question"]) != 1:
            continue
        raw_msgs = copy.deepcopy(e["question"][0])
        e2 = copy.deepcopy(e)
        inf = h._pre_query_processing_prompting(e2)
        msgs = e2["question"][0]
        bfcl_prompt = h._format_prompt(msgs, inf["function"])

        # 1. training formatter
        system = msgs[0]["content"] if msgs[0]["role"] == "system" else None
        turns = [m for m in msgs if m["role"] != "system"]
        ours = render_prompt(system, turns)
        if ours != bfcl_prompt:
            fail_train += 1

        # 2. production chat template, tools passed the OpenAI way
        # (entries with an empty function list are skipped: with no tools, production
        #  intentionally sends no tool system prompt, as the OpenAI API semantics imply)
        tools = [{"type": "function", "function": f} for f in e["function"]]
        if not tools:
            n += 1
            continue
        served = tpl.render(messages=raw_msgs, tools=tools, add_generation_prompt=True)
        if served != bfcl_prompt:
            fail_tpl += 1
            if fail_tpl == 1:
                import difflib
                print("first template mismatch:", e["id"])
                print("".join(list(difflib.unified_diff(bfcl_prompt.splitlines(1), served.splitlines(1)))[:40]))
        n += 1

# 3. multi-turn history with tool calls + tool results
hist = [
    {"role": "user", "content": "freeze card 8812"},
    {"role": "assistant", "content": None, "tool_calls": [{"type": "function", "function": {
        "name": "freeze_card", "arguments": {"card_id": "8812", "reason": "lost"}}}]},
    {"role": "tool", "content": '{"status": "frozen"}'},
]
out = tpl.render(messages=hist, tools=[{"type": "function", "function": {"name": "freeze_card"}}],
                 add_generation_prompt=True)
assert '<|im_start|>assistant\n[freeze_card(card_id="8812", reason="lost")]<|im_end|>\n' in out, out
assert out.endswith('<|im_start|>user\n<tool_response>\n{"status": "frozen"}\n</tool_response><|im_end|>\n'
                    '<|im_start|>assistant\n'), out

print(f"checked {n} BFCL entries: training-format mismatches={fail_train}, chat-template mismatches={fail_tpl}")
sys.exit(1 if (fail_train or fail_tpl) else 0)
