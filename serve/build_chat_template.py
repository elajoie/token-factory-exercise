#!/usr/bin/env python
"""Generate serve/chat_template.jinja for production (/v1/chat/completions + tools).

The system prompt text is produced by BFCL's own formulate_system_prompt(), so live
traffic gets the exact prompt the model was trained/evaluated with. The model answers
in pythonic form ([f(a=1)]) and vLLM's `pythonic` tool parser turns that into standard
OpenAI `tool_calls` for the client.

Run in the eval venv (needs bfcl-eval):
    python serve/build_chat_template.py --out serve/chat_template.jinja
"""
import argparse
import json

from bfcl_eval.constants.default_prompts import DEFAULT_SYSTEM_PROMPT_FORMAT
from bfcl_eval.model_handler.utils import formulate_system_prompt

SENTINEL = "@@FUNCTIONS@@"


def bfcl_prefix_suffix():
    # Build with a sentinel function list, then split around its JSON rendering.
    probe = [{"name": SENTINEL}]
    rendered = json.dumps(probe, indent=4)
    full = formulate_system_prompt(DEFAULT_SYSTEM_PROMPT_FORMAT, probe)
    assert full.count(rendered) == 1, "unexpected BFCL prompt layout"
    prefix, suffix = full.split(rendered)
    return prefix, suffix


TEMPLATE = r"""{#- Production chat template: Qwen3 ChatML + BFCL prompting-mode system prompt. -#}
{%- set ns = namespace(docs=[], sys=none, start=0) -%}
{%- if messages and messages[0]['role'] == 'system' -%}
  {%- set ns.sys = messages[0]['content'] -%}
  {%- set ns.start = 1 -%}
{%- endif -%}
{%- if tools -%}
  {%- for t in tools -%}
    {%- if t['function'] is defined -%}{%- set ns.docs = ns.docs + [t['function']] -%}{%- else -%}{%- set ns.docs = ns.docs + [t] -%}{%- endif -%}
  {%- endfor -%}
  {%- set bfcl = PREFIX_LITERAL + (ns.docs | tojson(indent=4, ensure_ascii=True)) + SUFFIX_LITERAL -%}
  {%- if ns.sys -%}{%- set ns.sys = bfcl + '\n\n' + ns.sys -%}{%- else -%}{%- set ns.sys = bfcl -%}{%- endif -%}
{%- endif -%}
{%- if ns.sys is not none -%}{{- '<|im_start|>system\n' + ns.sys + '<|im_end|>\n' -}}{%- endif -%}
{%- set msgs = messages[ns.start:] -%}
{%- for m in msgs -%}
  {%- if m['role'] == 'user' -%}
    {{- '<|im_start|>user\n' + m['content'] + '<|im_end|>\n' -}}
  {%- elif m['role'] == 'assistant' -%}
    {{- '<|im_start|>assistant\n' -}}
    {%- if m['tool_calls'] -%}
      {{- '[' -}}
      {%- for tc in m['tool_calls'] -%}
        {%- set f = tc['function'] if tc['function'] is defined else tc -%}
        {{- f['name'] + '(' -}}
        {%- if f['arguments'] is mapping -%}
          {%- for k, v in f['arguments'].items() -%}{{- k + '=' + (v | tojson) -}}{%- if not loop.last -%}{{- ', ' -}}{%- endif -%}{%- endfor -%}
        {%- else -%}{{- f['arguments'] -}}{%- endif -%}
        {{- ')' -}}{%- if not loop.last -%}{{- ', ' -}}{%- endif -%}
      {%- endfor -%}
      {{- ']' -}}
    {%- elif m['content'] -%}
      {{- m['content'] -}}
    {%- endif -%}
    {{- '<|im_end|>\n' -}}
  {%- elif m['role'] == 'tool' -%}
    {%- if loop.first or msgs[loop.index0 - 1]['role'] != 'tool' -%}{{- '<|im_start|>user' -}}{%- endif -%}
    {{- '\n<tool_response>\n' + m['content'] + '\n</tool_response>' -}}
    {%- if loop.last or msgs[loop.index0 + 1]['role'] != 'tool' -%}{{- '<|im_end|>\n' -}}{%- endif -%}
  {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}{{- '<|im_start|>assistant\n' -}}{%- endif -%}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="serve/chat_template.jinja")
    a = ap.parse_args()
    prefix, suffix = bfcl_prefix_suffix()
    tpl = TEMPLATE.replace("PREFIX_LITERAL", json.dumps(prefix)).replace("SUFFIX_LITERAL", json.dumps(suffix))
    with open(a.out, "w") as f:
        f.write(tpl)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
