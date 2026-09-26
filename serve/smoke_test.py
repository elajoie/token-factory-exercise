#!/usr/bin/env python
"""End-to-end check of the production API: OpenAI client -> vLLM -> pythonic parser -> tool_calls.

    python serve/smoke_test.py --base-url http://localhost:8000/v1 --model toolace-qwen3-8b
"""
import argparse
import json
import time

from openai import OpenAI

TOOLS = [
    {"type": "function", "function": {
        "name": "freeze_card",
        "description": "Freeze a payment card to block all new transactions.",
        "parameters": {"type": "object", "properties": {
            "card_id": {"type": "string", "description": "Card identifier"},
            "reason": {"type": "string", "enum": ["fraud_suspected", "lost", "stolen", "customer_request"]}},
            "required": ["card_id", "reason"]}}},
    {"type": "function", "function": {
        "name": "flag_transaction",
        "description": "Flag a transaction for manual fraud review.",
        "parameters": {"type": "object", "properties": {
            "transaction_id": {"type": "string"},
            "risk_score": {"type": "number", "description": "0-1 model risk score"}},
            "required": ["transaction_id"]}}},
    {"type": "function", "function": {
        "name": "get_exchange_rate",
        "description": "Get the FX rate between two ISO-4217 currencies.",
        "parameters": {"type": "object", "properties": {
            "base": {"type": "string"}, "quote": {"type": "string"}}, "required": ["base", "quote"]}}},
]

CASES = [
    "Flag transaction TX-4471 with risk score 0.93 and freeze card 8812 because we suspect fraud.",
    "What's the EUR to USD rate right now?",
    "Tell me a joke about accountants.",   # should NOT call a tool (irrelevance)
]

ap = argparse.ArgumentParser()
ap.add_argument("--base-url", default="http://localhost:8000/v1")
ap.add_argument("--model", default="toolace-qwen3-8b")
a = ap.parse_args()
client = OpenAI(base_url=a.base_url, api_key="EMPTY")

for q in CASES:
    t0 = time.perf_counter()
    r = client.chat.completions.create(model=a.model, messages=[{"role": "user", "content": q}],
                                       tools=TOOLS, tool_choice="auto", temperature=0)
    dt = (time.perf_counter() - t0) * 1000
    msg = r.choices[0].message
    calls = [{"name": c.function.name, "arguments": json.loads(c.function.arguments)} for c in (msg.tool_calls or [])]
    print(f"\nQ: {q}\n  latency: {dt:.0f} ms")
    print(f"  tool_calls: {json.dumps(calls)}" if calls else f"  text: {msg.content!r}")
