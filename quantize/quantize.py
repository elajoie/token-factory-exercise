#!/usr/bin/env python
"""Post-training quantisation with llm-compressor (vLLM-native compressed-tensors format).

Schemes:
  fp8       FP8_DYNAMIC: FP8 E4M3 weights (per-channel scales) + FP8 activations (per-token,
            computed at runtime). No calibration data needed. Native on H100 tensor cores.
            Expected to be ~lossless -> our production default ("accuracy first").
  w4a16     INT4 weights / 16-bit activations via AWQ, calibrated on ToolACE prompts.
            Smallest & cheapest, but riskier for exact argument values -> must pass BFCL.

    python quantize/quantize.py --model models/lora_r16-merged --scheme fp8   --out models/lora_r16-fp8
    python quantize/quantize.py --model models/lora_r16-merged --scheme w4a16 --out models/lora_r16-w4a16
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.prompt_format import render_segments  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--scheme", choices=["fp8", "w4a16"], required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--calib-file", default="data/processed/train.jsonl")
ap.add_argument("--calib-samples", type=int, default=256)
ap.add_argument("--calib-max-len", type=int, default=4096)
a = ap.parse_args()

from llmcompressor import oneshot  # noqa: E402

model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="cuda")
tok = AutoTokenizer.from_pretrained(a.model)

if a.scheme == "fp8":
    from llmcompressor.modifiers.quantization import QuantizationModifier

    recipe = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=["lm_head"])
    oneshot(model=model, recipe=recipe)
else:
    from datasets import Dataset
    from llmcompressor.modifiers.transform import AWQModifier

    rows = [json.loads(l) for l in open(a.calib_file)]
    random.Random(0).shuffle(rows)
    texts = ["".join(s for s, _ in render_segments(r["system"], r["turns"])) for r in rows[: a.calib_samples]]
    ds = Dataset.from_dict({"text": texts}).map(
        lambda x: tok(x["text"], add_special_tokens=False, truncation=True, max_length=a.calib_max_len),
        remove_columns=["text"],
    )
    from llmcompressor.modifiers.quantization import QuantizationModifier
    # llm-compressor >= 0.14: AWQ only rescales; a QuantizationModifier must follow it
    recipe = [
        AWQModifier(duo_scaling="both"),
        QuantizationModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"]),
    ]
    oneshot(model=model, dataset=ds, recipe=recipe,
            max_seq_length=a.calib_max_len, num_calibration_samples=len(texts))

model.save_pretrained(a.out, save_compressed=True)
tok.save_pretrained(a.out)
print(f"quantised ({a.scheme}) model written to {a.out}")
