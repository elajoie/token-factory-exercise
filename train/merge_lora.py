#!/usr/bin/env python
"""Merge a LoRA/DoRA/QLoRA adapter into the bf16 base model -> plain HF checkpoint.

A merged model has zero adapter overhead at inference and is the input for FP8 quantisation.
For QLoRA we merge into the *bf16* base (not the 4-bit one) - standard practice.

    python train/merge_lora.py --adapter runs/lora_r16/final --out models/lora_r16-merged
"""
import argparse

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="Qwen/Qwen3-8B")
ap.add_argument("--adapter", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()

base = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="cuda")
model = PeftModel.from_pretrained(base, a.adapter).merge_and_unload()
model.save_pretrained(a.out, safe_serialization=True, max_shard_size="5GB")
AutoTokenizer.from_pretrained(a.adapter).save_pretrained(a.out)
print(f"merged model written to {a.out}")
