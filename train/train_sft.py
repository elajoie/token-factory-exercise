#!/usr/bin/env python
"""Supervised fine-tuning of Qwen3 on ToolACE with LoRA / QLoRA / DoRA / full FT.

Key best practices baked in:
  * prompt format identical to BFCL eval (common/prompt_format.py)
  * loss only on assistant tokens (every assistant turn of a multi-turn dialog)
  * over-long samples are DROPPED, never truncated (truncation can cut the answer)
  * bf16, gradient checkpointing, FlashAttention-2 + padding-free packing
  * cosine LR schedule with warmup, eval on held-out split, best checkpoint kept
  * fixed seeds, all hyper-parameters written to <output_dir>/run_config.json

Examples:
  python train/train_sft.py --method lora  --r 16 --output-dir runs/lora_r16
  python train/train_sft.py --method qlora --r 16 --output-dir runs/qlora_r16
  python train/train_sft.py --method dora  --r 16 --output-dir runs/dora_r16
  python train/train_sft.py --method full  --lr 1e-5 --optim paged_adamw_8bit --output-dir runs/full
"""
import argparse
import json
import os
import sys
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.prompt_format import render_segments  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-8B")
    ap.add_argument("--method", choices=["lora", "qlora", "dora", "full"], required=True)
    ap.add_argument("--train-file", default="data/processed/train.jsonl")
    ap.add_argument("--val-file", default="data/processed/val.jsonl")
    ap.add_argument("--exclude-ids", default="data/processed/contaminated_ids.txt",
                    help="dialog ids flagged by eval/contamination_check.py (optional)")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=None, help="default: 2*r")
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=None, help="default: 2e-4 (adapters) / 1e-5 (full)")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-length", type=int, default=8192)
    ap.add_argument("--batch-size", type=int, default=2, help="per-device micro batch (packed 8k-token rows)")
    ap.add_argument("--grad-accum", type=int, default=4, help="=> ~64k tokens per optimizer step")
    ap.add_argument("--warmup", type=float, default=0.03, help="ratio of total steps")
    ap.add_argument("--weight-decay", type=float, default=0.0)
    ap.add_argument("--optim", default=None, help="default: adamw_torch_fused (adapters) / paged_adamw_8bit (full)")
    ap.add_argument("--attn", default="kernels-community/flash-attn2",
                    help="'kernels-community/flash-attn2', 'flash_attention_2' or 'sdpa'")
    ap.add_argument("--no-packing", action="store_true")
    ap.add_argument("--liger", action="store_true", help="Liger kernels (saves memory for full FT)")
    ap.add_argument("--eval-steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--report-to", default="tensorboard", help="tensorboard | wandb | mlflow | none")
    ap.add_argument("--max-train-samples", type=int, default=None, help="for smoke tests")
    return ap.parse_args()


def load_jsonl(path, exclude=frozenset(), limit=None):
    rows = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            if d["id"] not in exclude:
                rows.append(d)
    return rows[:limit] if limit else rows


def tokenize_dialogs(rows, tok, max_length):
    """Tokenise segment by segment so the assistant mask is exact."""
    out, dropped = {"input_ids": [], "assistant_masks": []}, 0
    for d in rows:
        ids, mask = [], []
        for text, trainable in render_segments(d["system"], d["turns"]):
            t = tok(text, add_special_tokens=False)["input_ids"]
            ids += t
            mask += [1 if trainable else 0] * len(t)
        if len(ids) > max_length:
            dropped += 1
            continue
        out["input_ids"].append(ids)
        out["assistant_masks"].append(mask)
    return Dataset.from_dict(out), dropped


def main():
    a = parse_args()
    set_seed(a.seed)
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTConfig, SFTTrainer

    is_adapter = a.method != "full"
    lr = a.lr or (2e-4 if is_adapter else 1e-5)
    optim = a.optim or ("adamw_torch_fused" if is_adapter else "paged_adamw_8bit")
    alpha = a.alpha or 2 * a.r

    tok = AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    exclude = set()
    if a.exclude_ids and os.path.exists(a.exclude_ids):
        exclude = set(Path(a.exclude_ids).read_text().split())
        print(f"excluding {len(exclude)} contaminated dialogs")
    train_rows = load_jsonl(a.train_file, exclude, a.max_train_samples)
    val_rows = load_jsonl(a.val_file, exclude)
    train_ds, drop_tr = tokenize_dialogs(train_rows, tok, a.max_length)
    val_ds, drop_va = tokenize_dialogs(val_rows, tok, a.max_length)
    n_tok = sum(len(x) for x in train_ds["input_ids"])
    n_sup = sum(sum(m) for m in train_ds["assistant_masks"])
    print(f"train dialogs={len(train_ds)} (dropped {drop_tr} > {a.max_length} tok), val={len(val_ds)} "
          f"(dropped {drop_va}); train tokens={n_tok:,}, supervised tokens={n_sup:,}")

    quant = None
    if a.method == "qlora":
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        a.model, dtype=torch.bfloat16, attn_implementation=a.attn, quantization_config=quant,
    )
    model.config.use_cache = False
    if a.method == "qlora":
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    peft_config = None
    if is_adapter:
        peft_config = LoraConfig(
            r=a.r, lora_alpha=alpha, lora_dropout=a.dropout, bias="none", task_type="CAUSAL_LM",
            target_modules="all-linear",           # attention + MLP projections
            use_dora=(a.method == "dora"),
        )

    cfg = SFTConfig(
        output_dir=a.output_dir,
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch_size,
        per_device_eval_batch_size=a.batch_size,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_steps=a.warmup,                 # float < 1 == ratio (transformers >= 5)
        weight_decay=a.weight_decay,
        max_grad_norm=1.0,
        optim=optim,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=a.max_length,
        packing=not a.no_packing,              # BFD packing => padding-free with FlashAttention
        use_liger_kernel=a.liger,
        eval_strategy="steps",
        eval_steps=a.eval_steps,
        save_strategy="steps",
        save_steps=a.eval_steps,
        save_total_limit=2,                    # disk is limited on the VM
        save_only_model=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=10,
        report_to=None if a.report_to == "none" else [a.report_to],
        seed=a.seed,
        data_seed=a.seed,
        dataloader_num_workers=4,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=train_ds, eval_dataset=val_ds,
                         processing_class=tok, peft_config=peft_config)
    if is_adapter:
        trainer.model.print_trainable_parameters()

    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    metrics = trainer.evaluate()

    final_dir = Path(a.output_dir) / "final"
    trainer.save_model(str(final_dir))       # adapter only for LoRA-family, full weights for full FT
    tok.save_pretrained(final_dir)

    summary = {
        "args": vars(a), "lr": lr, "optim": optim, "lora_alpha": alpha if is_adapter else None,
        "train_dialogs": len(train_ds), "val_dialogs": len(val_ds),
        "dropped_overlength": {"train": drop_tr, "val": drop_va},
        "train_tokens": n_tok, "supervised_tokens": n_sup,
        "train_runtime_s": result.metrics.get("train_runtime"),
        "train_loss": result.metrics.get("train_loss"),
        "best_eval_loss": metrics.get("eval_loss"),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
        "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
    }
    (Path(a.output_dir) / "run_config.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
