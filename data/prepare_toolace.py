#!/usr/bin/env python
"""Download Team-ACE/ToolACE, normalise it, dedupe, split train/val by DIALOG.

Output (JSONL, one dialog per line):
    {"id": str, "system": str, "turns": [{"role": "user|assistant|tool", "content": str}, ...]}

We keep ToolACE's native output format ([func(a=1)]), which is exactly what BFCL's
prompting mode expects. Tokenisation / loss masks happen in train/train_sft.py.

Usage:
    python data/prepare_toolace.py --out-dir data/processed --val-ratio 0.05 --seed 42
"""
import argparse
import hashlib
import json
import random
import statistics
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROLE_MAP = {"user": "user", "human": "user", "assistant": "assistant", "gpt": "assistant",
            "tool": "tool", "function": "tool", "observation": "tool"}


def normalise(row, idx, stats):
    turns = []
    for m in row["conversations"]:
        role = ROLE_MAP.get(str(m.get("from", "")).lower())
        if role is None:
            stats["unknown_role"] += 1
            return None
        content = m.get("value")
        if not isinstance(content, str) or not content.strip():
            stats["empty_turn"] += 1
            return None
        turns.append({"role": role, "content": content.strip()})
    if not turns or turns[0]["role"] != "user":
        stats["not_starting_with_user"] += 1
        return None
    if not any(t["role"] == "assistant" for t in turns):
        stats["no_assistant_turn"] += 1
        return None
    return {"id": f"toolace_{idx}", "system": row["system"], "turns": turns}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="Team-ACE/ToolACE")
    ap.add_argument("--revision", default=None, help="pin a dataset commit for reproducibility")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--val-ratio", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    ds = load_dataset(args.dataset, split="train", revision=args.revision)
    print(f"raw rows: {len(ds)}")

    stats, seen, dialogs = Counter(), set(), []
    for i, row in enumerate(ds):
        d = normalise(row, i, stats)
        if d is None:
            continue
        h = hashlib.sha256(json.dumps([d["system"], d["turns"]], sort_keys=True).encode()).hexdigest()
        if h in seen:
            stats["duplicate"] += 1
            continue
        seen.add(h)
        dialogs.append(d)

    random.Random(args.seed).shuffle(dialogs)
    n_val = max(1, int(len(dialogs) * args.val_ratio))
    val, train = dialogs[:n_val], dialogs[n_val:]

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, part in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w") as f:
            for d in part:
                f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ---- dataset report (goes into results/ and the presentation) ----
    first_answers = [next(t for t in d["turns"] if t["role"] == "assistant")["content"] for d in dialogs]
    n_calls = sum(a.startswith("[") and a.endswith("]") for a in first_answers)
    n_turns = [len(d["turns"]) for d in dialogs]
    report = {
        "raw_rows": len(ds),
        "kept_dialogs": len(dialogs),
        "train_dialogs": len(train),
        "val_dialogs": len(val),
        "dropped": dict(stats),
        "first_assistant_turn_is_function_call_pct": round(100 * n_calls / len(dialogs), 1),
        "first_assistant_turn_is_text_pct (irrelevance / clarification)": round(100 * (1 - n_calls / len(dialogs)), 1),
        "turns_per_dialog_mean": round(statistics.mean(n_turns), 2),
        "turns_per_dialog_max": max(n_turns),
        "system_prompt_chars_median": int(statistics.median(len(d["system"]) for d in dialogs)),
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
