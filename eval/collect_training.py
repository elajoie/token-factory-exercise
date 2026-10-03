#!/usr/bin/env python
"""Collect runs/*/run_config.json into results/training_summary.md (+ .csv)."""
import csv
import json
from pathlib import Path

rows = []
for f in sorted(Path("runs").glob("*/run_config.json")):
    c = json.loads(f.read_text())
    a = c["args"]
    rows.append({
        "run": f.parent.name, "method": a["method"], "base": a["model"],
        "r": a["r"] if a["method"] != "full" else "", "lr": c["lr"], "epochs": a["epochs"],
        "trainable_params_M": round(c["trainable_params"] / 1e6, 1),
        "train_loss": round(c["train_loss"], 4) if c.get("train_loss") else "",
        "best_eval_loss": round(c["best_eval_loss"], 4) if c.get("best_eval_loss") else "",
        "train_time_min": round(c["train_runtime_s"] / 60, 1) if c.get("train_runtime_s") else "",
        "peak_gpu_mem_GB": c["peak_gpu_mem_gb"],
    })
if rows:
    cols = list(rows[0])
    Path("results").mkdir(exist_ok=True)
    with open("results/training_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    Path("results/training_summary.md").write_text("# Training runs\n\n" + "\n".join(md) + "\n")
    print("\n".join(md))
else:
    print("no finished runs yet")
