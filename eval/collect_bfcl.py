#!/usr/bin/env python
"""Collect BFCL per-category scores of every evaluated variant into one table.

Reads eval/bfcl_runs/<variant>/score/**/BFCL_v4_<category>_score.json (first line holds
{"accuracy", "correct_count", "total_count"}) and writes Markdown + CSV.

Aggregates (clearly labelled, BFCL's own CSVs are also in each run's score/ folder):
  non_live_ast   = unweighted mean of simple_python, multiple, parallel, parallel_multiple
  live_ast       = sample-weighted mean of live_simple/multiple/parallel/parallel_multiple
  hallucination  = irrelevance + live_irrelevance (correct = no call when none fits)
  python_overall = micro average over all python categories
"""
import argparse
import csv
import json
from pathlib import Path

CATS = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
        "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
        "live_irrelevance", "live_relevance"]


def read_scores(run_dir: Path):
    out = {}
    for f in run_dir.glob("score/**/BFCL_v4_*_score.json"):
        cat = f.name[len("BFCL_v4_"):-len("_score.json")]
        head = json.loads(f.open().readline())
        out[cat] = (head["correct_count"], head["total_count"])
    return out


def pct(c, t):
    return 100.0 * c / t if t else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="eval/bfcl_runs")
    ap.add_argument("--out", default="results/bfcl_summary.md")
    a = ap.parse_args()

    rows = []
    for run in sorted(p for p in Path(a.runs_dir).iterdir() if p.is_dir()):
        s = read_scores(run)
        if not s:
            continue
        r = {"variant": run.name}
        for c in CATS:
            r[c] = round(pct(*s[c]), 2) if c in s else None
        nl = [pct(*s[c]) for c in CATS[:4] if c in s]
        r["non_live_ast"] = round(sum(nl) / len(nl), 2) if nl else None
        lv = [s[c] for c in CATS[5:9] if c in s]
        r["live_ast"] = round(pct(sum(c for c, _ in lv), sum(t for _, t in lv)), 2) if lv else None
        hz = [s[c] for c in ("irrelevance", "live_irrelevance") if c in s]
        r["hallucination_avoidance"] = round(pct(sum(c for c, _ in hz), sum(t for _, t in hz)), 2) if hz else None
        r["python_overall"] = round(pct(sum(c for c, _ in s.values()), sum(t for _, t in s.values())), 2)
        r["n"] = sum(t for _, t in s.values())
        rows.append(r)

    if not rows:
        print("no scores found yet")
        return
    cols = ["variant", "python_overall", "non_live_ast", "live_ast", "hallucination_avoidance"] + CATS + ["n"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out.with_suffix(".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append("| " + " | ".join("" if r.get(c) is None else str(r[c]) for c in cols) + " |")
    out.write_text("# BFCL v4 - python collection (accuracy %)\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
