#!/usr/bin/env python
"""Flag ToolACE dialogs whose user turns overlap BFCL python test questions.

Two checks: exact match after normalisation, and >=50% shared word 8-grams
(for questions with >= 8 words). Flagged ids are written to
data/processed/contaminated_ids.txt and excluded by train/train_sft.py.

Run in the eval venv (needs bfcl-eval for the test data):
    python eval/contamination_check.py
"""
import argparse
import json
import re
from pathlib import Path

import bfcl_eval

CATS = ["simple_python", "multiple", "parallel", "parallel_multiple", "irrelevance",
        "live_simple", "live_multiple", "live_parallel", "live_parallel_multiple",
        "live_irrelevance", "live_relevance"]
N = 8


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", s.lower())).strip()


def ngrams(words, n=N):
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def bfcl_questions():
    data_dir = Path(bfcl_eval.__file__).parent / "data"
    for c in CATS:
        for line in open(data_dir / f"BFCL_v4_{c}.json"):
            e = json.loads(line)
            for turn in e["question"]:
                for m in turn:
                    if m["role"] == "user" and isinstance(m["content"], str):
                        yield e["id"], norm(m["content"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/processed")
    a = ap.parse_args()

    exact, grams = {}, {}
    for qid, q in bfcl_questions():
        exact[q] = qid
        w = q.split()
        if len(w) >= N:
            grams[qid] = ngrams(w)
    index = {}
    for qid, g in grams.items():
        for x in g:
            index.setdefault(x, set()).add(qid)

    flagged, examples = set(), []
    for split in ("train", "val"):
        for line in open(Path(a.data_dir) / f"{split}.jsonl"):
            d = json.loads(line)
            for t in d["turns"]:
                if t["role"] != "user":
                    continue
                q = norm(t["content"])
                hit = exact.get(q)
                if not hit:
                    g = ngrams(q.split())
                    counts = {}
                    for x in g:
                        for qid in index.get(x, ()):
                            counts[qid] = counts.get(qid, 0) + 1
                    for qid, c in counts.items():
                        if c / len(grams[qid]) >= 0.5:
                            hit = qid
                            break
                if hit:
                    flagged.add(d["id"])
                    examples.append({"toolace_id": d["id"], "bfcl_id": hit, "text": t["content"][:160]})
                    break

    out = Path(a.data_dir) / "contaminated_ids.txt"
    out.write_text("\n".join(sorted(flagged)) + ("\n" if flagged else ""))
    (Path(a.data_dir) / "contamination_report.json").write_text(
        json.dumps({"flagged_dialogs": len(flagged), "examples": examples[:50]}, indent=2))
    print(f"flagged {len(flagged)} ToolACE dialogs overlapping BFCL python questions -> {out}")


if __name__ == "__main__":
    main()
