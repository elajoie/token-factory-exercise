#!/usr/bin/env python
"""Summarise `vllm bench serve` JSON results into a Markdown table + PNG charts.

    python bench/summarize_bench.py --results-dir results/bench --out results/bench_summary.md
"""
import argparse
import json
import re
from pathlib import Path

COLS = [("request_throughput", "req/s"), ("output_throughput", "out tok/s"),
        ("p50_ttft_ms", "TTFT p50"), ("p95_ttft_ms", "TTFT p95"), ("p99_ttft_ms", "TTFT p99"),
        ("p50_tpot_ms", "TPOT p50"), ("p95_tpot_ms", "TPOT p95"),
        ("p50_e2el_ms", "E2E p50"), ("p95_e2el_ms", "E2E p95"), ("p99_e2el_ms", "E2E p99"),
        ("failed", "failed")]
# categorical slots in fixed order (reference palette, light mode)
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def load(results_dir):
    rows = []
    for f in sorted(Path(results_dir).glob("*/*.json")):
        m = re.match(r"(natural|fixed128)_c(\d+)\.json", f.name)
        if not m:
            continue
        r = json.loads(f.read_text())
        rows.append({"variant": f.parent.name, "workload": m[1], "concurrency": int(m[2]), **r})
    return sorted(rows, key=lambda r: (r["workload"], r["variant"], r["concurrency"]))


def fmt(v):
    return "" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))


def charts(rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nat = [r for r in rows if r["workload"] == "natural"]
    variants = sorted({r["variant"] for r in nat})
    panels = [("p95_ttft_ms", "p95 time to first token (ms)"),
              ("p95_e2el_ms", "p95 end-to-end latency (ms)"),
              ("output_throughput", "Output throughput (tokens/s)")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, panels):
        ax.set_facecolor(SURFACE)
        for i, v in enumerate(variants):
            pts = [(r["concurrency"], r.get(key)) for r in nat if r["variant"] == v and r.get(key) is not None]
            if not pts:
                continue
            x, y = zip(*pts)
            ax.plot(x, y, color=SERIES[i % len(SERIES)], lw=2, marker="o", ms=5, label=v,
                    markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axvspan(16, 32, color=GRID, alpha=0.5, lw=0)
        ax.text(24, ax.get_ylim()[1] * 0.97, "target 16-32", ha="center", va="top", color=INK2, fontsize=8)
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.set_xlabel("concurrent requests", color=INK2, fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=9, labelcolor=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out_png, dpi=150, facecolor=SURFACE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/bench")
    ap.add_argument("--out", default="results/bench_summary.md")
    a = ap.parse_args()
    rows = load(a.results_dir)
    if not rows:
        print("no results yet")
        return
    head = ["variant", "workload", "conc."] + [c[1] for c in COLS]
    md = ["# Serving benchmark (latencies in ms)", "",
          "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        md.append("| " + " | ".join([r["variant"], r["workload"], str(r["concurrency"])] +
                                    [fmt(r.get(k)) for k, _ in COLS]) + " |")
    png = Path(a.out).with_suffix(".png")
    try:
        charts(rows, png)
        md += ["", f"![latency and throughput vs concurrency]({png.name})"]
    except ImportError:
        pass
    Path(a.out).write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
