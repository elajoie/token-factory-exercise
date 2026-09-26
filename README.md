# ToolACE function-calling fine-tune (Qwen3-8B, 1× H100)

Take-home solution: fine-tune → evaluate on BFCL (python) → optimize (FP8 / INT4) → deploy with vLLM → benchmark at 16–32 concurrent requests.

**Start here: [RUNBOOK.md](RUNBOOK.md)**, which has every command in order.

Quick path: `make setup download data baseline train eval quantize bench report` (set `WINNER=<run>` for quantize/bench).

Results land in `results/`: training_summary, bfcl_summary and bench_summary (with a .png chart).
