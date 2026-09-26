# Runbook: ToolACE → Qwen3-8B function-calling fine-tune on 1× H100

Every command, in order, to go from an empty Nebius VM to the deliverables the assignment asks for: training and eval scripts, BFCL results, latency and throughput metrics, and a production-style deployment.

The versions pinned here were current in September 2026: vLLM 0.30.0, TRL 1.14.0, PEFT 0.21.0, transformers 5.17.0, llm-compressor 0.14.0 and bfcl-eval 2026.3.23. `env/setup_vm.sh` freezes the exact environment into `env/lock/`.

---

## 0. The pipeline at a glance

```
ToolACE (HF) ──► data/prepare_toolace.py ──► contamination check ──► train/train_sft.py  (LoRA r16 / r64 / QLoRA / DoRA / full)
                                                                          │
Qwen3-8B ──► BFCL baseline                                   train/merge_lora.py
                                                                          │
                              eval/run_bfcl.sh (vLLM server + BFCL "python") ◄── every variant
                                                                          │
                                                   quantize/quantize.py (FP8, W4A16) ──► BFCL again
                                                                          │
                                                   bench/run_bench.sh (1…48 concurrent) ──► TTFT / TPOT / E2E / throughput
```

**Design decision that drives everything: one prompt format everywhere.** ToolACE's output format (`[func(a=1)]`) is exactly BFCL's *prompting mode*. `common/prompt_format.py` renders training data byte-for-byte like BFCL's Qwen3 handler. The production chat template is generated from BFCL's own prompt code. `tests/test_prompt_parity.py` proves both match on all 3,491 BFCL python test prompts: 0 mismatches. vLLM's `pythonic` tool parser then turns the model's output into standard OpenAI `tool_calls` for clients.

**Time budget (rough, H100 SXM):** setup ~30 min · data ~10 min · each BFCL run ~15–40 min (the base model is slower because it "thinks") · each LoRA training run ~1.5–3 h for 2 epochs · full FT ~3–5 h · quantization ~10–40 min · each benchmark sweep ~20–30 min. Run everything long in **tmux**.

---

## 1. Create the VM (Nebius console)

1. Accept the invite and log in at console.nebius.com.
2. Go to **Compute → Virtual machines → Create**.
3. Set these values **strictly**:
   - Project `default-project` (eu-north1)
   - **With GPU**, VM type **Regular**
   - Platform **NVIDIA H100 NVLink with Intel Sapphire Rapids (gpu-h100-sxm)**
   - Preset **1 GPU – 16 CPUs – 200 GiB RAM**
4. In the **Access** tab, add your username and SSH public key. If you don't have a key yet, run `ssh-keygen -t ed25519 -C "you@example.com"` on your laptop and paste in `~/.ssh/id_ed25519.pub`.
5. Leave everything else at the defaults.
6. Name it `Firstname_Lastname_dd_2026_mm_dd` (your demo date) and click **Create VM**.
7. Start the VM and copy its public IP.

From your laptop:
```bash
ssh <username>@<VM_PUBLIC_IP>
```

## 2. Get the code onto the VM

**Option A (recommended): your private GitHub repo.** Graders need GitHub access anyway.
```bash
# on your laptop, inside the unzipped toolace-fc folder
git init && git add . && git commit -m "ToolACE function-calling pipeline"
gh repo create toolace-fc --private --source . --push
# give the graders read access (they must accept the invite)
for u in glebberjoskin maxgreat hazirliver Alex-Hanley Farrukhmustafa aktsvigun nikitml; do
  gh api -X PUT "repos/<your-gh-user>/toolace-fc/collaborators/$u" -f permission=pull
done
```
Then on the VM:
```bash
git clone https://github.com/<your-gh-user>/toolace-fc.git && cd toolace-fc
```

**Option B: copy directly.**
```bash
scp -r toolace-fc <username>@<VM_PUBLIC_IP>:~/
```

## 3. Bootstrap the environment (~30 min)

```bash
tmux new -s work             # detach: Ctrl-b d  | re-attach: tmux attach -t work
cd ~/toolace-fc
./env/setup_vm.sh 2>&1 | tee logs_setup.txt
```

This checks that `nvidia-smi` shows an **H100 80GB** and reports free disk space. It then installs `uv` and builds four venvs: `.venv-train`, `.venv-quant`, `.venv-serve` and `.venv-eval`. They are kept separate because vLLM, llm-compressor and BFCL each pin conflicting versions of torch, transformers or numpy. Finally it writes `env/lock/*.txt`. Commit those files: that's your reproducibility proof.

At the end you should see `train ok … True`, `quant ok`, `serve ok` and `eval ok`.

**Disk.** The whole plan needs roughly 150 GB. If `df -h ~` shows less:
- Train fewer variants.
- Delete `runs/*/checkpoint-*` after each run finishes (`final/` is all you need).
- Delete merged bf16 copies of the losing variants once they're evaluated.
- Set `HF_HOME` to the largest mount.

Optional: `hf auth login` avoids HF rate limits. Both Qwen3-8B and ToolACE are public.

## 4. Download the base model (~5 min)

```bash
./scripts/01_download_base.sh
```
This puts Qwen3-8B in `models/Qwen3-8B` and prints the model and dataset commit SHAs. Put them in your README. For fully pinned re-runs, use `BASE_REV=<sha> ./scripts/01_download_base.sh`.

**Why Qwen3-8B:**
- Apache-2.0 license.
- Strong native tool use.
- Supported by Nebius Token Factory for LoRA and full fine-tuning and for hosting.
- Its 8B size trains on one H100 and serves 16–32 concurrent users at low latency.

## 5. Prepare the data and prove format parity (~10 min)

```bash
./scripts/02_prepare_data.sh
```

What it does:

| Step | Output | What to check |
|---|---|---|
| `data/prepare_toolace.py` | `data/processed/{train,val}.jsonl`, `dataset_report.json` | ~11.3k dialogs, 95/5 split **by dialog** (no leakage), duplicates dropped, % of answers that are function calls vs. text (irrelevance/clarification) |
| `eval/contamination_check.py` | `contaminated_ids.txt`, `contamination_report.json` | ToolACE dialogs overlapping BFCL test questions (exact or ≥50% 8-gram); these are **excluded from training** automatically |
| `serve/build_chat_template.py` | `serve/chat_template.jinja` | production template built from BFCL's own prompt code |
| `tests/test_prompt_parity.py` | console | must print `training-format mismatches=0, chat-template mismatches=0` |
| `bench/make_bench_prompts.py` | `bench/prompts_bfcl.jsonl` | 3,491 real prompts, median ~3.5k characters (~900 tokens), p95 ~11k characters |

## 6. Baseline BFCL of the untouched model (~30–60 min)

```bash
./scripts/03_baseline_eval.sh
cat results/bfcl_summary.md
```
`eval/run_bfcl.sh` does four things:

1. Starts **the production vLLM config** (`serve/serve.sh`) in the background.
2. Waits for `/health`.
3. Runs:
   ```bash
   bfcl generate --model Qwen/Qwen3-8B --test-category python --skip-server-setup --local-model-path <dir>
   bfcl evaluate --model Qwen/Qwen3-8B --test-category python
   ```
   with `BFCL_PROJECT_ROOT=eval/bfcl_runs/<variant>`, so each variant's results are kept separate.
4. Stops the server and updates `results/bfcl_summary.md`.

The BFCL **"python" collection** has 11 categories: `simple_python`, `multiple`, `parallel`, `parallel_multiple`, `irrelevance`, `live_simple`, `live_multiple`, `live_parallel`, `live_parallel_multiple`, `live_irrelevance` and `live_relevance`. That's about 3.5k test cases, all scored by AST matching.

Logs go to `logs/serve_<variant>.log`, `logs/bfcl_generate_<variant>.log` and `logs/bfcl_evaluate_<variant>.log`.

## 7. Fine-tuning experiments (the long part, ~10–15 h total)

```bash
./scripts/04_train_all.sh
# watch it:
tail -f logs/train_lora_r16.log
nvidia-smi -l 5          # or: nvtop
.venv-train/bin/tensorboard --logdir runs --port 6006   # from your laptop: ssh -L 6006:localhost:6006 <user>@<ip>
```

The script runs a **5-minute smoke test first** (200 dialogs) to catch out-of-memory errors and format bugs early. Then it runs the grid, most informative first, so you can stop when time runs out:

| Run | Method | Why it's in the grid |
|---|---|---|
| `lora_r16` | LoRA r=16, α=32, all linear layers, lr 2e-4 | Default production recipe; adapter hosted natively on Token Factory |
| `lora_r64` | LoRA r=64, lr 1e-4 | Does more adapter capacity help? |
| `qlora_r16` | 4-bit NF4 base + LoRA | Memory/quality trade-off (expect slower and slightly worse; shows why it isn't needed on an H100) |
| `dora_r16` | DoRA | Weight-decomposed LoRA; sometimes +accuracy at small cost |
| `full` | Full fine-tuning, lr 1e-5, paged 8-bit AdamW, Liger kernels | Quality ceiling. 16 GB weights + 16 GB grads + ~16 GB optimizer state fits in 80 GB with gradient checkpointing |

Best practices built into `train/train_sft.py`, each worth a line on your slide:

- The prompt is identical to evaluation, and the **loss is computed only on assistant tokens**, across every assistant turn of a multi-turn dialog.
- Over-long dialogs (over 8,192 tokens) are **dropped, not truncated**. Truncation could cut the answer, which is the part being learned.
- bf16 · gradient checkpointing · FlashAttention-2 (from the HF kernels hub) · **padding-free BFD packing** (roughly 2× throughput).
- Cosine schedule, 3% warmup, grad-norm clip 1.0, 2 epochs, about 64k tokens per optimizer step.
- Validation every 50 steps, **best checkpoint by eval loss** restored at the end.
- Fixed seeds. `run_config.json` records every hyper-parameter, token count, runtime and peak GPU memory.

To run a single experiment by hand:
```bash
source .venv-train/bin/activate
python train/train_sft.py --model models/Qwen3-8B --method lora --r 16 --output-dir runs/lora_r16
```

**If a run hits out-of-memory:** add `--batch-size 1 --grad-accum 8`. For full FT, also try `--max-length 4096`. If the FlashAttention kernel fails to load, use `--attn sdpa --no-packing`, which is slower but always works.

## 8. Merge adapters and evaluate every variant (~30 min per variant)

```bash
./scripts/05_merge_and_eval.sh            # or: ./scripts/05_merge_and_eval.sh lora_r16 dora_r16
.venv-train/bin/python eval/collect_training.py
cat results/training_summary.md results/bfcl_summary.md
```
Each adapter is merged into the bf16 base (`train/merge_lora.py`), giving one plain model with zero adapter overhead. That merged model then goes through the same live-server BFCL run.

**Pick the winner by `python_overall`.** Also look at `hallucination_avoidance` (the irrelevance categories), because an agent that calls a tool when it shouldn't is dangerous in finance.

## 9. Optimize: quantize the winner and check accuracy again (~1.5 h)

```bash
./scripts/06_quantize_and_eval.sh models/lora_r16-merged lora_r16     # substitute your winner
cat results/bfcl_summary.md results/model_sizes.txt
```

| Variant | What it is | Expectation |
|---|---|---|
| `*_fp8` | FP8 E4M3 weights + dynamic per-token FP8 activations (llm-compressor `FP8_DYNAMIC`, no calibration needed) | Half the memory, faster; roughly lossless. **Default production pick** if BFCL is within ~0.5 pt of bf16 |
| `*_fp8_kvfp8` | + FP8 KV cache | More room for concurrent requests; check the BFCL cost |
| `*_w4a16` | INT4 AWQ weights, calibrated on 256 ToolACE dialogs | Smallest and cheapest; accept only if BFCL holds up (exact argument values are fragile) |

Decision rule, following the client's priority order (accuracy > latency > cost): **choose the fastest variant whose BFCL score is statistically indistinguishable from bf16.** With about 3.5k samples, a ±1 pt difference is within noise.

## 10. Deploy like production and smoke-test the API

```bash
MODEL_PATH=models/lora_r16-fp8 ./serve/serve.sh      # foreground; Ctrl-C to stop
# in a second tmux pane:
.venv-serve/bin/python serve/smoke_test.py --model toolace-qwen3-8b
curl -s localhost:8000/metrics | grep -E "vllm:(num_requests_running|gpu_cache_usage_perc|time_to_first_token)" | head
```

The server runs vLLM with these settings:

- OpenAI-compatible `/v1/chat/completions` with `tools`, the same contract as Token Factory.
- Continuous batching and `max-num-seqs 64` (headroom above 32 concurrent).
- **Prefix caching**: the tool schemas repeat across requests, so their processing is reused (a large TTFT win).
- Chunked prefill.
- The `pythonic` tool parser and the generated chat template.
- `/health` and Prometheus `/metrics` endpoints.

The smoke test checks a multi-call request (flag the transaction and freeze the card), a single call (FX rate) and an **irrelevant** request (a joke). The irrelevant one must return text, not a tool call.

To build a container for the "as for production" story:
```bash
docker build -f serve/Dockerfile -t toolace-qwen3-8b:fp8 --build-arg MODEL_DIR=models/lora_r16-fp8 .
docker run --gpus all -p 8000:8000 toolace-qwen3-8b:fp8
```

**Mapping to Nebius Token Factory:** Token Factory serves Qwen3-8B behind the same OpenAI-compatible API and supports LoRA and full fine-tuning of it. The production path is either to import your adapter or weights, if your account supports custom-model upload (check the current docs), or to re-run the same recipe with Token Factory's own fine-tuning on `data/processed/train.jsonl` and deploy it to a dedicated endpoint. Clients and `serve/smoke_test.py` work unchanged apart from `--base-url`.

## 11. Benchmark TTFT, latency and throughput (~2–3 h for all variants)

```bash
./scripts/07_benchmark.sh lora_r16
cat results/bench_summary.md          # table + results/bench_summary.png
```

For each variant, `bench/run_bench.sh`:

1. Starts the production server.
2. Warms it up.
3. Sweeps **1, 8, 16, 24, 32 and 48 concurrent clients** with `vllm bench serve`. It uses the **real BFCL prompts** (`--dataset-name custom --skip-chat-template`), because tool-schema-heavy inputs make TTFT the dominant cost.

Two workloads are run:

- **natural**: the model stops on its own, as with real short function calls. Use this for E2E latency.
- **fixed128**: exactly 128 output tokens (`--ignore-eos`). Use this for stable TPOT and throughput comparisons.

The variants compared are bf16, FP8, FP8 with FP8 KV cache, FP8 **without** prefix caching (to prove its value) and W4A16.

**Metrics to report** (focus on the 16–32 band):

- TTFT p50/p95/p99
- TPOT/ITL p50/p95
- E2E latency p50/p95/p99
- Request/s and output tokens/s
- Failed requests (must be 0)
- GPU memory (`gpu_mem.csv`)

Derive **cost per 1M output tokens** as `(VM $/h) / (output tok/s × 3600) × 1e6` at 32 concurrent users.

## 12. Collect the results and hand in

```bash
./scripts/08_report.sh
git add -A && git commit -m "Results: training, BFCL, benchmarks" && git push
```

Hand in **at least 3 days before the demo day**:

- Reply to the interview confirmation email with the repo link and the numbers.
- Confirm all 7 graders have access.

Things to commit:

- `results/training_summary.md` and `results/bfcl_summary.md`
- `results/bench_summary.md` and `results/bench_summary.png`
- `data/processed/dataset_report.json` and `data/processed/contamination_report.json`
- `env/lock/*.txt`
- The `logs/` you want to show (the `.gitignore` excludes them by default, so add them explicitly with `git add -f logs/...`)

The assignment says "you may share the terminal commands you used". This runbook is that list.

To stop paying for the GPU, stop the VM in the console when you're idle. The disk persists.

---

## Repo map

```
common/prompt_format.py       single source of truth for the prompt (== BFCL Qwen3 prompting mode)
data/prepare_toolace.py       ToolACE → normalised dialogs, dedupe, dialog-level split, report
eval/contamination_check.py   BFCL ↔ ToolACE overlap → excluded ids
train/train_sft.py            LoRA / QLoRA / DoRA / full FT (TRL SFTTrainer, assistant-only loss)
train/merge_lora.py           adapter → merged bf16 checkpoint
quantize/quantize.py          FP8_DYNAMIC and AWQ W4A16 via llm-compressor
serve/build_chat_template.py  production chat template generated from BFCL's prompt code
serve/serve.sh | Dockerfile   production vLLM config (pythonic tool parser, prefix caching, metrics)
serve/smoke_test.py           OpenAI-client end-to-end tool-call check (incl. irrelevance)
eval/run_bfcl.sh              live-server BFCL "python" run for one variant
eval/collect_*.py             result tables
bench/*.py|sh                 realistic prompt set, concurrency sweep, summary + charts
tests/test_prompt_parity.py   proves train == eval == serve prompt formatting
scripts/01…08_*.sh            the pipeline, in order (also: make setup/data/baseline/train/eval/quantize/bench/report)
```

## Troubleshooting

| Symptom | Fix |
|---|---|
| `bfcl generate` gets 404 "model not found" | The server must be started with the **same absolute path** BFCL sends. `run_bfcl.sh` handles this via `realpath` and the second `--served-model-name`. |
| Base model BFCL run is slow | Expected: Qwen3 thinks before answering. Fine-tuned models answer directly. |
| Out-of-memory during training | `--batch-size 1 --grad-accum 8`; full FT: `--max-length 4096`; last resort: train `Qwen/Qwen3-4B` |
| `kernels-community/flash-attn2` fails to load | `--attn sdpa --no-packing` |
| vLLM refuses FP8 KV cache | Drop `KV_FP8=1` and report bf16 KV |
| Parity test fails after upgrading bfcl-eval | BFCL changed its prompt: re-run `serve/build_chat_template.py`, and re-check the training data format before training |
| Disk full | `rm -rf runs/*/checkpoint-*`, delete merged copies of losing variants, `uv cache prune` |

## Candidates for "the biggest challenge" (pick the real one)

1. **Prompt-format parity** across training, BFCL and serving, and getting BFCL to evaluate a custom checkpoint. Solved by rendering from one module and generating the serving template from BFCL's own code, with a test.
2. **Fitting full fine-tuning of an 8B model into 80 GB.** Solved with 8-bit paged AdamW, gradient checkpointing and Liger kernels.
3. **Proving quantization didn't cost accuracy** on exact argument values, not just on loss.
