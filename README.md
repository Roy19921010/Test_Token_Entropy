# Token-Level Uncertainty for LLM-Generated Test-Suite Quality

Open-source reference implementation accompanying the paper
*"Token-Level Uncertainty for LLM-Generated Test-Suite Quality"*
(authors anonymous during review).

---

## Overview of repo structure

```
src/        Python scripts for phase 1, 2 and 3
chains/     Bash drivers that call src/ with the right flags, grouped by phase
paper/      Scripts that process traces to produce the paper's tables + figures
```

---

## LLM endpoint

> ⚠️ **Important**: the original paper used a proprietary internal LLM
> gateway to run the four benchmarked models. This open-source branch
> ships with an **OpenAI-compatible client as a drop-in replacement**,
> but we have **not** verified the full pipeline against public endpoints.
> You should expect to adapt `src/llm_loader.py` to your own setup.

As shipped, `src/llm_loader.py` uses the `openai` Python package and reads
two env vars:

- `OPENAI_API_KEY` — API key (set to any non-empty string if your endpoint
  doesn't check).
- `OPENAI_BASE_URL` — e.g. `https://api.openai.com/v1`, or any
  OpenAI-compatible endpoint (vLLM, Together, Groq, Fireworks, Ollama's
  OpenAI shim, ...).

Set these in your shell before running anything in `chains/`.

**If your LLM endpoint is NOT OpenAI-compatible**, replace
`src/llm_loader.py` with a thin shim that exposes the same `client` and
`async_client` objects. The pipeline only needs streaming
`chat.completions.create(..., logprobs=True, top_logprobs=TOP_LOGPROBS)`
— adapt accordingly.

---

## Environment set up

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

### Fetching benchmarks

The three benchmarks used in the paper (HumanEval, MBPP+, ClassEval) are
loaded from `datasets/*.jsonl.gz`. The following command creates the
`datasets/` folder and populates it with the downloaded benchmarks:

```bash
bash chains/dataset_download/fetch_datasets.sh
```

Idempotent — skips any dataset already present.

---

## Steps to Run

Three phases (phase 1b optional): **1.** baseline generation → **1b.** split
re-evaluation → **2.** analyzer loop → **3.** hit-rate enrichment.
Each phase is a bash chain under `chains/<N>_*/`.

### Phase 1 — baseline generation (`chains/1_baseline/`)

```bash
bash chains/1_baseline/chain_run_baseline.sh
```

For each (LLM, benchmark) pair, generate one test suite per task at
temperature 0.0 with per-token logprobs (top-20). Writes to
`data/<llm>/<bench>/results.jsonl`.

Edit the `MODELS` array in the chain script with your own model ids —
the four defaults are the model names from the original paper.

### Phase 1b (optional) — split re-evaluation (`chains/1b_split_reeval/`)

Required only for the paper's RQ1 **per-assertion** tables (RQ1 at
suite-level and per-testbody granularity do not need it).

```bash
bash chains/1b_split_reeval/chain_split_reeval.sh
```

For each baseline suite, rewrites every pytest test into N single-assertion
tests and reruns pytest to get per-assertion pass/fail. Writes to
`data/<llm>/<bench>/results_split.jsonl`.

### Phase 2 — analyzer loop (`chains/2_analyzer/`)

For every suite in the parseable-baseline failing cohort (`cohorts/<llm>_<bench>_all.txt`,
generate your own from the Phase 1 output), run the Analyzer under both
token-selection modes (unfiltered vs. filtered):

```bash
bash chains/2_analyzer/chain_rerun_no_stop_on_success.sh    # unfiltered
bash chains/2_analyzer/chain_rerun_righteq.sh               # filtered
```

Both chains run `src/analyzer_loop.py` with `--no-guard --no-stop-on-success
--max-iterations 5`, differing only in `--rank testbody_mean_entropy`
(unfiltered) vs `--rank testbody_mean_entropy_right_eq` (filtered).

Output traces land at:
- `data/<llm>/<bench>/traces/traces_full_K{K}_optout_5iter_nostopos.jsonl`
- `data/<llm>/<bench>/traces/traces_righteq_K{K}_optout_5iter.jsonl`

### Phase 3 — hit-rate enrichment (`chains/3_hit_rate/`)

```bash
bash chains/3_hit_rate/chain_hitrate_noguard.sh
```

For each iteration of each trace, run pytest on the previous iteration's
code to recover the per-testbody FAIL/ERROR set, then attach
`hit_rate` / `n_hits` / `fix_rate` columns per iteration. Writes to
`data/<llm>/<bench>/traces_noguard_hr/<same-filename>`.

---

## Plotting tables and figures

All regeneration scripts live in `paper/`. Each one reads traces from
`data/` and writes either a `.tex` artifact, raw numbers to stdout, or an
audit log into `paper/logs/`.

---

## License

Mozilla Public License 2.0 — see [`LICENSE`](LICENSE).

## Citation

Paper citation will be added after review.
