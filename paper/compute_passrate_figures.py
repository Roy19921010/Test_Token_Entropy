"""Compute per-iteration pass-rate curves for the RQ3 right-eq ablation.

Produces coordinates for two figures (both appear in paper/section_rq3_righteq.tex):

  (A) fig:rq3-re-passrate-pooled
      Pooled pass rate: ``sum(n_passed_i) / sum(n_collected_i)`` across the
      parseable-baseline failing cohort of each (model, benchmark) cell.
      Weights each pytest-collected test case equally.

  (B) fig:rq3-re-passrate-mean
      Mean per-suite pass rate: ``mean(n_passed_i / n_collected_i)`` across
      the same cohort. Weights each suite equally regardless of its size.

Both figures plot 8 curves per panel (4 LLMs × {full-testbody, right-of-==}),
3 panels per figure (HumanEval / MBPP+ / ClassEval).

Semantics:
  - Final-iter: for iteration i, use the latest valid iteration state <= i.
    A "valid" iteration has n_collected > 0 and all count fields set.
    Rejected rewrites (n_collected == 0) are skipped; the previous valid
    state carries over.
  - Cohort: parseable-baseline failing suites only (baseline iter has
    n_collected > 0, 0 < n_failed <= n_collected).

Inputs: no-guard analyzer traces at
  data/<llm>/<bench>/traces_noguard_hr/
      traces_full_K{K}_optout_5iter_nostopos.jsonl   (full-testbody ranker)
      traces_righteq_K{K}_optout_5iter.jsonl         (right-of-== ranker)

Outputs: writes the raw numbers to paper/logs/compute_passrate_figures.log
and prints to stdout. Re-run after any trace file changes to refresh the
numbers used in the paper's pass-rate figures.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent

# Edit this CELLS list for your (LLM, benchmark) matrix. The 4 LLM ids
# below are the specific model names used in the original paper.
CELLS = [
    ("Qwen",     "HE", "data/qwen/humaneval",         3),
    ("Qwen",     "MP", "data/qwen/mbpp_plus",         2),
    ("Qwen",     "CE", "data/qwen/classeval",         3),
    ("gpt-oss",  "HE", "data/gpt_oss/humaneval",      3),
    ("gpt-oss",  "MP", "data/gpt_oss/mbpp_plus",      2),
    ("gpt-oss",  "CE", "data/gpt_oss/classeval",      3),
    ("GLM-5.2",  "HE", "data/glm52/humaneval",        3),
    ("GLM-5.2",  "MP", "data/glm52/mbpp_plus",        2),
    ("GLM-5.2",  "CE", "data/glm52/classeval",        3),
    ("DeepSeek", "HE", "data/deepseek_v4/humaneval",  2),
    ("DeepSeek", "MP", "data/deepseek_v4/mbpp_plus",  2),
    ("DeepSeek", "CE", "data/deepseek_v4/classeval",  4),
]


def valid(it) -> bool:
    return (
        it.get("n_failed") is not None
        and it.get("n_collected", 0) > 0
        and it.get("pass_rate") is not None
        and it.get("n_passed") is not None
    )


def iter_state(iters, idx) -> tuple:
    """Return (n_passed, n_collected, pass_rate) at the latest valid iter<=idx.
    Falls back to baseline iter 0 if nothing valid is found."""
    for it in reversed(iters):
        if it["iter"] <= idx and valid(it):
            return it["n_passed"], it["n_collected"], it["pass_rate"]
    b = iters[0]
    return b.get("n_passed"), b.get("n_collected"), b.get("pass_rate")


def load_cohort(path: Path) -> list[dict]:
    """Return list of parseable-baseline failing-suite rows."""
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    kept = []
    for r in rows:
        b = r["iterations"][0]
        if not valid(b):
            continue
        if b["n_failed"] <= 0 or b["n_failed"] > b["n_collected"]:
            continue
        kept.append(r)
    return kept


def compute_cell(trace_path: Path):
    """Return per-iter (0..5) list of (pooled_pr, mean_pr) tuples."""
    kept = load_cohort(trace_path)
    N = len(kept)
    results = []
    for i in range(0, 6):
        sum_pass = 0
        sum_coll = 0
        per_suite_pr = []
        for r in kept:
            npa, ncl, pr = iter_state(r["iterations"], i)
            if npa is None or ncl is None or ncl <= 0 or pr is None:
                continue
            sum_pass += npa
            sum_coll += ncl
            per_suite_pr.append(pr)
        pooled = (sum_pass / sum_coll) * 100 if sum_coll else 0.0
        meanpr = (mean(per_suite_pr) * 100) if per_suite_pr else 0.0
        results.append((pooled, meanpr))
    return N, results


def main():
    log_lines = []

    def emit(s):
        log_lines.append(s)
        print(s)

    emit(f"{'LLM':9s} {'B':3s}  {'variant':8s}  {'N':>4s}  "
         f"{'pooled pr @ iter 0..5':>42s}  |  {'mean pr @ iter 0..5':>42s}")
    emit("-" * 125)
    for llm, bench, d, K in CELLS:
        for variant, fname in (
            ("full",     f"traces_full_K{K}_optout_5iter_nostopos.jsonl"),
            ("right-eq", f"traces_righteq_K{K}_optout_5iter.jsonl"),
        ):
            f = ROOT / d / "traces_noguard_hr" / fname
            if not f.exists():
                emit(f"{llm:9s} {bench:3s}  {variant:8s}  MISSING {fname}")
                continue
            N, results = compute_cell(f)
            pooled = "  ".join(f"{p:5.2f}" for p, _ in results)
            meanpr = "  ".join(f"{m:5.2f}" for _, m in results)
            emit(f"{llm:9s} {bench:3s}  {variant:8s}  {N:>4d}  {pooled}  |  {meanpr}")
        emit("")

    out_path = ROOT / "paper" / "logs" / "compute_passrate_figures.log"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nWrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
