"""Compute per-iteration RQ3 statistics for both the full-testbody and
right-of-``==`` variants under Static-E1.

Emits pgfplots coordinate strings for:
  - TC-fixed (fraction of broken test cases fixed by end of iter i)   -> fig:rq3-re-tcs-fixed
  - Reach-100% (cumulative final-iter semantics)                      -> fig:rq3-re-reach100
  - Cumulative delta pass-rate (pp)                                   -> fig:rq3-re-deltapr
  - Fix rate (net reduction in failing testbodies / K)
  - Mean n_failing at iter start

Final-iter semantics: at iteration i, use the latest valid iteration
state <= i. Rejected rewrites (n_collected==0) fall back to the previous
valid state.

Reads from the hit-rate-enriched trace files:
  data/<llm>/<bench>/traces_noguard_hr/traces_full_K{K}_optout_5iter_nostopos.jsonl
  data/<llm>/<bench>/traces_noguard_hr/traces_righteq_K{K}_optout_5iter.jsonl

Writes:
  paper/logs/compute_per_iter_figures.log   -- stdout mirror
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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
LLM_ORDER = ["Qwen", "gpt-oss", "GLM-5.2", "DeepSeek"]
BENCH_ORDER = ["HE", "MP", "CE"]
VARIANTS = [("full", "traces_full_K{K}_optout_5iter_nostopos.jsonl"),
            ("righteq", "traces_righteq_K{K}_optout_5iter.jsonl")]


def valid(it):
    return (
        it.get("n_failed") is not None
        and it.get("n_collected") is not None
        and it.get("n_collected", 0) > 0
        and it.get("pass_rate") is not None
        and it.get("n_passed") is not None
    )


def iter_state(iters, idx):
    for it in reversed(iters):
        if it["iter"] <= idx and valid(it):
            return it["n_passed"], it["n_failed"], it["n_collected"], it["pass_rate"]
    b = iters[0]
    return b["n_passed"], b["n_failed"], b["n_collected"], b["pass_rate"]


def filter_cohort(rows):
    kept = []
    for r in rows:
        base = r["iterations"][0]
        if not valid(base):
            continue
        if base["n_collected"] <= 0 or base["n_failed"] <= 0:
            continue
        if base["n_failed"] > base["n_collected"]:
            continue
        kept.append(r)
    return kept


def cell_stats(rows, K):
    kept = filter_cohort(rows)
    N = len(kept)
    out = {"N": N, "K": K}
    sum_b_nf = sum(r["iterations"][0]["n_failed"] for r in kept)
    per_iter = {}
    for i in range(1, 6):
        n_reach100 = 0
        sum_f_nf = 0
        sum_delta_pp = 0.0
        sum_fix_count = 0
        sum_n_flagged = 0
        sum_nf_at_start = 0
        for r in kept:
            iters = r["iterations"]
            _, b_nf, _, b_pr = iter_state(iters, 0)
            _, nf_i, _, pr_i = iter_state(iters, i)
            _, nf_prev, _, _ = iter_state(iters, i - 1)
            if pr_i >= 1.0 - 1e-9:
                n_reach100 += 1
            sum_f_nf += nf_i
            sum_delta_pp += (pr_i - b_pr) * 100.0
            sum_fix_count += (nf_prev - nf_i)
            sum_n_flagged += K
            sum_nf_at_start += nf_prev
        per_iter[i] = {
            "reach100_pct": n_reach100 / N * 100.0 if N else 0.0,
            "tc_fixed_pct": (sum_b_nf - sum_f_nf) / sum_b_nf * 100.0 if sum_b_nf > 0 else 0.0,
            "delta_pp":    sum_delta_pp / N if N else 0.0,
            "fix_rate_pct": sum_fix_count / sum_n_flagged * 100.0 if sum_n_flagged > 0 else 0.0,
            "mean_nf_at_start": sum_nf_at_start / N if N else 0.0,
        }
    out["per_iter"] = per_iter
    out["sum_b_nf"] = sum_b_nf
    return out


def main():
    log_lines = []
    def emit(s=""):
        log_lines.append(s)
        print(s)

    cells = {}
    for llm, bench, d, K in CELLS:
        for variant, fname_tmpl in VARIANTS:
            fname = fname_tmpl.format(K=K)
            f = ROOT / d / "traces_noguard_hr" / fname
            if not f.exists():
                emit(f"[MISSING] {f}")
                continue
            rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
            cells[(llm, bench, variant)] = cell_stats(rows, K)

    def emit_coords(metric, variant, fmt="{:.2f}"):
        for bench in BENCH_ORDER:
            emit(f"% {bench}")
            for llm in LLM_ORDER:
                key = (llm, bench, variant)
                if key not in cells:
                    emit(f"  {llm:>8s}: MISSING")
                    continue
                s = cells[key]
                vals = [s["per_iter"][i][metric] for i in range(1, 6)]
                coords = " ".join(f"({i},{fmt.format(v)})" for i, v in enumerate(vals, 1))
                emit(f"  {llm:>8s}: {coords}")
            emit()

    for variant in ("full", "righteq"):
        emit(f"\n{'='*72}")
        emit(f"VARIANT: {variant}")
        emit(f"{'='*72}")
        emit("\n### TC-fixed (%)")
        emit_coords("tc_fixed_pct", variant)
        emit("### Reach-100 (%)")
        emit_coords("reach100_pct", variant)
        emit("### Delta pass-rate (pp)")
        emit_coords("delta_pp", variant)
        emit("### Fix rate (%)")
        emit_coords("fix_rate_pct", variant)
        emit("### Mean n_failing at iter start")
        emit_coords("mean_nf_at_start", variant)

    log_path = ROOT / "paper" / "logs" / "compute_per_iter_figures.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_lines) + "\n")
    print(f"\nWrote {log_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
