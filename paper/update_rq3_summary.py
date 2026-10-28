"""Compute per-LLM Static-E1 RQ3 summary stats (full-testbody vs right-of-``==``)
and emit the ``tab:rq3-summary`` LaTeX table to stdout + the paper logs.

Layout: two rows per LLM (full, right-of-==), plus an Overall block.

Columns: LLM, Variant, Improved, Reach-100%, TC-fixed, Mean pass rate,
Calls/fix.

* Mean pass rate = arithmetic mean of per-suite pass rates at iteration 5
  across the cohort. The pooled version is shown in the figure.
* All other columns computed under final-iter semantics on the parseable-
  baseline failing cohort (b.n_collected>0, 0<b.n_failed<=b.n_collected).

Trace files used (no-guard, static top-K, 5 iters):
  full:     <dir>/traces_noguard_hr/traces_full_K{K}_optout_5iter_nostopos.jsonl
  right-eq: <dir>/traces_noguard_hr/traces_righteq_K{K}_optout_5iter.jsonl
K is HE=3, MP=2, CE=3 for Qwen/gpt-oss/GLM and HE=2, MP=2, CE=4 for DeepSeek.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "paper" / "logs" / "update_rq3_summary.log"

# (display_llm, bench_tag, dir, K)
# Edit this CELLS list for your (LLM, benchmark) matrix. The 4 LLM ids
# below are the specific model names used in the original paper.
CELLS = [
    ("Qwen3-Coder",  "HE", "data/qwen/humaneval",         3),
    ("Qwen3-Coder",  "MP", "data/qwen/mbpp_plus",         2),
    ("Qwen3-Coder",  "CE", "data/qwen/classeval",         3),
    ("gpt-oss",      "HE", "data/gpt_oss/humaneval",      3),
    ("gpt-oss",      "MP", "data/gpt_oss/mbpp_plus",      2),
    ("gpt-oss",      "CE", "data/gpt_oss/classeval",      3),
    ("GLM-5.2",      "HE", "data/glm52/humaneval",        3),
    ("GLM-5.2",      "MP", "data/glm52/mbpp_plus",        2),
    ("GLM-5.2",      "CE", "data/glm52/classeval",        3),
    ("DeepSeek-V4",  "HE", "data/deepseek_v4/humaneval",  2),
    ("DeepSeek-V4",  "MP", "data/deepseek_v4/mbpp_plus",  2),
    ("DeepSeek-V4",  "CE", "data/deepseek_v4/classeval",  4),
]
LLM_ORDER = ["Qwen3-Coder", "gpt-oss", "GLM-5.2", "DeepSeek-V4"]


def _valid(it):
    return (
        it.get("n_failed") is not None
        and it.get("n_collected", 0) > 0
        and it.get("pass_rate") is not None
        and it.get("n_passed") is not None
    )


def _iter_state(iters, idx):
    """Return (n_passed, n_failed, n_collected, pass_rate) at the latest
    valid iter <= idx. Falls back to iter 0 if nothing valid is found."""
    for it in reversed(iters):
        if it["iter"] <= idx and _valid(it):
            return it["n_passed"], it["n_failed"], it["n_collected"], it["pass_rate"]
    b = iters[0]
    return b.get("n_passed"), b.get("n_failed"), b.get("n_collected"), b.get("pass_rate")


def _load_cohort(path: Path):
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    kept = []
    for r in rows:
        b = r["iterations"][0]
        if not _valid(b):
            continue
        if b["n_failed"] <= 0 or b["n_failed"] > b["n_collected"]:
            continue
        kept.append(r)
    return kept


def _trace_file(d: str, K: int, variant: str) -> Path:
    if variant == "full":
        return ROOT / d / "traces_noguard_hr" / f"traces_full_K{K}_optout_5iter_nostopos.jsonl"
    elif variant == "right-eq":
        return ROOT / d / "traces_noguard_hr" / f"traces_righteq_K{K}_optout_5iter.jsonl"
    else:
        raise ValueError(variant)


def compute_per_llm():
    """Aggregate stats pooled across the 3 benches of each LLM, for both
    variants. Returns ``{(llm, variant): {metrics}}`` plus
    ``{("Overall", variant): {metrics}}``."""
    variants = ("full", "right-eq")
    agg = {(llm, v): {"N": 0, "improved": 0, "reach": 0,
                      "sum_bnf": 0, "sum_fnf": 0,
                      "pr_values_at5": [],
                      "sum_iters": 0}
           for llm in LLM_ORDER for v in variants}
    for llm, bench, d, K in CELLS:
        for v in variants:
            f = _trace_file(d, K, v)
            if not f.exists():
                raise RuntimeError(f"missing trace: {f}")
            kept = _load_cohort(f)
            a = agg[(llm, v)]
            for r in kept:
                iters = r["iterations"]
                _, bnf, _, bpr = _iter_state(iters, 0)
                _, fnf, _, fpr = _iter_state(iters, 5)
                a["N"] += 1
                if fpr is not None and bpr is not None and fpr > bpr + 1e-9:
                    a["improved"] += 1
                if fpr is not None and fpr >= 1.0 - 1e-9:
                    a["reach"] += 1
                a["sum_bnf"] += bnf
                a["sum_fnf"] += fnf
                if fpr is not None:
                    a["pr_values_at5"].append(fpr)
                a["sum_iters"] += r.get("n_iterations_attempted", 0)
    out = {}
    for llm in LLM_ORDER:
        for v in variants:
            a = agg[(llm, v)]
            N = a["N"]
            prs = a["pr_values_at5"]
            out[(llm, v)] = {
                "N": N,
                "improved_pct": a["improved"] / N * 100 if N else 0.0,
                "reach_pct":    a["reach"]    / N * 100 if N else 0.0,
                "tc_pct":       (a["sum_bnf"] - a["sum_fnf"]) / a["sum_bnf"] * 100
                                 if a["sum_bnf"] else 0.0,
                "passrate_pct": sum(prs) / len(prs) * 100 if prs else 0.0,
                "calls_per_fix": a["sum_iters"] / a["improved"]
                                 if a["improved"] else float("inf"),
            }
    for v in variants:
        tot_N = sum(agg[(l, v)]["N"] for l in LLM_ORDER)
        tot_impr = sum(agg[(l, v)]["improved"] for l in LLM_ORDER)
        tot_rch  = sum(agg[(l, v)]["reach"]    for l in LLM_ORDER)
        tot_bnf  = sum(agg[(l, v)]["sum_bnf"]  for l in LLM_ORDER)
        tot_fnf  = sum(agg[(l, v)]["sum_fnf"]  for l in LLM_ORDER)
        all_prs = [p for l in LLM_ORDER for p in agg[(l, v)]["pr_values_at5"]]
        tot_iters = sum(agg[(l, v)]["sum_iters"] for l in LLM_ORDER)
        out[("Overall", v)] = {
            "N": tot_N,
            "improved_pct": tot_impr / tot_N * 100 if tot_N else 0.0,
            "reach_pct":    tot_rch / tot_N * 100 if tot_N else 0.0,
            "tc_pct":       (tot_bnf - tot_fnf) / tot_bnf * 100 if tot_bnf else 0.0,
            "passrate_pct": sum(all_prs) / len(all_prs) * 100 if all_prs else 0.0,
            "calls_per_fix": tot_iters / tot_impr if tot_impr else float("inf"),
        }
    return out


def _ff(v): return "---" if v is None else f"{v:.2f}"
def _pct(v, sign=False):
    if sign:
        return f"$+{v:.1f}\\%$" if v >= 0 else f"${v:.1f}\\%$"
    return f"{v:.1f}\\%"


def build_table(stats, label: str = "tab:rq3-summary") -> str:
    """Build a side-by-side (full vs right-of-==) RQ3 Static-E1 summary table.

    Two rows per LLM: the first is full-testbody ranking, the second is
    right-of-==. An Overall block at the bottom summarizes both variants.
    """
    lines = [
        r"\begin{table}[!tbp]",
        r"\centering\footnotesize",
        r"\setlength{\tabcolsep}{5pt}",
        r"\caption{Per-LLM aggregate RQ3 outcomes under Static top-$K$ on the",
        r"parseable-baseline failing suites (no-guard opt-out prompt;",
        r"final-iteration pass rate --- the pr the loop ends at, consistent",
        r"with oracle-free deployment). Two rows per LLM compare the ranking",
        r"on full-testbody tokens (``full'') against the right-of-\texttt{==}",
        r"denoised variant. \emph{Improved} and \emph{Reach-100\%} are",
        r"fractions of the per-LLM failing-suite cohort; \emph{TC-fixed}",
        r"is the fraction of broken test cases \emph{within} that cohort that",
        r"moved from fail to pass by iteration~$5$ (negative values arise when",
        r"cascading regressions break more previously-passing test cases than",
        r"the loop fixes); \emph{Mean pass rate} is the arithmetic mean of",
        r"per-suite pass rates at iteration~$5$; \emph{Calls/fix} is the mean",
        r"number of LLM calls per successfully improved suite.",
        r"Per-iteration dynamics are shown in Figures~\ref{fig:rq3-reach100}",
        r"and~\ref{fig:rq3-tcs-fixed}.}",
        f"\\label{{{label}}}",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"LLM & Variant & Improved & Reach-100\% & TC-fixed & Mean pass rate & Calls/fix \\",
        r"\midrule",
    ]
    for llm in LLM_ORDER:
        for i, v in enumerate(("full", "right-eq")):
            s = stats[(llm, v)]
            llm_cell = llm if i == 0 else ""
            v_label = r"full" if v == "full" else r"right-of-\texttt{==}"
            lines.append(
                f" {llm_cell} & {v_label} & {_pct(s['improved_pct'])} & "
                f"{_pct(s['reach_pct'])} & {_pct(s['tc_pct'], sign=True)} & "
                f"{_pct(s['passrate_pct'])} & {_ff(s['calls_per_fix'])} \\\\"
            )
        lines.append(r"\cmidrule(l){2-7}" if llm != LLM_ORDER[-1] else r"\midrule")
    for i, v in enumerate(("full", "right-eq")):
        s = stats[("Overall", v)]
        o_cell = r"\emph{Overall}" if i == 0 else ""
        v_label = r"full" if v == "full" else r"right-of-\texttt{==}"
        lines.append(
            f" {o_cell} & {v_label} & \\textbf{{{s['improved_pct']:.1f}\\%}} & "
            f"\\textbf{{{s['reach_pct']:.1f}\\%}} & "
            f"$\\mathbf{{+{s['tc_pct']:.1f}\\%}}$ & "
            f"\\textbf{{{s['passrate_pct']:.1f}\\%}} & "
            f"\\textbf{{{s['calls_per_fix']:.2f}}} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    return "\n".join(lines)


def main():
    log_lines = []
    def emit(s):
        log_lines.append(s)
        print(s)

    emit("Computing per-LLM stats for both variants from no-guard traces ...")
    stats = compute_per_llm()
    emit("")
    emit(f"{'LLM':12s}  {'variant':10s}  {'N':>5s}  {'Improv':>8s}  "
         f"{'Reach100':>9s}  {'TC-fixed':>9s}  {'MeanPR':>7s}  {'Calls/fix':>10s}")
    for llm in LLM_ORDER + ["Overall"]:
        for v in ("full", "right-eq"):
            s = stats[(llm, v)]
            emit(f"{llm:12s}  {v:10s}  {s['N']:>5d}  {s['improved_pct']:>7.1f}%  "
                 f"{s['reach_pct']:>8.1f}%  {s['tc_pct']:>+8.1f}%  "
                 f"{s['passrate_pct']:>6.1f}%  {s['calls_per_fix']:>10.2f}")
    emit("")

    # Emit tab:rq3-summary (side-by-side comparison) to stdout and log.
    emit("")
    emit("=== tab:rq3-summary LaTeX ===")
    emit(build_table(stats, label="tab:rq3-summary"))

    LOG.parent.mkdir(parents=True, exist_ok=True)
    LOG.write_text("\n".join(log_lines) + "\n")
    print(f"Wrote {LOG}", file=sys.stderr)


if __name__ == "__main__":
    main()
