"""Re-evaluate a base sweep or analyzer trace file under assertion splitting.

For each row (base sweep) or iteration (analyzer trace), we:
    1. Take the generated_code / iteration.code
    2. Split it with split_code() → one assertion per test
    3. Re-run pytest against the reference module
    4. Update exec__* fields (or iteration fields) with the new counts

Preserves everything else in the row (entropy metrics, position info, etc.)
because splitting is purely a post-hoc evaluation transform — it does not
touch the token stream or the AST-tagged positions.

Outputs go to a parallel directory (e.g. outputs_bench_split/) so originals
are preserved.

Usage:
    python reevaluate_split.py --in outputs_bench/results.jsonl \\
                                --out outputs_bench_split/results.jsonl \\
                                --bench humaneval
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from tqdm import tqdm

from bench_pipeline import evaluate_generation
from split_assertions import split_code
from tasks import load_bench


def reevaluate_base_row(row: dict, task: dict, timeout_sec: int = 60) -> dict:
    """Re-run pytest with the split code, update exec__* fields in place.

    Preserves logit__*, pos__*, per_assertion_entropy, entries, etc.
    Adds split__* fields with the splitting stats.
    """
    code = row.get("generated_code", "")
    if not code:
        return row
    new_code, stats = split_code(code)
    exec_new = evaluate_generation(task, new_code, timeout_sec=timeout_sec)

    updated = dict(row)  # shallow copy
    for k, v in exec_new.items():
        updated[f"exec__{k}"] = v
    for k, v in stats.items():
        updated[f"split__{k}"] = v
    # Store the split code alongside the original for reproducibility
    updated["generated_code_split"] = new_code
    return updated


def reevaluate_analyzer_trace(trace: dict, task: dict, timeout_sec: int = 60) -> dict:
    """Re-run pytest under splitting for every iteration in a trace.

    Updates iterations[i].n_collected, n_passed, n_failed, pass_rate. Also
    updates baseline_pass_rate and final_pass_rate to reflect the split
    baseline and (last-accepted) split final. Preserves everything else.
    Note: this recomputes accept/reject with the *split* counts — a repair
    that was accepted at test-level might now be rejected at assertion-level
    (or vice versa), so we recompute the guard here for consistency.
    """
    updated = json.loads(json.dumps(trace))  # deep copy
    iterations = updated.get("iterations", [])
    if not iterations:
        return updated

    # Re-evaluate every iteration
    for it in iterations:
        code = it.get("code", "")
        if not code:
            # Malformed regen — set to zeros to mark as failed
            it["n_collected"] = 0
            it["n_passed"] = 0
            it["n_failed"] = 0
            it["pass_rate"] = None
            it["split_stats"] = {}
            continue
        new_code, stats = split_code(code)
        exec_new = evaluate_generation(task, new_code, timeout_sec=timeout_sec)
        it["n_collected"] = exec_new.get("n_collected", 0)
        it["n_passed"] = exec_new.get("n_passed", 0)
        it["n_failed"] = exec_new.get("n_failed", 0)
        it["pass_rate"] = exec_new.get("pass_rate")
        it["split_stats"] = stats

    # Recompute baseline_pass_rate and final_pass_rate under split counts
    updated["baseline_pass_rate"] = iterations[0].get("pass_rate") or 0.0
    # Re-apply accept/reject guard on split counts
    baseline_pr = updated["baseline_pass_rate"]
    accepted_iters = [it for it in iterations if it.get("accepted", True)]
    # For analyzer traces we already recorded which iterations were accepted at
    # test-level. Under split-level evaluation, an iteration should still be
    # rejected if pass_rate dropped below baseline. Recompute:
    current_pr = baseline_pr
    for it in iterations[1:]:
        it_pr = it.get("pass_rate")
        if it_pr is None:
            it["accepted_split"] = False
            it["reject_reason_split"] = "no_tests_collected"
            continue
        if it_pr < current_pr:
            it["accepted_split"] = False
            it["reject_reason_split"] = f"regressed({it_pr:.3f}<{current_pr:.3f})"
        else:
            it["accepted_split"] = True
            it["reject_reason_split"] = None
            current_pr = it_pr
    updated["final_pass_rate"] = current_pr
    updated["improved"] = current_pr > baseline_pr
    updated["delta_pass_rate"] = current_pr - baseline_pr
    return updated


def load_tasks_for_bench(bench: str, task_ids: set[str]) -> dict:
    """Load task dicts for the given task_ids by dispatching to the right
    benchmark loader. Assumes all task_ids share the same benchmark prefix."""
    # tasks.load_bench doesn't do a full-limit dispatch for a subset; use
    # limit large enough to cover all.
    tasks = load_bench(bench, limit=10_000)
    return {t["task_id"]: t for t in tasks if t["task_id"] in task_ids}


def _detect_bench_from_task_id(task_id: str) -> str:
    prefix = task_id.split("/")[0]
    return {
        "HumanEval": "humaneval",
        "Mbpp": "mbpp_plus",
        "BigCodeBench": "bigcodebench",
        "ClassEval": "classeval",
        "CoderEval": "codereval",
        "APPS": "apps_intro",
    }[prefix]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True,
                    help="input JSONL (base results.jsonl or analyzer traces)")
    ap.add_argument("--out", required=True, help="output JSONL")
    ap.add_argument("--mode", choices=("base", "traces"), required=True,
                    help="'base' for run_bench results.jsonl, 'traces' for analyzer_loop output")
    ap.add_argument("--timeout-sec", type=int, default=60)
    ap.add_argument("--resume", action="store_true",
                    help="skip task_ids already present in --out")
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = [json.loads(l) for l in in_path.read_text().split("\n") if l.strip()]
    print(f"[reeval] loaded {len(rows)} rows from {in_path}")

    # Load task dicts. Assume the file is single-benchmark.
    task_ids = set(r["task_id"] for r in rows)
    bench = _detect_bench_from_task_id(next(iter(task_ids)))
    print(f"[reeval] detected benchmark: {bench}")
    tasks = load_tasks_for_bench(bench, task_ids)
    print(f"[reeval] resolved {len(tasks)}/{len(task_ids)} task dicts")

    done: set[tuple[str, ...]] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().split("\n"):
            try:
                r = json.loads(line)
                if args.mode == "base":
                    done.add((r["task_id"], r.get("temperature"), r.get("seed")))
                else:
                    done.add((r["task_id"],))
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"[reeval] resume: skipping {len(done)} completed entries")

    open_mode = "a" if args.resume else "w"
    with out_path.open(open_mode) as out_f:
        for row in tqdm(rows, desc="reeval", dynamic_ncols=True):
            if args.mode == "base":
                key = (row["task_id"], row.get("temperature"), row.get("seed"))
            else:
                key = (row["task_id"],)
            if key in done:
                continue
            task = tasks.get(row["task_id"])
            if task is None:
                print(f"[reeval] SKIP {row['task_id']} — task dict missing", file=sys.stderr)
                continue
            try:
                if args.mode == "base":
                    updated = reevaluate_base_row(row, task, timeout_sec=args.timeout_sec)
                else:
                    updated = reevaluate_analyzer_trace(row, task, timeout_sec=args.timeout_sec)
            except Exception as e:
                print(f"[reeval] ERROR {row['task_id']}: {e!r}", file=sys.stderr)
                continue
            out_f.write(json.dumps(updated, ensure_ascii=False) + "\n")
            out_f.flush()
    print(f"[reeval] wrote {out_path}")


if __name__ == "__main__":
    main()
