"""Compute testbody-level hit rate for E1/E2 analyzer traces.

For each iteration i (>=1), we:
  1. Take iteration (i-1)'s code (the state the analyzer saw)
  2. Run pytest with -rfE to get per-testbody FAILED/ERROR lines
  3. Parse the failing testbody names
  4. Extract the K flagged testbody names from iterations[i].hot_assertions[*].test_name
  5. hit_rate = |flagged_testbodies ∩ failing_testbodies_at_(i-1)| / K

Also records:
  - n_failing_at_iter_start: |failing_testbodies_at_(i-1)|
  - k_used: K = len(hot_assertions) (or top_k_used field)
  - n_hits: |flagged ∩ failing|
  - fix_rate: for accepted iterations, (n_failing_at_(i-1) - n_failing_at_i) / K

Output: writes to a new file with the trace fields plus per-iter hit_rate,
n_hits, n_failing_at_iter_start, fix_rate.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from tqdm import tqdm

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from tasks import load_bench

# Match: "FAILED test_generated.py::test_foo - ..." or "ERROR test_generated.py::test_bar - ..."
_FAIL_TEST_RE = re.compile(r"^(?:FAILED|ERROR)\s+test_generated\.py::([^\s]+)", re.MULTILINE)


def _run_pytest_get_failing(task: dict, code: str, timeout_sec: int = 60) -> set[str]:
    """Run pytest on `code` and return the set of failing testbody names.

    Returns empty set if code doesn't parse. Testbody names are extracted from
    pytest's `-rfE` per-test failure summary.
    """
    if not code:
        return set()
    try:
        import ast
        ast.parse(code)
    except SyntaxError:
        return set()
    source_module = task["source_module_name"]
    with tempfile.TemporaryDirectory(prefix="p3hitrate_") as tmp:
        wd = Path(tmp)
        (wd / f"{source_module}.py").write_text(task["source_code"])
        (wd / "test_generated.py").write_text(code)
        cmd = [
            sys.executable, "-m", "pytest",
            "test_generated.py",
            "-q", "--tb=no", "--no-header",
            "-rfE",
            "--disable-warnings",
        ]
        try:
            proc = subprocess.run(
                cmd, cwd=wd, capture_output=True, text=True,
                timeout=timeout_sec,
            )
            stdout = proc.stdout or ""
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or "")
    # Extract test names, strip parametrization suffix
    names = set()
    for m in _FAIL_TEST_RE.findall(stdout):
        # Strip parametrization e.g. test_foo[case1] -> test_foo
        base = m.split("[")[0]
        names.add(base)
    return names


def compute_hit_rate_for_trace(trace: dict, task: dict, timeout_sec: int = 60) -> dict:
    """Augment each iteration with hit_rate, n_hits, n_failing_at_iter_start."""
    out = json.loads(json.dumps(trace))  # deep copy
    iterations = out.get("iterations", [])
    if len(iterations) < 2:
        # No analyzer iters to score
        return out

    # For each iter i>=1, we need the failing testbodies as-of iter (i-1)
    # Cache: we run pytest once per unique iter code
    failing_at = {}  # iter_idx -> set of failing test names

    for i in range(len(iterations)):
        it = iterations[i]
        code = it.get("code", "")
        if not code:
            failing_at[i] = None
        else:
            failing_at[i] = _run_pytest_get_failing(task, code, timeout_sec)

    # Now for each analyzer iter (i>=1), compute hit rate against iter (i-1)
    for i in range(1, len(iterations)):
        it = iterations[i]
        prev_failing = failing_at.get(i - 1)
        # Skip exception iters — no hot_assertions
        if it.get("reject_reason") == "exception":
            continue
        hot = it.get("hot_assertions", []) or []
        flagged_names = set()
        for ha in hot:
            name = ha.get("test_name")
            if name:
                flagged_names.add(name)
        k = it.get("top_k_used") or len(flagged_names)
        if k == 0:
            # E2 K1=0 iter — no meaningful hit rate
            it["n_failing_at_iter_start"] = len(prev_failing) if prev_failing is not None else None
            it["n_hits"] = 0
            it["hit_rate"] = None
            continue
        if prev_failing is None:
            it["hit_rate"] = None
            it["n_hits"] = None
            it["n_failing_at_iter_start"] = None
            continue
        hits = flagged_names & prev_failing
        it["n_hits"] = len(hits)
        it["hit_rate"] = len(hits) / k
        it["n_failing_at_iter_start"] = len(prev_failing)
        # Fix rate: |failing at i-1| - |failing at i|, over K
        cur_failing = failing_at.get(i)
        if cur_failing is not None:
            fixed = len(prev_failing) - len(cur_failing)
            it["n_fixed_this_iter"] = max(0, fixed)
            it["fix_rate"] = max(0, fixed) / k
        else:
            it["n_fixed_this_iter"] = None
            it["fix_rate"] = None

    # Also store the number of failing testbodies at baseline (iter 0)
    baseline_failing = failing_at.get(0)
    if baseline_failing is not None:
        out["n_failing_testbodies_baseline"] = len(baseline_failing)
    return out


def load_tasks_by_bench(task_ids: set[str]) -> dict:
    """Load task dicts for the given ids by inferring bench prefix."""
    tasks = {}
    prefixes = {tid.split("/")[0] for tid in task_ids}
    bench_map = {"HumanEval": "humaneval", "Mbpp": "mbpp_plus", "ClassEval": "classeval"}
    for pfx in prefixes:
        bench = bench_map.get(pfx)
        if bench is None:
            continue
        for t in load_bench(bench, limit=10_000):
            if t["task_id"] in task_ids:
                tasks[t["task_id"]] = t
    return tasks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", required=True, help="input analyzer trace JSONL")
    ap.add_argument("--out", required=True, help="output JSONL with hit_rate fields")
    ap.add_argument("--timeout-sec", type=int, default=60)
    ap.add_argument("--resume", action="store_true",
                    help="skip task_ids already present in --out")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    traces = [json.loads(l) for l in in_path.read_text().split("\n") if l.strip()]
    if args.limit:
        traces = traces[:args.limit]

    done_ids: set[str] = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text().split("\n"):
            try:
                t = json.loads(line)
                done_ids.add(t["task_id"])
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"[hitrate] resume: skipping {len(done_ids)} completed", flush=True)
        traces = [t for t in traces if t["task_id"] not in done_ids]

    if not traces:
        print(f"[hitrate] nothing to do")
        return

    task_ids = {t["task_id"] for t in traces}
    tasks_by_id = load_tasks_by_bench(task_ids)

    open_mode = "a" if args.resume else "w"
    with out_path.open(open_mode) as f:
        bar = tqdm(traces, desc="hit_rate", unit="trace", dynamic_ncols=True)
        for tr in bar:
            tid = tr["task_id"]
            task = tasks_by_id.get(tid)
            if task is None:
                # Cannot evaluate — write trace as-is
                f.write(json.dumps(tr, ensure_ascii=False) + "\n")
                f.flush()
                continue
            t0 = time.monotonic()
            out = compute_hit_rate_for_trace(tr, task, timeout_sec=args.timeout_sec)
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
            f.flush()
            elapsed = time.monotonic() - t0
            bar.set_postfix_str(f"{elapsed:.1f}s")

    print(f"[hitrate] wrote {out_path}")


if __name__ == "__main__":
    main()
