"""Sweep the benchmark: N tasks × M temperatures × K samples → one JSONL row each.

Row schema (one JSON object per line in outputs_bench/results.jsonl):
    task_id, model, temperature, seed, finish_reason,
    logit__perplexity, logit__max_per_token_ppl, logit__mean_entropy,
    logit__max_entropy, logit__entropy_p95, logit__n_high_entropy_spikes,
    logit__num_tokens,
    exec__parses, exec__n_collected, exec__n_passed, exec__n_failed,
    exec__n_errored, exec__statement_coverage, exec__branch_coverage,
    exec__pytest_returncode, exec__pytest_timed_out,
    exec__pytest_stdout_tail, exec__pytest_stderr_tail,
    generated_text, generated_code, entries

Also writes a compact `summary.csv` (everything except generated_text / entries)
for quick correlation analysis in pandas.

Run:
    python run_bench.py                       # tiny smoke run (see DEFAULTS)
    python run_bench.py --n-tasks 20 --samples 5 --temps 0.2,0.7,1.0
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

from bench_pipeline import run_one
from llm_loader import TOP_LOGPROBS, MODELS, client
from tasks import load_bench


DEFAULT_MODEL = MODELS[0]
CSV_SKIP_KEYS = {"generated_text", "generated_code", "entries",
                 "exec__pytest_stdout_tail", "exec__pytest_stderr_tail",
                 "mut__per_op_kill_rate",
                 "per_assertion_entropy", "high_entropy_tokens"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--bench", default="humaneval",
                    choices=("humaneval", "mbpp_plus", "classeval"),
                    help="which benchmark to load tasks from")
    ap.add_argument("--n-tasks", type=int, default=3)
    ap.add_argument("--task-ids", default=None,
                    help="comma-separated task IDs (e.g. '50,83,100'); overrides --n-tasks. "
                         "Bare numbers get prefixed with the benchmark name (HumanEval/N or Mbpp/N).")
    ap.add_argument("--samples", type=int, default=2, help="samples per (task, temp)")
    ap.add_argument("--temps", default="0.2,0.7",
                    help="comma-separated temperatures")
    ap.add_argument("--top-logprobs", type=int, default=TOP_LOGPROBS)
    ap.add_argument("--timeout-sec", type=int, default=60)
    ap.add_argument("--out-dir", default="outputs_bench")
    ap.add_argument("--resume", action="store_true",
                    help="skip (task_id, temp, seed) triples already in results.jsonl")
    ap.add_argument("--mutation", action="store_true",
                    help="run mutation testing per row (slow — ~30-60s extra per row)")
    ap.add_argument("--mutation-max", type=int, default=30,
                    help="max mutants per row when --mutation is on")
    ap.add_argument("--disable-thinking", action="store_true",
                    help="pass chat_template_kwargs.enable_thinking=False to the "
                         "model. Turns off reasoning-mode output on models that "
                         "support it (e.g. GLM-5.2, Qwen3-32B).")
    args = ap.parse_args()

    _REPO_ROOT = Path(__file__).resolve().parent.parent
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.jsonl"
    csv_path = out_dir / "summary.csv"

    done: set[tuple[str, float, int]] = set()
    if args.resume and results_path.exists():
        for line in results_path.read_text().split("\n"):
            try:
                row = json.loads(line)
                done.add((row["task_id"], row["temperature"], row["seed"]))
            except (json.JSONDecodeError, KeyError):
                pass
        print(f"[bench] resume: skipping {len(done)} completed rows")

    temps = [float(t) for t in args.temps.split(",")]
    prefix = {"humaneval": "HumanEval", "mbpp_plus": "Mbpp",
              "bigcodebench": "BigCodeBench",
              "classeval": "ClassEval", "codereval": "CoderEval",
              "apps_intro": "APPS"}[args.bench]
    if args.task_ids:
        ids = [f"{prefix}/{s.strip()}" if s.strip().isdigit() else s.strip()
               for s in args.task_ids.split(",")]
        tasks = load_bench(args.bench, task_ids=ids)
    else:
        tasks = load_bench(args.bench, limit=args.n_tasks)
    print(f"[bench] {len(tasks)} tasks × {len(temps)} temps × {args.samples} samples "
          f"= {len(tasks) * len(temps) * args.samples} runs")

    # `client` imported from llm_loader; replace llm_loader.py if your
    # endpoint isn't OpenAI-compatible.

    plan = [
        (task, temp, 1000 + k)
        for task in tasks
        for temp in temps
        for k in range(args.samples)
    ]
    pending = [(t, tp, s) for (t, tp, s) in plan if (t["task_id"], tp, s) not in done]

    total_start = time.monotonic()
    bar = tqdm(pending, desc="bench", unit="row", dynamic_ncols=True)
    for task, temp, seed in bar:
        t0 = time.monotonic()
        try:
            row = run_one(
                task, client=client, model=args.model,
                temperature=temp, seed=seed,
                top_logprobs=args.top_logprobs,
                timeout_sec=args.timeout_sec,
                run_mutation=args.mutation,
                mutation_max_mutants=args.mutation_max,
                disable_thinking=args.disable_thinking,
            )
        except Exception as e:
            tqdm.write(f"[bench] ERROR {task['task_id']} t={temp} s={seed}: {e!r}",
                       file=sys.stderr)
            continue
        elapsed = time.monotonic() - t0
        _append_jsonl(results_path, row)
        cov = row.get("exec__statement_coverage")
        ppl = row.get("logit__perplexity")
        pr = row.get("exec__pass_rate")
        mut = row.get("mut__score")
        extras = ""
        if pr is not None:
            extras += f" pass={pr:.2f}"
        if mut is not None:
            extras += f" mut={mut:.2f}"
        tqdm.write(f"[bench] {task['task_id']} t={temp} s={seed} "
                   f"parses={row['exec__parses']} "
                   f"cov={cov if cov is None else f'{cov:.1f}%'} "
                   f"ppl={ppl if ppl is None else f'{ppl:.3f}'}"
                   f"{extras} ({elapsed:.1f}s)")
        bar.set_postfix_str(f"last={elapsed:.0f}s")

    _write_summary_csv(results_path, csv_path)
    print(f"[bench] wrote {results_path}")
    print(f"[bench] wrote {csv_path}")
    print(f"[bench] total {time.monotonic() - total_start:.1f}s")


def _append_jsonl(path: Path, row: dict) -> None:
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_summary_csv(jsonl_path: Path, csv_path: Path) -> None:
    """Re-derive summary.csv from the JSONL so resume runs stay consistent.

    Streaming impl: never materialize the whole file in memory. The jsonl can
    be ~1 GB (full top-20 logprob entries per token), and reading it all at
    once has OOM-killed sweeps on WSL2.
    """
    if not jsonl_path.exists():
        return
    # First pass: just read the first row to lock the column set.
    with jsonl_path.open() as f:
        first = None
        for line in f:
            s = line.strip()
            if s:
                first = json.loads(s)
                break
    if first is None:
        return
    keys = [k for k in first.keys() if k not in CSV_SKIP_KEYS]
    # Second pass: stream row-by-row.
    with jsonl_path.open() as fin, csv_path.open("w", newline="") as fout:
        w = csv.DictWriter(fout, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for line in fin:
            s = line.strip()
            if not s:
                continue
            w.writerow(json.loads(s))


if __name__ == "__main__":
    main()
