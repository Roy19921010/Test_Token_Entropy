"""Radon-based complexity metrics for generated pytest suites.

Computes cyclomatic complexity, cognitive complexity, and Halstead
metrics on the generated_code field of each base-sweep row. Metrics
are aggregated at the file level (per-suite) with `_mean` variants
where meaningful.

Callable module:
    python complexity_metrics.py --in <results.jsonl> --out <same+cx.jsonl>

Programmatic use:
    from complexity_metrics import compute_complexity
    cx = compute_complexity(source_str)      # -> dict of cx__* fields
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

from radon.complexity import cc_visit
from radon.metrics import h_visit, mi_visit
from cognitive_complexity.api import get_cognitive_complexity


def compute_complexity(code: str) -> dict:
    """Return dict of complexity metrics. All None if code fails to parse."""
    if not code or not code.strip():
        return {k: None for k in _CX_KEYS}
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {k: None for k in _CX_KEYS}

    try:
        cc_blocks = cc_visit(code)
    except Exception:
        cc_blocks = []
    cc_vals = [b.complexity for b in cc_blocks]
    cc_total = sum(cc_vals) if cc_vals else 0
    cc_mean = (cc_total / len(cc_vals)) if cc_vals else None
    cc_max = max(cc_vals) if cc_vals else None

    cog_vals = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            try:
                cog_vals.append(get_cognitive_complexity(node))
            except Exception:
                continue
    cog_total = sum(cog_vals) if cog_vals else 0
    cog_mean = (cog_total / len(cog_vals)) if cog_vals else None
    cog_max = max(cog_vals) if cog_vals else None

    try:
        h = h_visit(code).total
        h_volume = h.volume
        h_difficulty = h.difficulty
        h_effort = h.effort
        h_bugs = h.bugs
    except Exception:
        h_volume = h_difficulty = h_effort = h_bugs = None

    try:
        mi = mi_visit(code, multi=False)
    except Exception:
        mi = None

    return {
        "cx__cc_total":       cc_total,
        "cx__cc_mean":        cc_mean,
        "cx__cc_max":         cc_max,
        "cx__cog_total":      cog_total,
        "cx__cog_mean":       cog_mean,
        "cx__cog_max":        cog_max,
        "cx__halstead_volume":     h_volume,
        "cx__halstead_difficulty": h_difficulty,
        "cx__halstead_effort":     h_effort,
        "cx__halstead_bugs":       h_bugs,
        "cx__maintainability_index": mi,
    }


_CX_KEYS = list(compute_complexity("def x():\n    pass\n").keys())


def annotate_file(in_path: Path, out_path: Path, slim: bool = True) -> None:
    """Annotate `in_path` with cx__* fields. If slim=True, keeps only
    task_id + logit__/exec__/smell__/struct__/pos__/mut__ + cx__ columns
    (drops the raw `generated_code`, `generated_text`, `entries` payload
    to keep the file tractable for downstream correlation analysis)."""
    rows = [json.loads(l) for l in in_path.read_text().split("\n") if l.strip()]
    with out_path.open("w") as f:
        for r in rows:
            cx = compute_complexity(r.get("generated_code", "") or "")
            if slim:
                keep = {"task_id", "model", "temperature", "seed", "source_module_name"}
                out = {k: v for k, v in r.items()
                       if k in keep or any(k.startswith(p) for p in
                            ("logit__","exec__","smell__","struct__","pos__","mut__"))}
                out["exec__pytest_stdout_tail"] = None  # drop the stdout string
            else:
                out = dict(r)
            out.update(cx)
            f.write(json.dumps(out) + "\n")
    print(f"[cx] {in_path} -> {out_path}  n={len(rows)}  slim={slim}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    annotate_file(Path(args.inp), Path(args.out))


if __name__ == "__main__":
    main()
