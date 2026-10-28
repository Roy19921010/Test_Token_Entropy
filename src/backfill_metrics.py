"""Recompute `logit__*` metrics from stored per-token entries.

Use this after adding new fields to `metrics.aggregate_metrics` — reads the
existing `outputs_bench/results.jsonl`, recomputes `logit__*` from the
`entries` field on each row, writes the row back. Also re-derives
`summary.csv`.

Run:
    python backfill_metrics.py
    python backfill_metrics.py --in-dir outputs_bench_run3_hard_32rows
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from metrics import aggregate_metrics
from test_smells import test_structure, test_smells
from token_positions import (
    high_entropy_locations, per_assertion_entropy, position_summary,
)
from run_bench import CSV_SKIP_KEYS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="outputs_bench")
    args = ap.parse_args()

    d = Path(args.in_dir)
    if not d.is_absolute():
        d = Path(__file__).resolve().parent.parent / d
    jsonl_path = d / "results.jsonl"
    if not jsonl_path.exists():
        raise SystemExit(f"missing: {jsonl_path}")

    rows = [json.loads(l) for l in jsonl_path.read_text().split("\n") if l.strip()]
    print(f"[backfill] {len(rows)} rows to recompute")

    for r in rows:
        entries = r.get("entries") or []
        code = r.get("generated_code", "")
        new_logit = aggregate_metrics(entries)
        new_struct = test_structure(code)
        new_smells = test_smells(code)
        new_pos = position_summary(entries)
        new_per_assert = per_assertion_entropy(entries)
        new_hot = high_entropy_locations(entries, top_k=10)
        for k in list(r.keys()):
            if k.startswith(("logit__", "struct__", "smell__", "pos__")):
                del r[k]
        for k, v in new_logit.items():
            r[f"logit__{k}"] = v
        for k, v in new_struct.items():
            r[f"struct__{k}"] = v
        for k, v in new_smells.items():
            r[f"smell__{k}"] = v
        for k, v in new_pos.items():
            r[f"pos__{k}"] = v
        r["per_assertion_entropy"] = new_per_assert
        r["high_entropy_tokens"] = new_hot

    with jsonl_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    csv_path = d / "summary.csv"
    all_keys: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen and k not in CSV_SKIP_KEYS:
                seen.add(k)
                all_keys.append(k)
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[backfill] wrote {jsonl_path}")
    print(f"[backfill] wrote {csv_path}")


if __name__ == "__main__":
    main()
