#!/usr/bin/env bash
# Fetch and normalize the three benchmark datasets into datasets/*.jsonl.gz.
# Idempotent: skips files that already exist. Run once before Phase 1.
#
# Prereqs: python 3.12 venv activated (uv pip install evalplus happens here
# as a one-shot — it's not a pipeline runtime dep).

set -euo pipefail
cd "$(dirname "$0")/../.."

mkdir -p datasets

# ----- HumanEval -----
if [ ! -f datasets/human-eval.jsonl.gz ]; then
  echo "[fetch] HumanEval"
  curl -sL -o datasets/human-eval.jsonl.gz \
    https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
fi

# ----- MBPP+ -----
if [ ! -f datasets/mbpp-plus.jsonl.gz ]; then
  echo "[fetch] MBPP+ (via evalplus)"
  uv pip install --quiet evalplus
  python -c "
import json, gzip
from evalplus.data import get_mbpp_plus
# default=str coerces the few test-input values that aren't JSON-native
# (complex numbers in a handful of MBPP+ test cases).
with gzip.open('datasets/mbpp-plus.jsonl.gz', 'wt') as f:
    for v in get_mbpp_plus().values():
        f.write(json.dumps(v, default=str) + '\n')
"
  # evalplus is only needed for this one-shot fetch; drop it so it doesn't
  # linger as a runtime dependency.
  uv pip uninstall --quiet evalplus || true
fi

# ----- ClassEval -----
if [ ! -f datasets/classeval.jsonl.gz ]; then
  echo "[fetch] ClassEval"
  TMP=$(mktemp --suffix=.json)
  curl -sL -o "$TMP" \
    https://raw.githubusercontent.com/FudanSELab/ClassEval/master/data/ClassEval_data.json
  python -c "
import json, gzip
rows = json.load(open('$TMP'))
with gzip.open('datasets/classeval.jsonl.gz', 'wt') as f:
    for r in rows:
        f.write(json.dumps(r) + '\n')
"
  rm -f "$TMP"
fi

echo "[fetch] done:"
for f in datasets/human-eval.jsonl.gz datasets/mbpp-plus.jsonl.gz datasets/classeval.jsonl.gz; do
  echo "  $f  $(zcat "$f" | wc -l) entries"
done
