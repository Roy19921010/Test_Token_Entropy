#!/bin/bash
# Phase 0 (optional, used by RQ2 per-assertion tables) — assertion-level
# re-evaluation.
#
# For each baseline suite in data/<llm>/<bench>/results.jsonl, run
# reevaluate_split.py which:
#   1. splits every pytest test into N single-assertion tests
#   2. reruns pytest to get per-assertion pass/fail
#   3. writes data/<llm>/<bench>/results_split.jsonl with updated exec__* fields
#
# The per-assertion correlation tables (tab:rq2-perassert*) read from
# results_split.jsonl. If you skip this phase, those tables cannot be
# regenerated.
#
# ~1-3 h total across 12 cells, serial.

set -e
cd "$(dirname "$0")/../.."
source "$(dirname "$0")/../../.venv/bin/activate"

LOG=/tmp/split_reeval.log
echo "=== [$(date +%Y-%m-%d\ %H:%M:%S)] split_reeval started ===" > $LOG

run_cell() {
  local DIR="$1"
  echo "" >> $LOG
  echo "--- [$(date +%H:%M:%S)] $DIR ---" >> $LOG
  python src/reevaluate_split.py \
    --in "$DIR/results.jsonl" \
    --out "$DIR/results_split.jsonl" \
    --mode base \
    --resume >> $LOG 2>&1
}

for LLM in qwen gpt_oss glm52 deepseek_v4; do
  for BENCH in humaneval mbpp_plus classeval; do
    run_cell "data/$LLM/$BENCH"
  done
done

echo "" >> $LOG
echo "=== [$(date +%Y-%m-%d\ %H:%M:%S)] split_reeval complete ===" >> $LOG
